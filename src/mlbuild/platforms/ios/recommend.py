"""
mlbuild.platforms.ios.recommend

Single-run recommendation engine for iOS benchmark runs.

Responsibilities:
- Synthesize delegate status, speedup, stability, thermal, consistency,
  and simulator state into one human-readable, actionable recommendation
- Provide a RecommendationResult consumed by device.py and the CLI
- Never make idb calls — pure synthesis over already-collected data

Design rules:
- One recommendation per run. Not a list.
- Plain language. No CoreML jargon. No internal enum values surfaced.
- Every recommendation ends with an action or decision.
- Priority order: safety > correctness > performance > noise
    1. INCONSISTENT delegate  → exclude immediately, reason explained
    2. FALLBACK delegate      → exclude, no benefit
    3. SKIPPED delegate       → simulator notice, not an error
    4. Thermal instability    → warn, rerun before trusting results
       (real device only — simulator never fires this path)
    5. Unreliable stability   → warn, rerun recommended
    6. Noisy but valid        → note variance, proceed with caution
    7. Stable + fast          → recommend delegate
    8. Stable + marginal      → note marginal gain, user decides
    9. CPU only               → record baseline, no delegate available
- Simulator notice appended to every recommendation when is_simulator=True.
- Structured JSON logging for CI traceability.

Key differences from Android:
- Consumes iOSBuildView instead of AndroidBuildView.
- No emulator check — simulator check replaces it. Different semantics:
  emulator = non-representative hardware. simulator = no ANE, CPU/GPU only.
  Simulator runs are valid CPU/GPU benchmarks, just not ANE benchmarks.
- Thermal caveat uses IOSThermalState strings, not delta_c / drift_pct.
- SKIPPED status handled explicitly — ANE on simulator is structural,
  not a failure. Surfaced in status table, not as a recommendation.
- compute_units_used surfaced in fallback message for clarity.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Optional

from mlbuild.platforms.ios.delegate import DelegateStatus
from mlbuild.platforms.ios.stability import StabilityReport, StabilityBand
from mlbuild.platforms.ios.result import iOSBuildView


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DEFAULT_SPEEDUP_MEANINGFUL    = 2.0   # ANE is typically 2x+ over CPU
DEFAULT_SPEEDUP_MARGINAL_LOW  = 1.2
DEFAULT_RERUN_SCORE_THRESHOLD = 0.5

DEBUG = os.getenv("MLBUILD_DEBUG") == "1"

_SIMULATOR_NOTICE = (
    "These are simulator results — ANE is unavailable. "
    "Connect a real device for ANE benchmarks."
)


# ---------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------

_logger_lock:     threading.Lock           = threading.Lock()
_logger_instance: Optional[logging.Logger] = None


def _get_logger() -> logging.Logger:
    global _logger_instance
    with _logger_lock:
        if _logger_instance is None:
            logger = logging.getLogger("mlbuild.ios.recommend")
            logger.propagate = False
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter(
                '{"time": "%(asctime)s", "level": "%(levelname)s", '
                '"logger": "%(name)s", "msg": %(message)s}',
                datefmt="%Y-%m-%dT%H:%M:%SZ",
            ))
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG if DEBUG else logging.WARNING)
            _logger_instance = logger
    return _logger_instance


def _log(
    msg:     str,
    level:   str = "INFO",
    context: Optional[dict] = None,
) -> None:
    logger = _get_logger()
    payload = {"msg": msg}
    if context:
        payload.update(context)
    try:
        getattr(logger, level.lower(), logger.info)(
            json.dumps(payload, ensure_ascii=False, default=str)
        )
    except Exception as exc:
        logger.error(json.dumps({"msg": f"Logging failed: {exc}"}))


# ---------------------------------------------------------------------
# Recommendation Enums and Data
# ---------------------------------------------------------------------

class RecommendationKind(str, Enum):
    USE_DELEGATE     = "use_delegate"
    USE_CPU          = "use_cpu"
    RERUN            = "rerun"
    EXCLUDE_DELEGATE = "exclude_delegate"


@dataclass(frozen=True)
class RecommendationResult:
    kind:       RecommendationKind
    message:    str
    delegate:   str
    speedup:    Optional[float]
    confidence: str
    caveat:     Optional[str] = None

    def __str__(self) -> str:
        base = f"[{self.kind.value.upper()}] {self.message}"
        if self.caveat:
            base += f" | Note: {self.caveat}"
        return base


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _is_valid(x: Optional[float]) -> bool:
    return x is not None and isfinite(x)


def _speedup_str(speedup: Optional[float]) -> str:
    return f"{speedup:.1f}x" if _is_valid(speedup) else "unknown"


def _confidence(stability: StabilityReport, speedup: Optional[float]) -> str:
    try:
        rerun = getattr(stability, "rerun_recommendation_score", 0.0)
        if not _is_valid(rerun) or rerun >= DEFAULT_RERUN_SCORE_THRESHOLD:
            return "low"
        band = stability.stability_band or StabilityBand.UNRELIABLE
        if band == StabilityBand.UNRELIABLE:
            return "low"
        if band == StabilityBand.NOISY or (speedup and speedup < DEFAULT_SPEEDUP_MEANINGFUL):
            return "medium"
        return "high"
    except Exception:
        return "low"


def _thermal_caveat(stability: StabilityReport, is_simulator: bool) -> Optional[str]:
    """
    iOS thermal caveat uses discrete state strings — no delta_c.
    Simulator runs never fire this path.
    """
    if is_simulator:
        return None

    if not getattr(stability, "thermally_unstable", False):
        return None

    pre  = getattr(stability, "thermal_state_pre",  None)
    post = getattr(stability, "thermal_state_post", None)
    drift = getattr(stability, "latency_drift_pct", None)

    parts = []
    if pre and post:
        parts.append(f"thermal state: {pre.value} → {post.value}")
    elif post:
        parts.append(f"thermal state post-run: {post.value}")

    if _is_valid(drift):
        parts.append(f"latency drifted {drift:.0%}")

    detail = ", ".join(parts) if parts else "thermal throttling detected"
    return (
        f"Run was thermally unstable ({detail}) — "
        "results may not reflect steady-state performance."
    )


def _append_simulator_notice(message: str, is_simulator: bool) -> str:
    """
    Append simulator notice to every recommendation when on simulator.
    Single canonical location — not repeated in each helper.
    """
    if not is_simulator:
        return message
    return f"{message} {_SIMULATOR_NOTICE}"


# ---------------------------------------------------------------------
# Recommendation Constructors
# ---------------------------------------------------------------------

def _recommend_inconsistent(
    delegate:     str,
    is_simulator: bool,
) -> RecommendationResult:
    return RecommendationResult(
        kind       = RecommendationKind.EXCLUDE_DELEGATE,
        message    = _append_simulator_notice(
            f"{delegate} output diverged from CPU. Use CPU.",
            is_simulator,
        ),
        delegate   = delegate,
        speedup    = None,
        confidence = "high",
    )


def _recommend_fallback(
    delegate:           str,
    cpu_ms:             Optional[float],
    del_ms:             Optional[float],
    compute_units_used: Optional[str],
    is_simulator:       bool,
) -> RecommendationResult:
    cpu_fmt   = f"{cpu_ms:.1f}ms"  if _is_valid(cpu_ms)  else "unknown"
    del_fmt   = f"{del_ms:.1f}ms"  if _is_valid(del_ms)  else "unknown"
    units_str = f" (ran on {compute_units_used})" if compute_units_used else ""
    return RecommendationResult(
        kind       = RecommendationKind.EXCLUDE_DELEGATE,
        message    = _append_simulator_notice(
            f"{delegate} fell back to CPU{units_str} "
            f"(latency {del_fmt} ≈ CPU {cpu_fmt}). No benefit. Use CPU.",
            is_simulator,
        ),
        delegate   = delegate,
        speedup    = None,
        confidence = "high",
    )


def _recommend_rerun(
    stability:    StabilityReport,
    delegate:     str,
    is_simulator: bool,
) -> RecommendationResult:
    reasons = []
    if getattr(stability, "thermally_unstable", False) and not is_simulator:
        reasons.append("thermal instability")
    score = getattr(stability, "stability_score", None)
    if score is not None and _is_valid(score) and stability.stability_band == StabilityBand.UNRELIABLE:
        reasons.append(f"high variance (stability score {score:.2f})")
    reason_str = " and ".join(reasons) if reasons else "measurement instability"

    return RecommendationResult(
        kind       = RecommendationKind.RERUN,
        message    = _append_simulator_notice(
            f"Results unreliable due to {reason_str}. "
            "Let device cool and rerun.",
            is_simulator,
        ),
        delegate   = delegate,
        speedup    = None,
        confidence = "low",
        caveat     = _thermal_caveat(stability, is_simulator),
    )


def _recommend_cpu_only(is_simulator: bool) -> RecommendationResult:
    return RecommendationResult(
        kind       = RecommendationKind.USE_CPU,
        message    = _append_simulator_notice(
            "No delegate available. CPU baseline recorded. Deploy with CPU.",
            is_simulator,
        ),
        delegate   = "CPU",
        speedup    = None,
        confidence = "high",
    )


def _recommend_delegate(
    delegate:     str,
    speedup:      Optional[float],
    stability:    StabilityReport,
    is_simulator: bool,
) -> RecommendationResult:
    speedup_val    = speedup if _is_valid(speedup) else 0.0
    band           = stability.stability_band or StabilityBand.UNRELIABLE
    caveat         = _thermal_caveat(stability, is_simulator)
    confidence_val = _confidence(stability, speedup_val)

    if speedup_val >= DEFAULT_SPEEDUP_MEANINGFUL:
        if band == StabilityBand.STABLE:
            msg = (
                f"{delegate} is {_speedup_str(speedup_val)} faster than CPU "
                f"with stable results. Use {delegate}."
            )
        else:
            msg = (
                f"{delegate} is {_speedup_str(speedup_val)} faster than CPU "
                "but results are noisy. Rerun to confirm."
            )
        return RecommendationResult(
            kind       = RecommendationKind.USE_DELEGATE,
            message    = _append_simulator_notice(msg, is_simulator),
            delegate   = delegate,
            speedup    = speedup_val,
            confidence = confidence_val,
            caveat     = caveat,
        )

    if speedup_val >= DEFAULT_SPEEDUP_MARGINAL_LOW:
        return RecommendationResult(
            kind       = RecommendationKind.USE_DELEGATE,
            message    = _append_simulator_notice(
                f"{delegate} is {_speedup_str(speedup_val)} faster than CPU — "
                "marginal gain. Decide if worth the complexity.",
                is_simulator,
            ),
            delegate   = delegate,
            speedup    = speedup_val,
            confidence = "medium",
            caveat     = caveat,
        )

    return RecommendationResult(
        kind       = RecommendationKind.USE_CPU,
        message    = _append_simulator_notice(
            f"{delegate} shows negligible speedup ({_speedup_str(speedup_val)}). "
            "Use CPU.",
            is_simulator,
        ),
        delegate   = delegate,
        speedup    = speedup_val,
        confidence = confidence_val,
        caveat     = caveat,
    )


# ---------------------------------------------------------------------
# Public Entry Point
# ---------------------------------------------------------------------

def recommend(view: iOSBuildView) -> RecommendationResult:
    """
    Synthesize one actionable recommendation from an iOSBuildView.

    Priority order documented in module docstring.
    Never raises — falls back to CPU-only on any internal error.
    """
    try:
        delegate        = getattr(view, "delegate",        "CPU")
        delegate_status = getattr(view, "delegate_status", None)
        stability       = getattr(view, "stability",       None)
        speedup         = getattr(view, "speedup",         None)
        is_simulator    = getattr(view, "is_simulator",    False)
        units_used      = getattr(view, "compute_units_used", None)

        if stability is None:
            return _recommend_cpu_only(is_simulator)

        _log(
            "Generating recommendation",
            context={
                "run_id":             getattr(view, "run_id", None),
                "delegate":           delegate,
                "delegate_status":    getattr(delegate_status, "value", None),
                "compute_units_used": units_used,
                "speedup":            speedup,
                "stability_band":     getattr(stability.stability_band, "value", None),
                "rerun_score":        getattr(stability, "rerun_recommendation_score", None),
                "is_simulator":       is_simulator,
            },
        )

        # Priority 1: Inconsistent
        if delegate_status == DelegateStatus.INCONSISTENT:
            result = _recommend_inconsistent(delegate, is_simulator)
            _log(f"Recommendation: INCONSISTENT → {result.kind.value}")
            return result

        # Priority 2: Fallback
        if delegate_status == DelegateStatus.FALLBACK:
            result = _recommend_fallback(
                delegate,
                cpu_ms             = getattr(view, "cpu_avg_ms",      None),
                del_ms             = getattr(view, "delegate_avg_ms",  None),
                compute_units_used = units_used,
                is_simulator       = is_simulator,
            )
            _log(f"Recommendation: FALLBACK → {result.kind.value}")
            return result

        # Priority 3: SKIPPED (ANE on simulator)
        # Not surfaced as a recommendation — shown in delegate status table only.
        # If the only delegate was SKIPPED, fall through to CPU-only.
        if delegate_status == DelegateStatus.SKIPPED:
            result = _recommend_cpu_only(is_simulator)
            _log(f"Recommendation: SKIPPED delegate → CPU only")
            return result

        # Priority 4: Rerun (thermal or stability)
        rerun_score = getattr(stability, "rerun_recommendation_score", 0.0)
        if not _is_valid(rerun_score) or rerun_score >= DEFAULT_RERUN_SCORE_THRESHOLD:
            result = _recommend_rerun(stability, delegate, is_simulator)
            _log(f"Recommendation: RERUN → {result.kind.value}")
            return result

        # Priority 5: CPU only
        if delegate == "CPU" or delegate_status is None:
            result = _recommend_cpu_only(is_simulator)
            _log(f"Recommendation: CPU only → {result.kind.value}")
            return result

        # Priority 6: Missing speedup
        if not _is_valid(speedup):
            result = RecommendationResult(
                kind       = RecommendationKind.USE_CPU,
                message    = _append_simulator_notice(
                    f"{delegate} ran but speedup could not be computed. "
                    "Rerun with a clean CPU baseline.",
                    is_simulator,
                ),
                delegate   = delegate,
                speedup    = None,
                confidence = "low",
                caveat     = _thermal_caveat(stability, is_simulator),
            )
            _log(f"Recommendation: missing speedup → {result.kind.value}")
            return result

        # Priority 7: Delegate evaluation
        result = _recommend_delegate(delegate, speedup, stability, is_simulator)
        _log(
            f"Recommendation: delegate eval → {result.kind.value}",
            context={
                "speedup":    speedup,
                "confidence": result.confidence,
                "kind":       result.kind.value,
            },
        )
        return result

    except Exception as exc:
        _log(f"Critical failure in recommendation engine: {exc}", "ERROR")
        return _recommend_cpu_only(is_simulator=False)