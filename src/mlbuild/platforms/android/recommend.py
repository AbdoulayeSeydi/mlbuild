"""
mlbuild.platforms.android.recommend

Single-run recommendation engine.

Responsibilities:
- Synthesize delegate status, speedup, stability, thermal, and consistency
  into one human-readable, actionable recommendation per run
- Provide a RecommendationResult consumed by device.py and the CLI
- Never make ADB calls — pure synthesis over already-collected data

Design rules:
- One recommendation per run. Not a list.
- Plain language. No jargon. No internal enum values surfaced to users.
- Every recommendation ends with an action or decision.
- Priority order: safety > correctness > performance > noise
    1. INCONSISTENT delegate → exclude immediately, reason explained
    2. FALLBACK delegate     → exclude, no benefit
    3. Thermal instability   → warn, rerun before trusting results
    4. Unreliable stability  → warn, rerun recommended
    5. Noisy but valid       → note variance, proceed with caution
    6. Stable + fast         → recommend delegate
    7. Stable + marginal     → note marginal gain, user decides
    8. CPU only              → record baseline, no delegate available
- Structured JSON logging for CI traceability.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from dataclasses import dataclass, asdict, replace
from enum import Enum
from math import isnan, isfinite
from multiprocessing import get_logger as mp_get_logger, Lock
from typing import Optional

from mlbuild.platforms.android import stability
from mlbuild.platforms.android.delegate import DelegateStatus
from mlbuild.platforms.android.stability import StabilityReport, StabilityBand
from mlbuild.platforms.android.result import AndroidBuildView

# ---------------------------------------------------------------------
# Configurable thresholds (per environment or device type)
# ---------------------------------------------------------------------
DEFAULT_SPEEDUP_MEANINGFUL    = 1.5
DEFAULT_SPEEDUP_MARGINAL_LOW  = 1.1
DEFAULT_RERUN_SCORE_THRESHOLD = 0.5
DEBUG = os.getenv("MLBUILD_DEBUG") == "1"

# ---------------------------------------------------------------------
# Multiprocess-safe logger (singleton)
# ---------------------------------------------------------------------
_logger_lock: threading.Lock = Lock()
_logger_instance: Optional[logging.Logger] = None

def _get_logger() -> logging.Logger:
    global _logger_instance
    with _logger_lock:
        if _logger_instance is None:
            logger = mp_get_logger()
            logger.propagate = False
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter(
                '{"time": "%(asctime)s", "level": "%(levelname)s", '
                '"logger": "%(name)s", "msg": %(message)s}',
                datefmt="%Y-%m-%dT%H:%M:%SZ"  # UTC
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG if DEBUG else logging.WARNING)
            _logger_instance = logger
    return _logger_instance

def _safe_log(msg: str, level: str = "INFO", context: Optional[dict] = None) -> None:
    """Safe multiprocess JSON logging with exception fallback."""
    logger = _get_logger()
    payload = {"msg": msg}
    if context:
        payload.update(context)
    try:
        logger_method = getattr(logger, level.lower(), logger.info)
        # json.dumps with ensure_ascii=False for unicode, fallback to str on failure
        safe_msg = json.dumps(payload, ensure_ascii=False, default=str)
        logger_method(safe_msg)
    except Exception as e:
        logger.error(json.dumps({"msg": f"Failed logging: {e}", "original_msg": str(msg)}))

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
    kind: RecommendationKind
    message: str
    delegate: str
    speedup: Optional[float]
    confidence: str
    caveat: Optional[str] = None

    def __str__(self) -> str:
        base = f"[{self.kind.value.upper()}] {self.message}"
        if self.caveat:
            base += f" | Note: {self.caveat}"
        return base

# ---------------------------------------------------------------------
# Helpers: Validation and Safe Formatting
# ---------------------------------------------------------------------
def _is_valid_number(x: Optional[float]) -> bool:
    return x is not None and isfinite(x)

def _speedup_str(speedup: Optional[float]) -> str:
    return f"{speedup:.1f}x" if _is_valid_number(speedup) else "unknown"

def _confidence(stability: StabilityReport, speedup: Optional[float]) -> str:
    """Derive confidence safely from stability and speedup."""
    try:
        if not _is_valid_number(stability.rerun_recommendation_score) or \
           stability.rerun_recommendation_score >= DEFAULT_RERUN_SCORE_THRESHOLD:
            return "low"
        band = stability.stability_band or StabilityBand.UNRELIABLE
        if band == StabilityBand.UNRELIABLE:
            return "low"
        if band == StabilityBand.NOISY or (speedup and speedup < DEFAULT_SPEEDUP_MEANINGFUL):
            return "medium"
        return "high"
    except Exception:
        return "low"

def _thermal_caveat(stability: StabilityReport) -> Optional[str]:
    if not getattr(stability, "thermally_unstable", False):
        return None
    parts = []
    temp = getattr(stability, "thermal_delta_c", None)
    drift = getattr(stability, "latency_drift_pct", None)
    if _is_valid_number(temp):
        parts.append(f"temp rose {temp:.1f}°C")
    if _is_valid_number(drift):
        parts.append(f"latency drifted {drift:.0%}")
    detail = ", ".join(parts) if parts else "thermal signal detected"
    return f"Run was thermally unstable ({detail}) — results may not reflect steady-state performance."

# ---------------------------------------------------------------------
# Core Recommendation Functions
# ---------------------------------------------------------------------
def _recommend_inconsistent(delegate: str) -> RecommendationResult:
    return RecommendationResult(
        kind=RecommendationKind.EXCLUDE_DELEGATE,
        message=f"{delegate} output diverged from CPU. Use CPU.",
        delegate=delegate,
        speedup=None,
        confidence="high",
    )

def _recommend_fallback(delegate: str, cpu_ms: Optional[float], del_ms: Optional[float]) -> RecommendationResult:
    cpu_ms_fmt = f"{cpu_ms:.1f}" if _is_valid_number(cpu_ms) else "unknown"
    del_ms_fmt = f"{del_ms:.1f}" if _is_valid_number(del_ms) else "unknown"
    return RecommendationResult(
        kind=RecommendationKind.EXCLUDE_DELEGATE,
        message=f"{delegate} ran on CPU (latency {del_ms_fmt}ms ≈ CPU {cpu_ms_fmt}ms). No benefit. Use CPU.",
        delegate=delegate,
        speedup=None,
        confidence="high",
    )

def _recommend_rerun(stability: StabilityReport, delegate: str) -> RecommendationResult:
    reasons = []
    if getattr(stability, "thermally_unstable", False):
        reasons.append("thermal instability")
    score = getattr(stability, "stability_score", None)
    if score is not None and _is_valid_number(score) and stability.stability_band == StabilityBand.UNRELIABLE:
        reasons.append(f"high variance (stability score {score:.2f})")
    reason_str = " and ".join(reasons) if reasons else "measurement instability"
    return RecommendationResult(
        kind=RecommendationKind.RERUN,
        message=f"Results unreliable due to {reason_str}. Let device cool and rerun.",
        delegate=delegate,
        speedup=None,
        confidence="low",
        caveat=_thermal_caveat(stability),
    )

def _recommend_cpu_only() -> RecommendationResult:
    return RecommendationResult(
        kind=RecommendationKind.USE_CPU,
        message="No delegate available. CPU baseline recorded. Deploy with CPU.",
        delegate="CPU",
        speedup=None,
        confidence="high",
    )

def _recommend_delegate(delegate: str, speedup: Optional[float], stability: StabilityReport) -> RecommendationResult:
    speedup_val = speedup if _is_valid_number(speedup) else 0.0
    band = stability.stability_band or StabilityBand.UNRELIABLE
    caveat = _thermal_caveat(stability)
    confidence_val = _confidence(stability, speedup_val)

    if speedup_val >= DEFAULT_SPEEDUP_MEANINGFUL:
        if band == StabilityBand.STABLE:
            return RecommendationResult(
                kind=RecommendationKind.USE_DELEGATE,
                message=f"{delegate} is {_speedup_str(speedup_val)} faster than CPU with stable results. Use {delegate}.",
                delegate=delegate,
                speedup=speedup_val,
                confidence=confidence_val,
                caveat=caveat,
            )
        else:
            return RecommendationResult(
                kind=RecommendationKind.USE_DELEGATE,
                message=f"{delegate} is {_speedup_str(speedup_val)} faster than CPU but results are noisy. Rerun to confirm.",
                delegate=delegate,
                speedup=speedup_val,
                confidence=confidence_val,
                caveat=caveat,
            )
    elif speedup_val >= DEFAULT_SPEEDUP_MARGINAL_LOW:
        return RecommendationResult(
            kind=RecommendationKind.USE_DELEGATE,
            message=f"{delegate} is {_speedup_str(speedup_val)} faster than CPU — marginal gain. Decide if worth it.",
            delegate=delegate,
            speedup=speedup_val,
            confidence="medium",
            caveat=caveat,
        )
    else:
        return RecommendationResult(
            kind=RecommendationKind.USE_CPU,
            message=f"{delegate} shows negligible speedup ({_speedup_str(speedup_val)}). Use CPU.",
            delegate=delegate,
            speedup=speedup_val,
            confidence=confidence_val,
            caveat=caveat,
        )

# ---------------------------------------------------------------------
# Public Recommendation Entry
# ---------------------------------------------------------------------
def recommend(view: AndroidBuildView) -> RecommendationResult:
    try:
        delegate = getattr(view, "delegate", "CPU")
        delegate_status = getattr(view, "delegate_status", None)
        # Immutable copy of stability for thread safety
        stability: StabilityReport = getattr(view, "stability", None)
        if stability is None:
            result = _recommend_cpu_only()
            return result
        speedup = getattr(view, "speedup", None)

        _safe_log("Generating recommendation", context={
            "run_id": getattr(view, "run_id", None),
            "delegate": delegate,
            "delegate_status": getattr(delegate_status, "value", None),
            "speedup": speedup,
            "stability_band": getattr(stability.stability_band, "value", None),
            "rerun_score": getattr(stability, "rerun_recommendation_score", None),
        })

        # Priority 0 — emulator, before any other check
        is_emulator = getattr(view.device, "is_emulator", False)

        if is_emulator:
            _safe_log(f"Recommendation: emulator detected → not representative")
            return RecommendationResult(
                kind       = RecommendationKind.USE_CPU,
                message    = (
                    "Running on an Android emulator. Latency numbers are not "
                    "representative of real device performance. "
                    "Connect a physical device for production benchmarks."
                ),
                delegate   = "CPU",
                speedup    = None,
                confidence = "low",
                caveat     = (
                    f"Emulator: {view.device.model} "
                    f"(chipset={view.device.chipset})"
                ),
            )

        # Priority 1: Inconsistent
        if delegate_status == DelegateStatus.INCONSISTENT:
            result = _recommend_inconsistent(delegate)
            _safe_log(f"Recommendation: INCONSISTENT → {result.kind.value}")
            return result

        # Priority 2: Fallback
        if delegate_status == DelegateStatus.FALLBACK:
            result = _recommend_fallback(
                delegate,
                cpu_ms=getattr(view, "cpu_avg_ms", None),
                del_ms=getattr(view, "delegate_avg_ms", None)
            )
            _safe_log(f"Recommendation: FALLBACK → {result.kind.value}")
            return result

        # Priority 3: Rerun
        rerun_score = getattr(stability, "rerun_recommendation_score", 0.0)
        if not _is_valid_number(rerun_score) or rerun_score >= DEFAULT_RERUN_SCORE_THRESHOLD:
            result = _recommend_rerun(stability, delegate)
            _safe_log(f"Recommendation: RERUN → {result.kind.value}")
            return result

        # Priority 4: CPU-only
        if delegate == "CPU" or delegate_status is None:
            result = _recommend_cpu_only()
            _safe_log(f"Recommendation: CPU only → {result.kind.value}")
            return result

        # Priority 5: Delegate evaluation
        if not _is_valid_number(speedup):
            result = RecommendationResult(
                kind=RecommendationKind.USE_CPU,
                message=f"{delegate} ran but speedup could not be computed. Rerun with clean CPU baseline.",
                delegate=delegate,
                speedup=None,
                confidence="low",
                caveat=_thermal_caveat(stability),
            )
            _safe_log(f"Recommendation: missing speedup → {result.kind.value}")
            return result

        result = _recommend_delegate(delegate, speedup, stability)
        _safe_log(f"Recommendation: delegate eval → {result.kind.value}", context={
            "speedup": speedup,
            "confidence": result.confidence,
            "kind": result.kind.value
        })
        return result

    except Exception as e:
        # Catch-all: fallback to CPU baseline
        _safe_log(f"Critical failure in recommendation engine: {e}", level="ERROR")
        return _recommend_cpu_only()