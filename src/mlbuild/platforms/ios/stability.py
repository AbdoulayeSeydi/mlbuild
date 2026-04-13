"""
mlbuild.platforms.ios.stability

Stability scoring and thermal risk analysis for iOS targets.

Features:
- Computes continuous stability scores (p90/p50) with band classification.
- Thermal instability derived from discrete IOSThermalState transitions
  and latency drift — not continuous temperature delta.
- Simulator runs: ThermalScore is None — thermal logic skipped entirely,
  no phantom instability warnings.
- Thread-safe, CI-ready structured JSON logging.
- Reproducibility hashes per run (device/model/delegate).

Key differences from Android:
- No thermal_delta_c — iOS exposes discrete states, not raw temperatures.
- Thermal instability uses state_degraded + latency_drift_pct.
- ThermalScore is Optional — None on simulator, not an error condition.
- IOSThermalState.SERIOUS / CRITICAL map directly to unstable=True
  regardless of latency drift — state alone is sufficient signal.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from enum import Enum
from math import isnan
from typing import Any, Dict, Optional, Tuple

from mlbuild.platforms.ios.thermal import IOSThermalState, ThermalScore


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DEFAULT_LATENCY_DRIFT_PCT    = 0.10   # 10%
DEFAULT_STABILITY_STABLE_MAX = 1.15  # mobile-optimized: 0.85 score threshold
DEFAULT_STABILITY_NOISY_MAX  = 1.43  # mobile-optimized: 0.70 score threshold

# iOS thermal states that alone constitute instability
_CRITICAL_THERMAL_STATES = {IOSThermalState.SERIOUS, IOSThermalState.CRITICAL}

DEBUG = os.getenv("MLBUILD_DEBUG") == "1"


# ---------------------------------------------------------------------
# Stability Band
# ---------------------------------------------------------------------

class StabilityBand(str, Enum):
    STABLE     = "stable"
    NOISY      = "noisy"
    UNRELIABLE = "unreliable"


# ---------------------------------------------------------------------
# Thread-Safe Logger
# ---------------------------------------------------------------------

_logger_lock:     threading.Lock           = threading.Lock()
_logger_instance: Optional[logging.Logger] = None


def _get_logger() -> logging.Logger:
    global _logger_instance
    with _logger_lock:
        if _logger_instance is None:
            logger = logging.getLogger("mlbuild.ios.stability")
            logger.propagate = False
            logger.setLevel(logging.DEBUG if DEBUG else logging.WARNING)
            if not logger.handlers:
                handler = logging.StreamHandler(sys.stderr)
                handler.setFormatter(logging.Formatter(
                    '{"time":"%(asctime)s","level":"%(levelname)s",'
                    '"logger":"%(name)s","msg":%(message)s}',
                    datefmt="%Y-%m-%dT%H:%M:%S",
                ))
                logger.addHandler(handler)
            _logger_instance = logger
    return _logger_instance


def _log(
    msg:     str,
    level:   str = "INFO",
    context: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {"msg": msg}
    if context:
        payload.update(context)
    logger = _get_logger()
    lvl = getattr(logging, level.upper(), logging.INFO)
    logger.log(lvl, json.dumps(payload, ensure_ascii=False))


# ---------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class StabilityReport:
    """
    Unified stability report for an iOS benchmark run.

    thermal_state_pre / thermal_state_post: discrete iOS states.
    No thermal_delta_c — iOS doesn't expose raw temperatures.

    thermally_unstable=False and thermal_state_pre=None both indicate
    a simulator run — distinguish via is_simulator in BuildView,
    not by inspecting these fields.
    """
    stability_score:            Optional[float]
    stability_band:             StabilityBand = field(default=StabilityBand.UNRELIABLE)
    thermally_unstable:         bool          = False
    thermal_state_pre:          Optional[IOSThermalState] = None
    thermal_state_post:         Optional[IOSThermalState] = None
    state_degraded:             bool          = False
    latency_drift_pct:          Optional[float] = None
    rerun_recommendation_score: float         = 0.0
    run_hash:                   Optional[str] = None

    def __str__(self) -> str:
        score = f"{self.stability_score:.3f}" if self.stability_score is not None else "N/A"
        pre   = self.thermal_state_pre.value  if self.thermal_state_pre  else "N/A"
        post  = self.thermal_state_post.value if self.thermal_state_post else "N/A"
        return (
            f"StabilityReport(score={score}, band={self.stability_band.value}, "
            f"thermally_unstable={self.thermally_unstable}, "
            f"thermal={pre}->{post}, "
            f"rerun_score={self.rerun_recommendation_score:.2f})"
        )


# ---------------------------------------------------------------------
# Validation Helpers
# ---------------------------------------------------------------------

def _validate_latency(val: Optional[float], name: str) -> Optional[float]:
    if val is None:
        _log(f"{name} latency missing", "WARN")
        return None
    if val <= 0 or isnan(val):
        _log(f"{name} latency invalid: {val}", "WARN")
        return None
    return val


# ---------------------------------------------------------------------
# Stability Score
# ---------------------------------------------------------------------

def compute_stability_score(
    p50_ms: Optional[float],
    p90_ms: Optional[float],
) -> Optional[float]:
    p50 = _validate_latency(p50_ms, "p50")
    p90 = _validate_latency(p90_ms, "p90")
    if p50 is None or p90 is None:
        return None
    score = round(p90 / p50, 4)
    _log(
        f"Stability score: {score}",
        context={"p50": p50, "p90": p90, "stability_score": score},
    )
    return score


def classify_stability_band(
    score:      Optional[float],
    stable_max: float = DEFAULT_STABILITY_STABLE_MAX,
    noisy_max:  float = DEFAULT_STABILITY_NOISY_MAX,
) -> StabilityBand:
    if score is None:
        return StabilityBand.UNRELIABLE
    if score <= stable_max:
        return StabilityBand.STABLE
    if score <= noisy_max:
        return StabilityBand.NOISY
    return StabilityBand.UNRELIABLE


# ---------------------------------------------------------------------
# Thermal Instability (iOS discrete states)
# ---------------------------------------------------------------------

def compute_thermal_instability(
    thermal_score:   Optional[ThermalScore],
    drift_threshold: float = DEFAULT_LATENCY_DRIFT_PCT,
) -> Tuple[bool, Optional[IOSThermalState], Optional[IOSThermalState], bool, Optional[float]]:
    """
    Returns:
        (unstable, pre_state, post_state, state_degraded, latency_drift_pct)

    iOS thermal instability is derived from two signals:

    1. post_state in {SERIOUS, CRITICAL} — device is actively throttling.
       This alone is sufficient for unstable=True regardless of latency drift.

    2. state_degraded + latency_drift > threshold — thermal state worsened
       during the run AND latency trended upward. Both required for this path.

    Simulator: ThermalScore is None — returns all-None, unstable=False.
    This is not an error. Caller knows it's a simulator via BuildView.
    """
    if thermal_score is None:
        _log("ThermalScore is None — simulator run, skipping thermal instability", "DEBUG")
        return False, None, None, False, None

    pre_state   = thermal_score.pre_state
    post_state  = thermal_score.post_state
    degraded    = thermal_score.state_degraded
    drift       = thermal_score.latency_drift_pct

    # Validate drift
    if drift is not None and (drift < 0 or isnan(drift)):
        _log(f"Invalid latency drift {drift} — ignoring", "WARN")
        drift = None

    # Signal 1: post state is SERIOUS or CRITICAL — hard unstable
    critical_state = post_state in _CRITICAL_THERMAL_STATES if post_state else False

    # Signal 2: state degraded + drift exceeded threshold
    drift_fired = drift is not None and drift > drift_threshold
    degraded_and_drifting = degraded and drift_fired

    unstable = critical_state or degraded_and_drifting

    _log(
        f"Thermal instability: unstable={unstable}",
        level="DEBUG" if unstable else "DEBUG",
        context={
            "pre_state":           pre_state.value  if pre_state  else None,
            "post_state":          post_state.value if post_state else None,
            "state_degraded":      degraded,
            "latency_drift_pct":   drift,
            "drift_threshold":     drift_threshold,
            "critical_state":      critical_state,
            "degraded_and_drift":  degraded_and_drifting,
            "thermally_unstable":  unstable,
        },
    )

    return unstable, pre_state, post_state, degraded, drift


# ---------------------------------------------------------------------
# Reproducibility Hash
# ---------------------------------------------------------------------

def compute_run_hash(
    device_id:     str,
    model_hash:    str,
    delegate_hash: str,
) -> str:
    m = hashlib.sha256()
    m.update(f"{device_id}|{model_hash}|{delegate_hash}".encode("utf-8"))
    return m.hexdigest()


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def compute_stability_report(
    p50_ms:          Optional[float],
    p90_ms:          Optional[float],
    thermal_score:   Optional[ThermalScore] = None,
    device_id:       str   = "unknown",
    model_hash:      str   = "unknown",
    delegate_hash:   str   = "unknown",
    variance:        Optional[float] = None,
    std_ms:          Optional[float] = None,
    avg_ms:          Optional[float] = None,
    low_confidence:  bool  = False,
    stable_max:      float = DEFAULT_STABILITY_STABLE_MAX,
    noisy_max:       float = DEFAULT_STABILITY_NOISY_MAX,
    drift_threshold: float = DEFAULT_LATENCY_DRIFT_PCT,
    sensitivity_w:   float = 1.0,
) -> StabilityReport:
    """
    Compute unified StabilityReport for an iOS benchmark run.

    Stability score from best available signal:
    1. p90/p50 ratio  — primary
    2. std/avg CV     — fallback if no percentiles
    3. 0.5 default    — unknown, not enough data

    Thermal instability from discrete state transitions — not delta_c.
    ThermalScore=None (simulator) → thermally_unstable=False, no warning.
    """

    # --- Stability score ---
    raw_score: Optional[float] = None

    if p50_ms and p90_ms and p50_ms > 0:
        ratio     = p90_ms / p50_ms
        raw_score = round(max(0.0, 1.0 - sensitivity_w * (ratio - 1.0)), 4)
        _log(
            f"Stability score from p90/p50: ratio={ratio:.3f} score={raw_score}",
            context={"p50": p50_ms, "p90": p90_ms, "ratio": ratio},
        )

    elif std_ms and avg_ms and avg_ms > 0:
        cv        = std_ms / avg_ms
        raw_score = round(max(0.0, 1.0 - cv), 4)
        _log(
            f"Stability score from std/avg: cv={cv:.3f} score={raw_score}",
            context={"std_ms": std_ms, "avg_ms": avg_ms, "cv": cv},
        )

    else:
        raw_score = 0.5
        _log("Stability score unknown — insufficient data, defaulting to 0.5", "WARN")

    if low_confidence and raw_score is not None:
        raw_score = round(raw_score * 0.8, 4)
        _log(f"Score downgraded for low sample count: {raw_score}", "DEBUG")

    # --- Band ---
    if raw_score is None:
        band = StabilityBand.UNRELIABLE
    elif raw_score >= 0.85:
        band = StabilityBand.STABLE
    elif raw_score >= 0.70:
        band = StabilityBand.NOISY
    else:
        band = StabilityBand.UNRELIABLE

    # --- Thermal instability (None-safe) ---
    unstable, pre_state, post_state, degraded, drift_pct = compute_thermal_instability(
        thermal_score, drift_threshold
    )

    # --- Rerun score ---
    rerun_score = 0.0
    if unstable:
        rerun_score += 0.5
    if band == StabilityBand.UNRELIABLE:
        rerun_score += 0.5
    if low_confidence:
        rerun_score = max(rerun_score, 0.3)

    report = StabilityReport(
        stability_score            = raw_score,
        stability_band             = band,
        thermally_unstable         = unstable,
        thermal_state_pre          = pre_state,
        thermal_state_post         = post_state,
        state_degraded             = degraded,
        latency_drift_pct          = drift_pct,
        rerun_recommendation_score = rerun_score,
        run_hash                   = compute_run_hash(device_id, model_hash, delegate_hash),
    )

    _log(
        f"StabilityReport: score={raw_score} band={band.value} "
        f"thermally_unstable={unstable} rerun={rerun_score}",
        level="DEBUG" if rerun_score > 0 else "DEBUG",
        context={
            "stability_score":    raw_score,
            "stability_band":     band.value,
            "thermally_unstable": unstable,
            "rerun_score":        rerun_score,
            "low_confidence":     low_confidence,
        },
    )

    return report