"""
mlbuild.platforms.android.stability

Stability scoring and thermal risk analysis.

Features:
- Computes continuous stability scores (p90/p50) with automatic band classification.
- Computes thermal instability with bounds validation and optional threshold overrides.
- Thread-safe, CI-ready structured JSON logging.
- Includes reproducibility hashes per run (device/model/delegate).
- Continuous risk score for weighted rerun recommendations.
"""

from __future__ import annotations
import sys

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from math import isnan
from typing import Optional, Tuple, Dict, Any

from mlbuild.platforms.android.thermal import ThermalScore


# ---------------------------------------------------------------------
# Config — Thresholds live here, can be overridden per run
# ---------------------------------------------------------------------

DEFAULT_TEMP_DELTA_C = 5.0         # °C
DEFAULT_LATENCY_DRIFT_PCT = 0.10  # 10%

DEFAULT_STABILITY_STABLE_MAX = 1.10
DEFAULT_STABILITY_NOISY_MAX = 1.30

DEBUG = os.getenv("MLBUILD_DEBUG") == "1"


# ---------------------------------------------------------------------
# Stability Band
# ---------------------------------------------------------------------

class StabilityBand(str, Enum):
    STABLE = "stable"
    NOISY = "noisy"
    UNRELIABLE = "unreliable"


# ---------------------------------------------------------------------
# Thread-safe singleton logger
# ---------------------------------------------------------------------

_logger_lock = threading.Lock()
_logger_instance: Optional[logging.Logger] = None

def get_logger() -> logging.Logger:
    global _logger_instance
    with _logger_lock:
        if _logger_instance is None:
            logger = logging.getLogger("mlbuild.stability")
            logger.propagate = False
            logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
            if not logger.handlers:
                handler = logging.StreamHandler(sys.stderr)
                handler.setFormatter(logging.Formatter(
                    '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":%(message)s}',
                    datefmt="%Y-%m-%dT%H:%M:%S"
                ))
                logger.addHandler(handler)
            _logger_instance = logger
    return _logger_instance


def log(msg: str, level: str = "INFO", context: Optional[Dict[str, Any]] = None) -> None:
    """
    Thread-safe structured logging in JSON format.
    """
    payload = {"msg": msg}
    if context:
        payload.update(context)
    logger = get_logger()
    lvl = getattr(logging, level.upper(), logging.INFO)
    logger.log(lvl, json.dumps(payload, ensure_ascii=False))


# ---------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class StabilityReport:
    stability_score: Optional[float]
    stability_band: StabilityBand = field(default=StabilityBand.UNRELIABLE)
    thermally_unstable: bool = False
    thermal_delta_c: Optional[float] = None
    latency_drift_pct: Optional[float] = None
    rerun_recommendation_score: float = 0.0
    run_hash: Optional[str] = None  # deterministic hash of device/model/delegate

    def __str__(self) -> str:
        score = f"{self.stability_score:.3f}" if self.stability_score is not None else "N/A"
        band = self.stability_band.value
        return (f"StabilityReport(score={score}, band={band}, "
                f"thermally_unstable={self.thermally_unstable}, "
                f"rerun_score={self.rerun_recommendation_score:.2f})")


# ---------------------------------------------------------------------
# Validation Helpers
# ---------------------------------------------------------------------

def _validate_latency(latency_ms: Optional[float], name: str) -> Optional[float]:
    if latency_ms is None:
        log(f"{name} latency missing", "WARN")
        return None
    if latency_ms <= 0 or isnan(latency_ms):
        log(f"{name} latency invalid: {latency_ms}", "WARN")
        return None
    return latency_ms


# ---------------------------------------------------------------------
# Compute Stability Score
# ---------------------------------------------------------------------

def compute_stability_score(p50_ms: Optional[float], p90_ms: Optional[float]) -> Optional[float]:
    p50 = _validate_latency(p50_ms, "p50")
    p90 = _validate_latency(p90_ms, "p90")
    if p50 is None or p90 is None:
        return None
    score = round(p90 / p50, 4)
    log(f"Computed stability score: {score}", context={"p50": p50, "p90": p90, "stability_score": score})
    return score


def classify_stability_band(score: Optional[float],
                            stable_max: float = DEFAULT_STABILITY_STABLE_MAX,
                            noisy_max: float = DEFAULT_STABILITY_NOISY_MAX) -> StabilityBand:
    if score is None:
        return StabilityBand.UNRELIABLE
    if score <= stable_max:
        return StabilityBand.STABLE
    elif score <= noisy_max:
        return StabilityBand.NOISY
    else:
        return StabilityBand.UNRELIABLE


# ---------------------------------------------------------------------
# Thermal Instability
# ---------------------------------------------------------------------

def compute_thermal_instability(thermal_score: Optional[ThermalScore],
                                temp_threshold: float = DEFAULT_TEMP_DELTA_C,
                                drift_threshold: float = DEFAULT_LATENCY_DRIFT_PCT
                                ) -> Tuple[bool, Optional[float], Optional[float]]:
    """
    Returns (unstable_flag, thermal_delta_c, latency_drift_pct)
    """
    if thermal_score is None:
        log("ThermalScore unavailable, skipping thermal instability check", "DEBUG")
        return False, None, None

    delta = getattr(thermal_score, "thermal_delta_c", None)
    drift = getattr(thermal_score, "latency_drift_pct", None)

    # Validate numeric bounds
    if delta is not None and (delta < 0 or isnan(delta)):
        log(f"Invalid thermal delta {delta}, ignoring", "WARN")
        delta = None
    if drift is not None and (drift < 0 or isnan(drift)):
        log(f"Invalid latency drift {drift}, ignoring", "WARN")
        drift = None

    temp_fired = delta is not None and delta > temp_threshold
    drift_fired = drift is not None and drift > drift_threshold

    unstable = temp_fired and drift_fired

    log(
        f"Thermal instability check: unstable={unstable}",
        level="WARN" if unstable else "INFO",
        context={
            "thermal_delta_c": delta,
            "latency_drift_pct": drift,
            "temp_threshold": temp_threshold,
            "drift_threshold": drift_threshold,
            "temp_fired": temp_fired,
            "drift_fired": drift_fired,
            "thermally_unstable": unstable
        }
    )

    return unstable, delta, drift


# ---------------------------------------------------------------------
# Reproducibility Hash
# ---------------------------------------------------------------------

def compute_run_hash(device_id: str, model_hash: str, delegate_hash: str) -> str:
    """
    Deterministic hash for reproducibility and CI correlation.
    """
    m = hashlib.sha256()
    m.update(f"{device_id}|{model_hash}|{delegate_hash}".encode("utf-8"))
    return m.hexdigest()


# ---------------------------------------------------------------------
# Public API: Stability Report
# ---------------------------------------------------------------------

def compute_stability_report(
    p50_ms:         Optional[float],
    p90_ms:         Optional[float],
    thermal_score:  Optional[ThermalScore] = None,
    device_id:      str = "unknown",
    model_hash:     str = "unknown",
    delegate_hash:  str = "unknown",
    variance:       Optional[float] = None,
    std_ms:         Optional[float] = None,
    avg_ms:         Optional[float] = None,
    count:          Optional[int]   = None,
    low_confidence: bool            = False,
    stable_max:     float = DEFAULT_STABILITY_STABLE_MAX,
    noisy_max:      float = DEFAULT_STABILITY_NOISY_MAX,
    temp_threshold: float = DEFAULT_TEMP_DELTA_C,
    drift_threshold: float = DEFAULT_LATENCY_DRIFT_PCT,
) -> StabilityReport:
    """
    Compute unified StabilityReport.

    Stability score is computed from the best available signal:
    1. p90/p50 ratio (most reliable — percentile spread)
    2. std/avg coefficient of variation (fallback if no percentiles)
    3. 0.5 default (unknown — not enough data)

    Score is normalized to 0.0-1.0:
    1.0 = perfectly stable
    0.5 = unknown / insufficient data
    0.0 = completely unreliable
    """

    # --- Stability score from best available signal ---
    raw_score: Optional[float] = None

    if p50_ms and p90_ms and p50_ms > 0:
        # Primary: p90/p50 ratio. 1.0 = stable, higher = noisier
        ratio = p90_ms / p50_ms
        # Normalize: ratio 1.0 → score 1.0, ratio 2.0 → score 0.0
        raw_score = round(max(0.0, 1.0 - (ratio - 1.0)), 4)
        log(f"Stability score from p90/p50: ratio={ratio:.3f} score={raw_score}")

    elif std_ms and avg_ms and avg_ms > 0:
        # Fallback: coefficient of variation (std/avg)
        cv = std_ms / avg_ms
        raw_score = round(max(0.0, 1.0 - cv), 4)
        log(f"Stability score from std/avg: cv={cv:.3f} score={raw_score}")

    else:
        # Unknown — not enough data
        raw_score = 0.5
        log("Stability score unknown — insufficient data, defaulting to 0.5", "WARN")

    # Downgrade if low sample count
    if low_confidence and raw_score is not None:
        raw_score = round(raw_score * 0.8, 4)
        log(f"Score downgraded for low sample count: {raw_score}", "WARN")

    # Band classification from raw_score
    # raw_score ≥ 0.9 → stable
    # raw_score ≥ 0.7 → noisy
    # raw_score < 0.7 → unreliable
    if raw_score is None:
        band = StabilityBand.UNRELIABLE
    elif raw_score >= 0.90:
        band = StabilityBand.STABLE
    elif raw_score >= 0.70:
        band = StabilityBand.NOISY
    else:
        band = StabilityBand.UNRELIABLE

    # Thermal instability
    unstable, delta_c, drift_pct = compute_thermal_instability(
        thermal_score, temp_threshold, drift_threshold
    )

    # Continuous rerun score
    rerun_score = 0.0
    if unstable:
        rerun_score += 0.5
    if band == StabilityBand.UNRELIABLE:
        rerun_score += 0.5
    if low_confidence:
        rerun_score = max(rerun_score, 0.3)

    report = StabilityReport(
        stability_score              = raw_score,
        stability_band               = band,
        thermally_unstable           = unstable,
        thermal_delta_c              = delta_c,
        latency_drift_pct            = drift_pct,
        rerun_recommendation_score   = rerun_score,
        run_hash                     = compute_run_hash(device_id, model_hash, delegate_hash),
    )

    log(
        f"StabilityReport: score={raw_score} band={band.value} "
        f"thermally_unstable={unstable} rerun={rerun_score}",
        level="WARN" if rerun_score > 0 else "INFO",
        context={
            "stability_score":  raw_score,
            "stability_band":   band.value,
            "thermally_unstable": unstable,
            "rerun_score":      rerun_score,
            "low_confidence":   low_confidence,
        },
    )

    return report