"""
mlbuild.platforms.ios.result

Final BuildView assembly for iOS benchmark runs.

Responsibilities:
- Accept all upstream outputs (baseline, delegate, stability, thermal, validation)
- Assemble into a unified iOSBuildView
- Serialize to dict for registry storage in .mlbuild/registry.db
- Compute final speedup, stability, and recommendation inputs

Design rules:
- No idb calls. No subprocess. Pure assembly and serialization.
- Every field is nullable — partial results are valid BuildViews.
- raw_stdout always preserved unconditionally.
- iOSBuildView is the canonical data contract for iOS runs.
  result.py produces it. Everything downstream consumes it.
- Frozen dataclass — immutable after assembly.
- Structured JSON logging for CI traceability.

Key differences from Android:
- compute_units + compute_units_used fields — requested vs actual CoreML path.
- thermal_state_pre / thermal_state_post are IOSThermalState enums, not floats.
- thermal_score is Optional[ThermalScore] — None on simulator, not an error.
- is_simulator and simulator_warning are first-class fields.
- No cpu_autocorr — iOS runner doesn't emit per-run CSV for autocorrelation.
- No cpu_count — low_confidence threshold differs (20 not 100).
- backend is "coreml" not "tflite".
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from mlbuild.platforms.ios.introspect import DeviceProfile
from mlbuild.platforms.ios.baseline import BaselineResult
from mlbuild.platforms.ios.benchmark import BenchmarkResult, OpStat
from mlbuild.platforms.ios.delegate import DelegateValidation, DelegateStatus
from mlbuild.platforms.ios.stability import StabilityReport, StabilityBand
from mlbuild.platforms.ios.thermal import ThermalSnapshot, ThermalScore, IOSThermalState


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

SCHEMA_VERSION = os.getenv("MLBUILD_SCHEMA_VERSION", "1.0")
DEBUG          = os.getenv("MLBUILD_DEBUG") == "1"


# ---------------------------------------------------------------------
# Thread-Safe Logger
# ---------------------------------------------------------------------

_logger_instance: Optional[logging.Logger] = None
_logger_lock      = threading.Lock()


def _get_logger() -> logging.Logger:
    global _logger_instance
    with _logger_lock:
        if _logger_instance is None:
            logger = logging.getLogger("mlbuild.ios.result")
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
    context: Optional[Dict[str, Any]] = None,
) -> None:
    logger = _get_logger()
    payload = {"msg": msg}
    if context:
        payload.update(context)
    getattr(logger, level.lower(), logger.info)(
        json.dumps(payload, default=str, ensure_ascii=False)
    )


# ---------------------------------------------------------------------
# Immutable Data Contract
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class iOSBuildView:
    """
    Canonical immutable data contract for iOS benchmark runs.

    compute_units:      what was requested (e.g. "cpuAndGPU")
    compute_units_used: what CoreML actually ran on — ground truth from runner.
                        May differ from compute_units on silent fallback.

    thermal_state_pre / post: discrete IOSThermalState — None on simulator.
    thermal_score:            None on simulator — no faked data.

    is_simulator:      True if run was on iOS Simulator.
    simulator_warning: Non-empty string if is_simulator=True.
                       Appended to CLI output and stored for audit.
    """

    # Identity
    schema_version: str
    run_id:         str
    run_hash:       str
    assembled_at:   str         # UTC ISO format

    # Run config
    backend:        str         # always "coreml"
    model_path:     str
    delegate:       str
    delegate_status: Optional[DelegateStatus]
    compute_units:      str             # requested
    compute_units_used: Optional[str]   # actual — CoreML ground truth
    binary_version: str

    # Device
    device: DeviceProfile

    # Simulator
    is_simulator:      bool
    simulator_warning: Optional[str]

    # CPU baseline
    cpu_avg_ms:      Optional[float]
    cpu_p50_ms:      Optional[float]
    cpu_p90_ms:      Optional[float]
    cpu_p99_ms:      Optional[float]
    cpu_init_ms:     Optional[float]
    cpu_peak_mem_mb: Optional[float]
    cpu_variance:    Optional[float]
    cpu_std_ms:      Optional[float]
    cpu_min_ms:      Optional[float]
    cpu_max_ms:      Optional[float]
    cpu_low_confidence: bool
    cpu_raw_stdout:  str

    # Delegate result
    delegate_avg_ms:      Optional[float]
    delegate_p50_ms:      Optional[float]
    delegate_p90_ms:      Optional[float]
    delegate_p99_ms:      Optional[float]
    delegate_init_ms:     Optional[float]
    delegate_peak_mem_mb: Optional[float]
    delegate_variance:    Optional[float]
    delegate_raw_stdout:  str
    ops:                  Tuple[OpStat, ...]
    latency_trend:        Optional[Tuple[float, ...]]

    # Computed
    speedup:   Optional[float]
    stability: StabilityReport

    # Thermal — discrete states, None on simulator
    thermal_pre:         Optional[ThermalSnapshot]
    thermal_post:        Optional[ThermalSnapshot]
    thermal_score:       Optional[ThermalScore]
    thermal_state_pre:   Optional[IOSThermalState]
    thermal_state_post:  Optional[IOSThermalState]

    def __str__(self) -> str:
        band = (
            self.stability.stability_band.value
            if self.stability and self.stability.stability_band
            else "N/A"
        )
        sim = " [SIMULATOR]" if self.is_simulator else ""
        return (
            f"iOSBuildView(run_id={self.run_id}, delegate={self.delegate}, "
            f"compute_units_used={self.compute_units_used}, "
            f"speedup={self.speedup}, stability={band}{sim})"
        )


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------

def serialize_dataclass(obj: Any) -> Any:
    """
    Recursively convert dataclass / Enum / tuple into JSON-safe types.
    """
    if obj is None:
        return None
    if isinstance(obj, tuple):
        return [serialize_dataclass(x) for x in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return {k: serialize_dataclass(getattr(obj, k)) for k in obj.__dataclass_fields__}
    if isinstance(obj, Enum):
        return obj.value
    return obj


# ---------------------------------------------------------------------
# Run Hash
# ---------------------------------------------------------------------

def compute_run_hash(
    model_path:         str,
    delegate:           str,
    device_fingerprint: str,
    run_id:             str,
) -> str:
    payload = f"{model_path}|{delegate.lower()}|{device_fingerprint}|{run_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------
# Speedup
# ---------------------------------------------------------------------

def compute_speedup(
    cpu_avg_ms:      Optional[float],
    delegate_avg_ms: Optional[float],
) -> Optional[float]:
    if cpu_avg_ms is None or delegate_avg_ms is None:
        _log("Missing CPU or delegate average for speedup", "DEBUG")
        return None
    if cpu_avg_ms <= 0 or delegate_avg_ms <= 0:
        _log(
            f"Invalid avg for speedup: cpu={cpu_avg_ms} delegate={delegate_avg_ms}",
            "WARNING",
        )
        return None
    return round(cpu_avg_ms / delegate_avg_ms, 2)


# ---------------------------------------------------------------------
# Metric Sanitization
# ---------------------------------------------------------------------

def _sanitize(val: Optional[float], name: str) -> Optional[float]:
    if val is None:
        return None
    if not isinstance(val, (int, float)) or val < 0:
        _log(f"Invalid metric {name}: {val}", "WARNING")
        return None
    return val


def _sanitize_metrics(src: Any, keys: list[str]) -> Dict[str, Optional[float]]:
    return {k: _sanitize(getattr(src, k, None), k) for k in keys}


# ---------------------------------------------------------------------
# Public: Assemble
# ---------------------------------------------------------------------

def assemble(
    *,
    model_path:      str,
    device:          DeviceProfile,
    baseline:        BaselineResult,
    binary_version:  str,
    stability:       StabilityReport,
    delegate_result: Optional[BenchmarkResult]   = None,
    validation:      Optional[DelegateValidation] = None,
    thermal_pre:     Optional[ThermalSnapshot]    = None,
    thermal_post:    Optional[ThermalSnapshot]    = None,
    thermal_score:   Optional[ThermalScore]       = None,
) -> iOSBuildView:
    """
    Assemble a fully validated, immutable iOSBuildView.

    thermal_score=None is valid — it means simulator run.
    delegate_result=None is valid — CPU-only run.
    All fields nullable — partial results are valid BuildViews.
    """
    now = datetime.now(timezone.utc).isoformat()

    delegate        = delegate_result.delegate if delegate_result else "CPU"
    delegate_status = validation.status        if validation      else None

    compute_units      = delegate_result.compute_units      if delegate_result else "cpuOnly"
    compute_units_used = (delegate_result.compute_units_used if delegate_result else None) or (baseline.compute_units if baseline else "cpuOnly")

    is_simulator      = device.is_simulator
    simulator_warning = delegate_result.simulator_warning if delegate_result else None

    run_hash = compute_run_hash(
        model_path, delegate, device.fingerprint, baseline.run_id
    )

    speedup = compute_speedup(
        baseline.avg_ms,
        delegate_result.avg_ms if delegate_result else None,
    )

    # Thermal state extraction from ThermalScore
    thermal_state_pre:  Optional[IOSThermalState] = None
    thermal_state_post: Optional[IOSThermalState] = None
    if thermal_score is not None:
        thermal_state_pre  = thermal_score.pre_state
        thermal_state_post = thermal_score.post_state

    # Sanitize metrics
    cpu_keys = [
        "avg_ms", "p50_ms", "p90_ms", "p99_ms",
        "init_ms", "peak_mem_mb", "variance", "std_ms", "min_ms", "max_ms",
    ]
    delegate_keys = [
        "avg_ms", "p50_ms", "p90_ms", "p99_ms",
        "init_ms", "peak_mem_mb", "variance",
    ]

    cpu_metrics      = _sanitize_metrics(baseline, cpu_keys)
    delegate_metrics = (
        _sanitize_metrics(delegate_result, delegate_keys)
        if delegate_result
        else {k: None for k in delegate_keys}
    )

    ops           = tuple(delegate_result.ops)           if delegate_result and delegate_result.ops           else ()
    latency_trend = tuple(delegate_result.latency_trend) if delegate_result and delegate_result.latency_trend else None

    view = iOSBuildView(
        schema_version        = SCHEMA_VERSION,
        run_id                = baseline.run_id,
        run_hash              = run_hash,
        assembled_at          = now,
        backend               = "coreml",
        model_path            = model_path,
        delegate              = delegate,
        delegate_status       = delegate_status,
        compute_units         = compute_units,
        compute_units_used    = compute_units_used,
        binary_version        = binary_version,
        device                = replace(device),
        is_simulator          = is_simulator,
        simulator_warning     = simulator_warning,
        cpu_avg_ms            = cpu_metrics["avg_ms"],
        cpu_p50_ms            = cpu_metrics["p50_ms"],
        cpu_p90_ms            = cpu_metrics["p90_ms"],
        cpu_p99_ms            = cpu_metrics["p99_ms"],
        cpu_init_ms           = cpu_metrics["init_ms"],
        cpu_peak_mem_mb       = cpu_metrics["peak_mem_mb"],
        cpu_variance          = cpu_metrics["variance"],
        cpu_std_ms            = cpu_metrics["std_ms"],
        cpu_min_ms            = cpu_metrics["min_ms"],
        cpu_max_ms            = cpu_metrics["max_ms"],
        cpu_low_confidence    = baseline.low_confidence,
        cpu_raw_stdout        = baseline.raw_stdout or "",
        delegate_avg_ms       = delegate_metrics["avg_ms"],
        delegate_p50_ms       = delegate_metrics["p50_ms"],
        delegate_p90_ms       = delegate_metrics["p90_ms"],
        delegate_p99_ms       = delegate_metrics["p99_ms"],
        delegate_init_ms      = delegate_metrics["init_ms"],
        delegate_peak_mem_mb  = delegate_metrics["peak_mem_mb"],
        delegate_variance     = delegate_metrics["variance"],
        delegate_raw_stdout   = delegate_result.raw_stdout if delegate_result else "",
        ops                   = ops,
        latency_trend         = latency_trend,
        speedup               = speedup,
        stability             = stability,
        thermal_pre           = replace(thermal_pre)  if thermal_pre  else None,
        thermal_post          = replace(thermal_post) if thermal_post else None,
        thermal_score         = thermal_score,
        thermal_state_pre     = thermal_state_pre,
        thermal_state_post    = thermal_state_post,
    )

    _log(
        "iOSBuildView assembled",
        context={
            "run_id":             view.run_id,
            "run_hash":           view.run_hash,
            "delegate":           view.delegate,
            "compute_units_used": view.compute_units_used,
            "speedup":            view.speedup,
            "stability_band":     view.stability.stability_band.value if view.stability else None,
            "rerun_score":        view.stability.rerun_recommendation_score if view.stability else None,
            "is_simulator":       view.is_simulator,
        },
    )

    return view


# ---------------------------------------------------------------------
# Public: Serialize to Registry Dict
# ---------------------------------------------------------------------

def to_registry_dict(view: iOSBuildView) -> dict:
    """
    Serialize iOSBuildView into a JSON-safe dict for registry storage.
    Handles nested dataclasses, Enums, IOSThermalState, and optional fields.
    """
    return serialize_dataclass(view)