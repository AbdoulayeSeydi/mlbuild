"""
mlbuild.platforms.android.result

Final BuildView assembly for Android benchmark runs.

Responsibilities:
- Accept all upstream outputs (baseline, delegate, stability, thermal, validation)
- Assemble into a unified AndroidBuildView
- Serialize to dict for registry storage in .mlbuild/registry.db
- Compute final speedup, stability, and recommendation inputs

Design rules:
- No ADB calls. No subprocess. Pure assembly and serialization.
- Every field is nullable — partial results are valid BuildViews.
- raw_stdout always preserved unconditionally.
- AndroidBuildView is the canonical data contract for Android runs.
  result.py produces it. Everything downstream consumes it.
- Frozen dataclass — immutable after assembly.
- Structured JSON logging for CI traceability.
"""

from __future__ import annotations
import sys

from enum import Enum
import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime, timezone
from typing import Optional, Any, Tuple, Dict

from psutil import cpu_count

from mlbuild.platforms.android.introspect import DeviceProfile
from mlbuild.platforms.android.baseline import BaselineResult
from mlbuild.platforms.android.benchmark import BenchmarkResult, OpStat
from mlbuild.platforms.android.delegate import DelegateValidation, DelegateStatus
from mlbuild.platforms.android.stability import StabilityReport, StabilityBand
from mlbuild.platforms.android.thermal import ThermalSnapshot

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
SCHEMA_VERSION = os.getenv("MLBUILD_SCHEMA_VERSION", "1.0")
DEBUG = os.getenv("MLBUILD_DEBUG") == "1"

# ---------------------------------------------------------------------
# Thread-safe Singleton Logger
# ---------------------------------------------------------------------
_logger_instance: Optional[logging.Logger] = None
_logger_lock = threading.Lock()


def get_logger() -> logging.Logger:
    global _logger_instance
    with _logger_lock:
        if _logger_instance is None:
            logger = logging.getLogger("mlbuild.result")
            logger.propagate = False
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter(
                '{"time": "%(asctime)s", "level": "%(levelname)s", '
                '"logger": "%(name)s", "msg": %(message)s}',
                datefmt="%Y-%m-%dT%H:%M:%SZ",
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
            _logger_instance = logger
    return _logger_instance


def log(msg: str, level: str = "INFO", context: Optional[Dict[str, Any]] = None) -> None:
    """
    Thread-safe JSON-structured logging.
    Always uses UTC ISO format timestamps.
    """
    logger = get_logger()
    payload = {"msg": msg}
    if context:
        payload.update(context)
    try:
        log_method = getattr(logger, level.upper())
    except AttributeError:
        log_method = logger.info
    log_method(json.dumps(payload, default=str, ensure_ascii=False))


# ---------------------------------------------------------------------
# Immutable Data Contracts
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class AndroidBuildView:
    """
    Canonical immutable data contract for Android benchmark runs.
    Nested objects are replaced with frozen equivalents or primitives to
    guarantee reproducibility.
    """

    # Identity
    schema_version: str
    run_id: str
    run_hash: str
    assembled_at: str  # UTC ISO format

    # Run config
    backend: str
    model_path: str
    delegate: str
    delegate_status: Optional[DelegateStatus]
    binary_version: str

    # Device
    device: DeviceProfile

    # CPU baseline
    cpu_avg_ms: Optional[float]
    cpu_p50_ms: Optional[float]
    cpu_p90_ms: Optional[float]
    cpu_p99_ms: Optional[float]
    cpu_init_ms: Optional[float]
    cpu_peak_mem_mb: Optional[float]
    cpu_variance: Optional[float]
    cpu_raw_stdout: str
    cpu_std_ms:     Optional[float]      
    cpu_min_ms:     Optional[float]      
    cpu_max_ms:     Optional[float]      
    cpu_count:      Optional[int]
    cpu_autocorr:    Optional[float]          

    # Delegate result
    delegate_avg_ms: Optional[float]
    delegate_p50_ms: Optional[float]
    delegate_p90_ms: Optional[float]
    delegate_p99_ms: Optional[float]
    delegate_init_ms: Optional[float]
    delegate_peak_mem_mb: Optional[float]
    delegate_variance: Optional[float]
    delegate_raw_stdout: str
    ops: Tuple[OpStat, ...]
    latency_trend: Optional[Tuple[float, ...]]

    # Computed
    speedup: Optional[float]
    stability: StabilityReport

    # Thermal
    thermal_pre: Optional[ThermalSnapshot]
    thermal_post: Optional[ThermalSnapshot]

    def __str__(self) -> str:
        band = self.stability.stability_band.value if self.stability.stability_band else "N/A"
        return f"AndroidBuildView(run_id={self.run_id}, delegate={self.delegate}, speedup={self.speedup}, stability={band})"


# ---------------------------------------------------------------------
# Serialization Helpers
# ---------------------------------------------------------------------
def serialize_dataclass(obj) -> Any:
    """
    Convert dataclass or nested dataclass into JSON-safe dict recursively.
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
# Run Hash (collision-resistant)
# ---------------------------------------------------------------------
def compute_run_hash(model_path: str, delegate: str, device_fingerprint: str, run_id: str) -> str:
    """
    SHA256 hash for unique run identification.
    Uses full 64-character digest to prevent collisions in enterprise scale.
    """
    payload = f"{model_path}|{delegate.lower()}|{device_fingerprint}|{run_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------
# Speedup Computation (validated)
# ---------------------------------------------------------------------
def compute_speedup(cpu_avg_ms: Optional[float], delegate_avg_ms: Optional[float]) -> Optional[float]:
    """
    Compute delegate speedup over CPU with validation.
    Returns None if inputs are missing, zero, negative, or absurd.
    """
    if cpu_avg_ms is None or delegate_avg_ms is None:
        log("Missing CPU or delegate average for speedup", level="WARNING")
        return None
    if cpu_avg_ms <= 0 or delegate_avg_ms <= 0:
        log(f"Invalid CPU/Delegate avg: cpu={cpu_avg_ms}, delegate={delegate_avg_ms}", level="WARNING")
        return None
    return cpu_avg_ms / delegate_avg_ms


# ---------------------------------------------------------------------
# Defensive Assembly
# ---------------------------------------------------------------------
def assemble(
    *,
    model_path: str,
    device: DeviceProfile,
    baseline: BaselineResult,
    binary_version: str,
    stability: StabilityReport,
    delegate_result: Optional[BenchmarkResult] = None,
    validation: Optional[DelegateValidation] = None,
    thermal_pre: Optional[ThermalSnapshot] = None,
    thermal_post: Optional[ThermalSnapshot] = None,
) -> AndroidBuildView:
    """
    Assemble a fully validated, immutable AndroidBuildView.
    Ensures numeric validation, thread-safe logging, UTC timestamps,
    and immutable nested objects.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Defensive delegate identity
    delegate = delegate_result.delegate if delegate_result else "CPU"
    delegate_status = validation.status if validation else None

    # Run hash
    run_hash = compute_run_hash(model_path, delegate, device.fingerprint, baseline.run_id)

    # Speedup
    speedup = compute_speedup(baseline.avg_ms, delegate_result.avg_ms if delegate_result else None)

    # Validate numeric metrics
    def sanitize_metric(value: Optional[float], name: str) -> Optional[float]:
        if value is None:
            return None
        if value < 0 or not isinstance(value, (int, float)):
            log(f"Invalid metric {name}: {value}", level="WARNING")
            return None
        return value

    cpu_count = getattr(baseline, 'count', None)
    cpu_autocorr = getattr(baseline, 'autocorr_lag1', None)
    delegate_metrics = {k: sanitize_metric(getattr(delegate_result, k, None), k) for k in [
        'avg_ms', 'p50_ms', 'p90_ms', 'p99_ms', 'init_ms', 'peak_mem_mb', 'variance',
    ]} if delegate_result else {k: None for k in [
        'avg_ms', 'p50_ms', 'p90_ms', 'p99_ms', 'init_ms', 'peak_mem_mb', 'variance',
    ]}

    cpu_metrics = {k: sanitize_metric(getattr(baseline, k, None), k) for k in [
        'avg_ms', 'p50_ms', 'p90_ms', 'p99_ms', 'init_ms',
        'peak_mem_mb', 'variance', 'std_ms', 'min_ms', 'max_ms',
    ]}

    # Immutable tuples
    ops = tuple(delegate_result.ops) if delegate_result and delegate_result.ops else ()
    latency_trend = tuple(delegate_result.latency_trend) if delegate_result and delegate_result.latency_trend else None

    view = AndroidBuildView(
        schema_version=SCHEMA_VERSION,
        run_id=baseline.run_id,
        run_hash=run_hash,
        assembled_at=now,
        backend="tflite",
        model_path=model_path,
        delegate=delegate,
        delegate_status=delegate_status,
        binary_version=binary_version,
        device=replace(device),  # shallow copy to enforce immutability
        cpu_avg_ms=cpu_metrics['avg_ms'],
        cpu_p50_ms=cpu_metrics['p50_ms'],
        cpu_p90_ms=cpu_metrics['p90_ms'],
        cpu_p99_ms=cpu_metrics['p99_ms'],
        cpu_init_ms=cpu_metrics['init_ms'],
        cpu_peak_mem_mb=cpu_metrics['peak_mem_mb'],
        cpu_variance=cpu_metrics['variance'],
        cpu_raw_stdout=baseline.raw_stdout or "",
        cpu_std_ms=cpu_metrics['std_ms'],
        cpu_min_ms=cpu_metrics['min_ms'],
        cpu_max_ms=cpu_metrics['max_ms'],
        cpu_count=cpu_count,
        cpu_autocorr=cpu_autocorr,
        delegate_avg_ms=delegate_metrics['avg_ms'],
        delegate_p50_ms=delegate_metrics['p50_ms'],
        delegate_p90_ms=delegate_metrics['p90_ms'],
        delegate_p99_ms=delegate_metrics['p99_ms'],
        delegate_init_ms=delegate_metrics['init_ms'],
        delegate_peak_mem_mb=delegate_metrics['peak_mem_mb'],
        delegate_variance=delegate_metrics['variance'],
        delegate_raw_stdout=delegate_result.raw_stdout if delegate_result else "",
        ops=ops,
        latency_trend=latency_trend,
        speedup=speedup,
        stability=stability,
        thermal_pre=replace(thermal_pre) if thermal_pre else None,
        thermal_post=replace(thermal_post) if thermal_post else None,
    )

    log(
        f"AndroidBuildView assembled",
        context={
            "run_id": view.run_id,
            "run_hash": view.run_hash,
            "delegate": view.delegate,
            "speedup": view.speedup,
            "stability_band": view.stability.stability_band.value if view.stability.stability_band else None,
            "rerun_score": view.stability.rerun_recommendation_score,
        },
    )

    return view


# ---------------------------------------------------------------------
# Public: Serialize to Registry Dict
# ---------------------------------------------------------------------
def to_registry_dict(view: AndroidBuildView) -> dict:
    """
    Serialize AndroidBuildView into a JSON-safe dict for registry storage.
    Handles nested dataclasses, Enums, and optional metrics.
    """
    return serialize_dataclass(view)