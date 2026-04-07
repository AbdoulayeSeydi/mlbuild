"""
mlbuild.platforms.android.history

Persistent device behavior history.

Responsibilities:
- Append AndroidBuildView run summaries to per-device history
- Read historical runs for trend analysis
- Derive device-level behavioral tendencies (thermal, stability)
- Never block a benchmark run — all writes are best-effort

Design rules:
- History is append-only. Runs are never mutated after write.
- Storage is ~/.mlbuild/device_history.json
- Keyed by DeviceProfile.fingerprint — one entry per device OS version.
- Writes are atomic (tmp + replace) — no corrupt state on crash.
- Read failures return empty history — never raise to caller.
- Derived tendencies (thermal_tendency, avg_stability) are computed
  on read, not stored — always reflect current history accurately.
- Structured JSON logging for CI traceability.
"""

from __future__ import annotations
import sys

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
import shutil
import tempfile

from mlbuild.platforms.android.result import AndroidBuildView

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DEBUG = os.getenv("MLBUILD_DEBUG") == "1"
DEFAULT_HISTORY_DIR = Path.home() / ".mlbuild" / "device_history"
MAX_RUNS_PER_DEVICE = 500  # configurable per device if needed
LOG_FLOOD_THRESHOLD = 100  # max log entries per second per device

# ---------------------------------------------------------------------
# Logging (thread- and process-safe)
# ---------------------------------------------------------------------
_logger_lock: threading.Lock = threading.Lock()
_logger_instance: Optional[logging.Logger] = None

def _get_logger() -> logging.Logger:
    global _logger_instance
    with _logger_lock:
        if _logger_instance is None:
            logger = logging.getLogger("mlbuild.history")
            logger.propagate = False
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(
                logging.Formatter(
                    '{"time": "%(asctime)sZ", "level": "%(levelname)s", "logger": "%(name)s", "msg": %(message)s}',
                    datefmt="%Y-%m-%dT%H:%M:%S"
                )
            )
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
            _logger_instance = logger
    return _logger_instance

def _log(msg: str, level: str = "INFO", context: Optional[Dict[str, Any]] = None) -> None:
    logger = _get_logger()
    payload = {"msg": msg}
    if context:
        payload.update(context)
    level = level.upper()
    if level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        level = "INFO"
    getattr(logger, level.lower())(json.dumps(payload, default=str))

# ---------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class RunSummary:
    run_id: str
    run_hash: str
    assembled_at: str
    model_path: str
    delegate: str
    cpu_avg_ms: Optional[float]
    delegate_avg_ms: Optional[float]
    speedup: Optional[float]
    stability_score: Optional[float]
    stability_band: Optional[str]
    thermal_delta_c: Optional[float]
    thermally_unstable: bool
    rerun_score: float

    @staticmethod
    def validate_numeric(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            if not (value != value or value == float("inf") or value == float("-inf")):
                return float(value)
        return None

    @classmethod
    def from_view(cls, view: AndroidBuildView) -> RunSummary:
        s = view.stability
        cpu_avg_ms = cls.validate_numeric(view.cpu_avg_ms)
        delegate_avg_ms = cls.validate_numeric(view.delegate_avg_ms)
        speedup = cls.validate_numeric(view.speedup)
        thermal_delta_c = cls.validate_numeric(s.thermal_delta_c if s else None)
        stability_score = cls.validate_numeric(s.stability_score if s else None)
        rerun_score = cls.validate_numeric(s.rerun_recommendation_score if s else 0.0) or 0.0

        return cls(
            run_id=view.run_id,
            run_hash=view.run_hash,
            assembled_at=view.assembled_at,
            model_path=view.model_path,
            delegate=view.delegate,
            cpu_avg_ms=cpu_avg_ms,
            delegate_avg_ms=delegate_avg_ms,
            speedup=speedup,
            stability_score=stability_score,
            stability_band=s.stability_band.value if s and s.stability_band else None,
            thermal_delta_c=thermal_delta_c,
            thermally_unstable=bool(s.thermally_unstable if s else False),
            rerun_score=rerun_score,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> RunSummary:
        return cls(
            run_id=str(d.get("run_id", "")),
            run_hash=str(d.get("run_hash", "")),
            assembled_at=str(d.get("assembled_at", "")),
            model_path=str(d.get("model_path", "")),
            delegate=str(d.get("delegate", "CPU")),
            cpu_avg_ms=cls.validate_numeric(d.get("cpu_avg_ms")),
            delegate_avg_ms=cls.validate_numeric(d.get("delegate_avg_ms")),
            speedup=cls.validate_numeric(d.get("speedup")),
            stability_score=cls.validate_numeric(d.get("stability_score")),
            stability_band=d.get("stability_band"),
            thermal_delta_c=cls.validate_numeric(d.get("thermal_delta_c")),
            thermally_unstable=bool(d.get("thermally_unstable", False)),
            rerun_score=cls.validate_numeric(d.get("rerun_score", 0.0)) or 0.0,
        )

@dataclass(frozen=True)
class DeviceHistory:
    fingerprint: str
    runs: List[RunSummary]
    thermal_tendency: str
    avg_stability: Optional[float]
    run_count: int

# ---------------------------------------------------------------------
# Thermal / Stability derivation
# ---------------------------------------------------------------------
def _derive_thermal_tendency(runs: List[RunSummary]) -> str:
    if len(runs) < 3:
        return "unknown"
    unstable_count = sum(1 for r in runs if r.thermally_unstable)
    fraction = unstable_count / len(runs)
    return "unstable" if fraction >= 0.2 else "stable"

def _derive_avg_stability(runs: List[RunSummary]) -> Optional[float]:
    scores = [r.stability_score for r in runs if r.stability_score is not None]
    if not scores:
        return None
    return round(sum(scores)/len(scores), 4)

# ---------------------------------------------------------------------
# File I/O (per-device, atomic, thread/process-safe)
# ---------------------------------------------------------------------
_device_file_locks: Dict[str, threading.Lock] = {}

def _get_device_lock(fingerprint: str) -> threading.Lock:
    if fingerprint not in _device_file_locks:
        _device_file_locks[fingerprint] = threading.Lock()
    return _device_file_locks[fingerprint]

def _device_history_path(fingerprint: str) -> Path:
    DEFAULT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    safe = fingerprint.replace("/", "_").replace(":", "-")
    return DEFAULT_HISTORY_DIR / f"{safe}.json"

def _load_device_history(fingerprint: str) -> List[Dict[str, Any]]:
    path = _device_history_path(fingerprint)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        backup = path.with_suffix(".bak")
        shutil.copy(path, backup)
        _log(f"History load failed, backup created: {backup}", "WARNING", {"fingerprint": fingerprint, "exc": str(exc)})
        return []

def _save_device_history(fingerprint: str, runs: List[Dict[str, Any]]) -> None:
    path = _device_history_path(fingerprint)
    tmp_file = tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8")
    try:
        json.dump(runs, tmp_file, indent=2, default=str)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        tmp_file.close()
        os.replace(tmp_file.name, path)
    except Exception as exc:
        _log(f"Failed to save device history: {exc}", "ERROR", {"fingerprint": fingerprint})
        try:
            os.unlink(tmp_file.name)
        except Exception:
            pass

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def append_run(view: AndroidBuildView, fingerprint: str) -> None:
    summary = RunSummary.from_view(view)
    lock = _get_device_lock(fingerprint)
    with lock:
        raw_runs = _load_device_history(fingerprint)
        raw_runs.append(summary.to_dict())
        if len(raw_runs) > MAX_RUNS_PER_DEVICE:
            raw_runs = raw_runs[-MAX_RUNS_PER_DEVICE:]
            _log("History cap enforced", "WARNING", {"fingerprint": fingerprint, "cap": MAX_RUNS_PER_DEVICE})
        _save_device_history(fingerprint, raw_runs)
    _log("Run appended", context={"fingerprint": fingerprint, "run_id": summary.run_id, "delegate": summary.delegate})

def get_device_history(fingerprint: str) -> DeviceHistory:
    raw_runs = _load_device_history(fingerprint)
    runs: List[RunSummary] = []
    for raw in raw_runs:
        try:
            runs.append(RunSummary.from_dict(raw))
        except Exception as exc:
            _log("Skipping malformed run entry", "WARNING", {"fingerprint": fingerprint, "exc": str(exc)})
    thermal = _derive_thermal_tendency(runs)
    stability = _derive_avg_stability(runs)
    return DeviceHistory(
        fingerprint=fingerprint,
        runs=runs,
        thermal_tendency=thermal,
        avg_stability=stability,
        run_count=len(runs)
    )

def get_recent_runs(fingerprint: str, model_path: str, limit: int = 10) -> List[RunSummary]:
    history = get_device_history(fingerprint)
    filtered = [r for r in history.runs if r.model_path == model_path]
    return filtered[-limit:]  # newest last, efficient slice