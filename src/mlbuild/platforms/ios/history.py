"""
mlbuild.platforms.ios.history

Persistent device behavior history for iOS benchmark runs.

Responsibilities:
- Append iOSBuildView run summaries to per-device history
- Read historical runs for trend analysis
- Derive device-level behavioral tendencies (thermal, stability)
- Never block a benchmark run — all writes are best-effort

Design rules:
- History is append-only. Runs are never mutated after write.
- Storage is ~/.mlbuild/device_history/ — one JSON file per device fingerprint.
- Keyed by DeviceProfile.fingerprint — one entry per device iOS version.
- Writes are atomic (tmp + replace) — no corrupt state on crash.
- Read failures return empty history — never raise to caller.
- Derived tendencies computed on read, not stored — always current.
- Structured JSON logging for CI traceability.

Key differences from Android:
- Consumes iOSBuildView instead of AndroidBuildView.
- thermal_tendency derived from IOSThermalState transitions, not delta_c.
- thermal_delta_c replaced by thermal_state_post (discrete enum value).
- thermally_unstable sourced from StabilityReport — same field, same logic.
- is_simulator stored per run — simulator runs excluded from thermal tendency.
- No thermal_delta_c field — iOS doesn't expose raw temperatures.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from mlbuild.platforms.ios.result import iOSBuildView


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DEBUG                = os.getenv("MLBUILD_DEBUG") == "1"
DEFAULT_HISTORY_DIR  = Path.home() / ".mlbuild" / "device_history"
MAX_RUNS_PER_DEVICE  = 500


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

_logger_lock:     threading.Lock           = threading.Lock()
_logger_instance: Optional[logging.Logger] = None


def _get_logger() -> logging.Logger:
    global _logger_instance
    with _logger_lock:
        if _logger_instance is None:
            logger = logging.getLogger("mlbuild.ios.history")
            logger.propagate = False
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter(
                '{"time": "%(asctime)sZ", "level": "%(levelname)s", '
                '"logger": "%(name)s", "msg": %(message)s}',
                datefmt="%Y-%m-%dT%H:%M:%S",
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
    level = level.upper()
    if level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        level = "INFO"
    getattr(logger, level.lower())(json.dumps(payload, default=str))


# ---------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class RunSummary:
    """
    Immutable per-run summary stored in device history.

    thermal_state_post: raw string value of IOSThermalState enum, or None.
    is_simulator: stored so simulator runs can be excluded from
                  thermal tendency derivation — they never throttle.
    """
    run_id:             str
    run_hash:           str
    assembled_at:       str
    model_path:         str
    delegate:           str
    compute_units_used: Optional[str]
    cpu_avg_ms:         Optional[float]
    delegate_avg_ms:    Optional[float]
    speedup:            Optional[float]
    stability_score:    Optional[float]
    stability_band:     Optional[str]
    thermal_state_post: Optional[str]       # IOSThermalState.value or None
    thermally_unstable: bool
    rerun_score:        float
    is_simulator:       bool

    @staticmethod
    def _safe_float(val: Any) -> Optional[float]:
        if val is None:
            return None
        try:
            f = float(val)
            if f != f or f == float("inf") or f == float("-inf"):
                return None
            return f
        except (TypeError, ValueError):
            return None

    @classmethod
    def from_view(cls, view: iOSBuildView) -> RunSummary:
        s = view.stability

        # thermal_state_post — discrete enum, stored as string
        thermal_state_post: Optional[str] = None
        if view.thermal_state_post is not None:
            thermal_state_post = view.thermal_state_post.value

        return cls(
            run_id             = view.run_id,
            run_hash           = view.run_hash,
            assembled_at       = view.assembled_at,
            model_path         = view.model_path,
            delegate           = view.delegate,
            compute_units_used = view.compute_units_used,
            cpu_avg_ms         = cls._safe_float(view.cpu_avg_ms),
            delegate_avg_ms    = cls._safe_float(view.delegate_avg_ms),
            speedup            = cls._safe_float(view.speedup),
            stability_score    = cls._safe_float(s.stability_score if s else None),
            stability_band     = s.stability_band.value if s and s.stability_band else None,
            thermal_state_post = thermal_state_post,
            thermally_unstable = bool(s.thermally_unstable if s else False),
            rerun_score        = cls._safe_float(s.rerun_recommendation_score if s else 0.0) or 0.0,
            is_simulator       = view.is_simulator,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> RunSummary:
        return cls(
            run_id             = str(d.get("run_id", "")),
            run_hash           = str(d.get("run_hash", "")),
            assembled_at       = str(d.get("assembled_at", "")),
            model_path         = str(d.get("model_path", "")),
            delegate           = str(d.get("delegate", "CPU")),
            compute_units_used = d.get("compute_units_used"),
            cpu_avg_ms         = cls._safe_float(d.get("cpu_avg_ms")),
            delegate_avg_ms    = cls._safe_float(d.get("delegate_avg_ms")),
            speedup            = cls._safe_float(d.get("speedup")),
            stability_score    = cls._safe_float(d.get("stability_score")),
            stability_band     = d.get("stability_band"),
            thermal_state_post = d.get("thermal_state_post"),
            thermally_unstable = bool(d.get("thermally_unstable", False)),
            rerun_score        = cls._safe_float(d.get("rerun_score", 0.0)) or 0.0,
            is_simulator       = bool(d.get("is_simulator", False)),
        )


@dataclass(frozen=True)
class DeviceHistory:
    fingerprint:       str
    runs:              List[RunSummary]
    thermal_tendency:  str              # "stable" | "unstable" | "unknown"
    avg_stability:     Optional[float]
    run_count:         int
    simulator_run_count: int            # how many runs were on simulator


# ---------------------------------------------------------------------
# Tendency Derivation
# ---------------------------------------------------------------------

def _derive_thermal_tendency(runs: List[RunSummary]) -> str:
    """
    Derived from real device runs only — simulator runs excluded.
    Requires at least 3 real device runs for a meaningful tendency.

    iOS thermal tendency uses thermally_unstable flag (derived in
    stability.py from discrete state transitions) rather than delta_c.
    """
    real_runs = [r for r in runs if not r.is_simulator]

    if len(real_runs) < 3:
        return "unknown"

    unstable_count = sum(1 for r in real_runs if r.thermally_unstable)
    fraction = unstable_count / len(real_runs)
    return "unstable" if fraction >= 0.2 else "stable"


def _derive_avg_stability(runs: List[RunSummary]) -> Optional[float]:
    scores = [r.stability_score for r in runs if r.stability_score is not None]
    if not scores:
        return None
    return round(sum(scores) / len(scores), 4)


# ---------------------------------------------------------------------
# File I/O (per-device, atomic, thread-safe)
# ---------------------------------------------------------------------

_device_file_locks: Dict[str, threading.Lock] = {}
_locks_mutex = threading.Lock()


def _get_device_lock(fingerprint: str) -> threading.Lock:
    with _locks_mutex:
        if fingerprint not in _device_file_locks:
            _device_file_locks[fingerprint] = threading.Lock()
        return _device_file_locks[fingerprint]


def _device_history_path(fingerprint: str) -> Path:
    DEFAULT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    safe = fingerprint.replace("/", "_").replace(":", "-")
    return DEFAULT_HISTORY_DIR / f"ios_{safe}.json"


def _load_device_history(fingerprint: str) -> List[Dict[str, Any]]:
    path = _device_history_path(fingerprint)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        backup = path.with_suffix(".bak")
        shutil.copy(path, backup)
        _log(
            f"History load failed — backup created: {backup}",
            "WARNING",
            {"fingerprint": fingerprint, "exc": str(exc)},
        )
        return []


def _save_device_history(
    fingerprint: str,
    runs: List[Dict[str, Any]],
) -> None:
    path = _device_history_path(fingerprint)
    tmp = tempfile.NamedTemporaryFile(
        "w", delete=False, dir=path.parent, encoding="utf-8"
    )
    try:
        json.dump(runs, tmp, indent=2, default=str)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        os.replace(tmp.name, path)
    except Exception as exc:
        _log(
            f"Failed to save device history: {exc}",
            "ERROR",
            {"fingerprint": fingerprint},
        )
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def append_run(view: iOSBuildView, fingerprint: str) -> None:
    """
    Append a run summary to device history. Best-effort — never raises.
    Simulator runs are appended but flagged with is_simulator=True.
    """
    try:
        summary = RunSummary.from_view(view)
        lock = _get_device_lock(fingerprint)

        with lock:
            raw_runs = _load_device_history(fingerprint)
            raw_runs.append(summary.to_dict())

            if len(raw_runs) > MAX_RUNS_PER_DEVICE:
                raw_runs = raw_runs[-MAX_RUNS_PER_DEVICE:]
                _log(
                    "History cap enforced",
                    "WARNING",
                    {"fingerprint": fingerprint, "cap": MAX_RUNS_PER_DEVICE},
                )

            _save_device_history(fingerprint, raw_runs)

        _log(
            "Run appended",
            context={
                "fingerprint":       fingerprint,
                "run_id":            summary.run_id,
                "delegate":          summary.delegate,
                "compute_units_used": summary.compute_units_used,
                "is_simulator":      summary.is_simulator,
            },
        )

    except Exception as exc:
        # Never block the caller — history write failure is non-fatal
        _log(
            f"append_run failed (non-fatal): {exc}",
            "ERROR",
            {"fingerprint": fingerprint},
        )


def get_device_history(fingerprint: str) -> DeviceHistory:
    """
    Load and derive full DeviceHistory for a fingerprint.
    Returns empty history on any read failure — never raises.
    """
    raw_runs = _load_device_history(fingerprint)
    runs: List[RunSummary] = []

    for raw in raw_runs:
        try:
            runs.append(RunSummary.from_dict(raw))
        except Exception as exc:
            _log(
                "Skipping malformed run entry",
                "WARNING",
                {"fingerprint": fingerprint, "exc": str(exc)},
            )

    return DeviceHistory(
        fingerprint         = fingerprint,
        runs                = runs,
        thermal_tendency    = _derive_thermal_tendency(runs),
        avg_stability       = _derive_avg_stability(runs),
        run_count           = len(runs),
        simulator_run_count = sum(1 for r in runs if r.is_simulator),
    )


def get_recent_runs(
    fingerprint: str,
    model_path:  str,
    limit:       int = 10,
    *,
    real_device_only: bool = False,
) -> List[RunSummary]:
    """
    Return the most recent runs for a given model on this device.

    real_device_only=True filters out simulator runs — useful for
    comparing against ANE benchmark history only.
    """
    history = get_device_history(fingerprint)
    filtered = [r for r in history.runs if r.model_path == model_path]

    if real_device_only:
        filtered = [r for r in filtered if not r.is_simulator]

    return filtered[-limit:]