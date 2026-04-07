"""
mlbuild.platforms.android.device

Main public interface for Android benchmarking.

Responsibilities:
- Orchestrate the full benchmark pipeline end-to-end
- Expose a clean ADBDevice interface to the CLI
- Coordinate: introspect → thermal → deploy → baseline →
              delegate validation → benchmark → stability →
              result assembly → history → recommendation

Design rules:
- This is the only module the CLI touches directly.
- All complexity lives in the modules it coordinates.
- Every step is logged with timing for CI observability.
- Cleanup always runs — even on failure.
- No raw exceptions escape — all errors are typed MLBuild errors.
- Stateless per run — ADBDevice owns no mutable runtime state.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import threading
import time
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from mlbuild.platforms.android import adb
from mlbuild.platforms.android.introspect import DeviceProfile, build_profile
from mlbuild.platforms.android.thermal import capture_snapshot, compute_thermal_score
from mlbuild.platforms.android.deploy import deploy, cleanup, DeployedRun
from mlbuild.platforms.android.baseline import run_cpu_baseline, BaselineResult
from mlbuild.platforms.android.delegate import (
    validate_delegates,
    DelegateValidation,
    DelegateStatus,
)
from mlbuild.platforms.android.benchmark import run_benchmark, BenchmarkResult
from mlbuild.platforms.android.stability import compute_stability_report
from mlbuild.platforms.android.result import assemble, to_registry_dict, AndroidBuildView
from mlbuild.platforms.android.history import append_run
from mlbuild.platforms.android.recomend import recommend, RecommendationResult
from mlbuild.core.errors import DeployError, ExecutionError

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DEBUG = os.getenv("MLBUILD_DEBUG") == "1"

# Defaults can be overridden in BenchmarkConfig
DEFAULT_NUM_RUNS    = 50
DEFAULT_WARMUP_RUNS = 10
DEFAULT_NUM_THREADS = 4
DEFAULT_TIMEOUT     = 600


# ---------------------------------------------------------------------
# Multiprocess-safe Logger
# ---------------------------------------------------------------------

_logger_instance: Optional[logging.Logger] = None
_logger_lock: threading.Lock = threading.Lock()

def _get_logger() -> logging.Logger:
    global _logger_instance
    with _logger_lock:
        if _logger_instance is None:
            logger = multiprocessing.get_logger()
            logger.propagate = False
            handler = logging.StreamHandler(sys.stderr) 
            handler.setFormatter(logging.Formatter(
                '{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "msg": %(message)s}',
                datefmt="%Y-%m-%dT%H:%M:%S",
            ))
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
            _logger_instance = logger
    return _logger_instance

def _log(msg: str, level: str = "INFO", context: Optional[dict] = None):
    logger = _get_logger()
    payload = {"msg": msg}
    if context:
        payload.update(context)
    # Escape quotes safely
    safe_msg = json.dumps(payload, ensure_ascii=False)
    getattr(logger, level.lower(), logger.info)(safe_msg)


# ---------------------------------------------------------------------
# BenchmarkConfig
# ---------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    model_path: Path
    delegate: Optional[str] = None
    num_runs: int = DEFAULT_NUM_RUNS
    warmup_runs: int = DEFAULT_WARMUP_RUNS
    num_threads: int = DEFAULT_NUM_THREADS
    timeout_s: int = DEFAULT_TIMEOUT
    force_validate: bool = False
    history_path: Optional[Path] = None


# ---------------------------------------------------------------------
# RunResult
# ---------------------------------------------------------------------

@dataclass
class RunResult:
    view: Optional[AndroidBuildView]
    recommendation: Optional[RecommendationResult]
    registry_dict: Optional[dict]
    error: Optional[str] = None  # capture partial failures


# ---------------------------------------------------------------------
# ADBDevice
# ---------------------------------------------------------------------

class ADBDevice:
    """Multiprocess-safe, production-ready Android benchmarking interface."""

    def __init__(self, profile: DeviceProfile) -> None:
        self._profile = profile

    @classmethod
    def connect(cls, serial: Optional[str] = None) -> "ADBDevice":
        start = time.time()
        _log("Connecting to device", context={"serial": serial})
        profile = build_profile(serial=serial)
        _log(
            "Device connected",
            context={
                "serial": profile.serial,
                "model": profile.model,
                "api_level": profile.api_level,
                "chipset": profile.chipset,
                "primary_abi": profile.primary_abi,
                "candidates": profile.delegate_candidates,
                "duration_s": round(time.time() - start, 2),
            },
        )
        return cls(profile)

    @property
    def profile(self) -> DeviceProfile:
        return self._profile

    @property
    def serial(self) -> str:
        return self._profile.serial

    @property
    def fingerprint(self) -> str:
        return self._profile.fingerprint

    def run(self, config: BenchmarkConfig) -> RunResult:
        """Full production-safe benchmark pipeline."""
        start = time.time()
        model_path = Path(config.model_path)
        serial = self._profile.serial

        _log("Run started", context={"model": model_path.name, "delegate": config.delegate, "serial": serial})

        deployed: Optional[DeployedRun] = None

        try:
            self._validate_config(config)

            # ---------------- Deploy ----------------
            try:
                deployed = deploy(model_path, self._profile.primary_abi, serial)
                deployed_copy = replace(deployed)  # immutable copy for downstream
                _log("Deploy complete", context={
                    "run_id": deployed_copy.run_id,
                    "binary_version": deployed_copy.binary_version,
                    "remote_dir": deployed_copy.remote_dir
                })
            except Exception as exc:
                return RunResult(None, None, None, error=f"Deploy failed: {exc}")

            # ---------------- Pre-run thermal ----------------
            thermal_pre = self._safe_capture_snapshot(serial)

            # ---------------- CPU baseline ----------------
            try:
                baseline = self._run_baseline(deployed_copy, config)
            except Exception as exc:
                return RunResult(None, None, None, error=f"CPU baseline failed: {exc}")

            # ---------------- Delegate validation ----------------
            validations: Dict[str, DelegateValidation] = {}
            if self._profile.delegate_candidates:
                validations = self._safe_validate_delegates(deployed_copy, baseline, config.force_validate)

            # ---------------- Delegate benchmark ----------------
            delegate_result, active_validation = self._safe_run_delegate(deployed_copy, baseline, config, validations)

            latency_trend = getattr(delegate_result, "latency_trend", None)
            if latency_trend and not isinstance(latency_trend, list):
                latency_trend = None

            # ---------------- Post-run thermal ----------------
            thermal_post = self._safe_capture_snapshot(serial)

            # ---------------- Thermal score ----------------
            thermal_score = compute_thermal_score(
                pre=thermal_pre, post=thermal_post, latency_trend=latency_trend
            )

            # ---------------- Stability ----------------
            p50 = getattr(delegate_result, "p50_ms", None) or baseline.p50_ms
            p90 = getattr(delegate_result, "p90_ms", None) or baseline.p90_ms

            stability = compute_stability_report(
                p50_ms         = p50,
                p90_ms         = p90,
                thermal_score  = thermal_score,
                device_id      = self.fingerprint,
                model_hash     = str(model_path),
                delegate_hash  = config.delegate or "CPU",
                variance       = baseline.variance,
                std_ms         = baseline.std_ms,
                avg_ms         = baseline.avg_ms,
                count          = baseline.count,
                low_confidence = baseline.low_confidence,
            )

            # ---------------- Assemble view ----------------
            view = assemble(
                model_path=str(model_path),
                device=self._profile,
                baseline=baseline,
                binary_version=deployed_copy.binary_version,
                stability=stability,
                delegate_result=delegate_result,
                validation=active_validation,
                thermal_pre=thermal_pre,
                thermal_post=thermal_post,
            )

            self._safe_append_history(view, config.history_path)

            # ---------------- Recommendation ----------------
            rec = recommend(view)

            _log(
                "Run complete",
                context={
                    "run_id": view.run_id,
                    "duration_s": round(time.time() - start, 2),
                    "speedup": getattr(view, "speedup", None),
                    "rec": rec.kind.value if rec else None,
                },
            )

            return RunResult(
                view=view,
                recommendation=rec,
                registry_dict=to_registry_dict(view),
            )

        finally:
            if deployed:
                self._safe_cleanup(deployed)

    # ----------------- Internal Helpers -----------------

    def _validate_config(self, config: BenchmarkConfig):
        model_path = Path(config.model_path)
        if not model_path.exists():
            raise DeployError(f"Model file not found: {model_path}")
        if model_path.suffix.lower() not in (".tflite", ".lite"):
            raise DeployError(f"Unsupported model format {model_path.suffix}")
        if config.delegate and config.delegate.upper() not in ("GPU","NNAPI","HEXAGON","HEXAGON_HTP"):
            raise DeployError(f"Unknown delegate {config.delegate}")

    def _run_baseline(self, deployed: DeployedRun, config: BenchmarkConfig) -> BaselineResult:
        return run_cpu_baseline(
            deployed=deployed,
            num_runs=config.num_runs,
            warmup_runs=config.warmup_runs,
            num_threads=config.num_threads,
            timeout_s=config.timeout_s
        )

    def _safe_validate_delegates(self, deployed: DeployedRun, baseline: BaselineResult, force: bool) -> Dict[str, DelegateValidation]:
        try:
            result = validate_delegates(
                deployed=deployed,
                candidates=self._profile.delegate_candidates,
                baseline=baseline,
                fingerprint=self.fingerprint,
                force=force
            )
            return result or {}
        except Exception as exc:
            _log(f"Delegate validation failed: {exc}", "WARN")
            return {}

    def _safe_run_delegate(
        self, deployed: DeployedRun, baseline: BaselineResult,
        config: BenchmarkConfig, validations: Dict[str, DelegateValidation]
    ) -> Tuple[Optional[BenchmarkResult], Optional[DelegateValidation]]:
        delegate = config.delegate.upper() if config.delegate else None
        if not delegate:
            return None, None
        validation = validations.get(delegate)
        if not validation or validation.status != DelegateStatus.SUPPORTED:
            reason = getattr(validation, "reason", None)
            _log(f"Skipping delegate {delegate}, unsupported: {reason}", "WARN")
            return None, validation
        try:
            result = run_benchmark(
                deployed=deployed,
                delegate=delegate,
                num_runs=config.num_runs,
                warmup_runs=config.warmup_runs,
                num_threads=config.num_threads,
                timeout_s=config.timeout_s,
                cpu_avg_ms=baseline.avg_ms,
                device_fingerprint=self.fingerprint
            )
            return result, validation
        except Exception as exc:
            _log(f"Delegate benchmark failed: {exc}", "WARN")
            return None, validation

    def _safe_capture_snapshot(self, serial: str):
        try:
            return capture_snapshot(serial)
        except Exception as exc:
            _log(f"Thermal snapshot failed (non-fatal): {exc}", "WARN")
            return None

    def _safe_append_history(self, view: AndroidBuildView, history_path: Optional[Path]):
        try:
            kwargs = {"fingerprint": self.fingerprint}
            if history_path:
                kwargs["history_path"] = history_path
            append_run(view, **kwargs)
        except Exception as exc:
            _log(f"History append failed (non-fatal): {exc}", "WARN")

    def _safe_cleanup(self, deployed: DeployedRun):
        try:
            cleanup(deployed)
            _log("Cleanup complete", context={"run_id": deployed.run_id, "remote_dir": deployed.remote_dir})
        except Exception as exc:
            _log(f"Cleanup failed (non-fatal): {exc}", "WARN")

    def __str__(self):
        return f"ADBDevice({self._profile.manufacturer} {self._profile.model}, API {self._profile.api_level}, {self._profile.primary_abi}, serial={self.serial})"

    def __repr__(self):
        return self.__str__()