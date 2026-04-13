"""
mlbuild.platforms.ios.device

Main public interface for iOS benchmarking.

Responsibilities:
- Orchestrate the full benchmark pipeline end-to-end
- Expose a clean IDBDevice interface to the CLI
- Coordinate: companion check → introspect → thermal → deploy →
              baseline → delegate validation → benchmark →
              stability → result assembly → history → recommendation

Design rules:
- This is the only module the CLI touches directly.
- All complexity lives in the modules it coordinates.
- Every step is logged with timing for CI observability.
- Cleanup always runs — even on failure.
- No raw exceptions escape — all errors are typed MLBuild errors.
- Stateless per run — IDBDevice owns no mutable runtime state.

Key differences from Android:
- idb.ensure_companion() called before profile build — companion must
  be running before any idb command is issued.
- deploy() takes is_simulator and signed_app — no ABI selection.
- thermal snapshots are None on simulator — capture_snapshot handles this.
- validate_delegates() receives is_simulator — ANE auto-skipped there.
- No num_threads in BenchmarkConfig — CoreML manages its own threading.
- Model format validation accepts .mlpackage and .mlmodelc.
- SKIPPED delegates are logged in status table, not treated as failures.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Optional, Tuple

from mlbuild.platforms.ios import idb
from mlbuild.platforms.ios.introspect import DeviceProfile, build_profile
from mlbuild.platforms.ios.thermal import capture_snapshot, compute_thermal_score
from mlbuild.platforms.ios.deploy import deploy, cleanup, DeployedRun, BUNDLE_ID
from mlbuild.platforms.ios.baseline import run_cpu_baseline, BaselineResult
from mlbuild.platforms.ios.delegate import (
    validate_delegates,
    DelegateValidation,
    DelegateStatus,
)
from mlbuild.platforms.ios.benchmark import run_benchmark, BenchmarkResult
from mlbuild.platforms.ios.stability import compute_stability_report
from mlbuild.platforms.ios.result import assemble, to_registry_dict, iOSBuildView
from mlbuild.platforms.ios.history import append_run
from mlbuild.platforms.ios.recommend import recommend, RecommendationResult
from mlbuild.core.errors import IDBDeployError, IDBExecutionError


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DEBUG               = os.getenv("MLBUILD_DEBUG") == "1"
DEFAULT_NUM_RUNS    = 50
DEFAULT_WARMUP_RUNS = 10
DEFAULT_TIMEOUT     = 600

_VALID_MODEL_SUFFIXES = {".mlpackage", ".mlmodelc", ".mlmodel"}
_VALID_DELEGATES      = {"CPU", "GPU", "ANE", "ANE_EXPLICIT"}


# ---------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------

_logger_instance: Optional[logging.Logger] = None
_logger_lock      = threading.Lock()


def _get_logger() -> logging.Logger:
    global _logger_instance
    with _logger_lock:
        if _logger_instance is None:
            logger = logging.getLogger("mlbuild.ios.device")
            logger.propagate = False
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter(
                '{"time": "%(asctime)s", "level": "%(levelname)s", '
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
    context: Optional[dict] = None,
) -> None:
    logger = _get_logger()
    payload = {"msg": msg}
    if context:
        payload.update(context)
    getattr(logger, level.lower(), logger.info)(
        json.dumps(payload, ensure_ascii=False, default=str)
    )


# ---------------------------------------------------------------------
# BenchmarkConfig
# ---------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    model_path:     Path
    delegate:       Optional[str] = None       # GPU, ANE, ANE_EXPLICIT, or None (CPU)
    num_runs:       int           = DEFAULT_NUM_RUNS
    warmup_runs:    int           = DEFAULT_WARMUP_RUNS
    timeout_s:      int           = DEFAULT_TIMEOUT
    force_validate: bool          = False
    signed_app:     Optional[Path] = None      # required for real device
    history_path:   Optional[Path] = None


# ---------------------------------------------------------------------
# RunResult
# ---------------------------------------------------------------------

@dataclass
class RunResult:
    view:           Optional[iOSBuildView]
    recommendation: Optional[RecommendationResult]
    registry_dict:  Optional[dict]
    error:          Optional[str] = None       # captures partial failures


# ---------------------------------------------------------------------
# IDBDevice
# ---------------------------------------------------------------------

class IDBDevice:
    """
    Clean public interface for iOS benchmarking.
    The CLI instantiates this and calls run().
    All pipeline complexity is in the modules this coordinates.
    """

    def __init__(self, profile: DeviceProfile) -> None:
        self._profile = profile

    @classmethod
    def connect(cls, udid: Optional[str] = None) -> "IDBDevice":
        """
        Ensure idb_companion is running, then build DeviceProfile.
        companion check must happen before any idb command.
        """
        start = time.time()
        _log("Connecting to iOS target", context={"udid": udid})

        idb.ensure_companion()

        profile = build_profile(udid=udid)

        _log(
            "Target connected",
            context={
                "udid":         profile.udid,
                "name":         profile.name,
                "model":        profile.model,
                "ios_version":  profile.ios_version,
                "chip":         profile.chip,
                "is_simulator": profile.is_simulator,
                "has_ane":      profile.has_ane,
                "candidates":   profile.delegate_candidates,
                "duration_s":   round(time.time() - start, 2),
            },
        )

        return cls(profile)

    @property
    def profile(self) -> DeviceProfile:
        return self._profile

    @property
    def udid(self) -> str:
        return self._profile.udid

    @property
    def fingerprint(self) -> str:
        return self._profile.fingerprint

    def run(self, config: BenchmarkConfig) -> RunResult:
        """
        Full benchmark pipeline — end to end.

        Steps:
        1.  Validate config
        2.  Deploy (install app + push model)
        3.  Pre-run thermal snapshot
        4.  CPU baseline
        5.  Delegate validation
        6.  Delegate benchmark
        7.  Post-run thermal snapshot
        8.  Thermal score
        9.  Stability report
        10. Assemble iOSBuildView
        11. Append to history
        12. Recommendation
        13. Cleanup (always)
        """
        start      = time.time()
        model_path = Path(config.model_path)
        is_sim     = self._profile.is_simulator

        _log(
            "Run started",
            context={
                "model":        model_path.name,
                "delegate":     config.delegate,
                "udid":         self._profile.udid,
                "is_simulator": is_sim,
            },
        )

        deployed: Optional[DeployedRun] = None

        try:
            self._validate_config(config)

            # ---- 2. Deploy ----
            try:
                deployed = deploy(
                    model_path,
                    is_simulator = is_sim,
                    udid         = self._profile.udid,
                    signed_app   = config.signed_app,
                )
                _log(
                    "Deploy complete",
                    context={
                        "run_id":         deployed.run_id,
                        "binary_version": deployed.binary_version,
                        "remote_dir":     deployed.remote_dir,
                    },
                )
            except Exception as exc:
                return RunResult(None, None, None, error=f"Deploy failed: {exc}")

            # ---- 3. Pre-run thermal snapshot ----
            thermal_pre = self._safe_capture_snapshot(deployed)

            # ---- 4. CPU baseline ----
            try:
                baseline = run_cpu_baseline(
                    deployed,
                    num_runs    = config.num_runs,
                    warmup_runs = config.warmup_runs,
                    timeout_s   = config.timeout_s,
                )
            except Exception as exc:
                return RunResult(None, None, None, error=f"CPU baseline failed: {exc}")

            # ---- 5. Delegate validation ----
            validations: Dict[str, DelegateValidation] = {}
            if self._profile.delegate_candidates:
                validations = self._safe_validate_delegates(
                    deployed, baseline, config.force_validate
                )
                self._log_delegate_status_table(validations)

            # ---- 6. Delegate benchmark ----
            delegate_result, active_validation = self._safe_run_delegate(
                deployed, baseline, config, validations
            )

            latency_trend = getattr(delegate_result, "latency_trend", None)
            if latency_trend and not isinstance(latency_trend, list):
                latency_trend = None
            if not latency_trend and baseline:
                latency_trend = getattr(baseline, "latency_trend", None)

            # ---- 7. Post-run thermal snapshot ----
            thermal_post = self._safe_capture_snapshot(deployed, runner_stdout=baseline.raw_stdout)

            # Override pre-snapshot state from thermal_boot event
            if thermal_pre is not None and getattr(baseline, "thermal_state_pre", None):
                from mlbuild.platforms.ios.thermal import IOSThermalState, ThermalSnapshot
                pre_state = IOSThermalState.from_string(baseline.thermal_state_pre)
                thermal_pre = ThermalSnapshot(state=pre_state, is_simulated=self._profile.is_simulator)

            # ---- 8. Thermal score ----
            # compute_thermal_score returns None on simulator — handled inside
            thermal_score = compute_thermal_score(
                pre           = thermal_pre,
                post          = thermal_post,
                latency_trend = latency_trend,
            )

            # ---- 9. Stability ----
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
                low_confidence = baseline.low_confidence,
            )

            # ---- 10. Assemble view ----
            view = assemble(
                model_path      = str(model_path),
                device          = self._profile,
                baseline        = baseline,
                binary_version  = deployed.binary_version,
                stability       = stability,
                delegate_result = delegate_result,
                validation      = active_validation,
                thermal_pre     = thermal_pre,
                thermal_post    = thermal_post,
                thermal_score   = thermal_score,
            )

            # ---- 11. History ----
            self._safe_append_history(view)

            # ---- 12. Recommendation ----
            rec = recommend(view)

            _log(
                "Run complete",
                context={
                    "run_id":     view.run_id,
                    "duration_s": round(time.time() - start, 2),
                    "speedup":    getattr(view, "speedup", None),
                    "rec":        rec.kind.value if rec else None,
                    "is_simulator": is_sim,
                },
            )

            return RunResult(
                view           = view,
                recommendation = rec,
                registry_dict  = to_registry_dict(view),
            )

        finally:
            if deployed:
                self._safe_cleanup(deployed)

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _validate_config(self, config: BenchmarkConfig) -> None:
        model_path = Path(config.model_path)

        if not model_path.exists():
            raise IDBDeployError(detail=f"Model file not found: {model_path}")

        if model_path.suffix.lower() not in _VALID_MODEL_SUFFIXES:
            raise IDBDeployError(
                detail=(
                    f"Unsupported model format '{model_path.suffix}'. "
                    f"Expected: {', '.join(_VALID_MODEL_SUFFIXES)}"
                )
            )

        if config.delegate and config.delegate.upper() not in _VALID_DELEGATES:
            raise IDBDeployError(detail=f"Unknown delegate '{config.delegate}'")

        # Real device requires signed app
        if not self._profile.is_simulator and config.signed_app is None:
            from mlbuild.core.errors import UnsignedBinaryError
            raise UnsignedBinaryError()

    def _safe_capture_snapshot(self, deployed: DeployedRun, runner_stdout: str = ""):
        """
        Thermal snapshot from last runner stdout on real device.
        Returns None on simulator or any failure — always non-fatal.
        """
        try:
            return capture_snapshot(
                runner_stdout = runner_stdout,
                is_simulated  = self._profile.is_simulator,
            )
        except Exception as exc:
            _log(f"Thermal snapshot failed (non-fatal): {exc}", "WARN")
            return None

    def _safe_validate_delegates(
        self,
        deployed:       DeployedRun,
        baseline:       BaselineResult,
        force_validate: bool,
    ) -> Dict[str, DelegateValidation]:
        try:
            return validate_delegates(
                deployed     = deployed,
                candidates   = self._profile.delegate_candidates,
                baseline     = baseline,
                fingerprint  = self.fingerprint,
                is_simulator = self._profile.is_simulator,
                force        = force_validate,
            ) or {}
        except Exception as exc:
            _log(f"Delegate validation failed (non-fatal): {exc}", "WARN")
            return {}

    def _log_delegate_status_table(
        self,
        validations: Dict[str, DelegateValidation],
    ) -> None:
        """
        Emit delegate status table to log for CLI to display.
        SKIPPED delegates are shown explicitly — not silently omitted.
        """
        for delegate, v in validations.items():
            _log(
                f"Delegate status: {delegate} → {v.status.value}",
                context={
                    "delegate":           delegate,
                    "status":             v.status.value,
                    "avg_ms":             v.avg_ms,
                    "compute_units_used": v.compute_units_used,
                    "reason":             v.reason,
                },
            )

    def _safe_run_delegate(
        self,
        deployed:    DeployedRun,
        baseline:    BaselineResult,
        config:      BenchmarkConfig,
        validations: Dict[str, DelegateValidation],
    ) -> Tuple[Optional[BenchmarkResult], Optional[DelegateValidation]]:
        delegate = config.delegate.upper() if config.delegate else None

        if not delegate:
            # Auto-select fastest delegate that beats CPU
            cpu_avg = baseline.avg_ms or float("inf")
            faster = {
                k: v for k, v in validations.items()
                if v.status == DelegateStatus.SUPPORTED
                and v.avg_ms is not None
                and v.avg_ms < cpu_avg
            }
            if not faster:
                _log("No delegate faster than CPU — using CPU baseline")
                return None, None
            delegate = min(faster, key=lambda k: faster[k].avg_ms)
            _log(f"Auto-selected delegate: {delegate} ({faster[delegate].avg_ms:.2f}ms vs CPU {cpu_avg:.2f}ms)")

        validation = validations.get(delegate)

        if not validation or validation.status != DelegateStatus.SUPPORTED:
            reason = getattr(validation, "reason", "not validated")
            status = getattr(validation, "status", None)
            _log(
                f"Skipping delegate benchmark: {delegate}",
                "WARN",
                context={
                    "status": status.value if status else None,
                    "reason": reason,
                },
            )
            return None, validation

        try:
            result = run_benchmark(
                deployed           = deployed,
                delegate           = delegate,
                num_runs           = config.num_runs,
                warmup_runs        = config.warmup_runs,
                timeout_s          = config.timeout_s,
                cpu_avg_ms         = baseline.avg_ms,
                device_fingerprint = self.fingerprint,
            )
            return result, validation

        except Exception as exc:
            _log(f"Delegate benchmark failed (non-fatal): {exc}", "WARN")
            return None, validation

    def _safe_append_history(self, view: iOSBuildView) -> None:
        try:
            append_run(view, fingerprint=self.fingerprint)
        except Exception as exc:
            _log(f"History append failed (non-fatal): {exc}", "WARN")

    def _safe_cleanup(self, deployed: DeployedRun) -> None:
        try:
            cleanup(deployed)
            _log(
                "Cleanup complete",
                context={
                    "run_id":     deployed.run_id,
                    "remote_dir": deployed.remote_dir,
                },
            )
        except Exception as exc:
            _log(f"Cleanup failed (non-fatal): {exc}", "WARN")



    def __str__(self) -> str:
        sim = " [SIMULATOR]" if self._profile.is_simulator else ""
        return (
            f"IDBDevice({self._profile.name}, "
            f"{self._profile.chip}, "
            f"iOS {self._profile.ios_version}, "
            f"udid={self.udid}{sim})"
        )

    def __repr__(self) -> str:
        return self.__str__()