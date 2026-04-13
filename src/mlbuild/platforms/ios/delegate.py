"""
mlbuild.platforms.ios.delegate

Runtime delegate validation and caching.

Responsibilities:
- Validate delegate candidates against actual device behavior
- Cache results per device fingerprint to avoid re-running validation
- Detect silent fallback via adaptive latency-based threshold
- Detect output divergence via consistency.py
- Auto-skip ANE/ANE_EXPLICIT on simulator — SKIPPED is a first-class status
- Return structured DelegateValidation results

Design rules:
- Cache is checked first. Validation runs once per device per iOS version.
- Fallback detection is latency-based, not log-based.
- Threshold is adaptive: max(0.10, cpu_variance * 2)
- INCONSISTENT is a first-class status — not collapsed into UNSUPPORTED.
- SKIPPED is a first-class status — ANE on simulator, never an error.
- Validation uses a short run (5 iterations) — not a full benchmark.
- compute_units_used from runner output is ground truth for fallback detection.
- All status decisions are logged with their reasoning.

Key differences from Android:
- No adb shell — runner launched via idb launch with --compute_units arg.
- Delegate expressed as MLComputeUnits string, not binary flag.
- compute_units_used parsed from runner stdout — CoreML can silently
  fall back even when a specific compute unit is requested.
- ANE/ANE_EXPLICIT auto-skipped on simulator via is_simulator gate.
- No process group kill — idb launch killed directly.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

from mlbuild.platforms.ios.deploy import DeployedRun
from mlbuild.platforms.ios.baseline import BaselineResult
from mlbuild.platforms.ios.consistency import check_consistency, ConsistencyResult
from mlbuild.core.errors import IDBNotFoundError


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DEBUG = os.getenv("MLBUILD_DEBUG") == "1"

DEFAULT_CACHE_PATH      = Path.home() / ".mlbuild" / "device_cache.json"
DEFAULT_NUM_RUNS        = 5
DEFAULT_WARMUP          = 2
DEFAULT_TIMEOUT         = 60
MIN_FALLBACK_THRESHOLD  = 0.10

BUNDLE_ID = "com.mlbuild.MLBuildRunner"

# Delegate → MLComputeUnits string passed to runner
_COMPUTE_UNITS: Dict[str, str] = {
    "CPU":          "cpuOnly",
    "GPU":          "cpuAndGPU",
    "ANE":          "all",
    "ANE_EXPLICIT": "cpuAndNeuralEngine",
}

# Delegates that require real ANE hardware — auto-skipped on simulator
_ANE_DELEGATES = {"ANE", "ANE_EXPLICIT"}


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def _log(msg: str, udid: Optional[str] = None, level: str = "INFO") -> None:
    if not DEBUG: return
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    prefix = f"[{udid}]" if udid else ""
    print(f"{ts} [{level}]{prefix} {msg}", flush=True, file=sys.stderr)


# ---------------------------------------------------------------------
# Delegate Status
# ---------------------------------------------------------------------

class DelegateStatus(str, Enum):
    SUPPORTED    = "supported"
    FALLBACK     = "fallback"       # ran on CPU despite delegate request
    UNSUPPORTED  = "unsupported"    # crashed or rejected
    INCONSISTENT = "inconsistent"   # passed but output diverged
    SKIPPED      = "skipped"        # ANE on simulator — structural, not a failure


# ---------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------

@dataclass
class DelegateValidation:
    delegate:         str
    status:           DelegateStatus
    avg_ms:           Optional[float]
    compute_units_used: Optional[str]   # what CoreML actually ran on
    consistency:      Optional[ConsistencyResult]
    reason:           str
    validated_at:     str

    def to_cache_dict(self) -> dict:
        return {
            "status":             self.status.value,
            "avg_ms":             self.avg_ms,
            "compute_units_used": self.compute_units_used,
            "reason":             self.reason,
            "validated_at":       self.validated_at,
        }

    def __str__(self) -> str:
        return (
            f"DelegateValidation(delegate={self.delegate}, "
            f"status={self.status.value}, avg_ms={self.avg_ms}, "
            f"compute_units_used={self.compute_units_used}, "
            f"reason={self.reason!r})"
        )


# ---------------------------------------------------------------------
# Cache (atomic writes)
# ---------------------------------------------------------------------

@contextmanager
def _atomic_write(path: Path):
    tmp = path.with_suffix(".tmp")
    yield tmp
    tmp.replace(path)


def _load_cache(cache_path: Path) -> Dict[str, Any]:
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text())
    except Exception as exc:
        _log(f"Cache load failed — starting fresh: {exc}", level="WARN")
        return {}


def _save_cache(cache_path: Path, cache: Dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with _atomic_write(cache_path) as tmp:
            tmp.write_text(json.dumps(cache, indent=2))
    except Exception as exc:
        _log(f"Cache save failed: {exc}", level="ERROR")


def _cache_key(fingerprint: str, delegate: str) -> str:
    return f"{fingerprint}:{delegate.upper()}"


def _read_cached(
    cache_path: Path,
    fingerprint: str,
    delegate: str,
) -> Optional[DelegateValidation]:
    cache = _load_cache(cache_path)
    entry = cache.get(_cache_key(fingerprint, delegate))
    if not entry:
        return None
    try:
        return DelegateValidation(
            delegate=delegate,
            status=DelegateStatus(entry["status"]),
            avg_ms=entry.get("avg_ms"),
            compute_units_used=entry.get("compute_units_used"),
            consistency=None,
            reason=entry.get("reason", "loaded from cache"),
            validated_at=entry.get("validated_at", "unknown"),
        )
    except Exception as exc:
        _log(f"Malformed cache entry for {delegate}: {exc}", level="WARN")
        return None


def _write_cached(
    cache_path: Path,
    fingerprint: str,
    delegate: str,
    validation: DelegateValidation,
) -> None:
    cache = _load_cache(cache_path)
    cache[_cache_key(fingerprint, delegate)] = validation.to_cache_dict()
    _save_cache(cache_path, cache)
    _log(f"Cached: {delegate} = {validation.status.value}")


# ---------------------------------------------------------------------
# Mini Benchmark
# ---------------------------------------------------------------------

def _idb_bin() -> str:
    path = shutil.which("idb")
    if not path:
        raise IDBNotFoundError()
    return path


def _run_mini_benchmark(
    deployed:      DeployedRun,
    compute_units: str,
    num_runs:      int = DEFAULT_NUM_RUNS,
    warmup:        int = DEFAULT_WARMUP,
    timeout:       int = DEFAULT_TIMEOUT,
) -> tuple[Optional[float], Optional[str], str]:
    """
    Run a short benchmark with the given compute_units.

    Returns (avg_ms, compute_units_used, stdout).

    avg_ms is None on crash or timeout.
    compute_units_used is what CoreML actually ran on — parsed from
    runner stdout. This is the ground truth for fallback detection,
    more reliable than latency comparison alone.
    """
    if not deployed.is_simulator and deployed.udid:
        _binary = "com.mlbuild.MLBuildRunner"
        cmd = [
            "xcrun", "devicectl", "device", "process", "launch",
            "--device", deployed.udid, "--console", "--terminate-existing",
            _binary,
            f"--model={deployed.remote_model_path}",
            f"--num_runs={num_runs}",
            f"--warmup_runs={warmup}",
            f"--compute_units={compute_units}",
        ]
    elif deployed.is_simulator and deployed.udid:
        _r = subprocess.run(
            ["xcrun", "simctl", "get_app_container", deployed.udid, BUNDLE_ID, "app"],
            capture_output=True, text=True, timeout=10,
        )
        _binary = _r.stdout.strip() + "/MLBuildRunner"
        _dr = subprocess.run(
            ["xcrun", "simctl", "get_app_container", deployed.udid, BUNDLE_ID, "data"],
            capture_output=True, text=True, timeout=10,
        )
        _data = _dr.stdout.strip()
        cmd = [
            "xcrun", "simctl", "spawn", deployed.udid, _binary,
            f"--model={_data}/{deployed.remote_model_path}",
            f"--num_runs={num_runs}",
            f"--warmup_runs={warmup}",
            f"--compute_units={compute_units}",
        ]
    else:
        cmd = [_idb_bin(), "launch"]
        if deployed.udid:
            cmd += ["--udid", deployed.udid]
        cmd += [
            BUNDLE_ID,
            f"--model={deployed.remote_model_path}",
            f"--num_runs={num_runs}",
            f"--warmup_runs={warmup}",
            f"--compute_units={compute_units}",
        ]

    _log(f"Mini benchmark: {' '.join(cmd)}", deployed.udid)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stdout_lines: List[str] = []
        start = time.time()

        while True:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                if time.time() - start > timeout:
                    _log("Mini benchmark timed out — killing", deployed.udid, "WARN")
                    proc.kill()
                    proc.communicate()
                    return None, None, "\n".join(stdout_lines)
                continue
            stdout_lines.append(line.rstrip())
            _log(f"mini: {line.rstrip()}", deployed.udid)

        proc.wait()
        stdout = "\n".join(stdout_lines)

        if proc.returncode != 0:
            _log(f"Mini benchmark exited {proc.returncode}", deployed.udid, "WARN")
            return None, None, stdout

        avg_ms = _parse_avg_ms(stdout)
        units_used = _parse_compute_units_used(stdout)

        return avg_ms, units_used, stdout

    except Exception as exc:
        _log(f"Mini benchmark error: {exc}", deployed.udid, "ERROR")
        return None, None, ""


def _parse_avg_ms(stdout: str) -> Optional[float]:
    import json as _json
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                obj = _json.loads(line)
                if obj.get("event") == "result" and "avg_ms" in obj:
                    return float(obj["avg_ms"])
            except Exception:
                pass
    match = re.search(r"avg_ms:\s*(\d+\.?\d*)", stdout, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


def _parse_compute_units_used(stdout: str) -> Optional[str]:
    """
    Parse what CoreML actually ran on from runner output.
    Primary: JSON result event. Fallback: flat text.
    """
    import json as _json
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                obj = _json.loads(line)
                if obj.get("event") == "result" and "compute_units_used" in obj:
                    return obj["compute_units_used"]
            except Exception:
                pass
    for line in stdout.splitlines():
        line = line.strip().lower()
        if line.startswith("compute_units_used:"):
            _, _, raw = line.partition(":")
            return raw.strip()
    return None


# ---------------------------------------------------------------------
# Fallback Detection
# ---------------------------------------------------------------------

def _fallback_threshold(cpu_variance: float) -> float:
    if not cpu_variance or cpu_variance != cpu_variance:  # None or NaN
        cpu_variance = 0.0
    return max(MIN_FALLBACK_THRESHOLD, cpu_variance * 2)


def _is_fallback(
    delegate_avg_ms:   float,
    cpu_avg_ms:        float,
    cpu_variance:      float,
    compute_units_used: Optional[str],
    requested_units:   str,
) -> bool:
    """
    Two-signal fallback detection:

    1. compute_units_used.lower() != requested_units.lower() — CoreML explicitly ran on
       something else. Hard fallback signal, no latency comparison needed.

    2. Latency delta < adaptive threshold — delegate is so close to CPU
       that it's probably running on CPU anyway.
    """
    # Signal 1: explicit units mismatch
    if compute_units_used and compute_units_used.lower() != requested_units.lower():
        _log(
            f"compute_units_used={compute_units_used} != "
            f"requested={requested_units} — hard fallback"
        )
        return True

    # Signal 2: latency-based
    if cpu_avg_ms <= 0:
        return False
    threshold = _fallback_threshold(cpu_variance)
    delta = abs(delegate_avg_ms - cpu_avg_ms) / (cpu_avg_ms + 1e-8)
    return delta < threshold


# ---------------------------------------------------------------------
# Failure Classification
# ---------------------------------------------------------------------

def _classify_failure(stdout: str, exit_code: Optional[int], delegate: str) -> str:
    s = stdout.lower()

    if "model not found" in s:
        return "Model not found in sandbox — deploy may have failed"

    if "unsupported" in s or "not supported" in s:
        return f"Model contains ops unsupported by {delegate}"

    if "neural engine" in s and "unavailable" in s:
        return "ANE unavailable on this device"

    if "metal" in s and "failed" in s:
        return "Metal GPU initialization failed"

    if exit_code is None:
        return "Benchmark timed out"

    return f"Runner exited with code {exit_code}"


# ---------------------------------------------------------------------
# Single Delegate Validation
# ---------------------------------------------------------------------

def _validate_single(
    deployed:    DeployedRun,
    delegate:    str,
    baseline:    BaselineResult,
    fingerprint: str,
    num_runs:    int = DEFAULT_NUM_RUNS,
    warmup:      int = DEFAULT_WARMUP,
    timeout:     int = DEFAULT_TIMEOUT,
) -> DelegateValidation:
    now = time.strftime("%Y-%m-%dT%H:%M:%S")

    def make(
        status: DelegateStatus,
        reason: str,
        avg_ms: Optional[float] = None,
        units_used: Optional[str] = None,
        consistency: Optional[ConsistencyResult] = None,
    ) -> DelegateValidation:
        return DelegateValidation(
            delegate=delegate,
            status=status,
            avg_ms=avg_ms,
            compute_units_used=units_used,
            consistency=consistency,
            reason=reason,
            validated_at=now,
        )

    compute_units = _COMPUTE_UNITS.get(delegate.upper())
    if not compute_units:
        return make(DelegateStatus.UNSUPPORTED, f"Unknown delegate '{delegate}'")

    avg_ms, units_used, stdout = _run_mini_benchmark(
        deployed, compute_units, num_runs, warmup, timeout
    )

    if avg_ms is None:
        reason = _classify_failure(stdout, None, delegate)
        _log(f"{delegate}: {reason} → UNSUPPORTED", deployed.udid, "WARN")
        return make(DelegateStatus.UNSUPPORTED, reason, units_used=units_used)

    cpu_avg_ms   = baseline.avg_ms or 0.0
    cpu_variance = baseline.variance or 0.0

    # Fallback check — two signals
    if _is_fallback(avg_ms, cpu_avg_ms, cpu_variance, units_used, compute_units):
        reason = (
            f"Fell back to CPU "
            f"(avg={avg_ms:.2f}ms vs cpu={cpu_avg_ms:.2f}ms, "
            f"compute_units_used={units_used})"
        )
        _log(f"{delegate}: {reason} → FALLBACK", deployed.udid, "WARN")
        return make(DelegateStatus.FALLBACK, reason, avg_ms=avg_ms, units_used=units_used)

    # Consistency check
    consistency = check_consistency(deployed, delegate)
    if not consistency.passed:
        reason = (
            f"Output diverged from CPU "
            f"(rel_diff={consistency.relative_diff}, tol={consistency.tolerance})"
        )
        _log(f"{delegate}: {reason} → INCONSISTENT", deployed.udid, "WARN")
        return make(
            DelegateStatus.INCONSISTENT,
            reason,
            avg_ms=avg_ms,
            units_used=units_used,
            consistency=consistency,
        )

    reason = (
        f"Validated: {avg_ms:.2f}ms vs CPU {cpu_avg_ms:.2f}ms "
        f"on {units_used or compute_units}"
    )
    _log(f"{delegate}: {reason} → SUPPORTED", deployed.udid)
    return make(
        DelegateStatus.SUPPORTED,
        reason,
        avg_ms=avg_ms,
        units_used=units_used,
        consistency=consistency,
    )


# ---------------------------------------------------------------------
# Public: Validate Delegates
# ---------------------------------------------------------------------

def validate_delegates(
    deployed:     DeployedRun,
    candidates:   List[str],
    baseline:     BaselineResult,
    fingerprint:  str,
    is_simulator: bool,
    *,
    cache_path: Path = DEFAULT_CACHE_PATH,
    force:      bool = False,
    num_runs:   int  = DEFAULT_NUM_RUNS,
    warmup:     int  = DEFAULT_WARMUP,
    timeout:    int  = DEFAULT_TIMEOUT,
) -> Dict[str, DelegateValidation]:
    """
    Validate all candidates. Returns a dict of delegate → DelegateValidation.

    ANE/ANE_EXPLICIT are auto-skipped on simulator — SKIPPED status,
    never written to cache (no hardware to cache against).
    """
    results: Dict[str, DelegateValidation] = {}
    now = time.strftime("%Y-%m-%dT%H:%M:%S")

    for delegate in candidates:
        # Simulator gate — structural skip, not a validation failure
        if is_simulator and delegate.upper() in _ANE_DELEGATES:
            _log(f"{delegate}: simulator target → SKIPPED", deployed.udid)
            results[delegate] = DelegateValidation(
                delegate=delegate,
                status=DelegateStatus.SKIPPED,
                avg_ms=None,
                compute_units_used=None,
                consistency=None,
                reason="ANE not available on simulator",
                validated_at=now,
            )
            continue

        _log(f"Validating: {delegate}", deployed.udid)

        if not force:
            cached = _read_cached(cache_path, fingerprint, delegate)
            if cached:
                _log(f"{delegate}: cache hit → {cached.status.value}", deployed.udid)
                results[delegate] = cached
                continue

        validation = _validate_single(
            deployed, delegate, baseline, fingerprint,
            num_runs, warmup, timeout,
        )
        results[delegate] = validation
        _write_cached(cache_path, fingerprint, delegate, validation)
        _log(f"{delegate}: {validation}", deployed.udid)

    return results


def supported_delegates(validations: Dict[str, DelegateValidation]) -> List[str]:
    return [
        name for name, v in validations.items()
        if v.status == DelegateStatus.SUPPORTED
    ]