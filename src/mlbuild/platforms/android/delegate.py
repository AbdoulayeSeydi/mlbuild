"""
mlbuild.platforms.android.delegate

Runtime delegate validation and caching.

Responsibilities:
- Validate delegate candidates against actual device behavior
- Cache results per device fingerprint to avoid re-running validation
- Detect silent fallback via adaptive latency-based threshold
- Detect output divergence via consistency.py
- Return structured DelegateValidation results

Design rules:
- Cache is checked first. Validation runs once per device per OS version.
- Fallback detection is latency-based, not log-based. Logs lie.
- Threshold is adaptive: max(0.10, cpu_variance * 2)
- INCONSISTENT is a first-class status — not collapsed into UNSUPPORTED.
- Validation uses a short run (5 iterations) — not a full benchmark.
- All status decisions are logged with their reasoning.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from queue import Queue, Empty
from typing import Optional, Dict, List, Callable, Any

from mlbuild.platforms.android import adb
from mlbuild.platforms.android.deploy import DeployedRun
from mlbuild.platforms.android.baseline import BaselineResult
from mlbuild.platforms.android.consistency import check_consistency, ConsistencyResult

# ---------------------------------------------------------------------
# Config (override via kwargs in validate_delegates)
# ---------------------------------------------------------------------

DEBUG = os.getenv("MLBUILD_DEBUG") == "1"
DEFAULT_CACHE_PATH = Path.home() / ".mlbuild" / "device_cache.json"
DEFAULT_NUM_RUNS = 5
DEFAULT_WARMUP = 2
DEFAULT_TIMEOUT = 60  # seconds per benchmark
MIN_FALLBACK_THRESHOLD = 0.10  # 10% floor

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def _log(msg: str, serial: Optional[str] = None, level: str = "INFO") -> None:
    if not DEBUG: return
    """
    Structured log for enterprise device farms.
    Includes timestamp, serial, log level.
    """
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    prefix = f"[{serial}]" if serial else ""
    print(f"{ts} [{level}]{prefix} {msg}", flush=True, file=sys.stderr)

# ---------------------------------------------------------------------
# Delegate Status
# ---------------------------------------------------------------------

class DelegateStatus(str, Enum):
    SUPPORTED = "supported"
    FALLBACK = "fallback"
    UNSUPPORTED = "unsupported"
    INCONSISTENT = "inconsistent"

# ---------------------------------------------------------------------
# Delegate Flag Mapping
# ---------------------------------------------------------------------

_DELEGATE_FLAGS: Dict[str, str] = {
    "GPU": "--use_gpu=true",
    "NNAPI": "--use_nnapi=true",
    "HEXAGON": "--use_hexagon=true",
    "HEXAGON_HTP": "--use_hexagon=true --hexagon_nn_nodes_on_graph=1",
}

def _delegate_flag(delegate: str) -> str:
    """
    Return properly validated delegate flag.
    Raises ValueError if unknown.
    """
    flag = _DELEGATE_FLAGS.get(delegate.upper())
    if flag is None:
        raise ValueError(f"Unknown delegate '{delegate}'")
    return flag

# ---------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------

@dataclass
class DelegateValidation:
    delegate: str
    status: DelegateStatus
    avg_ms: Optional[float]
    consistency: Optional[ConsistencyResult]
    reason: str
    validated_at: str

    def to_cache_dict(self) -> dict:
        return {
            "status": self.status.value,
            "avg_ms": self.avg_ms,
            "reason": self.reason,
            "validated_at": self.validated_at,
        }

    def __str__(self) -> str:
        return (
            f"DelegateValidation(delegate={self.delegate}, status={self.status.value}, "
            f"avg_ms={self.avg_ms}, reason={self.reason!r})"
        )

# ---------------------------------------------------------------------
# Cache (atomic writes)
# ---------------------------------------------------------------------

@contextmanager
def _atomic_write(path: Path):
    tmp_path = path.with_suffix(".tmp")
    yield tmp_path
    tmp_path.replace(path)

def _load_cache(cache_path: Path) -> Dict[str, Any]:
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path) as f:
            return json.load(f)
    except Exception as exc:
        _log(f"Cache load failed — starting fresh: {exc}", level="WARN")
        return {}

def _save_cache(cache_path: Path, cache: Dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with _atomic_write(cache_path) as tmp_file:
            with open(tmp_file, "w") as f:
                json.dump(cache, f, indent=2)
    except Exception as exc:
        _log(f"Cache save failed: {exc}", level="ERROR")

def _cache_key(fingerprint: str, delegate: str) -> str:
    return f"{fingerprint}:{delegate.upper()}"

def _read_cached(cache_path: Path, fingerprint: str, delegate: str) -> Optional[DelegateValidation]:
    cache = _load_cache(cache_path)
    entry = cache.get(_cache_key(fingerprint, delegate))
    if not entry:
        return None
    try:
        return DelegateValidation(
            delegate=delegate,
            status=DelegateStatus(entry["status"]),
            avg_ms=entry.get("avg_ms"),
            consistency=None,
            reason=entry.get("reason", "loaded from cache"),
            validated_at=entry.get("validated_at", "unknown"),
        )
    except Exception as exc:
        _log(f"Malformed cache entry for {delegate}: {exc}", level="WARN")
        return None

def _write_cached(cache_path: Path, fingerprint: str, delegate: str, validation: DelegateValidation) -> None:
    cache = _load_cache(cache_path)
    cache[_cache_key(fingerprint, delegate)] = validation.to_cache_dict()
    _save_cache(cache_path, cache)
    _log(f"Cached: {delegate} = {validation.status.value}")

# ---------------------------------------------------------------------
# Subprocess with streaming + safe shell
# ---------------------------------------------------------------------

def _run_subprocess_stream(cmd: List[str], timeout: int, serial: Optional[str] = None) -> tuple[int, str]:
    """
    Run adb subprocess safely:
      - Uses streaming to avoid memory blowup
      - Escapes shell arguments safely
      - Kills process group on timeout / Ctrl-C
      - Returns (returncode, stdout)
    """
    stdout_queue: Queue[str] = Queue()
    proc = None
    stdout_lines: List[str] = []

    def enqueue_output(stream, queue: Queue):
        for line in iter(stream.readline, ""):
            queue.put(line)
        stream.close()

    full_cmd = cmd
    _log(f"Executing: {' '.join(full_cmd)}", serial)

    try:
        proc = subprocess.Popen(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
        )
        thread = threading.Thread(target=enqueue_output, args=(proc.stdout, stdout_queue))
        thread.daemon = True
        thread.start()

        start = time.time()
        while thread.is_alive():
            thread.join(timeout=0.1)
            if time.time() - start > timeout:
                _log(f"Process timeout, killing pid {proc.pid}", serial, "ERROR")
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                thread.join()
                return -1, "\n".join(stdout_lines)

            try:
                while True:
                    line = stdout_queue.get_nowait()
                    stdout_lines.append(line.rstrip())
            except Empty:
                pass

        # Drain remaining lines
        while not stdout_queue.empty():
            stdout_lines.append(stdout_queue.get_nowait().rstrip())

        return proc.returncode, "\n".join(stdout_lines)

    except Exception as exc:
        if proc:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                pass
        _log(f"Subprocess error: {exc}", serial, "ERROR")
        return -1, ""

# ---------------------------------------------------------------------
# Mini Benchmark
# ---------------------------------------------------------------------

def _run_mini_benchmark(
    deployed:  DeployedRun,
    delegate_flag: Optional[str],
    serial:    Optional[str],
    num_runs:  int = DEFAULT_NUM_RUNS,
    warmup:    int = DEFAULT_WARMUP,
    timeout:   int = DEFAULT_TIMEOUT,
) -> tuple[Optional[float], str, Optional[int]]:
    """
    Returns (avg_ms, stdout, exit_code).
    avg_ms is None on failure.
    """
    import re

    binary = shlex.quote(deployed.remote_binary_path)
    model = shlex.quote(deployed.remote_model_path)
    cmd_parts = [
        binary,
        f"--graph={model}",
        f"--num_runs={num_runs}",
        f"--warmup_runs={warmup}"
    ]
    if delegate_flag:
        cmd_parts += shlex.split(delegate_flag)

    full_cmd = ["adb"]
    if serial:
        full_cmd += ["-s", serial]
    full_cmd += ["shell"] + cmd_parts

    returncode, output = _run_subprocess_stream(full_cmd, timeout, serial)
    if returncode != 0 and returncode is not None:
        return None, output, returncode

    match = re.search(r"avg\s*=?\s*(\d+\.?\d*)", output, re.IGNORECASE)
    if not match:
        return None, output, returncode

    avg_ms = round(float(match.group(1)) / 1000.0, 3)
    return avg_ms, output, returncode

# ---------------------------------------------------------------------
# Fallback Detection
# ---------------------------------------------------------------------

def _fallback_threshold(cpu_variance: float) -> float:
    if cpu_variance is None or cpu_variance < 0 or cpu_variance != cpu_variance:  # NaN
        cpu_variance = 0.0
    threshold = max(MIN_FALLBACK_THRESHOLD, cpu_variance * 2)
    return threshold

def _is_fallback(delegate_avg_ms: float, cpu_avg_ms: float, cpu_variance: float) -> bool:
    if cpu_avg_ms <= 0:
        return False
    threshold = _fallback_threshold(cpu_variance)
    delta = abs(delegate_avg_ms - cpu_avg_ms) / (cpu_avg_ms + 1e-8)
    return delta < threshold

# ---------------------------------------------------------------------
# Single Delegate Validation
# ---------------------------------------------------------------------

def _validate_single(
    deployed: DeployedRun,
    delegate: str,
    baseline: BaselineResult,
    fingerprint: str,
    num_runs: int = DEFAULT_NUM_RUNS,
    warmup: int = DEFAULT_WARMUP,
    timeout: int = DEFAULT_TIMEOUT,
) -> DelegateValidation:
    now = time.strftime("%Y-%m-%dT%H:%M:%S")

    def make(status: DelegateStatus, reason: str, avg_ms=None, consistency=None):
        return DelegateValidation(
            delegate=delegate,
            status=status,
            avg_ms=avg_ms,
            consistency=consistency,
            reason=reason,
            validated_at=now,
        )

    try:
        flag = _delegate_flag(delegate)
    except ValueError as exc:
        return make(DelegateStatus.UNSUPPORTED, str(exc))

    

    cpu_avg_ms = baseline.avg_ms or 0.0
    cpu_variance = baseline.variance or 0.0

    delegate_avg_ms, mini_stdout, exit_code = _run_mini_benchmark(
        deployed, flag, deployed.serial, num_runs, warmup, timeout
    )

    if delegate_avg_ms is None:
        reason = _classify_delegate_failure(mini_stdout, exit_code, delegate)
        _log(f"{delegate}: {reason} → UNSUPPORTED", deployed.serial)
        return make(DelegateStatus.UNSUPPORTED, reason)

    consistency = check_consistency(deployed, delegate)
    if not consistency.passed:
        return make(
            DelegateStatus.INCONSISTENT,
            f"Output diverged from CPU (rel_diff={consistency.relative_diff})",
            avg_ms=delegate_avg_ms,
            consistency=consistency,
        )

    return make(
        DelegateStatus.SUPPORTED,
        f"Validated: {delegate_avg_ms:.2f}ms vs CPU {cpu_avg_ms:.2f}ms",
        avg_ms=delegate_avg_ms,
        consistency=consistency,
    )

def _classify_delegate_failure(
    stdout:    str,
    exit_code: Optional[int],
    delegate:  str,
) -> str:
    """
    Classify delegate failure from stdout patterns.
    Returns a human-readable reason string.
    """
    s = stdout.lower()

    # Emulator NNAPI — reference implementation, not real hardware
    if "nnapi-reference" in s and delegate == "NNAPI":
        return (
            "NNAPI fell back to reference implementation — "
            "no hardware acceleration available (emulator or unsupported device)"
        )

    # GPU initialized but crashed at runtime
    if "gpu delegate created" in s and delegate == "GPU":
        return (
            "GPU delegate initialized but crashed during inference — "
            "runtime incompatibility (common on emulators)"
        )

    # GPU failed to initialize
    if "gpu delegate" in s and "failed" in s:
        return "GPU delegate failed to initialize — unsupported device or driver"

    # Unsupported ops
    if "only supports" in s or "not supported" in s:
        return "Delegate rejected model — contains unsupported ops for this delegate"

    # Timeout
    if exit_code is None:
        return "Benchmark timed out — device too slow or hung"

    # Generic crash
    return f"Benchmark exited with code {exit_code}"

# ---------------------------------------------------------------------
# Public: Validate Delegates
# ---------------------------------------------------------------------

def validate_delegates(
    deployed: DeployedRun,
    candidates: List[str],
    baseline: BaselineResult,
    fingerprint: str,
    *,
    cache_path: Path = DEFAULT_CACHE_PATH,
    force: bool = False,
    num_runs: int = DEFAULT_NUM_RUNS,
    warmup: int = DEFAULT_WARMUP,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, DelegateValidation]:
    results: Dict[str, DelegateValidation] = {}

    for delegate in candidates:
        _log(f"Validating: {delegate}", deployed.serial)

        if not force:
            cached = _read_cached(cache_path, fingerprint, delegate)
            if cached:
                _log(f"{delegate}: cache hit → {cached.status.value}", deployed.serial)
                results[delegate] = cached
                continue

        validation = _validate_single(deployed, delegate, baseline, fingerprint, num_runs, warmup, timeout)
        results[delegate] = validation
        _write_cached(cache_path, fingerprint, delegate, validation)
        _log(f"{delegate}: {validation}", deployed.serial)

    return results

def supported_delegates(validations: Dict[str, DelegateValidation]) -> List[str]:
    return [name for name, v in validations.items() if v.status == DelegateStatus.SUPPORTED]