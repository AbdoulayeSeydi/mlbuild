"""
mlbuild.platforms.ios.baseline

CPU baseline benchmark enforcement.

Responsibilities:
- Run the benchmark runner app with MLComputeUnits.cpuOnly
- Stream stdout back via non-blocking reads through idb launch
- Parse latency percentiles from runner output
- Compute variance (p90/p50 - 1) for downstream adaptive thresholds
- Return a structured BaselineResult

Design rules:
- CPU baseline always runs. It is never optional.
- variance is a first-class output — delegate.py and stability.py depend on it.
- Parsing is defensive — missing fields → None, not crash.
- Raw stdout is always stored unconditionally.
- Never raises IDBParseError for a partial result — partial is better than nothing.

Key differences from Android:
- No adb shell command — runner launched via idb launch with args.
- MLComputeUnits.cpuOnly passed as a launch arg, not a binary flag.
- CoreML runner reports milliseconds directly — no microsecond conversion.
- No remote binary path — runner app is pre-installed, model path is sandbox-relative.
- No process group kill — idb launch process is killed directly.
- thermal_state line parsed here and passed to thermal.py downstream.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Iterator, Optional

from mlbuild.platforms.ios.deploy import DeployedRun
from mlbuild.core.errors import IDBExecutionError, IDBTimeoutError


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DEBUG = os.getenv("MLBUILD_DEBUG") == "1"

DEFAULT_NUM_RUNS    = 50
DEFAULT_WARMUP_RUNS = 10
DEFAULT_CHUNK_SIZE  = 1024
DEFAULT_TIMEOUT     = 600

BUNDLE_ID = "com.mlbuild.MLBuildRunner"

# CoreML runner arg for CPU-only compute units
COMPUTE_UNITS_CPU = "cpuOnly"


def _log(msg: str, udid: Optional[str] = None) -> None:
    if not DEBUG: return
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    prefix = f"[mlbuild.ios.baseline][{udid}]" if udid else "[mlbuild.ios.baseline]"
    print(f"{ts} {prefix} {msg}", file=sys.stderr)


# ---------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------

@dataclass
class BaselineResult:
    avg_ms:         Optional[float]
    p50_ms:         Optional[float]
    p90_ms:         Optional[float]
    p99_ms:         Optional[float]
    init_ms:        Optional[float]
    peak_mem_mb:    Optional[float]
    variance:       Optional[float]
    std_ms:         Optional[float]
    min_ms:         Optional[float]
    max_ms:         Optional[float]
    count:          Optional[int]
    low_confidence: bool
    compute_units:  str              # always "cpuOnly" for baseline
    thermal_state:     Optional[str]    # post-run thermal state
    thermal_state_pre: Optional[str]    # pre-run thermal state (from thermal_boot event)
    raw_stdout:     str
    run_id:         str
    num_runs:       int
    duration_s:     float

    @property
    def is_noisy(self) -> bool:
        return self.variance is not None and self.variance > 0.2

    def __str__(self) -> str:
        var_str = f"{self.variance:.3f}" if self.variance is not None else "N/A"
        return (
            f"BaselineResult(avg={self.avg_ms}ms, "
            f"p50={self.p50_ms}ms, p90={self.p90_ms}ms, "
            f"variance={var_str})"
        )


# ---------------------------------------------------------------------
# Command Builder
# ---------------------------------------------------------------------

def _build_launch_args(
    deployed: DeployedRun,
    num_runs: int,
    warmup_runs: int,
) -> list[str]:
    """
    Args passed to idb launch <bundle_id> <args...>.
    Runner interprets these to configure the CoreML benchmark.
    """
    return [
        f"--model={deployed.remote_model_path}",
        f"--num_runs={num_runs}",
        f"--warmup_runs={warmup_runs}",
        f"--compute_units={COMPUTE_UNITS_CPU}",
        "--report_peak_memory=true",
        "--report_thermal=true",
    ]



def _sim_binary(udid: str) -> str:
    result = subprocess.run(
        ["xcrun", "simctl", "get_app_container", udid, BUNDLE_ID, "app"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"App not installed on simulator {udid}")
    return result.stdout.strip() + "/MLBuildRunner"

def _build_idb_cmd(
    deployed: DeployedRun,
    launch_args: list[str],
) -> list[str]:
    if not deployed.is_simulator and deployed.udid:
        binary = "com.mlbuild.MLBuildRunner"
        return ["xcrun", "devicectl", "device", "process", "launch",
                "--device", deployed.udid, "--console", "--terminate-existing",
                binary] + launch_args
    if deployed.is_simulator and deployed.udid:
        binary = _sim_binary(deployed.udid)
        data_container = subprocess.run(
            ["xcrun", "simctl", "get_app_container", deployed.udid, BUNDLE_ID, "data"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        # Rewrite --model= arg to absolute path — simctl spawn runs outside sandbox
        abs_args = []
        for a in launch_args:
            if a.startswith("--model="):
                rel = a[len("--model="):]
                abs_args.append(f"--model={data_container}/{rel}")
            else:
                abs_args.append(a)
        return ["xcrun", "simctl", "spawn", deployed.udid, binary] + abs_args

    # Real device: idb gRPC path
    idb_bin = shutil.which("idb")
    if not idb_bin:
        from mlbuild.core.errors import IDBNotFoundError
        raise IDBNotFoundError()

    cmd = [idb_bin, "launch", "--udid", deployed.udid, BUNDLE_ID] if deployed.udid \
        else [idb_bin, "launch", BUNDLE_ID]
    cmd += launch_args
    return cmd


# ---------------------------------------------------------------------
# Non-Blocking Output Streaming
# ---------------------------------------------------------------------

def _stream_output(
    proc: subprocess.Popen,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[str]:
    """
    Non-blocking read of stdout lines.
    Mirrors Android implementation exactly.
    """
    stdout_buffer = ""

    while True:
        chunk = proc.stdout.read(chunk_size) or ""
        if not chunk and proc.poll() is not None:
            break

        stdout_buffer += chunk
        lines = stdout_buffer.split("\n")
        stdout_buffer = lines[-1]
        yield from lines[:-1]

    if stdout_buffer.strip():
        yield stdout_buffer


# ---------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------

def _parse_float(pattern: str, text: str) -> Optional[float]:
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _parse_int(pattern: str, text: str) -> Optional[int]:
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _parse_thermal_state(stdout: str) -> Optional[str]:
    """
    Extract raw thermal state string from runner output.
    thermal.py does the interpretation — return raw value only.

    Expected runner line:
        thermal_state: nominal
    """
    for line in stdout.splitlines():
        line = line.strip().lower()
        if line.startswith("thermal_state:"):
            _, _, raw = line.partition(":")
            return raw.strip()
    return None


def _parse_baseline_output(stdout: str) -> dict:
    """
    Parse CoreML runner output.

    Primary: JSON event parsing — app emits {"event":"result","p50_ms":...}
    Fallback: flat text parsing for simulator (simctl spawn path).
    """
    import json as _json

    # Primary: scan for JSON result event
    avg_ms = p50_ms = p90_ms = p99_ms = None
    init_ms = std_ms = min_ms = max_ms = None
    count = None
    peak_mem_mb = None
    thermal_state_json = None
    thermal_boot_state = None

    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = _json.loads(line)
        except Exception:
            continue
        if obj.get("event") == "thermal_boot":
            thermal_boot_state = obj.get("state")
            continue
        if obj.get("event") == "result":
            avg_ms      = obj.get("avg_ms")
            p50_ms      = obj.get("p50_ms")
            p90_ms      = obj.get("p90_ms")
            p99_ms      = obj.get("p99_ms")
            init_ms     = obj.get("init_ms")
            std_ms      = obj.get("std_ms") if "std_ms" in obj else None
            min_ms      = obj.get("min_ms")
            max_ms      = obj.get("max_ms")
            count       = obj.get("count")
            peak_mem_mb = obj.get("peak_memory_mb")
            break

    # Fallback: flat text parsing for simulator
    if p50_ms is None:
        avg_ms      = _parse_float(r"avg_ms:\s*(\d+\.?\d*)", stdout)
        p50_ms      = _parse_float(r"p50_ms:\s*(\d+\.?\d*)", stdout)
        p90_ms      = _parse_float(r"p90_ms:\s*(\d+\.?\d*)", stdout)
        p99_ms      = _parse_float(r"p99_ms:\s*(\d+\.?\d*)", stdout)
        init_ms     = _parse_float(r"init_ms:\s*(\d+\.?\d*)", stdout)
        std_ms      = _parse_float(r"std_ms:\s*(\d+\.?\d*)", stdout)
        min_ms      = _parse_float(r"min_ms:\s*(\d+\.?\d*)", stdout)
        max_ms      = _parse_float(r"max_ms:\s*(\d+\.?\d*)", stdout)
        count       = _parse_int(r"count:\s*(\d+)", stdout)
        peak_mem_mb = _parse_float(r"peak_memory_mb:\s*(\d+\.?\d*)", stdout)

    # Variance: p90/p50 - 1 primary, std/avg fallback
    variance: Optional[float] = None
    if p50_ms and p90_ms and p50_ms > 0:
        variance = round((p90_ms / p50_ms) - 1, 4)
    elif std_ms and avg_ms and avg_ms > 0:
        variance = round(std_ms / avg_ms, 4)

    low_confidence = count is not None and count < 20

    thermal_state = _parse_thermal_state(stdout)

    _log(
        f"Parsed: avg={avg_ms}ms p50={p50_ms}ms p90={p90_ms}ms "
        f"std={std_ms}ms count={count} variance={variance} "
        f"thermal={thermal_state} low_confidence={low_confidence}"
    )

    return {
        "avg_ms":              avg_ms,
        "p50_ms":              p50_ms,
        "p90_ms":              p90_ms,
        "p99_ms":              p99_ms,
        "init_ms":             init_ms,
        "peak_mem_mb":         peak_mem_mb,
        "variance":            variance,
        "std_ms":              std_ms,
        "min_ms":              min_ms,
        "max_ms":              max_ms,
        "count":               count,
        "low_confidence":      low_confidence,
        "thermal_state":       thermal_state,
        "thermal_state_pre":   thermal_boot_state,
    }


# ---------------------------------------------------------------------
# CPU Baseline Execution
# ---------------------------------------------------------------------

def run_cpu_baseline(
    deployed: DeployedRun,
    *,
    num_runs: int = DEFAULT_NUM_RUNS,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    timeout_s: int = DEFAULT_TIMEOUT,
) -> BaselineResult:
    """
    Launch the runner app via idb with cpuOnly compute units.
    Stream stdout, parse results, return BaselineResult.

    Raises:
        IDBTimeoutError  — benchmark exceeded timeout_s
        IDBExecutionError — runner exited non-zero
    """
    launch_args = _build_launch_args(deployed, num_runs, warmup_runs)
    cmd = _build_idb_cmd(deployed, launch_args)

    _log(f"CPU baseline command: {' '.join(cmd)}", deployed.udid)

    start = time.time()
    stdout_lines: list[str] = []

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    try:
        for line in _stream_output(proc, chunk_size):
            stdout_lines.append(line)
            _log(line, deployed.udid)

        proc.wait(timeout=timeout_s)

    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raw = "\n".join(stdout_lines)
        raise IDBTimeoutError(command=" ".join(cmd))

    except Exception as exc:
        proc.kill()
        raise exc

    duration_s = time.time() - start
    raw_stdout = "\n".join(stdout_lines)

    if proc.returncode != 0:
        stderr = proc.stderr.read() if proc.stderr else ""
        raise IDBExecutionError(
            raw_stdout=raw_stdout + "\n" + stderr,
            run_id=deployed.run_id,
        )

    metrics = _parse_baseline_output(raw_stdout)

    result = BaselineResult(
        avg_ms         = metrics["avg_ms"],
        p50_ms         = metrics["p50_ms"],
        p90_ms         = metrics["p90_ms"],
        p99_ms         = metrics["p99_ms"],
        init_ms        = metrics["init_ms"],
        peak_mem_mb    = metrics["peak_mem_mb"],
        variance       = metrics["variance"],
        std_ms         = metrics["std_ms"],
        min_ms         = metrics["min_ms"],
        max_ms         = metrics["max_ms"],
        count          = metrics["count"],
        low_confidence = metrics["low_confidence"],
        compute_units  = COMPUTE_UNITS_CPU,
        thermal_state     = metrics["thermal_state"],
        thermal_state_pre = metrics.get("thermal_state_pre"),
        raw_stdout     = raw_stdout,
        run_id         = deployed.run_id,
        num_runs       = num_runs,
        duration_s     = duration_s,
    )

    _log(f"CPU baseline complete in {duration_s:.1f}s: {result}", deployed.udid)
    return result