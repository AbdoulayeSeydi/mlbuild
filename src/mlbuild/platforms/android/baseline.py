"""
mlbuild.platforms.android.baseline

CPU baseline benchmark enforcement.

Responsibilities:
- Run the benchmark binary with CPU only (no delegate)
- Stream stdout back via non-blocking reads
- Parse latency percentiles from raw output
- Compute variance (p90/p50 - 1) for downstream adaptive thresholds
- Return a structured BaselineResult

Design rules:
- CPU baseline always runs. It is never optional.
- variance is a first-class output — delegate.py and stability.py depend on it.
- Parsing is defensive — missing fields → None, not crash.
- Raw stdout is always stored unconditionally.
- Never raises ParseError for a partial result — partial is better than nothing.
"""

from __future__ import annotations


import sys
import os
import re
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from mlbuild.platforms.android import adb
from mlbuild.platforms.android.deploy import DeployedRun
from mlbuild.core.errors import ExecutionError, ADBTimeoutError

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DEBUG = os.getenv("MLBUILD_DEBUG") == "1"

DEFAULT_NUM_RUNS    = 50
DEFAULT_WARMUP_RUNS = 10
DEFAULT_NUM_THREADS = 4
DEFAULT_CHUNK_SIZE  = 1024  # bytes per read
DEFAULT_TIMEOUT     = 600   # seconds per benchmark

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def _log(msg: str, serial: Optional[str] = None) -> None:
    if not DEBUG: return
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    prefix = f"[mlbuild.baseline][{serial}]" if serial else "[mlbuild.baseline]"
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
    autocorr_lag1:  Optional[float]
    raw_stdout:     str
    run_id:         str
    num_runs:       int
    num_threads:    int
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

def _sanitize_path(path: str) -> str:
    """Escape shell-sensitive characters in paths."""
    return shlex.quote(path)

def _build_command(
    deployed: DeployedRun,
    num_runs: int,
    warmup_runs: int,
    num_threads: int,
) -> str:
    graph_path  = _sanitize_path(deployed.remote_model_path)
    binary_path = _sanitize_path(deployed.remote_binary_path)
    return (
        f"{binary_path} "
        f"--graph={graph_path} "
        f"--num_runs={num_runs} "
        f"--warmup_runs={warmup_runs} "
        f"--num_threads={num_threads} "
        f"--enable_op_profiling=true "
        f"--report_peak_memory_footprint=true"
    )

# ---------------------------------------------------------------------
# Non-blocking output streaming
# ---------------------------------------------------------------------

def _stream_output(proc: subprocess.Popen, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Iterator[str]:
    """Non-blocking read of stdout and stderr lines."""
    stdout_buffer = ""
    stderr_buffer = ""
    while True:
        out_chunk = proc.stdout.read(chunk_size) or ""
        err_chunk = proc.stderr.read(chunk_size) or ""
        if not out_chunk and not err_chunk and proc.poll() is not None:
            break

        stdout_buffer += out_chunk
        stderr_buffer += err_chunk

        lines = stdout_buffer.split("\n")
        stdout_buffer = lines[-1]
        yield from lines[:-1]

    if stdout_buffer.strip():
        yield stdout_buffer

# ---------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------

def _parse_ms(pattern: str, text: str) -> Optional[float]:
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
            pass
    return None


def _parse_baseline_output(stdout: str) -> dict:
    def us_to_ms(val: Optional[float]) -> Optional[float]:
        return round(val / 1000.0, 3) if val is not None else None

    # TFLite outputs median not P50, and p95 not P90
    avg_us  = _parse_ms(r"avg\s*=\s*(\d+\.?\d*)", stdout)
    p50_us  = _parse_ms(r"median\s*=\s*(\d+\.?\d*)", stdout)
    p90_us  = _parse_ms(r"p95\s*=\s*(\d+\.?\d*)", stdout)
    p99_us  = _parse_ms(r"p99\s*=\s*(\d+\.?\d*)", stdout)
    init_us = _parse_ms(r"Init:\s*(\d+\.?\d*)", stdout)
    std_us  = _parse_ms(r"std\s*=\s*(\d+\.?\d*)", stdout)
    min_us  = _parse_ms(r"min\s*=\s*(\d+\.?\d*)", stdout)
    max_us  = _parse_ms(r"max\s*=\s*(\d+\.?\d*)", stdout)
    count   = _parse_int(r"count\s*=\s*(\d+)", stdout)

    avg_ms  = us_to_ms(avg_us)
    p50_ms  = us_to_ms(p50_us)
    p90_ms  = us_to_ms(p90_us)
    p99_ms  = us_to_ms(p99_us)
    init_ms = us_to_ms(init_us)
    std_ms  = us_to_ms(std_us)
    min_ms  = us_to_ms(min_us)
    max_ms  = us_to_ms(max_us)

    # Value from TFLite is already in MB — do not convert
    peak_mem_mb = _parse_ms(
        r"Overall peak memory footprint \(MB\).*?:\s*(\d+\.?\d*)", stdout
    )
    peak_mem_mb = round(peak_mem_mb, 2) if peak_mem_mb is not None else None

    # Variance: p90/p50 - 1 is primary signal
    # Fall back to std/avg (coefficient of variation) if percentiles missing
    variance: Optional[float] = None
    if p50_ms and p90_ms and p50_ms > 0:
        variance = round((p90_ms / p50_ms) - 1, 4)
    elif std_ms and avg_ms and avg_ms > 0:
        variance = round(std_ms / avg_ms, 4)

    # Low confidence flag — too few samples for reliable variance
    low_confidence = count is not None and count < 100

    _log(
        f"Parsed: avg={avg_ms}ms p50={p50_ms}ms p90={p90_ms}ms "
        f"std={std_ms}ms count={count} variance={variance} "
        f"low_confidence={low_confidence}"
    )

    return {
        "avg_ms":         avg_ms,
        "p50_ms":         p50_ms,
        "p90_ms":         p90_ms,
        "p99_ms":         p99_ms,
        "init_ms":        init_ms,
        "peak_mem_mb":    peak_mem_mb,
        "variance":       variance,
        "std_ms":         std_ms,
        "min_ms":         min_ms,
        "max_ms":         max_ms,
        "count":          count,
        "low_confidence": low_confidence,
    }

# ---------------------------------------------------------------------
# CPU Baseline Execution
# ---------------------------------------------------------------------


def _compute_autocorr_from_csv(
    deployed: DeployedRun,
    serial: Optional[str],
) -> Optional[float]:
    import tempfile
    import csv
    from pathlib import Path

    remote_csv = f"{deployed.remote_dir}/latencies.csv"
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            local_csv = Path(tmp.name)

        adb.pull(remote_csv, local_csv, serial=serial)  # raises DeployError on failure

        latencies = []
        with open(local_csv) as f:
            reader = csv.reader(f)
            for row in reader:
                for cell in row:
                    try:
                        latencies.append(float(cell.strip()))
                    except ValueError:
                        continue

        if len(latencies) < 10:
            return None

        import statistics
        n    = len(latencies)
        mean = statistics.mean(latencies)
        var  = statistics.variance(latencies)
        if var == 0:
            return 0.0

        cov = sum(
            (latencies[i] - mean) * (latencies[i + 1] - mean)
            for i in range(n - 1)
        ) / (n - 1)

        return round(cov / var, 4)

    except Exception:
        return None

def run_cpu_baseline(
    deployed: DeployedRun,
    *,
    num_runs: int = DEFAULT_NUM_RUNS,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    num_threads: int = DEFAULT_NUM_THREADS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    timeout_s: int = DEFAULT_TIMEOUT,
) -> BaselineResult:

    cmd = _build_command(deployed, num_runs, warmup_runs, num_threads)
    full_cmd = ["adb"]
    if deployed.serial:
        full_cmd += ["-s", deployed.serial]
    full_cmd += ["shell", cmd]

    _log(f"CPU baseline command: {' '.join(full_cmd)}", deployed.serial)

    start = time.time()
    stdout_lines: list[str] = []
    proc = subprocess.Popen(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
        preexec_fn=os.setsid  # allow killing entire process group
    )

    try:
        for line in _stream_output(proc, chunk_size):
            stdout_lines.append(line)
            _log(f"{line}", deployed.serial)

        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        stdout, stderr = proc.communicate()
        raw_output = "\n".join(stdout_lines) + "\n" + stdout + "\n" + stderr
        raise ADBTimeoutError(command=" ".join(full_cmd), output=raw_output)
    except Exception as exc:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        raise exc

    duration_s = time.time() - start
    raw_stdout = "\n".join(stdout_lines)

    if proc.returncode != 0:
        stderr = proc.stderr.read() if proc.stderr else ""
        raise ExecutionError(
            raw_stdout=raw_stdout + "\n" + stderr,
            run_id=deployed.run_id,
        )

    metrics = _parse_baseline_output(raw_stdout)
    autocorr: Optional[float] = None

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
        autocorr_lag1  = autocorr,
        raw_stdout     = raw_stdout,
        run_id         = deployed.run_id,
        num_runs       = num_runs,
        num_threads    = num_threads,
        duration_s     = duration_s,
    )

    _log(f"CPU baseline complete in {duration_s:.1f}s: {result}", deployed.serial)
    return result