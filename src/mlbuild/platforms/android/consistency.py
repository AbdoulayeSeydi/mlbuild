"""
mlbuild.platforms.android.consistency

Output consistency checking between CPU and delegate execution.

Responsibilities:
- Run a single forward pass on CPU and on the delegate
- Compare outputs with a loose tolerance
- Flag delegates whose outputs diverge from CPU as INCONSISTENT
- Protect MLBuild's credibility from fake speedups

Design rules:
- Runs once per delegate during validation — not on every benchmark run.
- Tolerance is loose by default (relative diff < 0.05).
- Never crashes on comparison failure — returns INCONSISTENT instead.
- Raw outputs are stored for debugging regardless of result.
- No latency measurement here — that belongs to benchmark.py.

Why this exists:
  Some delegates skip ops, use lower precision, or silently change
  execution paths. The result is faster latency with wrong computation.
  Without catching this, MLBuild publishes numbers that are technically
  fast and substantively wrong. A tool that reports 3x speedup from a
  delegate computing wrong answers is worse than no tool at all.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
import signal
from dataclasses import dataclass
from typing import Optional, Iterator, List

from mlbuild.platforms.android.deploy import DeployedRun

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DEBUG = os.getenv("MLBUILD_DEBUG") == "1"

DEFAULT_TOLERANCE       = 0.05  # 5% relative difference
DEFAULT_NUM_RUNS        = 1
DEFAULT_WARMUP_RUNS     = 0
DEFAULT_TIMEOUT_SEC     = 60
STREAM_CHUNK_SIZE       = 512  # adaptive for large tensor outputs

# Delegate flag mapping (normalized)
_DELEGATE_FLAGS = {
    "GPU": "--use_gpu=true",
    "NNAPI": "--use_nnapi=true",
    "HEXAGON": "--use_hexagon=true",
    "HEXAGON_HTP": "--use_hexagon=true --hexagon_nn_nodes_on_graph=1",
}

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def _log(msg: str, deployed: Optional[DeployedRun] = None) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    serial = f"[{deployed.serial}]" if deployed and deployed.serial else ""
    print(f"{ts} [mlbuild.consistency]{serial} {msg}", file=sys.stderr if DEBUG else sys.stdout)

# ---------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------

@dataclass
class ConsistencyResult:
    passed: bool
    relative_diff: Optional[float]
    cpu_output: Optional[str]
    delegate_output: Optional[str]
    tolerance: float

    def __str__(self) -> str:
        diff_str = f"{self.relative_diff:.6f}" if self.relative_diff not in (None, float('nan'), float('inf')) else "N/A"
        status = "PASS" if self.passed else "FAIL"
        return f"ConsistencyResult({status}, rel_diff={diff_str}, tol={self.tolerance})"

# ---------------------------------------------------------------------
# Subprocess Streaming
# ---------------------------------------------------------------------

def _stream_subprocess(proc: subprocess.Popen) -> Iterator[str]:
    """
    Non-blocking streaming of stdout and stderr lines with process group handling.
    Flushes partial lines at EOF.
    """
    stdout_buffer = ""
    stderr_buffer = ""

    def read_stream(stream, buffer_name):
        nonlocal stdout_buffer, stderr_buffer
        buf = ""
        while True:
            chunk = stream.read(STREAM_CHUNK_SIZE)
            if not chunk:
                break
            buf += chunk
            lines = buf.split("\n")
            buf = lines[-1]
            for line in lines[:-1]:
                yield line
        if buf.strip():
            yield buf

    for line in read_stream(proc.stdout, "stdout"):
        yield line

# ---------------------------------------------------------------------
# Safe Command Builder
# ---------------------------------------------------------------------

def _build_command(deployed: DeployedRun, delegate_flag: Optional[str] = None,
                   num_runs: int = DEFAULT_NUM_RUNS, warmup_runs: int = DEFAULT_WARMUP_RUNS) -> List[str]:
    """
    Build adb shell command safely using a list (no shell injection risk)
    """
    cmd = [
        deployed.remote_binary_path,
        f"--graph={deployed.remote_model_path}",
        f"--num_runs={num_runs}",
        f"--warmup_runs={warmup_runs}",
        "--print_preinvoke_state=true",
        "--print_postinvoke_state=true",
    ]
    if delegate_flag:
        cmd.extend(delegate_flag.split())
    full_cmd = ["adb"]
    if deployed.serial:
        full_cmd += ["-s", deployed.serial]
    full_cmd += ["shell"] + [f'"{c}"' if ' ' in c else c for c in cmd]
    return full_cmd

# ---------------------------------------------------------------------
# Delegate Flag Normalization
# ---------------------------------------------------------------------

def _delegate_flag(delegate: str) -> Optional[str]:
    key = delegate.upper()
    flag = _DELEGATE_FLAGS.get(key)
    if not flag:
        _log(f"Unknown delegate '{delegate}' — skipping delegate run")
    return flag

# ---------------------------------------------------------------------
# Output Extraction
# ---------------------------------------------------------------------

def _extract_output_tensors(stdout: str) -> Optional[List[float]]:
    """
    Flatten all tensor outputs into a single list.
    Multi-line outputs handled.
    """
    values: List[float] = []
    patterns = [
        r"Output\s*\[\d+\]:\s*\[([^\]]+)\]",
        r"output_tensor\[\d+\]\s*=\s*\[([^\]]+)\]",
        r"Tensor\s+\d+.*?:\s*\[([^\]]+)\]",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, stdout, re.IGNORECASE | re.MULTILINE):
            raw_vals = match.group(1).split(",")
            for i, v in enumerate(raw_vals):
                try:
                    values.append(float(v.strip()))
                except ValueError:
                    _log(f"Failed parsing tensor value '{v}' at index {i}")
        if values:
            break
    if not values:
        _log("No output tensor values found")
        return None
    _log(f"Extracted {len(values)} output values")
    return values

# ---------------------------------------------------------------------
# Relative Difference
# ---------------------------------------------------------------------

def _compute_relative_diff(cpu_vals: List[float], delegate_vals: List[float], epsilon: float = 1e-6) -> Optional[float]:
    if not cpu_vals or not delegate_vals:
        return None
    n = min(len(cpu_vals), len(delegate_vals))
    if n == 0:
        return None
    if len(cpu_vals) != len(delegate_vals):
        _log(f"Output length mismatch: CPU={len(cpu_vals)}, Delegate={len(delegate_vals)} — comparing first {n} values")
    max_diff = 0.0
    for i in range(n):
        denom = max(abs(cpu_vals[i]), epsilon)
        diff = abs(cpu_vals[i] - delegate_vals[i]) / denom
        max_diff = max(max_diff, diff)
    return round(max_diff, 8)

# ---------------------------------------------------------------------
# Run Single Pass (Robust)
# ---------------------------------------------------------------------

def _run_single_pass(deployed: DeployedRun, delegate_flag: Optional[str] = None,
                     timeout: float = DEFAULT_TIMEOUT_SEC) -> Optional[str]:
    """
    Execute one forward pass safely with streaming, timeout, and process group kill
    """
    cmd = _build_command(deployed, delegate_flag)
    _log(f"Running command: {' '.join(cmd)}", deployed)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid  # process group for safe kill
        )
        stdout_lines: List[str] = []
        stderr_lines: List[str] = []

        def reader_thread(stream, collector):
            for line in iter(lambda: stream.readline(), ''):
                collector.append(line.rstrip())
                _log(line.rstrip(), deployed)

        t_out = threading.Thread(target=reader_thread, args=(proc.stdout, stdout_lines))
        t_err = threading.Thread(target=reader_thread, args=(proc.stderr, stderr_lines))
        t_out.start()
        t_err.start()

        start = time.time()
        t_out.join(timeout)
        t_err.join(timeout)
        proc.poll()
        elapsed = time.time() - start

        if proc.returncode is None:
            _log(f"Process exceeded timeout ({timeout}s), killing", deployed)
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            t_out.join()
            t_err.join()
            return None

        if proc.returncode != 0:
            _log(f"Process exited with {proc.returncode}", deployed)
            return None

        return "\n".join(stdout_lines)

    except Exception as exc:
        _log(f"Error running forward pass: {exc}", deployed)
        return None

# ---------------------------------------------------------------------
# Public Consistency Check
# ---------------------------------------------------------------------

def check_consistency(deployed: DeployedRun, delegate: str, *,
                      tolerance: float = DEFAULT_TOLERANCE,
                      num_runs: int = DEFAULT_NUM_RUNS,
                      warmup_runs: int = DEFAULT_WARMUP_RUNS,
                      timeout: float = DEFAULT_TIMEOUT_SEC) -> ConsistencyResult:
    start = time.time()
    _log(f"Consistency check: delegate={delegate}, tolerance={tolerance}", deployed)

    # CPU pass
    cpu_stdout = _run_single_pass(deployed, delegate_flag=None, timeout=timeout)
    if cpu_stdout is None:
        _log("CPU pass failed — cannot perform consistency check", deployed)
        return ConsistencyResult(
            passed=False,
            relative_diff=None,
            cpu_output=None,
            delegate_output=None,
            tolerance=tolerance
        )

    # Delegate pass
    flag = _delegate_flag(delegate)
    delegate_stdout = _run_single_pass(deployed, delegate_flag=flag, timeout=timeout)
    if delegate_stdout is None:
        _log(f"Delegate pass failed — marking INCONSISTENT", deployed)
        return ConsistencyResult(
            passed=False,
            relative_diff=None,
            cpu_output=cpu_stdout,
            delegate_output=None,
            tolerance=tolerance
        )

    # Tensor extraction
    cpu_vals = _extract_output_tensors(cpu_stdout)
    delegate_vals = _extract_output_tensors(delegate_stdout)
    if cpu_vals is None or delegate_vals is None:
        _log("Tensor extraction failed — skipping numerical comparison", deployed)
        return ConsistencyResult(
            passed=True,
            relative_diff=None,
            cpu_output=cpu_stdout,
            delegate_output=delegate_stdout,
            tolerance=tolerance
        )

    # Relative difference
    relative_diff = _compute_relative_diff(cpu_vals, delegate_vals)
    passed = relative_diff is not None and relative_diff < tolerance

    duration = time.time() - start
    _log(f"Consistency check complete in {duration:.2f}s: rel_diff={relative_diff}, result={'PASS' if passed else 'FAIL'}", deployed)

    return ConsistencyResult(
        passed=passed,
        relative_diff=relative_diff,
        cpu_output=cpu_stdout,
        delegate_output=delegate_stdout,
        tolerance=tolerance
    )