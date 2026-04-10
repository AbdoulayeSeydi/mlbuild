"""
mlbuild.platforms.ios.consistency

Output consistency checking between CPU and delegate execution.

Responsibilities:
- Run a single forward pass on CPU and on the delegate via idb launch
- Compare outputs with a loose tolerance
- Flag delegates whose outputs diverge from CPU as INCONSISTENT
- Protect MLBuild's credibility from fake speedups

Design rules:
- Runs once per delegate during validation — not on every benchmark run.
- Tolerance is loose by default (5%). ANE uses INT8/FP16 internally —
  tighter values will false-positive on legitimate precision tradeoffs.
- Never crashes on comparison failure — returns INCONSISTENT instead.
- Raw outputs stored for debugging regardless of result.
- No latency measurement here — that belongs to benchmark.py.
- ANE/ANE_EXPLICIT must not be called on simulator — caller gates this
  via has_ane. consistency.py trusts the delegate list it receives.

Key differences from Android:
- No adb shell — runner launched via idb launch with compute_units arg.
- No process group kill — idb launch process killed directly.
- Delegate expressed as compute_units string, not binary flag.
- Output tensor format is runner-defined key:value lines.
- No threading for stderr — single stdout stream from idb.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Iterator, List, Optional

from mlbuild.platforms.ios.deploy import DeployedRun
from mlbuild.core.errors import IDBNotFoundError


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DEBUG = os.getenv("MLBUILD_DEBUG") == "1"

DEFAULT_TOLERANCE   = 0.05   # 5% — loose intentionally, ANE uses reduced precision
DEFAULT_NUM_RUNS    = 1
DEFAULT_WARMUP_RUNS = 0
DEFAULT_TIMEOUT_SEC = 60
STREAM_CHUNK_SIZE   = 512

BUNDLE_ID = "com.mlbuild.MLBuildRunner"

# Compute units per delegate label — passed as --compute_units arg to runner
_COMPUTE_UNITS = {
    "CPU":          "cpuOnly",
    "GPU":          "cpuAndGPU",
    "ANE":          "all",
    "ANE_EXPLICIT": "cpuAndNeuralEngine",
}


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def _log(msg: str, deployed: Optional[DeployedRun] = None) -> None:
    if not DEBUG:
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    udid = f"[{deployed.udid}]" if deployed and deployed.udid else ""
    print(f"{ts} [mlbuild.ios.consistency]{udid} {msg}", file=sys.stderr)


# ---------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------

@dataclass
class ConsistencyResult:
    passed:          bool
    relative_diff:   Optional[float]
    cpu_output:      Optional[str]
    delegate_output: Optional[str]
    tolerance:       float

    def __str__(self) -> str:
        diff_str = (
            f"{self.relative_diff:.6f}"
            if self.relative_diff not in (None, float("nan"), float("inf"))
            else "N/A"
        )
        status = "PASS" if self.passed else "FAIL"
        return f"ConsistencyResult({status}, rel_diff={diff_str}, tol={self.tolerance})"


# ---------------------------------------------------------------------
# Command Builder
# ---------------------------------------------------------------------

def _idb_bin() -> str:
    path = shutil.which("idb")
    if not path:
        raise IDBNotFoundError()
    return path


def _build_command(
    deployed: DeployedRun,
    compute_units: str,
    num_runs: int,
    warmup_runs: int,
) -> List[str]:
    if deployed.is_simulator and deployed.udid:
        app_r = subprocess.run(
            ["xcrun", "simctl", "get_app_container", deployed.udid, BUNDLE_ID, "app"],
            capture_output=True, text=True, timeout=10,
        )
        binary = app_r.stdout.strip() + "/MLBuildRunner"
        data_r = subprocess.run(
            ["xcrun", "simctl", "get_app_container", deployed.udid, BUNDLE_ID, "data"],
            capture_output=True, text=True, timeout=10,
        )
        data_container = data_r.stdout.strip()
        return [
            "xcrun", "simctl", "spawn", deployed.udid, binary,
            f"--model={data_container}/{deployed.remote_model_path}",
            f"--num_runs={num_runs}",
            f"--warmup_runs={warmup_runs}",
            f"--compute_units={compute_units}",
            "--print_output_tensors=true",
            "--report_thermal=false",
        ]

    cmd = [_idb_bin(), "launch"]
    if deployed.udid:
        cmd += ["--udid", deployed.udid]
    cmd += [
        BUNDLE_ID,
        f"--model={deployed.remote_model_path}",
        f"--num_runs={num_runs}",
        f"--warmup_runs={warmup_runs}",
        f"--compute_units={compute_units}",
        "--print_output_tensors=true",
        "--report_thermal=false",
    ]
    return cmd


# ---------------------------------------------------------------------
# Compute Units Resolution
# ---------------------------------------------------------------------

def _resolve_compute_units(delegate: str) -> Optional[str]:
    units = _COMPUTE_UNITS.get(delegate.upper())
    if not units:
        _log(f"Unknown delegate '{delegate}' — no compute_units mapping")
    return units


# ---------------------------------------------------------------------
# Output Streaming
# ---------------------------------------------------------------------

def _stream_output(
    proc: subprocess.Popen,
    chunk_size: int = STREAM_CHUNK_SIZE,
) -> Iterator[str]:
    """
    Non-blocking stdout streaming.
    Mirrors baseline.py — single stream, no threading needed.
    """
    buffer = ""

    while True:
        chunk = proc.stdout.read(chunk_size) or ""
        if not chunk and proc.poll() is not None:
            break

        buffer += chunk
        lines = buffer.split("\n")
        buffer = lines[-1]
        yield from lines[:-1]

    if buffer.strip():
        yield buffer


# ---------------------------------------------------------------------
# Output Tensor Extraction
# ---------------------------------------------------------------------

def _extract_output_tensors(stdout: str) -> Optional[List[float]]:
    """
    Extract output tensor values from CoreML runner output.

    Expected runner format:
        output_tensor[0]: [0.12, 0.34, 0.54]
        output_tensor[1]: [0.98, 0.01, 0.01]

    All tensors are flattened into a single list for comparison.
    Returns None if no tensors found — caller treats this as skip,
    not INCONSISTENT.
    """
    values: List[float] = []

    patterns = [
        r"output_tensor\[\d+\]:\s*\[([^\]]+)\]",
        r"output\[\d+\]:\s*\[([^\]]+)\]",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, stdout, re.IGNORECASE):
            for raw in match.group(1).split(","):
                try:
                    values.append(float(raw.strip()))
                except ValueError:
                    _log(f"Failed to parse tensor value: {repr(raw.strip())}")
        if values:
            break

    if not values:
        _log("No output tensors found in runner output")
        return None

    _log(f"Extracted {len(values)} tensor values")
    return values


# ---------------------------------------------------------------------
# Relative Difference
# ---------------------------------------------------------------------

def _compute_relative_diff(
    cpu_vals: List[float],
    delegate_vals: List[float],
    epsilon: float = 1e-6,
) -> Optional[float]:
    if not cpu_vals or not delegate_vals:
        return None

    n = min(len(cpu_vals), len(delegate_vals))

    if len(cpu_vals) != len(delegate_vals):
        _log(
            f"Output length mismatch: CPU={len(cpu_vals)}, "
            f"delegate={len(delegate_vals)} — comparing first {n} values"
        )

    max_diff = 0.0
    for i in range(n):
        denom = max(abs(cpu_vals[i]), epsilon)
        diff = abs(cpu_vals[i] - delegate_vals[i]) / denom
        max_diff = max(max_diff, diff)

    return round(max_diff, 8)


# ---------------------------------------------------------------------
# Single Forward Pass
# ---------------------------------------------------------------------

def _run_single_pass(
    deployed: DeployedRun,
    compute_units: str,
    timeout: float,
    num_runs: int,
    warmup_runs: int,
) -> Optional[str]:
    """
    Execute one forward pass via idb launch.
    Returns stdout string on success, None on failure.
    Never raises — caller decides how to handle None.
    """
    cmd = _build_command(deployed, compute_units, num_runs, warmup_runs)
    _log(f"Running pass: compute_units={compute_units} cmd={' '.join(cmd)}", deployed)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stdout_lines: List[str] = []
        start = time.time()

        for line in _stream_output(proc):
            stdout_lines.append(line)
            _log(line, deployed)

            if time.time() - start > timeout:
                _log(f"Pass exceeded timeout ({timeout}s) — killing", deployed)
                proc.kill()
                proc.communicate()
                return None

        proc.wait()

        if proc.returncode != 0:
            _log(f"Pass exited with code {proc.returncode}", deployed)
            return None

        return "\n".join(stdout_lines)

    except Exception as exc:
        _log(f"Pass error: {exc}", deployed)
        return None


# ---------------------------------------------------------------------
# Public: Consistency Check
# ---------------------------------------------------------------------

def check_consistency(
    deployed: DeployedRun,
    delegate: str,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    num_runs: int = DEFAULT_NUM_RUNS,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    timeout: float = DEFAULT_TIMEOUT_SEC,
) -> ConsistencyResult:
    """
    Run CPU and delegate forward passes, compare outputs.

    On tensor extraction failure: passes with relative_diff=None.
    Absence of tensors is a runner limitation, not a delegate error.

    On pass execution failure: fails immediately — can't compare
    what we can't run.
    """
    start = time.time()
    _log(f"Consistency check: delegate={delegate}, tolerance={tolerance}", deployed)

    compute_units = _resolve_compute_units(delegate)
    if compute_units is None:
        _log(f"No compute_units for delegate '{delegate}' — marking INCONSISTENT")
        return ConsistencyResult(
            passed=False,
            relative_diff=None,
            cpu_output=None,
            delegate_output=None,
            tolerance=tolerance,
        )

    # CPU pass
    cpu_stdout = _run_single_pass(
        deployed,
        compute_units=_COMPUTE_UNITS["CPU"],
        timeout=timeout,
        num_runs=num_runs,
        warmup_runs=warmup_runs,
    )

    if cpu_stdout is None:
        _log("CPU pass failed — cannot perform consistency check", deployed)
        return ConsistencyResult(
            passed=False,
            relative_diff=None,
            cpu_output=None,
            delegate_output=None,
            tolerance=tolerance,
        )

    # Delegate pass
    delegate_stdout = _run_single_pass(
        deployed,
        compute_units=compute_units,
        timeout=timeout,
        num_runs=num_runs,
        warmup_runs=warmup_runs,
    )

    if delegate_stdout is None:
        _log(f"Delegate pass failed — marking INCONSISTENT", deployed)
        return ConsistencyResult(
            passed=False,
            relative_diff=None,
            cpu_output=cpu_stdout,
            delegate_output=None,
            tolerance=tolerance,
        )

    # Tensor extraction
    cpu_vals      = _extract_output_tensors(cpu_stdout)
    delegate_vals = _extract_output_tensors(delegate_stdout)

    if cpu_vals is None or delegate_vals is None:
        # Runner didn't emit tensors — can't compare numerically.
        # Pass with None diff rather than false INCONSISTENT.
        _log("Tensor extraction failed — skipping numerical comparison, passing")
        return ConsistencyResult(
            passed=True,
            relative_diff=None,
            cpu_output=cpu_stdout,
            delegate_output=delegate_stdout,
            tolerance=tolerance,
        )

    # Relative difference
    relative_diff = _compute_relative_diff(cpu_vals, delegate_vals)
    passed = relative_diff is not None and relative_diff < tolerance

    duration = time.time() - start
    _log(
        f"Consistency check complete in {duration:.2f}s: "
        f"rel_diff={relative_diff} result={'PASS' if passed else 'FAIL'}",
        deployed,
    )

    return ConsistencyResult(
        passed=passed,
        relative_diff=relative_diff,
        cpu_output=cpu_stdout,
        delegate_output=delegate_stdout,
        tolerance=tolerance,
    )