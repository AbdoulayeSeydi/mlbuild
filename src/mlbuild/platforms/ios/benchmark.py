"""
mlbuild.platforms.ios.benchmark

Delegate benchmark execution for iOS targets.

Responsibilities:
- Run the benchmark runner app with a specified delegate via idb launch
- Stream stdout via memory-safe bounded queue
- Parse latency percentiles, per-op breakdown, and latency trend
- Parse compute_units_used and thermal_state from runner output
- Return a structured BenchmarkResult with full run provenance

Design rules:
- Always runs after CPU baseline — never standalone.
- Delegate must be SUPPORTED or SKIPPED — caller enforces this.
- Parsing is defensive — missing fields → None, not crash.
- Raw stdout always stored unconditionally.
- Per-run latency trend extracted for stability.py and thermal.py.
- Simulator warning attached to result — never blocks execution.
- No num_threads arg — CoreML manages its own threading.

Key differences from Android:
- No adb shell — runner launched via idb launch with --compute_units arg.
- No process group kill — idb launch killed directly, no setsid needed.
- CoreML runner reports milliseconds directly — no microsecond conversion.
- compute_units_used parsed from stdout — CoreML ground truth for what ran.
- thermal_state parsed from stdout — passed to thermal.py downstream.
- No --enable_op_profiling flag — runner emits op stats if supported.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mlbuild.platforms.ios.deploy import DeployedRun
from mlbuild.core.errors import IDBExecutionError, IDBTimeoutError, IDBNotFoundError


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DEBUG               = os.getenv("MLBUILD_DEBUG") == "1"
DEFAULT_NUM_RUNS    = 50
DEFAULT_WARMUP_RUNS = 10
DEFAULT_TIMEOUT     = 600
QUEUE_MAXSIZE       = 2048
NOISY_THRESHOLD     = 0.2

BUNDLE_ID = "com.mlbuild.MLBuildRunner"

_COMPUTE_UNITS: dict[str, str] = {
    "CPU":          "cpuOnly",
    "GPU":          "cpuAndGPU",
    "ANE":          "all",
    "ANE_EXPLICIT": "cpuAndNeuralEngine",
}


# ---------------------------------------------------------------------
# Structured Logging
# ---------------------------------------------------------------------

def _make_logger(udid: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(f"mlbuild.ios.benchmark.{udid or 'default'}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            '{"time": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "msg": %(message)s}',
            datefmt="%Y-%m-%dT%H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if DEBUG else logging.WARNING)
    return logger


def _log(
    msg:     str,
    udid:    Optional[str] = None,
    level:   str = "INFO",
    context: Optional[dict] = None,
) -> None:
    logger = _make_logger(udid)
    payload = {"msg": msg}
    if context:
        payload.update(context)
    getattr(logger, level.lower(), logger.info)(json.dumps(payload))


# ---------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------

@dataclass
class OpStat:
    name:      str
    avg_ms:    Optional[float]
    pct_total: Optional[float]


@dataclass
class BenchmarkResult:
    """
    Delegate benchmark result.

    compute_units_used: what CoreML actually ran on — may differ from
    requested delegate. Ground truth from runner stdout.

    thermal_state: raw string from runner — thermal.py interprets it.
    None on simulator (no thermal data).

    simulator_warning: non-empty if run was on simulator. Attached to
    result for CLI display and BuildView storage.
    """
    delegate:             str
    compute_units:        str                 # requested
    compute_units_used:   Optional[str]       # actual — CoreML ground truth
    avg_ms:               Optional[float]
    p50_ms:               Optional[float]
    p90_ms:               Optional[float]
    p99_ms:               Optional[float]
    init_ms:              Optional[float]
    peak_mem_mb:          Optional[float]
    variance:             Optional[float]
    latency_trend:        Optional[list[float]]
    ops:                  list[OpStat] = field(default_factory=list)
    raw_stdout:           str          = ""
    run_id:               str          = ""
    run_hash:             str          = ""
    num_runs:             int          = DEFAULT_NUM_RUNS
    duration_s:           float        = 0.0
    cpu_avg_ms:           Optional[float] = None
    speedup:              Optional[float] = None
    thermal_state:        Optional[str]  = None
    is_simulator:         bool           = False
    simulator_warning:    Optional[str]  = None

    @property
    def is_noisy(self) -> bool:
        return self.variance is not None and self.variance > NOISY_THRESHOLD

    def __str__(self) -> str:
        var_str = f"{self.variance:.3f}" if self.variance is not None else "N/A"
        return (
            f"BenchmarkResult("
            f"delegate={self.delegate}, avg={self.avg_ms}ms, "
            f"p50={self.p50_ms}ms, p90={self.p90_ms}ms, "
            f"variance={var_str}, speedup={self.speedup}x, "
            f"compute_units_used={self.compute_units_used})"
        )


# ---------------------------------------------------------------------
# Run Provenance Hash
# ---------------------------------------------------------------------

def _compute_run_hash(
    model_path:         str,
    delegate:           str,
    device_fingerprint: str,
) -> str:
    payload = f"{model_path}|{delegate.upper()}|{device_fingerprint}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------
# Command Construction
# ---------------------------------------------------------------------

def _idb_bin() -> str:
    path = shutil.which("idb")
    if not path:
        raise IDBNotFoundError()
    return path


def _build_idb_command(
    deployed:      DeployedRun,
    compute_units: str,
    num_runs:      int,
    warmup_runs:   int,
) -> list[str]:
    if deployed.is_simulator and deployed.udid:
        app_result = subprocess.run(
            ["xcrun", "simctl", "get_app_container", deployed.udid, BUNDLE_ID, "app"],
            capture_output=True, text=True, timeout=10,
        )
        binary = app_result.stdout.strip() + "/MLBuildRunner"
        data_result = subprocess.run(
            ["xcrun", "simctl", "get_app_container", deployed.udid, BUNDLE_ID, "data"],
            capture_output=True, text=True, timeout=10,
        )
        data_container = data_result.stdout.strip()
        return [
            "xcrun", "simctl", "spawn", deployed.udid, binary,
            f"--model={data_container}/{deployed.remote_model_path}",
            f"--num_runs={num_runs}",
            f"--warmup_runs={warmup_runs}",
            f"--compute_units={compute_units}",
            "--report_peak_memory=true",
            "--report_thermal=true",
            "--report_op_stats=true",
            "--report_latency_trend=true",
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
        "--report_peak_memory=true",
        "--report_thermal=true",
        "--report_op_stats=true",
        "--report_latency_trend=true",
    ]
    return cmd


# ---------------------------------------------------------------------
# Memory-Safe Streaming (mirrors Android exactly)
# ---------------------------------------------------------------------

class _BoundedStreamCollector:
    """
    Memory-safe stdout collector with bounded queue.
    Spills to temp file if queue fills on extremely verbose output.
    Mirrors Android implementation — no process group needed for idb.
    """

    def __init__(self, udid: Optional[str] = None):
        self._queue: queue.Queue[Optional[str]] = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._lines: list[str]                  = []
        self._udid:  Optional[str]              = udid
        self._lock:  threading.Lock             = threading.Lock()
        self._spill: Optional[Path]             = None

    def producer(self, stream) -> None:
        try:
            for line in iter(stream.readline, ""):
                stripped = line.rstrip()
                try:
                    self._queue.put(stripped, timeout=5)
                except queue.Full:
                    self._spill_line(stripped)
            self._queue.put(None)
        except Exception as exc:
            _log(f"Producer thread error: {exc}", self._udid, "ERROR")
            self._queue.put(None)
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def _spill_line(self, line: str) -> None:
        import tempfile
        if self._spill is None:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".mlbuild_spill", delete=False
            )
            self._spill = Path(tmp.name)
            _log(f"Queue full — spilling to {self._spill}", self._udid, "WARN")
        with open(self._spill, "a") as f:
            f.write(line + "\n")

    def drain(self, timeout_per_item: float = 0.1) -> Optional[str]:
        try:
            return self._queue.get(timeout=timeout_per_item)
        except queue.Empty:
            return ""

    def collect(self, line: str) -> None:
        with self._lock:
            self._lines.append(line)

    def all_lines(self) -> list[str]:
        with self._lock:
            lines = list(self._lines)
        if self._spill and self._spill.exists():
            try:
                lines += self._spill.read_text().splitlines()
                self._spill.unlink()
            except Exception:
                pass
        return lines


def _stream_process(
    proc:    subprocess.Popen,
    timeout: int,
    udid:    Optional[str] = None,
) -> tuple[int, str]:
    """
    Stream subprocess stdout safely via bounded queue.

    Key difference from Android: proc.kill() instead of os.killpg().
    idb launch doesn't spawn a detached child — killing the proc is enough.
    """
    collector = _BoundedStreamCollector(udid)

    thread = threading.Thread(
        target=collector.producer,
        args=(proc.stdout,),
        daemon=True,
    )
    thread.start()

    start      = time.time()
    timed_out  = False

    while True:
        line = collector.drain()

        if line is None:
            break

        if line:
            collector.collect(line)
            if DEBUG:
                _log(line, udid, context={"stream": "stdout"})

        if time.time() - start > timeout:
            _log(
                f"Benchmark timeout after {time.time() - start:.1f}s — killing",
                udid, "ERROR",
                context={"timeout_s": timeout, "pid": proc.pid},
            )
            timed_out = True
            proc.kill()
            proc.communicate()
            thread.join(timeout=3)
            break

    if not timed_out:
        thread.join(timeout=10)
        if thread.is_alive():
            _log("Producer thread still alive after join — abandoning", udid, "WARN")

    proc.poll()
    full_stdout = "\n".join(collector.all_lines())

    if timed_out:
        return -1, full_stdout

    return proc.returncode or 0, full_stdout


# ---------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------

def _parse_float(pattern: str, text: str) -> Optional[float]:
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


def _parse_int(pattern: str, text: str) -> Optional[int]:
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass
    return None


def _parse_latency_trend(
    stdout: str,
    udid:   Optional[str] = None,
) -> Optional[list[float]]:
    """
    Extract per-run latencies in execution order (ms).

    CoreML runner emits one line per inference:
        run_latency_ms: 12.34

    Returns list in execution order for stability.py and thermal.py.
    """
    pattern = r"run_latency_ms:\s*(\d+\.?\d*)"
    matches = re.findall(pattern, stdout, re.IGNORECASE)

    if matches:
        try:
            trend = [round(float(v), 3) for v in matches]
            _log(
                f"Latency trend: {len(trend)} points",
                udid,
                context={"trend_count": len(trend)},
            )
            return trend
        except ValueError as exc:
            _log(f"Trend parse failed: {exc}", udid, "WARN")

    _log(
        "No latency trend found — stability analysis will be limited",
        udid, "WARN",
    )
    return None


def _parse_ops(
    stdout: str,
    udid:   Optional[str] = None,
) -> list[OpStat]:
    """
    Parse per-op stats from runner output.

    Expected runner format:
        op_stat: Conv2D avg_ms=1.23 pct=34.5
    """
    ops: list[OpStat] = []
    failed = 0

    pattern = re.compile(
        r"op_stat:\s*([\w][\w\s/]*?)\s+avg_ms=(\d+\.?\d*)\s+pct=(\d+\.?\d*)",
        re.IGNORECASE,
    )

    for line in stdout.splitlines():
        m = pattern.search(line)
        if m:
            try:
                ops.append(OpStat(
                    name      = m.group(1).strip(),
                    avg_ms    = round(float(m.group(2)), 4),
                    pct_total = float(m.group(3)),
                ))
            except ValueError:
                failed += 1

    if failed:
        _log(f"Op parse: {failed} lines failed", udid, "WARN")

    _log(
        f"Op profiling: {len(ops)} ops extracted",
        udid,
        context={"op_count": len(ops), "parse_failures": failed},
    )
    return ops


def _parse_benchmark_output(
    stdout:   str,
    delegate: str,
    udid:     Optional[str] = None,
) -> dict:
    """
    Parse CoreML runner stdout into a metrics dict.
    All fields nullable. No microsecond conversion — runner emits ms.

    Expected runner output lines:
        avg_ms: 12.34
        p50_ms: 12.10
        p90_ms: 13.20
        p99_ms: 14.50
        init_ms: 45.00
        peak_memory_mb: 23.4
        compute_units_used: cpuAndGPU
        thermal_state: nominal
        run_latency_ms: 12.34    (one per run)
        op_stat: Conv2D avg_ms=1.23 pct=34.5
    """
    avg_ms      = _parse_float(r"avg_ms:\s*(\d+\.?\d*)", stdout)
    p50_ms      = _parse_float(r"p50_ms:\s*(\d+\.?\d*)", stdout)
    p90_ms      = _parse_float(r"p90_ms:\s*(\d+\.?\d*)", stdout)
    p99_ms      = _parse_float(r"p99_ms:\s*(\d+\.?\d*)", stdout)
    init_ms     = _parse_float(r"init_ms:\s*(\d+\.?\d*)", stdout)
    peak_mem_mb = _parse_float(r"peak_memory_mb:\s*(\d+\.?\d*)", stdout)

    # compute_units_used — CoreML ground truth
    compute_units_used: Optional[str] = None
    for line in stdout.splitlines():
        line = line.strip().lower()
        if line.startswith("compute_units_used:"):
            _, _, raw = line.partition(":")
            compute_units_used = raw.strip()
            break

    # thermal_state — raw string, thermal.py interprets
    thermal_state: Optional[str] = None
    for line in stdout.splitlines():
        line = line.strip().lower()
        if line.startswith("thermal_state:"):
            _, _, raw = line.partition(":")
            thermal_state = raw.strip()
            break

    variance: Optional[float] = None
    if p50_ms and p90_ms and p50_ms > 0:
        variance = round((p90_ms / p50_ms) - 1, 4)

    latency_trend = _parse_latency_trend(stdout, udid)
    ops           = _parse_ops(stdout, udid)

    _log(
        f"Parsed [{delegate}]: avg={avg_ms}ms p50={p50_ms}ms "
        f"p90={p90_ms}ms variance={variance} "
        f"compute_units_used={compute_units_used} "
        f"ops={len(ops)} trend={len(latency_trend) if latency_trend else 0}",
        udid,
        context={
            "delegate":           delegate,
            "avg_ms":             avg_ms,
            "p50_ms":             p50_ms,
            "p90_ms":             p90_ms,
            "p99_ms":             p99_ms,
            "variance":           variance,
            "compute_units_used": compute_units_used,
            "op_count":           len(ops),
            "trend_count":        len(latency_trend) if latency_trend else 0,
        },
    )

    return {
        "avg_ms":             avg_ms,
        "p50_ms":             p50_ms,
        "p90_ms":             p90_ms,
        "p99_ms":             p99_ms,
        "init_ms":            init_ms,
        "peak_mem_mb":        peak_mem_mb,
        "variance":           variance,
        "latency_trend":      latency_trend,
        "ops":                ops,
        "compute_units_used": compute_units_used,
        "thermal_state":      thermal_state,
    }


# ---------------------------------------------------------------------
# Simulator Warning
# ---------------------------------------------------------------------

_SIMULATOR_WARNING = (
    "Running on iOS Simulator — ANE is unavailable. "
    "Results reflect CPU/GPU only and will not match real device performance. "
    "Connect a real iPhone for ANE benchmarks."
)


# ---------------------------------------------------------------------
# Public: Run Delegate Benchmark
# ---------------------------------------------------------------------

def run_benchmark(
    deployed:           DeployedRun,
    delegate:           str,
    *,
    num_runs:           int   = DEFAULT_NUM_RUNS,
    warmup_runs:        int   = DEFAULT_WARMUP_RUNS,
    timeout_s:          int   = DEFAULT_TIMEOUT,
    cpu_avg_ms:         Optional[float] = None,
    device_fingerprint: str   = "",
) -> BenchmarkResult:
    """
    Run a delegate benchmark and return a structured BenchmarkResult.

    Always called after run_cpu_baseline() — never standalone.
    delegate must be a validated DelegateStatus.SUPPORTED delegate.

    Args:
        deployed:           DeployedRun from deploy.py
        delegate:           Validated delegate name (GPU, ANE, etc.)
        num_runs:           Benchmark iterations
        warmup_runs:        Warmup iterations before measurement
        timeout_s:          Hard timeout in seconds
        cpu_avg_ms:         CPU baseline avg for speedup computation
        device_fingerprint: DeviceProfile.fingerprint for run_hash

    Raises:
        ValueError:       Unknown delegate name
        IDBExecutionError: Runner exited non-zero
        IDBTimeoutError:  Benchmark exceeded timeout_s
    """
    udid = deployed.udid or None

    compute_units = _COMPUTE_UNITS.get(delegate.upper())
    if not compute_units:
        raise ValueError(
            f"Unknown delegate '{delegate}'. "
            f"Valid: {list(_COMPUTE_UNITS.keys())}"
        )

    run_hash = _compute_run_hash(
        deployed.remote_model_path, delegate, device_fingerprint
    )

    # Simulator warning — non-blocking, stored in result
    simulator_warning: Optional[str] = None
    if deployed.is_simulator:
        simulator_warning = _SIMULATOR_WARNING
        _log(_SIMULATOR_WARNING, udid, "WARN")

    _log(
        f"Starting {delegate} benchmark",
        udid,
        context={
            "run_id":        deployed.run_id,
            "run_hash":      run_hash,
            "delegate":      delegate,
            "compute_units": compute_units,
            "num_runs":      num_runs,
            "is_simulator":  deployed.is_simulator,
        },
    )

    cmd = _build_idb_command(deployed, compute_units, num_runs, warmup_runs)
    _log(f"Command: {' '.join(cmd)}", udid)

    start = time.time()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError:
        raise IDBNotFoundError()

    returncode, raw_stdout = _stream_process(proc, timeout_s, udid)
    duration_s = time.time() - start

    if returncode == -1:
        raise IDBTimeoutError(command=" ".join(cmd))

    if returncode != 0:
        raise IDBExecutionError(
            raw_stdout=raw_stdout,
            run_id=deployed.run_id,
        )

    metrics = _parse_benchmark_output(raw_stdout, delegate, udid)

    speedup: Optional[float] = None
    if cpu_avg_ms and metrics["avg_ms"] and cpu_avg_ms > 0:
        speedup = round(cpu_avg_ms / metrics["avg_ms"], 2)
        _log(
            f"Speedup: {speedup}x "
            f"({cpu_avg_ms}ms CPU → {metrics['avg_ms']}ms {delegate})",
            udid,
            context={"speedup": speedup, "cpu_avg_ms": cpu_avg_ms},
        )

    result = BenchmarkResult(
        delegate            = delegate,
        compute_units       = compute_units,
        compute_units_used  = metrics["compute_units_used"],
        avg_ms              = metrics["avg_ms"],
        p50_ms              = metrics["p50_ms"],
        p90_ms              = metrics["p90_ms"],
        p99_ms              = metrics["p99_ms"],
        init_ms             = metrics["init_ms"],
        peak_mem_mb         = metrics["peak_mem_mb"],
        variance            = metrics["variance"],
        latency_trend       = metrics["latency_trend"],
        ops                 = metrics["ops"],
        raw_stdout          = raw_stdout,
        run_id              = deployed.run_id,
        run_hash            = run_hash,
        num_runs            = num_runs,
        duration_s          = duration_s,
        cpu_avg_ms          = cpu_avg_ms,
        speedup             = speedup,
        thermal_state       = metrics["thermal_state"],
        is_simulator        = deployed.is_simulator,
        simulator_warning   = simulator_warning,
    )

    _log(
        f"{delegate} benchmark complete in {duration_s:.1f}s: {result}",
        udid,
        context={
            "run_id":    deployed.run_id,
            "run_hash":  run_hash,
            "duration_s": duration_s,
            "speedup":   speedup,
        },
    )

    return result