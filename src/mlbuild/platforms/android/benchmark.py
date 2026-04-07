"""
mlbuild.platforms.android.benchmark

Enterprise-grade delegate benchmark execution.

Responsibilities:
- Run the benchmark binary with a specified delegate
- Stream stdout via memory-safe bounded queue
- Parse latency percentiles, per-op breakdown, and latency trend
- Return a structured BenchmarkResult with full run provenance

Design rules:
- Always runs after CPU baseline — never standalone.
- Delegate flag is required — CPU-only runs belong to baseline.py.
- Parsing is defensive — missing fields → None, not crash.
- Raw stdout always stored unconditionally.
- Per-run latency trend extracted for stability.py and thermal.py.
- All subprocess interactions use process groups + SIGKILL.
- Shell arguments passed as list — no string concatenation, no injection risk.
- Structured JSON logging for CI/device farm aggregation.
- Memory-safe: bounded queue + disk spill for large runs.
"""

from __future__ import annotations
import sys

import hashlib
import json
import logging
import os
import queue
import re
import shlex
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mlbuild.platforms.android.deploy import DeployedRun
from mlbuild.core.errors import ExecutionError, ADBTimeoutError


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DEBUG               = os.getenv("MLBUILD_DEBUG") == "1"
DEFAULT_NUM_RUNS    = 50
DEFAULT_WARMUP_RUNS = 10
DEFAULT_NUM_THREADS = 4
DEFAULT_TIMEOUT     = 600        # seconds — long runs need minutes
QUEUE_MAXSIZE       = 2048       # bounded queue — prevents memory blowup
NOISY_THRESHOLD     = 0.2        # variance > this = noisy result

# Delegate flag registry
_DELEGATE_FLAGS: dict[str, str] = {
    "GPU":         "--use_gpu=true",
    "NNAPI":       "--use_nnapi=true",
    "HEXAGON":     "--use_hexagon=true",
    "HEXAGON_HTP": "--use_hexagon=true --hexagon_nn_nodes_on_graph=1",
}


# ---------------------------------------------------------------------
# Structured Logging
# ---------------------------------------------------------------------

def _make_logger(serial: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(f"mlbuild.benchmark.{serial or 'default'}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            '{"time": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "msg": %(message)s}',
            datefmt="%Y-%m-%dT%H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
    return logger


def _log(
    msg: str,
    serial:  Optional[str] = None,
    level:   str = "INFO",
    context: Optional[dict] = None,
) -> None:
    """
    Structured JSON log. CI/device farms can parse and aggregate these.
    context dict is merged into the log payload for per-run correlation.
    """
    logger = _make_logger(serial)
    payload = {"msg": msg}
    if context:
        payload.update(context)
    getattr(logger, level.lower(), logger.info)(json.dumps(payload))


# ---------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------

@dataclass
class OpStat:
    """Per-op profiling entry from --enable_op_profiling."""
    name:      str
    avg_ms:    Optional[float]
    pct_total: Optional[float]


@dataclass
class BenchmarkResult:
    """
    Delegate benchmark result.

    run_hash: SHA256 of (model_path + delegate + device_fingerprint).
    Enables cache lookups and cross-run reproducibility checks.

    latency_trend: per-run latencies in execution order (ms).
    Consumed by stability.py and thermal.py for drift detection.
    """
    delegate:            str
    avg_ms:              Optional[float]
    p50_ms:              Optional[float]
    p90_ms:              Optional[float]
    p99_ms:              Optional[float]
    init_ms:             Optional[float]
    peak_mem_mb:         Optional[float]
    variance:            Optional[float]
    latency_trend:       Optional[list[float]]
    ops:                 list[OpStat] = field(default_factory=list)
    raw_stdout:          str          = ""
    run_id:              str          = ""
    run_hash:            str          = ""
    num_runs:            int          = DEFAULT_NUM_RUNS
    num_threads:         int          = DEFAULT_NUM_THREADS
    duration_s:          float        = 0.0
    cpu_avg_ms:          Optional[float] = None   # baseline ref for speedup
    speedup:             Optional[float] = None   # cpu_avg / delegate_avg

    @property
    def is_noisy(self) -> bool:
        return self.variance is not None and self.variance > NOISY_THRESHOLD

    def __str__(self) -> str:
        var_str = f"{self.variance:.3f}" if self.variance is not None else "N/A"
        return (
            f"BenchmarkResult("
            f"delegate={self.delegate}, avg={self.avg_ms}ms, "
            f"p50={self.p50_ms}ms, p90={self.p90_ms}ms, "
            f"variance={var_str}, speedup={self.speedup}x)"
        )


# ---------------------------------------------------------------------
# Run Provenance Hash
# ---------------------------------------------------------------------

def _compute_run_hash(
    model_path:          str,
    delegate:            str,
    device_fingerprint:  str,
) -> str:
    """
    SHA256 of (model_path + delegate + device_fingerprint).
    Stable identifier for this exact run configuration.
    Enables caching and cross-run comparison.
    """
    payload = f"{model_path}|{delegate.upper()}|{device_fingerprint}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------
# Command Construction (injection-safe)
# ---------------------------------------------------------------------

def _build_adb_command(
    deployed:    DeployedRun,
    delegate:    str,
    num_runs:    int,
    warmup_runs: int,
    num_threads: int,
) -> list[str]:
    """
    Build the adb command as a list of strings.

    NEVER joins into a single shell string — prevents injection via
    model paths or binary paths containing spaces, quotes, or special chars.

    The remote command is passed as a single quoted argument to
    `adb shell` using shlex.join on the remote parts.
    """
    flag = _DELEGATE_FLAGS.get(delegate.upper())
    if not flag:
        raise ValueError(
            f"Unknown delegate '{delegate}'. "
            f"Valid: {list(_DELEGATE_FLAGS.keys())}"
        )

    # Build remote command parts as a list, then join for adb shell
    remote_parts = [
        deployed.remote_binary_path,
        f"--graph={deployed.remote_model_path}",
        f"--num_runs={num_runs}",
        f"--warmup_runs={warmup_runs}",
        f"--num_threads={num_threads}",
        "--enable_op_profiling=true",
        "--report_peak_memory_footprint=true",
    ] + shlex.split(flag)

    # shlex.join safely quotes every argument
    safe_remote_cmd = shlex.join(remote_parts)

    full_cmd = ["adb"]
    if deployed.serial:
        full_cmd += ["-s", deployed.serial]
    full_cmd += ["shell", safe_remote_cmd]

    return full_cmd


# ---------------------------------------------------------------------
# Memory-Safe Streaming
# ---------------------------------------------------------------------

class _BoundedStreamCollector:
    """
    Memory-safe stdout collector.

    Uses a bounded queue (QUEUE_MAXSIZE) to prevent blowup on
    large runs (50+ iterations × full op profiling).

    If queue fills (extremely verbose output), spills to a temp file
    and logs a warning. Main thread drains the queue continuously
    to prevent producer blocking.
    """

    def __init__(self, serial: Optional[str] = None):
        self._queue:  queue.Queue[Optional[str]] = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._lines:  list[str]                  = []
        self._serial: Optional[str]              = serial
        self._lock:   threading.Lock             = threading.Lock()
        self._spill:  Optional[Path]             = None

    def producer(self, stream) -> None:
        """Runs in a daemon thread. Puts lines into bounded queue."""
        try:
            for line in iter(stream.readline, ""):
                stripped = line.rstrip()
                try:
                    self._queue.put(stripped, timeout=5)
                except queue.Full:
                    # Queue full — spill to temp file
                    self._spill_line(stripped)
            self._queue.put(None)  # sentinel
        except Exception as exc:
            _log(f"Producer thread error: {exc}", self._serial, "ERROR")
            self._queue.put(None)
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def _spill_line(self, line: str) -> None:
        if self._spill is None:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".mlbuild_spill", delete=False
            )
            self._spill = Path(tmp.name)
            _log(
                f"Queue full — spilling to {self._spill}",
                self._serial, "WARN"
            )
        with open(self._spill, "a") as f:
            f.write(line + "\n")

    def drain(self, timeout_per_item: float = 0.1) -> Optional[str]:
        """Non-blocking drain — returns one line or None."""
        try:
            return self._queue.get(timeout=timeout_per_item)
        except queue.Empty:
            return ""   # distinguish from sentinel (None)

    def collect(self, line: str) -> None:
        with self._lock:
            self._lines.append(line)

    def all_lines(self) -> list[str]:
        """Return all collected lines including any spill file content."""
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
    serial:  Optional[str] = None,
) -> tuple[int, str]:
    """
    Stream subprocess stdout safely.

    - Bounded queue prevents memory blowup on large outputs
    - Producer thread is daemon — won't block process exit
    - SIGKILL (not SIGTERM) on timeout — adb ignores SIGTERM
    - Thread join with deadline enforces hard cleanup
    - Returns (-1, partial_stdout) on timeout
    - Returns (returncode, full_stdout) on completion
    """
    collector = _BoundedStreamCollector(serial)

    thread = threading.Thread(
        target=collector.producer,
        args=(proc.stdout,),
        daemon=True,
    )
    thread.start()

    start = time.time()
    timed_out = False

    while True:
        line = collector.drain()

        if line is None:
            # Sentinel — producer finished
            break

        if line:
            collector.collect(line)
            # Only log in debug mode to avoid slowing down tight loops
            if DEBUG:
                _log(line, serial, context={"stream": "stdout"})

        elapsed = time.time() - start
        if elapsed > timeout:
            _log(
                f"Benchmark timeout after {elapsed:.1f}s — killing",
                serial, "ERROR",
                context={"timeout_s": timeout, "pid": proc.pid},
            )
            timed_out = True
            _kill_process_group(proc, serial)
            # Give thread 3s to drain then abandon
            thread.join(timeout=3)
            break

    if not timed_out:
        thread.join(timeout=10)
        if thread.is_alive():
            _log("Producer thread still alive after join — abandoning", serial, "WARN")

    proc.poll()
    full_stdout = "\n".join(collector.all_lines())

    if timed_out:
        return -1, full_stdout

    return proc.returncode or 0, full_stdout


def _kill_process_group(proc: subprocess.Popen, serial: Optional[str]) -> None:
    """
    Kill the process group with SIGKILL.
    Verifies the kill succeeded — logs explicitly if it didn't.
    Never silently swallows failures.
    """
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGKILL)
        _log(f"SIGKILL sent to pgid={pgid}", serial)
    except ProcessLookupError:
        _log(f"Process {proc.pid} already gone before SIGKILL", serial, "DEBUG")
    except PermissionError as exc:
        _log(f"SIGKILL permission denied for pid={proc.pid}: {exc}", serial, "ERROR")
    except Exception as exc:
        _log(f"SIGKILL failed for pid={proc.pid}: {exc}", serial, "ERROR")


# ---------------------------------------------------------------------
# Parsing (Defensive + Multi-Format)
# ---------------------------------------------------------------------

def _parse_ms(pattern: str, text: str) -> Optional[float]:
    """
    Generic float extractor. Handles scientific notation (1.23e4).
    Returns None on no match or parse failure.
    """
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


def _us_to_ms(val: Optional[float]) -> Optional[float]:
    """Convert microseconds to milliseconds. Explicit — never implicit."""
    return round(val / 1000.0, 3) if val is not None else None


def _parse_latency_trend(stdout: str, serial: Optional[str] = None) -> Optional[list[float]]:
    """
    Extract per-run latencies in execution order (microseconds → ms).

    Supports multiple TFLite output formats:
    - curr=NNNNN           (standard)
    - Current=NNNNN        (some builds)
    - Run N: NNNNus        (older format)
    - Inference[N]=NNNN    (some custom builds)

    Logs explicitly if no trend found — farm engineers need visibility.
    """
    patterns = [
        r"\bcurr\s*=\s*(\d+\.?\d*(?:e[+-]?\d+)?)",        # curr=11234
        r"\bCurrent\s*=\s*(\d+\.?\d*(?:e[+-]?\d+)?)",     # Current=11234
        r"Run\s+\d+:\s*(\d+\.?\d*(?:e[+-]?\d+)?)us",      # Run 1: 11234us
        r"Inference\[\d+\]\s*=\s*(\d+\.?\d*(?:e[+-]?\d+)?)", # Inference[N]=NNNN
    ]

    for pattern in patterns:
        matches = re.findall(pattern, stdout, re.IGNORECASE)
        if matches:
            try:
                trend = [round(float(v) / 1000.0, 3) for v in matches]
                _log(
                    f"Latency trend extracted: {len(trend)} points "
                    f"via pattern '{pattern[:30]}'",
                    serial,
                    context={"trend_count": len(trend)},
                )
                return trend
            except ValueError as exc:
                _log(f"Trend parse failed for pattern '{pattern[:30]}': {exc}", serial, "WARN")
                continue

    _log(
        "No latency trend found in stdout — stability analysis will be limited",
        serial, "WARN",
    )
    return None


def _parse_ops(stdout: str, serial: Optional[str] = None) -> list[OpStat]:
    """
    Parse per-op profiling from --enable_op_profiling output.

    Multiple patterns handle format variation across TFLite versions.
    Logs count and any parse failures explicitly — CI needs visibility
    when per-op data is missing.
    """
    ops: list[OpStat] = []
    failed_lines = 0

    patterns = [
        # Standard: op_name  1234us  12.3%
        re.compile(
            r"^\s*([\w][\w\s/]*?)\s+(\d+\.?\d*(?:e[+-]?\d+)?)\s+(\d+\.?\d*)%",
            re.IGNORECASE,
        ),
        # Node format: Node N: op_name avg_ms=1.234 percentage=12.3%
        re.compile(
            r"Node\s+\d+:\s*([\w][\w\s/]*?)\s+avg_ms=(\d+\.?\d*)\s+percentage=(\d+\.?\d*)%",
            re.IGNORECASE,
        ),
    ]

    for line in stdout.splitlines():
        matched = False
        for pattern in patterns:
            m = pattern.search(line)
            if m:
                try:
                    ops.append(OpStat(
                        name      = m.group(1).strip(),
                        avg_ms    = round(float(m.group(2)) / 1000.0, 4),
                        pct_total = float(m.group(3)),
                    ))
                    matched = True
                    break
                except ValueError:
                    failed_lines += 1
                    continue

    if failed_lines:
        _log(f"Op parse: {failed_lines} lines failed to parse", serial, "WARN")

    _log(
        f"Op profiling: {len(ops)} ops extracted",
        serial,
        context={"op_count": len(ops), "parse_failures": failed_lines},
    )
    return ops


def _parse_benchmark_output(
    stdout:  str,
    delegate: str,
    serial:  Optional[str] = None,
) -> dict:
    """
    Parse full benchmark stdout into a metrics dict.
    All fields nullable — TFLite output format varies between versions.
    Logs each field extracted for CI traceability.
    """
    avg_ms  = _us_to_ms(_parse_ms(r"avg\s*=\s*(\d+\.?\d*(?:e[+-]?\d+)?)", stdout))
    p50_ms  = _us_to_ms(_parse_ms(r"P50\s*=\s*(\d+\.?\d*(?:e[+-]?\d+)?)", stdout))
    p90_ms  = _us_to_ms(_parse_ms(r"P90\s*=\s*(\d+\.?\d*(?:e[+-]?\d+)?)", stdout))
    p99_ms  = _us_to_ms(_parse_ms(r"P99\s*=\s*(\d+\.?\d*(?:e[+-]?\d+)?)", stdout))
    init_ms = _us_to_ms(_parse_ms(r"Init:\s*(\d+\.?\d*(?:e[+-]?\d+)?)", stdout))

    peak_mem_bytes = _parse_ms(
        r"Peak memory footprint.*?:\s*(\d+\.?\d*(?:e[+-]?\d+)?)", stdout
    )
    peak_mem_mb = (
        round(peak_mem_bytes / (1024 ** 2), 2)
        if peak_mem_bytes is not None else None
    )

    variance: Optional[float] = None
    if p50_ms and p90_ms and p50_ms > 0:
        variance = round((p90_ms / p50_ms) - 1, 4)

    latency_trend = _parse_latency_trend(stdout, serial)
    ops           = _parse_ops(stdout, serial)

    _log(
        f"Parsed [{delegate}]: avg={avg_ms}ms p50={p50_ms}ms "
        f"p90={p90_ms}ms variance={variance} "
        f"ops={len(ops)} trend={len(latency_trend) if latency_trend else 0}",
        serial,
        context={
            "delegate": delegate,
            "avg_ms": avg_ms,
            "p50_ms": p50_ms,
            "p90_ms": p90_ms,
            "p99_ms": p99_ms,
            "variance": variance,
            "op_count": len(ops),
            "trend_count": len(latency_trend) if latency_trend else 0,
        },
    )

    return {
        "avg_ms":        avg_ms,
        "p50_ms":        p50_ms,
        "p90_ms":        p90_ms,
        "p99_ms":        p99_ms,
        "init_ms":       init_ms,
        "peak_mem_mb":   peak_mem_mb,
        "variance":      variance,
        "latency_trend": latency_trend,
        "ops":           ops,
    }


# ---------------------------------------------------------------------
# Public: Run Delegate Benchmark
# ---------------------------------------------------------------------

def run_benchmark(
    deployed:            DeployedRun,
    delegate:            str,
    *,
    num_runs:            int   = DEFAULT_NUM_RUNS,
    warmup_runs:         int   = DEFAULT_WARMUP_RUNS,
    num_threads:         int   = DEFAULT_NUM_THREADS,
    timeout_s:           int   = DEFAULT_TIMEOUT,
    cpu_avg_ms:          Optional[float] = None,
    device_fingerprint:  str   = "",
) -> BenchmarkResult:
    """
    Run a delegate benchmark and return a structured BenchmarkResult.

    Always called after run_cpu_baseline() — never standalone.
    delegate must be a validated DelegateStatus.SUPPORTED delegate.

    Args:
        deployed:           DeployedRun from deploy.py
        delegate:           Validated delegate name (GPU, NNAPI, etc.)
        num_runs:           Number of benchmark iterations
        warmup_runs:        Warmup iterations before measurement
        num_threads:        CPU thread count (ignored for GPU/NNAPI)
        timeout_s:          Hard timeout in seconds
        cpu_avg_ms:         CPU baseline avg for speedup computation
        device_fingerprint: DeviceProfile.fingerprint for run_hash

    Raises:
        ValueError:      Unknown delegate name
        ExecutionError:  Benchmark process exited non-zero
        ADBTimeoutError: Benchmark exceeded timeout_s
    """
    serial = deployed.serial or None

    run_hash = _compute_run_hash(
        deployed.remote_model_path, delegate, device_fingerprint
    )

    _log(
        f"Starting {delegate} benchmark",
        serial,
        context={
            "run_id":    deployed.run_id,
            "run_hash":  run_hash,
            "delegate":  delegate,
            "num_runs":  num_runs,
        },
    )

    full_cmd = _build_adb_command(
        deployed, delegate, num_runs, warmup_runs, num_threads
    )
    _log(f"Command: {shlex.join(full_cmd)}", serial)

    start = time.time()

    try:
        proc = subprocess.Popen(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr into stdout
            text=True,
            preexec_fn=os.setsid,       # process group for SIGKILL
        )
    except FileNotFoundError:
        from mlbuild.core.errors import ADBNotFoundError
        raise ADBNotFoundError()

    returncode, raw_stdout = _stream_process(proc, timeout_s, serial)
    duration_s = time.time() - start

    # Timeout
    if returncode == -1:
        raise ADBTimeoutError(
            command=shlex.join(full_cmd),
        )

    # Non-zero exit
    if returncode != 0:
        raise ExecutionError(
            raw_stdout=raw_stdout,
            run_id=deployed.run_id,
        )

    metrics = _parse_benchmark_output(raw_stdout, delegate, serial)

    # Compute speedup if CPU baseline provided
    speedup: Optional[float] = None
    if cpu_avg_ms and metrics["avg_ms"] and cpu_avg_ms > 0:
        speedup = round(cpu_avg_ms / metrics["avg_ms"], 2)
        _log(
            f"Speedup: {speedup}x ({cpu_avg_ms}ms CPU → {metrics['avg_ms']}ms {delegate})",
            serial,
            context={"speedup": speedup, "cpu_avg_ms": cpu_avg_ms},
        )

    result = BenchmarkResult(
        delegate      = delegate,
        avg_ms        = metrics["avg_ms"],
        p50_ms        = metrics["p50_ms"],
        p90_ms        = metrics["p90_ms"],
        p99_ms        = metrics["p99_ms"],
        init_ms       = metrics["init_ms"],
        peak_mem_mb   = metrics["peak_mem_mb"],
        variance      = metrics["variance"],
        latency_trend = metrics["latency_trend"],
        ops           = metrics["ops"],
        raw_stdout    = raw_stdout,
        run_id        = deployed.run_id,
        run_hash      = run_hash,
        num_runs      = num_runs,
        num_threads   = num_threads,
        duration_s    = duration_s,
        cpu_avg_ms    = cpu_avg_ms,
        speedup       = speedup,
    )

    _log(
        f"{delegate} benchmark complete in {duration_s:.1f}s: {result}",
        serial,
        context={
            "run_id":    deployed.run_id,
            "run_hash":  run_hash,
            "duration_s": duration_s,
            "speedup":   speedup,
        },
    )

    return result