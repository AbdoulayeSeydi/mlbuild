"""
On-device benchmark runner for CoreML models.

Features:
- Latency profiling (p50/p95/p99)
- Memory tracking (peak usage)
- Warmup runs
- Statistical significance
- Device detection
- Hardware fingerprinting (audit-grade)
- A/B significance testing
- Regression detection for CI
"""

from __future__ import annotations

import os
import time
import json
import math
import queue
import threading
import statistics
import subprocess
import platform
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

from abc import ABC, abstractmethod
import numpy as np

try:
    import coremltools as ct
except ImportError:
    ct = None

try:
    import psutil
except ImportError:
    psutil = None

# ==============================
# Compute Unit (Structured Enum)
# ==============================

class ComputeUnit(Enum):
    CPU_ONLY = "CPU_ONLY"
    CPU_AND_GPU = "CPU_AND_GPU"
    ALL = "ALL"

    def to_coreml(self):
        mapping = {
            ComputeUnit.CPU_ONLY: ct.ComputeUnit.CPU_ONLY,
            ComputeUnit.CPU_AND_GPU: ct.ComputeUnit.CPU_AND_GPU,
            ComputeUnit.ALL: ct.ComputeUnit.ALL,
        }
        return mapping[self]


# ==============================
# Deterministic Environment
# ==============================

def configure_determinism(seed: int, ci_mode: bool):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Thread control
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

    if ci_mode:
        os.environ["COREML_ENABLE_PROFILING"] = "0"


# ==============================
# Robust Apple Chip Detection
# ==============================

def detect_apple_chip() -> str:
    if platform.system() != "Darwin":
        raise RuntimeError("Non-macOS system unsupported for CoreML benchmarking.")

    machine = platform.machine()

    try:
        model = subprocess.check_output(
            ["sysctl", "-n", "hw.model"],
            text=True,
        ).strip()
    except Exception:
        model = "unknown"

    try:
        brand = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
        ).strip().lower()
    except Exception:
        brand = "unknown"

    chip = "unknown"

    for token in ["m3", "m2", "m1", "a17", "a16"]:
        if token in brand:
            chip = f"apple_{token}"
            break

    if chip == "unknown":
        raise RuntimeError(
            f"Unable to reliably detect Apple chip. machine={machine}, model={model}, brand={brand}"
        )

    return chip


# ==============================
# Hardware Capability Fingerprint
# ==============================

@dataclass
class HardwareFingerprint:
    chip: str
    machine: str
    macos_version: str
    cpu_count_logical: int
    cpu_count_physical: int
    memory_gb: float

def hardware_fingerprint() -> HardwareFingerprint:
    if psutil is None:
        raise RuntimeError("psutil not installed - required for hardware fingerprinting")

    chip = detect_apple_chip()
    machine = platform.machine()
    macos_version = platform.mac_ver()[0]

    return HardwareFingerprint(
        chip=chip,
        machine=machine,
        macos_version=macos_version,
        cpu_count_logical=psutil.cpu_count(logical=True),
        cpu_count_physical=psutil.cpu_count(logical=False),
        memory_gb=round(psutil.virtual_memory().total / (1024**3), 2),
    )


# ==============================
# Memory Sampler Thread
# ==============================

class MemorySampler(threading.Thread):
    def __init__(self, interval_sec: float = 0.01):
        super().__init__()
        self.interval_sec = interval_sec
        self._stop_event = threading.Event()
        self.samples: List[float] = []

    def run(self):
        if psutil is None:
            return
        proc = psutil.Process()
        while not self._stop_event.is_set():
            rss = proc.memory_info().rss / (1024 * 1024)
            self.samples.append(rss)
            time.sleep(self.interval_sec)

    def stop(self):
        self._stop_event.set()

    def peak(self) -> float:
        return max(self.samples) if self.samples else 0.0


# ==============================
# Statistics
# ==============================

def mad_based_filter(data: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    if mad == 0:
        return data
    modified_z = 0.6745 * (data - median) / mad
    return data[np.abs(modified_z) <= threshold]


def bootstrap_ci(data: np.ndarray, percentile: float, n_boot: int = 1000):
    rng = np.random.default_rng(42)
    estimates = []
    for _ in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        estimates.append(np.percentile(sample, percentile))
    return (
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
    )


def autocorrelation_lag1(data: np.ndarray) -> float:
    if len(data) < 2:
        return 0.0
    return float(np.corrcoef(data[:-1], data[1:])[0, 1])


# ==============================
# A/B Significance Testing
# ==============================

def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Effect size: probability difference that one sample > another."""
    gt = 0
    lt = 0
    for x in a:
        gt += np.sum(x > b)
        lt += np.sum(x < b)
    n = len(a) * len(b)
    return (gt - lt) / n


def mann_whitney_u(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import mannwhitneyu
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    return p


@dataclass
class ABTestResult:
    p_value: float
    cliffs_delta: float
    mean_diff_ms: float
    significant: bool


def compare_benchmarks(
    baseline_latencies: np.ndarray,
    candidate_latencies: np.ndarray,
    alpha: float = 0.05,
) -> ABTestResult:

    p_value = mann_whitney_u(baseline_latencies, candidate_latencies)
    effect = cliffs_delta(candidate_latencies, baseline_latencies)
    mean_diff = float(np.mean(candidate_latencies) - np.mean(baseline_latencies))

    return ABTestResult(
        p_value=p_value,
        cliffs_delta=effect,
        mean_diff_ms=mean_diff,
        significant=p_value < alpha,
    )


# ==============================
# Benchmark Result
# ==============================

@dataclass
class BenchmarkResult:
    build_id: str
    chip: str
    compute_unit: str
    num_runs: int
    failures: int

    latency_p50: float
    latency_p95: float
    latency_p99: float
    latency_mean: float
    latency_std: float

    p50_ci_low: float
    p50_ci_high: float

    autocorr_lag1: float
    memory_peak_mb: float

    thermal_drift_ratio: float
    hardware: Dict[str, Any]


# ==============================
# CI Regression Detection
# ==============================

@dataclass
class RegressionResult:
    regression_detected: bool
    percent_change: float
    threshold_percent: float
    p_value: float


def detect_regression(
    baseline: BenchmarkResult,
    candidate: BenchmarkResult,
    baseline_latencies: np.ndarray,
    candidate_latencies: np.ndarray,
    threshold_percent: float = 5.0,
) -> RegressionResult:

    ab = compare_benchmarks(baseline_latencies, candidate_latencies)

    percent_change = (
        (candidate.latency_p50 - baseline.latency_p50)
        / baseline.latency_p50
        * 100
    )

    regression = (
        percent_change > threshold_percent
        and ab.significant
    )

    return RegressionResult(
        regression_detected=regression,
        percent_change=percent_change,
        threshold_percent=threshold_percent,
        p_value=ab.p_value,
    )


# ==============================
# Benchmark Runner Abstraction
# ==============================

class BenchmarkRunner(ABC):
    """Abstract base class for all benchmark runners."""

    def __init__(
        self,
        model_path: Path,
        compute_unit: ComputeUnit = ComputeUnit.ALL,
        warmup_runs: int = 20,
        benchmark_runs: int = 100,
        seed: int = 42,
        ci_mode: bool = False,
    ):
        # Note: no coremltools guard here — subclasses guard their own deps
        self.model_path = Path(model_path)
        self.compute_unit = compute_unit
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs

    @abstractmethod
    def run(self, build_id: str) -> BenchmarkResult: ...

    @abstractmethod
    def get_input_spec(self) -> list: ...

    @abstractmethod
    def predict(self, inputs: dict) -> dict: ...

# ==============================
# Benchmark Runner
# ==============================

class CoreMLBenchmarkRunner(BenchmarkRunner):

    def __init__(
        self,
        model_path: Path,
        compute_unit: ComputeUnit = ComputeUnit.ALL,
        warmup_runs: int = 20,
        benchmark_runs: int = 100,
        seed: int = 42,
        ci_mode: bool = False,
    ):
        if ct is None:
            raise RuntimeError("coremltools not installed")

        configure_determinism(seed, ci_mode)

        self.model_path = Path(model_path)
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.compute_unit = compute_unit

        self.model = ct.models.MLModel(
            str(self.model_path),
            compute_units=self.compute_unit.to_coreml(),
        )

        spec = self.model.get_spec()
        self.inputs = {
            i.name: tuple(i.type.multiArrayType.shape)
            for i in spec.description.input
        }
        # Store full input spec with dtype for accuracy checker
        self._input_spec = self._build_input_spec(spec)

    # CoreML protobuf dataType → numpy dtype
    _COREML_DTYPE_MAP = {
        65552: np.float32,   # FLOAT32
        65553: np.float16,   # FLOAT16
        131104: np.int32,    # INT32
        131072: np.int32,    # INT32 (alternate)
        65536: np.float64,   # DOUBLE
    }

    def _build_input_spec(self, spec) -> list:
        from ..core.accuracy.inputs import InputSpec
        result = []
        for inp in spec.description.input:
            ma = inp.type.multiArrayType
            shape = tuple(int(d) for d in ma.shape)
            # Strip leading batch dim of 1 — generate_batch prepends its own
            if len(shape) > 1 and shape[0] == 1:
                shape = shape[1:]
            dtype = self._COREML_DTYPE_MAP.get(ma.dataType, np.float32)
            result.append(InputSpec(name=inp.name, shape=shape, dtype=dtype))
        return result

    def get_input_spec(self) -> list:
        """Return input specs for accuracy checker."""
        return self._input_spec

    def predict(self, inputs: dict) -> dict:
        """
        Run single inference and return output tensors as numpy arrays.
        Used by accuracy checker — separate from benchmarking loop.
        Re-adds batch dim if the model spec requires rank 4 but input is rank 3.
        """
        batched = {}
        for name, arr in inputs.items():
            expected_rank = len(self.inputs.get(name, arr.shape)) + 1  # +1 for batch
            if arr.ndim < expected_rank:
                arr = arr[np.newaxis, ...]
            batched[name] = arr
        raw = self.model.predict(batched)
        return {
            k: np.array(v, dtype=np.float32) if not isinstance(v, np.ndarray) else v
            for k, v in raw.items()
        }

    def _generate_inputs(self) -> Dict[str, np.ndarray]:
        inputs = {}
        for name, shape in self.inputs.items():
            inputs[name] = np.random.rand(*shape).astype(np.float32)
        return inputs

    def run(self, build_id: str, return_raw: bool = False):

        chip = detect_apple_chip()
        fingerprint = hardware_fingerprint()
        inputs = self._generate_inputs()

        # Warmup
        for _ in range(self.warmup_runs):
            try:
                self.model.predict(inputs)
            except Exception:
                pass

        latencies = []
        failures = 0

        memory_sampler = MemorySampler()
        memory_sampler.start()

        start_all = time.perf_counter()

        for _ in range(self.benchmark_runs):
            t0 = time.perf_counter()
            try:
                self.model.predict(inputs)
            except Exception:
                failures += 1
                continue
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

        end_all = time.perf_counter()

        memory_sampler.stop()
        memory_sampler.join()

        latencies = np.array(latencies)

        # Outlier removal
        latencies = mad_based_filter(latencies)

        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))
        mean = float(np.mean(latencies))
        std = float(np.std(latencies))

        ci_low, ci_high = bootstrap_ci(latencies, 50)

        autocorr = autocorrelation_lag1(latencies)

        # Thermal drift heuristic
        first_half = np.mean(latencies[: len(latencies)//2])
        second_half = np.mean(latencies[len(latencies)//2 :])
        drift_ratio = second_half / first_half if first_half else 1.0

        result = BenchmarkResult(
            build_id=build_id,
            chip=chip,
            compute_unit=self.compute_unit.value,
            num_runs=len(latencies),
            failures=failures,
            latency_p50=p50,
            latency_p95=p95,
            latency_p99=p99,
            latency_mean=mean,
            latency_std=std,
            p50_ci_low=ci_low,
            p50_ci_high=ci_high,
            autocorr_lag1=autocorr,
            memory_peak_mb=memory_sampler.peak(),
            thermal_drift_ratio=drift_ratio,
            hardware=asdict(fingerprint),
        )

        if return_raw:
            return result, latencies
        return result

    def export_json(self, result: BenchmarkResult, output_path: Path):
        with open(output_path, "w") as f:
            json.dump(asdict(result), f, indent=2)


# ==============================
# TFLite Benchmark Runner
# ==============================

class TFLiteBenchmarkRunner(BenchmarkRunner):
    """
    Minimal TFLite benchmark runner.

    Uses tflite_runtime if available, falls back to
    tensorflow.lite.Interpreter for dev environments.
    """

    def __init__(
        self,
        model_path: Path,
        warmup_runs: int = 20,
        benchmark_runs: int = 100,
    ):
        super().__init__(model_path, warmup_runs, benchmark_runs)
        self._interpreter = None

    def _get_interpreter(self):
        if self._interpreter is not None:
            return self._interpreter

        try:
            import tflite_runtime.interpreter as tflite
            self._interpreter = tflite.Interpreter(model_path=str(self.model_path))
        except ImportError:
            try:
                import tensorflow as tf
                self._interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
            except ImportError:
                raise RuntimeError(
                    "TFLite runtime not found. Install tflite-runtime or tensorflow."
                )

        self._interpreter.allocate_tensors()
        return self._interpreter

    def _generate_inputs(self, interpreter) -> list:
        input_details = interpreter.get_input_details()
        inputs = []
        for detail in input_details:
            shape = detail["shape"]
            dtype = detail["dtype"]
            data = np.ones(shape, dtype=dtype)
            inputs.append((detail["index"], data))
        return inputs

    def get_input_spec(self) -> list:
        """Return input specs for accuracy checker."""
        from ..core.accuracy.inputs import InputSpec
        interpreter = self._get_interpreter()
        result = []
        for detail in interpreter.get_input_details():
            shape = tuple(int(d) for d in detail["shape"])
            # TFLite gives us numpy dtype directly
            dtype = np.dtype(detail["dtype"])
            # Strip batch dim if shape starts with 1 and has >1 dims
            input_shape = shape[1:] if len(shape) > 1 and shape[0] == 1 else shape
            result.append(InputSpec(
                name=detail["name"],
                shape=input_shape,
                dtype=dtype,
            ))
        return result

    def predict(self, inputs: dict) -> dict:
        """
        Run single inference and return output tensors as numpy arrays.
        Used by accuracy checker — separate from benchmarking loop.
        """
        interpreter = self._get_interpreter()
        for detail in interpreter.get_input_details():
            name = detail["name"]
            if name in inputs:
                data = inputs[name]
            else:
                # fallback: match by index order
                data = list(inputs.values())[detail["index"]]
            # Add batch dim if needed
            if data.ndim == len(detail["shape"]) - 1:
                data = data[np.newaxis, ...]
            interpreter.set_tensor(detail["index"], data.astype(detail["dtype"]))

        interpreter.invoke()

        outputs = {}
        for detail in interpreter.get_output_details():
            outputs[detail["name"]] = interpreter.get_tensor(detail["index"]).copy()
        return outputs

    def run(self, build_id: str, return_raw: bool = False) -> BenchmarkResult:
        import time as _time

        interpreter = self._get_interpreter()
        inputs = self._generate_inputs(interpreter)

        def _invoke():
            for idx, data in inputs:
                interpreter.set_tensor(idx, data)
            interpreter.invoke()

        # Warmup
        for _ in range(self.warmup_runs):
            try:
                _invoke()
            except Exception:
                pass

        latencies = []
        failures = 0

        for _ in range(self.benchmark_runs):
            t0 = _time.perf_counter()
            try:
                _invoke()
            except Exception:
                failures += 1
                continue
            latencies.append((_time.perf_counter() - t0) * 1000)

        latencies = np.array(latencies)
        latencies = mad_based_filter(latencies)

        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))

        result = BenchmarkResult(
            build_id=build_id,
            chip="cpu",
            compute_unit="CPU_ONLY",
            num_runs=len(latencies),
            failures=failures,
            latency_p50=p50,
            latency_p95=p95,
            latency_p99=p99,
            latency_mean=float(np.mean(latencies)),
            latency_std=float(np.std(latencies)),
            p50_ci_low=p50,
            p50_ci_high=p50,
            autocorr_lag1=autocorrelation_lag1(latencies),
            memory_peak_mb=0.0,
            thermal_drift_ratio=1.0,
            hardware={},
        )

        if return_raw:
            return result, latencies
        return result


# ==============================
# Runner Registry
# ==============================

BENCHMARK_RUNNERS: dict[str, type[BenchmarkRunner]] = {
    "coreml": CoreMLBenchmarkRunner,
    "tflite": TFLiteBenchmarkRunner,
}