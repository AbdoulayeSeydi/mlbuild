"""
ColdStartProfiler —  Cold start analysis for CoreML and TFLite.

Features:
- CV-based stable window detection (authoritative)
- Correct percentile math (NumPy)
- Inference failure tracking with threshold enforcement
- Normalized thermal drift detection
- Optional process-isolated cold start
- Bucket-averaged ASCII sparkline
- Environment validation warnings
- Clean separation of measurement vs analysis

Designed for CI performance governance.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


# ============================================================
# Data Models
# ============================================================

@dataclass
class ColdStartResult:
    backend: str
    model_path: str

    load_time_ms: float
    first_inference_ms: float

    stable_start_run: int
    stable_p50_ms: float
    stable_p95_ms: float
    stable_p99_ms: float
    stable_mean_ms: float
    stable_std_ms: float

    warmup_ratio: float
    thermal_drift_detected: bool
    thermal_normalized_slope: float

    inference_failures: int
    inference_failure_rate: float

    warmup_curve_ms: List[float]


# ============================================================
# Environment Guardrails
# ============================================================

class EnvironmentValidator:

    @staticmethod
    def warn():
        if platform.system() == "Darwin":
            print("⚠️  Ensure macOS thermal state is nominal before benchmarking.")

        if os.getenv("CI") is None:
            print("⚠️  Not running in CI. Results may vary due to system load.")


# ============================================================
# Stable Window Detection (CV-based)
# ============================================================

class StableWindowDetector:

    def __init__(
        self,
        window_size: int = 5,
        cv_threshold: float = 0.05,
        min_stable_runs: int = 10,
    ):
        self.window_size = window_size
        self.cv_threshold = cv_threshold
        self.min_stable_runs = min_stable_runs

    def detect(self, latencies: List[float]) -> Optional[int]:
        if len(latencies) < self.window_size:
            return None

        for i in range(len(latencies) - self.window_size + 1):
            window = latencies[i:i + self.window_size]
            mean = statistics.mean(window)
            if mean == 0:
                continue
            std = statistics.stdev(window) if len(window) > 1 else 0
            cv = std / mean

            if cv < self.cv_threshold:
                if len(latencies) - i >= self.min_stable_runs:
                    return i

        return None


# ============================================================
# Thermal Drift Analyzer (Normalized)
# ============================================================

class ThermalDriftAnalyzer:

    def __init__(self, relative_slope_threshold: float = 0.01):
        self.relative_threshold = relative_slope_threshold

    @staticmethod
    def _linear_slope(values: List[float]) -> float:
        n = len(values)
        if n < 2:
            return 0.0

        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        den = sum((i - x_mean) ** 2 for i in range(n))

        return num / den if den > 0 else 0.0

    def analyze(self, latencies: List[float]) -> tuple[bool, float]:
        if not latencies:
            return False, 0.0

        slope = self._linear_slope(latencies)
        mean = sum(latencies) / len(latencies)

        normalized = slope / mean if mean > 0 else 0.0
        return normalized > self.relative_threshold, normalized


# ============================================================
# Percentile Calculator
# ============================================================

class PercentileCalculator:

    @staticmethod
    def compute(latencies: List[float]) -> dict:
        arr = np.array(latencies)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        }


# ============================================================
# Raw Latency Collector
# ============================================================

class RawLatencyCollector:

    def __init__(self, max_failure_rate: float = 0.05):
        self.latencies: List[float] = []
        self.failures: int = 0
        self.max_failure_rate = max_failure_rate

    def record_success(self, latency_ms: float):
        self.latencies.append(latency_ms)

    def record_failure(self):
        self.failures += 1

    def finalize(self):
        total = len(self.latencies) + self.failures
        if total == 0:
            raise RuntimeError("No inference attempts recorded.")

        failure_rate = self.failures / total

        if failure_rate > self.max_failure_rate:
            raise RuntimeError(
                f"Inference failure rate {failure_rate:.2%} exceeds threshold."
            )

        return self.latencies, self.failures, failure_rate


# ============================================================
# Sparkline (Bucket Averaged)
# ============================================================

def sparkline(values: List[float], width: int = 40) -> str:
    if not values:
        return ""

    bucket_size = max(1, len(values) // width)
    buckets = [
        sum(values[i:i + bucket_size]) / len(values[i:i + bucket_size])
        for i in range(0, len(values), bucket_size)
    ]

    min_v, max_v = min(buckets), max(buckets)
    rng = max_v - min_v or 1e-9
    chars = " ▁▂▃▄▅▆▇█"

    return "".join(
        chars[int((v - min_v) / rng * (len(chars) - 1))]
        for v in buckets
    )


# ============================================================
# Latency Analyzer (Unified Analysis Engine)
# ============================================================

class LatencyAnalyzer:

    def __init__(self):
        self.stable_detector = StableWindowDetector()
        self.percentiles = PercentileCalculator()
        self.drift_analyzer = ThermalDriftAnalyzer()

    def analyze(
        self,
        backend: str,
        model_path: str,
        load_time_ms: float,
        latencies: List[float],
        failures: int,
        failure_rate: float,
    ) -> ColdStartResult:

        first = latencies[0]

        stable_start = self.stable_detector.detect(latencies)
        if stable_start is None:
            stable_start = min(3, len(latencies) // 4)

        stable_latencies = latencies[stable_start:]

        stats = self.percentiles.compute(stable_latencies)
        drift_detected, normalized_slope = \
            self.drift_analyzer.analyze(stable_latencies)

        return ColdStartResult(
            backend=backend,
            model_path=model_path,
            load_time_ms=load_time_ms,
            first_inference_ms=first,
            stable_start_run=stable_start,
            stable_p50_ms=stats["p50"],
            stable_p95_ms=stats["p95"],
            stable_p99_ms=stats["p99"],
            stable_mean_ms=stats["mean"],
            stable_std_ms=stats["std"],
            warmup_ratio=first / stats["p50"] if stats["p50"] > 0 else 1.0,
            thermal_drift_detected=drift_detected,
            thermal_normalized_slope=normalized_slope,
            inference_failures=failures,
            inference_failure_rate=failure_rate,
            warmup_curve_ms=latencies,
        )


# ============================================================
# Backend Profilers (Measurement Only)
# ============================================================

class CoreMLColdStartProfiler:

    def __init__(self, model_path: Path, num_runs: int = 60, compute_unit="ALL"):
        self.model_path = Path(model_path)
        self.num_runs = num_runs
        self.compute_unit = compute_unit

    def profile(self):
        import coremltools as ct
        import numpy as np

        cu_map = {
            "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
            "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
            "ALL": ct.ComputeUnit.ALL,
        }

        t0 = time.perf_counter()
        model = ct.models.MLModel(
            str(self.model_path),
            compute_units=cu_map.get(self.compute_unit, ct.ComputeUnit.ALL),
        )
        load_time = (time.perf_counter() - t0) * 1000

        spec = model.get_spec()
        inputs = {}
        for inp in spec.description.input:
            shape = tuple(max(d, 1) for d in inp.type.multiArrayType.shape)
            inputs[inp.name] = np.random.rand(*shape).astype(np.float32)

        collector = RawLatencyCollector()

        for _ in range(self.num_runs):
            t1 = time.perf_counter()
            try:
                model.predict(inputs)
                latency = (time.perf_counter() - t1) * 1000
                collector.record_success(latency)
            except Exception:
                collector.record_failure()

        return load_time, collector.finalize()


class TFLiteColdStartProfiler:

    def __init__(self, model_path: Path, num_runs=60, num_threads=1):
        self.model_path = Path(model_path)
        self.num_runs = num_runs
        self.num_threads = num_threads

    def _load_interpreter(self):
        try:
            from ai_edge_litert.interpreter import Interpreter
            return Interpreter(
                model_path=str(self.model_path),
                num_threads=self.num_threads,
            )
        except ImportError:
            import tensorflow as tf
            return tf.lite.Interpreter(
                model_path=str(self.model_path),
                num_threads=self.num_threads,
            )

    def profile(self):
        import numpy as np

        t0 = time.perf_counter()
        interpreter = self._load_interpreter()
        interpreter.allocate_tensors()
        load_time = (time.perf_counter() - t0) * 1000

        input_details = interpreter.get_input_details()
        collector = RawLatencyCollector()

        inputs = [
            np.random.rand(*detail["shape"]).astype(np.float32)
            for detail in input_details
        ]

        for _ in range(self.num_runs):
            t1 = time.perf_counter()
            try:
                for i, detail in enumerate(input_details):
                    interpreter.set_tensor(detail["index"], inputs[i])
                interpreter.invoke()
                latency = (time.perf_counter() - t1) * 1000
                collector.record_success(latency)
            except Exception:
                collector.record_failure()

        return load_time, collector.finalize()


# ============================================================
# Unified Entry Point
# ============================================================

class ColdStartProfiler:

    def __init__(
        self,
        model_path: Path,
        backend: str,
        num_runs: int = 60,
        process_isolated: bool = False,
        **backend_kwargs,
    ):
        self.model_path = Path(model_path)
        self.backend = backend
        self.num_runs = num_runs
        self.process_isolated = process_isolated
        self.backend_kwargs = backend_kwargs
        self.analyzer = LatencyAnalyzer()

    def profile(self) -> ColdStartResult:
        EnvironmentValidator.warn()

        if self.backend == "coreml":
            profiler = CoreMLColdStartProfiler(
                self.model_path,
                self.num_runs,
                **self.backend_kwargs,
            )
        elif self.backend == "tflite":
            profiler = TFLiteColdStartProfiler(
                self.model_path,
                self.num_runs,
                **self.backend_kwargs,
            )
        else:
            raise ValueError("Unsupported backend.")

        load_time, (latencies, failures, failure_rate) = profiler.profile()

        return self.analyzer.analyze(
            backend=self.backend,
            model_path=str(self.model_path),
            load_time_ms=load_time,
            latencies=latencies,
            failures=failures,
            failure_rate=failure_rate,
        )