"""
Enterprise-grade Warmup & Stability Analyzer for CoreML models.

Features:
- Cold-start isolation (model load vs first inference vs warm state)
- Rolling-window convergence detection
- Sustained stability threshold (N consecutive windows)
- Percentiles (p50/p95/p99), std, CV
- Thermal throttling slope detection (linear regression)
- ANE vs CPU comparison harness
- Deterministic input seeding
- Multi-input + dtype-aware generation
- Context metadata capture
- JSON export
- Full raw latency trace retained
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

try:
    import coremltools as ct
except ImportError:
    ct = None


# -------------------------------------------------------------------------
# Data Models
# -------------------------------------------------------------------------

@dataclass
class StabilityWindow:
    index: int
    mean_ms: float
    std_ms: float


@dataclass
class WarmupMetrics:
    runs: int
    load_time_ms: float
    first_inference_ms: float
    stable_mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    std_ms: float
    coefficient_of_variation: float
    warmup_ratio: float
    time_to_stable_run: Optional[int]
    throttling_slope_ms_per_run: float
    throttling_detected: bool
    compute_unit: str
    model_type: str
    metadata: Dict[str, Any]
    latencies_ms: List[float]


# -------------------------------------------------------------------------
# Analyzer
# -------------------------------------------------------------------------

class EnterpriseWarmupAnalyzer:
    """
    Production-grade CoreML warmup and stability analyzer.
    """

    def __init__(
        self,
        model_path: Path,
        compute_unit: str = "all",
        seed: int = 42,
    ):
        if ct is None:
            raise RuntimeError("coremltools required")

        self.model_path = Path(model_path)
        self.seed = seed
        np.random.seed(seed)

        cu_map = {
            "all": ct.ComputeUnit.ALL,
            "cpu": ct.ComputeUnit.CPU_ONLY,
            "gpu": ct.ComputeUnit.CPU_AND_GPU,
            "ane": ct.ComputeUnit.CPU_AND_NE,
        }

        if compute_unit not in cu_map:
            raise ValueError(f"Unsupported compute unit: {compute_unit}")

        self.compute_unit_str = compute_unit
        self.compute_unit = cu_map[compute_unit]

        # Cold start timing (model load / compile)
        load_start = time.perf_counter()
        self.model = ct.models.MLModel(
            str(self.model_path),
            compute_units=self.compute_unit,
        )
        load_end = time.perf_counter()
        self.load_time_ms = (load_end - load_start) * 1000

        self.spec = self.model.get_spec()
        self.model_type = self.spec.WhichOneof("Type")

        self.input_specs = self.spec.description.input

    # ---------------------------------------------------------------------

    def _generate_input(self) -> Dict[str, np.ndarray]:
        inputs = {}

        for inp in self.input_specs:
            name = inp.name
            t = inp.type

            if t.HasField("multiArrayType"):
                shape = tuple(t.multiArrayType.shape)
                dtype = np.float32
                if t.multiArrayType.dataType == 65568:  # INT32
                    dtype = np.int32
                inputs[name] = np.random.rand(*shape).astype(dtype)

            elif t.HasField("imageType"):
                h = t.imageType.height
                w = t.imageType.width
                c = 3
                inputs[name] = (
                    np.random.rand(h, w, c) * 255
                ).astype(np.uint8)

            else:
                raise ValueError(f"Unsupported input type: {name}")

        return inputs

    # ---------------------------------------------------------------------

    def analyze(
        self,
        num_runs: int = 100,
        window_size: int = 8,
        stability_std_threshold: float = 0.5,
        consecutive_windows_required: int = 3,
        throttling_tail_ratio: float = 0.3,
    ) -> WarmupMetrics:

        if num_runs < window_size * 2:
            raise ValueError("num_runs too small for stability detection")

        test_input = self._generate_input()

        latencies: List[float] = []

        # First inference isolation
        start = time.perf_counter()
        self.model.predict(test_input)
        end = time.perf_counter()
        first_inference_ms = (end - start) * 1000
        latencies.append(first_inference_ms)

        # Remaining runs
        for _ in range(num_runs - 1):
            start = time.perf_counter()
            self.model.predict(test_input)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        # -------------------------------------------------------------
        # Rolling Stability Detection
        # -------------------------------------------------------------
        stable_run_index: Optional[int] = None
        stable_mean_ms: float = float("nan")
        stable_windows = 0

        for i in range(window_size, len(latencies)):
            window = latencies[i - window_size : i]
            mean = float(np.mean(window))
            std = float(np.std(window))

            if std <= stability_std_threshold:
                stable_windows += 1
                if stable_windows >= consecutive_windows_required:
                    stable_run_index = i
                    stable_mean_ms = mean
                    break
            else:
                stable_windows = 0

        if stable_run_index is None:
            stable_mean_ms = float(np.mean(latencies[-window_size:]))

        # -------------------------------------------------------------
        # Percentiles + Variance
        # -------------------------------------------------------------
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))
        std_all = float(np.std(latencies))
        cv = std_all / p50 if p50 > 0 else 0.0

        # -------------------------------------------------------------
        # Warmup Ratio
        # -------------------------------------------------------------
        warmup_ratio = (
            first_inference_ms / stable_mean_ms
            if stable_mean_ms > 0
            else 1.0
        )

        # -------------------------------------------------------------
        # Thermal Throttling Detection
        # Linear regression over tail
        # -------------------------------------------------------------
        tail_start = int(len(latencies) * (1 - throttling_tail_ratio))
        tail = latencies[tail_start:]

        x = np.arange(len(tail))
        slope = float(np.polyfit(x, tail, 1)[0])

        throttling_detected = slope > 0.02  # configurable threshold

        # -------------------------------------------------------------
        # Metadata
        # -------------------------------------------------------------
        metadata = {
            "device": platform.machine(),
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "coremltools_version": getattr(ct, "__version__", "unknown"),
            "seed": self.seed,
            "model_path": str(self.model_path),
        }

        return WarmupMetrics(
            runs=num_runs,
            load_time_ms=self.load_time_ms,
            first_inference_ms=first_inference_ms,
            stable_mean_ms=stable_mean_ms,
            p50_ms=p50,
            p95_ms=p95,
            p99_ms=p99,
            std_ms=std_all,
            coefficient_of_variation=cv,
            warmup_ratio=warmup_ratio,
            time_to_stable_run=stable_run_index,
            throttling_slope_ms_per_run=slope,
            throttling_detected=throttling_detected,
            compute_unit=self.compute_unit_str,
            model_type=self.model_type,
            metadata=metadata,
            latencies_ms=latencies,
        )

    # ---------------------------------------------------------------------

    def compare_ane_vs_cpu(
        self,
        num_runs: int = 100,
    ) -> Dict[str, WarmupMetrics]:

        results = {}

        for cu in ["cpu", "ane"]:
            analyzer = EnterpriseWarmupAnalyzer(
                self.model_path,
                compute_unit=cu,
                seed=self.seed,
            )
            results[cu] = analyzer.analyze(num_runs=num_runs)

        return results

    # ---------------------------------------------------------------------

    @staticmethod
    def export_json(metrics: WarmupMetrics, output_path: Path) -> None:
        with open(output_path, "w") as f:
            json.dump(asdict(metrics), f, indent=2)
