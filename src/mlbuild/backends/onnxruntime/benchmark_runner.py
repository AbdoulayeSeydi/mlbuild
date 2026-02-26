"""
ONNX Runtime benchmark runner.
Measures CPU/GPU inference performance.
"""

import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class ONNXRuntimeBenchmarkResult:
    """ONNX Runtime benchmark results."""
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    latency_std_ms: float
    
    throughput_fps: float
    
    memory_peak_mb: float
    num_runs: int
    failures: int
    
    execution_provider: str
    device: str


class ONNXRuntimeBenchmarkRunner:
    """Benchmark ONNX Runtime models."""
    
    def __init__(
        self,
        model_path: Path,
        execution_provider: str = "CPUExecutionProvider",
    ):
        """
        Initialize ONNX Runtime benchmark runner.
        
        Args:
            model_path: Path to .onnx model
            execution_provider: "CPUExecutionProvider", "CUDAExecutionProvider", etc.
        """
        import onnxruntime as ort
        
        self.model_path = model_path
        self.execution_provider = execution_provider
        
        # Create inference session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options,
            providers=[execution_provider]
        )
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
    
    def run(
        self,
        warmup_runs: int = 20,
        benchmark_runs: int = 100,
    ) -> ONNXRuntimeBenchmarkResult:
        """
        Run benchmark.
        
        Returns:
            ONNXRuntimeBenchmarkResult
        """
        import psutil
        import platform
        
        # Create dummy input
        # Handle dynamic dimensions (replace None with 1)
        shape = [dim if dim is not None else 1 for dim in self.input_shape]
        dummy_input = np.random.randn(*shape).astype(np.float32)
        
        # Warmup
        for _ in range(warmup_runs):
            self.session.run([self.output_name], {self.input_name: dummy_input})
        
        # Benchmark
        latencies = []
        failures = 0
        memory_start = psutil.Process().memory_info().rss / (1024 * 1024)
        peak_memory = memory_start
        
        for _ in range(benchmark_runs):
            try:
                start = time.perf_counter()
                self.session.run([self.output_name], {self.input_name: dummy_input})
                end = time.perf_counter()
                
                latency_ms = (end - start) * 1000.0
                latencies.append(latency_ms)
                
                # Track memory
                current_mem = psutil.Process().memory_info().rss / (1024 * 1024)
                peak_memory = max(peak_memory, current_mem)
                
            except Exception:
                failures += 1
        
        # Compute statistics
        latencies_arr = np.array(latencies)
        
        return ONNXRuntimeBenchmarkResult(
            latency_p50_ms=float(np.percentile(latencies_arr, 50)),
            latency_p95_ms=float(np.percentile(latencies_arr, 95)),
            latency_p99_ms=float(np.percentile(latencies_arr, 99)),
            latency_mean_ms=float(np.mean(latencies_arr)),
            latency_std_ms=float(np.std(latencies_arr)),
            throughput_fps=1000.0 / float(np.mean(latencies_arr)),
            memory_peak_mb=peak_memory - memory_start,
            num_runs=len(latencies),
            failures=failures,
            execution_provider=self.execution_provider,
            device=platform.machine(),
        )