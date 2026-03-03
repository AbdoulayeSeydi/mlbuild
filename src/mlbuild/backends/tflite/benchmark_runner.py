"""
Enterprise-Grade TFLite Benchmark Runner

Features:
- Multi-input model support
- Dynamic shape handling
- Deterministic benchmarking (seeded)
- User-provided input generator
- Batch size control
- Thread control
- Delegate detection and reporting
- Memory usage measurement (RSS)
- Throughput metrics
- Outlier trimming
- Failure isolation
- Timeout protection
"""

from __future__ import annotations

import gc
import logging
import os
import platform
import statistics
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Any

import numpy as np
import psutil


class TFLiteBenchmarkError(Exception):
    pass


class TFLiteBenchmarkRunner:
    def __init__(
        self,
        num_threads: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or logging.getLogger("mlbuild.tflite")

        # Strong TF validation
        try:
            import tensorflow as tf
        except ImportError as e:
            raise ImportError(
                "TensorFlow is required for TFLite benchmarking."
            ) from e

        if not hasattr(tf, "lite") or not hasattr(tf.lite, "Interpreter"):
            raise RuntimeError("TensorFlow installation does not include TFLite.")

        self.tf = tf
        self.num_threads = num_threads

    # ============================================================
    # PUBLIC API
    # ============================================================

    def benchmark(
        self,
        model_path: Path,
        runs: int = 50,
        warmup: int = 10,
        batch_size: int = 1,
        use_gpu: bool = False,
        seed: Optional[int] = 42,
        timeout_sec: Optional[float] = None,
        input_generator: Optional[
            Callable[[List[Dict[str, Any]]], Iterator[List[np.ndarray]]]
        ] = None,
        trim_outliers: bool = True,
    ) -> Dict[str, Any]:
        """
        Run benchmark.

        Args:
            model_path: Path to .tflite file
            runs: Number of measured runs
            warmup: Warmup runs
            batch_size: Batch size override
            use_gpu: Attempt GPU delegate
            seed: RNG seed for deterministic mode
            timeout_sec: Per-invoke timeout
            input_generator: Optional user-supplied generator
            trim_outliers: Remove top/bottom 5% latencies

        Returns:
            Dictionary of metrics
        """

        if not model_path.exists():
            raise FileNotFoundError(model_path)

        if seed is not None:
            np.random.seed(seed)

        interpreter, delegate_info = self._create_interpreter(
            model_path, use_gpu
        )

        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Handle dynamic shapes + batch override
        self._resize_inputs(interpreter, input_details, batch_size)

        # Create input iterator
        if input_generator:
            generator = input_generator(input_details)
        else:
            generator = self._default_input_generator(
                input_details, batch_size
            )

        # Warmup
        self.logger.info(f"Warmup: {warmup} runs")
        for _ in range(warmup):
            self._invoke_once(interpreter, input_details, next(generator))

        gc.collect()

        # Benchmark
        self.logger.info(f"Benchmark: {runs} runs")
        latencies = []
        failures = 0

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss

        total_start = time.perf_counter()

        for _ in range(runs):
            try:
                latency = self._invoke_timed(
                    interpreter,
                    input_details,
                    next(generator),
                    timeout_sec,
                )
                latencies.append(latency)
            except Exception as e:
                failures += 1
                self.logger.warning(f"Inference failure: {e}")

        total_end = time.perf_counter()

        mem_after = process.memory_info().rss

        if not latencies:
            raise TFLiteBenchmarkError("All benchmark runs failed.")

        latencies_arr = np.array(latencies)

        if trim_outliers and len(latencies_arr) > 10:
            low = np.percentile(latencies_arr, 5)
            high = np.percentile(latencies_arr, 95)
            latencies_arr = latencies_arr[
                (latencies_arr >= low) & (latencies_arr <= high)
            ]

        total_time = total_end - total_start
        throughput = len(latencies) / total_time

        results = {
            "p50_ms": float(np.percentile(latencies_arr, 50)),
            "p95_ms": float(np.percentile(latencies_arr, 95)),
            "p99_ms": float(np.percentile(latencies_arr, 99)),
            "mean_ms": float(np.mean(latencies_arr)),
            "std_ms": float(np.std(latencies_arr)),
            "min_ms": float(np.min(latencies_arr)),
            "max_ms": float(np.max(latencies_arr)),
            "throughput_inf_per_sec": float(throughput),
            "memory_rss_mb": max(0.0, (mem_after - mem_before) / (1024 * 1024)),
            "runs_requested": runs,
            "runs_completed": len(latencies),
            "failures": failures,
            "batch_size": batch_size,
            "num_threads": self.num_threads,
            "delegate": delegate_info,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        }

        self.logger.info(
            f"✓ p50={results['p50_ms']:.2f}ms | "
            f"throughput={throughput:.2f}/s | "
            f"failures={failures}"
        )

        return results

    # ============================================================
    # INTERNALS
    # ============================================================

    def _create_interpreter(
        self,
        model_path: Path,
        use_gpu: bool,
    ) -> Tuple[Any, Dict[str, Any]]:
        delegate_info = {"requested_gpu": use_gpu, "active": "cpu"}

        delegates = []

        if use_gpu:
            try:
                delegate = self.tf.lite.experimental.load_delegate(
                    self._detect_gpu_delegate_path()
                )
                delegates.append(delegate)
                delegate_info["active"] = "gpu"
            except Exception as e:
                self.logger.warning(f"GPU delegate unavailable: {e}")
                delegate_info["active"] = "cpu_fallback"

        interpreter = self.tf.lite.Interpreter(
            model_path=str(model_path),
            experimental_delegates=delegates or None,
            num_threads=self.num_threads,
        )

        return interpreter, delegate_info

    def _detect_gpu_delegate_path(self) -> str:
        system = platform.system()
        if system == "Linux":
            return "libtensorflowlite_gpu_delegate.so"
        elif system == "Darwin":
            return "libtensorflowlite_gpu_delegate.dylib"
        elif system == "Windows":
            return "tensorflowlite_gpu_delegate.dll"
        else:
            raise RuntimeError("Unsupported platform for GPU delegate.")

    def _resize_inputs(self, interpreter, input_details, batch_size: int):
        for detail in input_details:
            shape = detail["shape"]
            if shape[0] == -1 or shape[0] == 0:
                new_shape = [batch_size] + list(shape[1:])
                interpreter.resize_tensor_input(
                    detail["index"], new_shape
                )
        interpreter.allocate_tensors()

    def _default_input_generator(
        self,
        input_details: List[Dict[str, Any]],
        batch_size: int,
    ) -> Iterator[List[np.ndarray]]:
        while True:
            inputs = []
            for detail in input_details:
                shape = detail["shape"]
                dtype = detail["dtype"]
                if shape[0] in (-1, 0):
                    shape = [batch_size] + list(shape[1:])
                data = np.random.normal(size=shape).astype(dtype)
                inputs.append(data)
            yield inputs

    def _invoke_once(self, interpreter, input_details, inputs):
        for detail, data in zip(input_details, inputs):
            interpreter.set_tensor(detail["index"], data)
        interpreter.invoke()

    def _invoke_timed(
        self,
        interpreter,
        input_details,
        inputs,
        timeout_sec: Optional[float],
    ) -> float:
        result = {}

        def target():
            start = time.perf_counter()
            self._invoke_once(interpreter, input_details, inputs)
            end = time.perf_counter()
            result["latency"] = (end - start) * 1000

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=timeout_sec)

        if thread.is_alive():
            raise TimeoutError("Inference timeout")

        return result["latency"]