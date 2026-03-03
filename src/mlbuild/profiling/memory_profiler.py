"""
BasicMemoryProfiler — Week 1 implementation.

Captures memory behavior during model inference via background sampling thread.

What this measures (Week 1):
  - Baseline RSS before any inference
  - Peak RSS during inference (sampled at 2ms intervals)
  - Final RSS after inference
  - Peak delta (spike above baseline)
  - Python heap delta via tracemalloc

What this does NOT measure yet (Week 3):
  - True VM pressure (mach_task_basic_info)
  - GPU/ANE driver memory
  - Per-layer memory peaks

Why RSS sampling beats before/after:
  - Before/after only captures the delta at endpoints
  - Peak RSS during inference can be 2-5x the stable footprint
  - Especially true for CoreML where ANE driver allocates/frees mid-inference
  - 2ms polling interval catches most spikes for models > 5ms latency
    (for sub-ms models like ANE fp16, spikes are driver-internal and
     require mach_task_basic_info — that's Week 3)

Usage:
    with BasicMemoryProfiler() as mem:
        # ... run inferences ...
        pass
    print(mem.report())

Or manually:
    profiler = BasicMemoryProfiler(poll_interval_ms=2)
    profiler.start()
    run_inference()
    profiler.stop()
    report = profiler.report()
"""

from __future__ import annotations

import threading
import time
import tracemalloc
from dataclasses import dataclass
from typing import List, Optional
import warnings

# ============================================================
# Data Class
# ============================================================

@dataclass
class MemoryReport:
    """Memory usage report from a profiling session."""

    # RSS (Resident Set Size) — actual physical RAM used by process
    baseline_rss_mb: float
    peak_rss_mb: float
    final_rss_mb: float
    peak_delta_mb: float
    rss_samples: List[float]

    # Python heap (tracemalloc)
    heap_baseline_mb: float
    heap_peak_mb: float
    heap_delta_mb: float

    # Derived
    non_python_rss_delta_mb: float

    # Sampling metadata
    poll_interval_ms: float
    num_samples: int
    duration_ms: float

    # Flags
    tracemalloc_available: bool
    rss_available: bool
    sufficient_samples: bool


# ============================================================
# Profiler
# ============================================================

class BasicMemoryProfiler:
    """
    Enterprise-ready memory profiler for Python ML workloads.

    Features:
    - Monotonic interval RSS sampling (bounded drift)
    - True tracemalloc peak memory
    - Thread-safe sample collection
    - Minimum sample validation
    - Explicit handling when psutil is unavailable
    - Context manager support

    Limitations:
    - Sub-ms inference spikes (e.g., ANE fp16) may be missed
    - GPU/driver memory beyond Python process not captured
    """

    MIN_SAMPLES = 5  # for reliable peak detection

    def __init__(self, poll_interval_ms: float = 2.0):
        self.poll_interval_ms = poll_interval_ms
        self._poll_interval_s = poll_interval_ms / 1000.0

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._samples: List[float] = []
        self._samples_lock = threading.Lock()

        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._baseline_rss_mb: float = 0.0
        self._final_rss_mb: float = 0.0

        self._heap_baseline_mb: float = 0.0
        self._heap_peak_mb: float = 0.0
        self._tracemalloc_available: bool = False
        self._rss_available: bool = False

        self._process = None
        self._load_psutil()

    def _load_psutil(self):
        try:
            import psutil
            import os
            self._process = psutil.Process(os.getpid())
            self._rss_available = True
        except ImportError:
            self._process = None
            self._rss_available = False
            warnings.warn("psutil not available; RSS profiling disabled.", RuntimeWarning)

    def _get_rss_mb(self) -> float:
        if not self._rss_available:
            return 0.0
        try:
            return self._process.memory_info().rss / (1024 * 1024)
        except Exception as e:
            warnings.warn(f"RSS read failed: {e}", RuntimeWarning)
            return 0.0

    def _sample_loop(self):
        next_tick = time.perf_counter()
        try:
            while not self._stop_event.is_set():
                sample = self._get_rss_mb()
                with self._samples_lock:
                    self._samples.append(sample)
                next_tick += self._poll_interval_s
                sleep_time = next_tick - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except Exception as e:
            warnings.warn(f"Sampling thread exception: {e}", RuntimeWarning)

    def start(self):
        """Start memory profiling. Call before running inferences."""
        if self._tracemalloc_available or tracemalloc.is_tracing():
            tracemalloc.stop()

        self._samples = []
        self._stop_event.clear()

        # RSS baseline
        self._baseline_rss_mb = self._get_rss_mb()
        self._start_time = time.perf_counter()

        # Start tracemalloc
        try:
            tracemalloc.start()
            self._tracemalloc_available = True
            self._heap_baseline_mb, _ = tracemalloc.get_traced_memory()
            self._heap_baseline_mb /= 1024 * 1024
        except Exception:
            self._tracemalloc_available = False
            self._heap_baseline_mb = 0.0

        # Start background sampler
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop memory profiling. Call after inferences complete."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)

        self._end_time = time.perf_counter()
        self._final_rss_mb = self._get_rss_mb()

        # Capture tracemalloc peak
        if self._tracemalloc_available:
            try:
                _, peak = tracemalloc.get_traced_memory()
                self._heap_peak_mb = peak / (1024 * 1024)
                tracemalloc.stop()
            except Exception:
                self._heap_peak_mb = self._heap_baseline_mb

    def report(self) -> MemoryReport:
        """Build and return the memory report."""
        with self._samples_lock:
            samples = list(self._samples)

        if not samples:
            samples = [self._baseline_rss_mb]

        peak_rss_mb = max(samples)
        peak_delta_mb = peak_rss_mb - self._baseline_rss_mb
        heap_delta_mb = self._heap_peak_mb - self._heap_baseline_mb
        non_python_rss_delta_mb = max(0.0, peak_delta_mb - heap_delta_mb)

        sufficient_samples = len(samples) >= self.MIN_SAMPLES
        if not sufficient_samples:
            warnings.warn(
                f"Profiling collected only {len(samples)} samples; results may be unreliable",
                RuntimeWarning,
            )

        duration_ms = (self._end_time - self._start_time) * 1000.0

        return MemoryReport(
            baseline_rss_mb=self._baseline_rss_mb,
            peak_rss_mb=peak_rss_mb,
            final_rss_mb=self._final_rss_mb,
            peak_delta_mb=peak_delta_mb,
            rss_samples=samples,
            heap_baseline_mb=self._heap_baseline_mb,
            heap_peak_mb=self._heap_peak_mb,
            heap_delta_mb=heap_delta_mb,
            non_python_rss_delta_mb=non_python_rss_delta_mb,
            poll_interval_ms=self.poll_interval_ms,
            num_samples=len(samples),
            duration_ms=duration_ms,
            tracemalloc_available=self._tracemalloc_available,
            rss_available=self._rss_available,
            sufficient_samples=sufficient_samples,
        )

    # ── Context manager support ──────────────────────────────
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


# ============================================================
# Convenience: profile a callable
# ============================================================

def profile_memory(fn, poll_interval_ms: float = 2.0) -> tuple:
    """
    Profile memory usage of a callable.

    Returns (result, MemoryReport).
    """
    profiler = BasicMemoryProfiler(poll_interval_ms=poll_interval_ms)
    profiler.start()
    try:
        result = fn()
    finally:
        profiler.stop()
    return result, profiler.report()