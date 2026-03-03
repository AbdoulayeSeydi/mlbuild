"""
CoreMLDeepProfiler — Deep profiling for CoreML NeuralNetwork models.

Five features (NeuralNetwork only unless noted):

  1. Per-layer timing      Incremental p50 via CumulativeLayerProfiler slicing
  2. Memory flow           Estimated activation memory from weight dimensions
  3. Bottleneck            COMPUTE vs MEMORY per layer (FLOPs / bytes)
  4. Cold start            Full decomposition via ColdStartProfiler (both formats)
  5. Fusion detection      Conv+Activation, BN+Scale, etc. from spec layer types

MLProgram: cold start only — ANE-compiled, cannot be sliced without Xcode.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ============================================================
# Data structures
# ============================================================

@dataclass
class LayerTimingRow:
    index: int
    name: str
    layer_type: str
    incremental_p50_ms: float
    cumulative_p50_ms: float
    pct_total: float
    param_mb: float


@dataclass
class LayerMemoryRow:
    index: int
    name: str
    layer_type: str
    param_mb: float
    activation_mb: float       # estimated from weight dims, tagged (est) in display
    is_memory_peak: bool


@dataclass
class LayerBottleneckRow:
    index: int
    name: str
    layer_type: str
    flops: float
    bytes_moved: float
    arithmetic_intensity: float
    classification: str        # "COMPUTE", "MEMORY", "UNKNOWN"
    incremental_p50_ms: float


@dataclass
class FusionGroup:
    group_id: int
    layer_indices: List[int]
    layer_names: List[str]
    layer_types: List[str]
    note: str


@dataclass
class CoreMLDeepProfileResult:
    model_path: str
    model_type: str            # "neuralNetwork" or "mlProgram"
    total_layers: int
    total_time_ms: float

    # NeuralNetwork only (empty list for MLProgram)
    layer_timing: List[LayerTimingRow]
    memory_flow: List[LayerMemoryRow]
    peak_memory_mb: float
    peak_memory_layer: str
    bottlenecks: List[LayerBottleneckRow]
    compute_bound_layers: int
    memory_bound_layers: int
    fusion_groups: List[FusionGroup]
    unfused_opportunities: List[str]

    # Both formats
    cold_start: object         # ColdStartResult


# ============================================================
# FLOPs / bytes helpers
# ============================================================

def _estimate_flops_coreml(layer_type: str, layer_raw) -> float:
    lt = layer_type.lower()
    try:
        if lt == "convolution":
            p = layer_raw.convolution
            oc = p.outputChannels
            kh = p.kernelSize[0] if len(p.kernelSize) >= 1 else 1
            kw = p.kernelSize[1] if len(p.kernelSize) >= 2 else 1
            weight_len = len(p.weights.floatValue)
            if weight_len > 0 and oc > 0 and kh > 0 and kw > 0:
                kc = weight_len // (oc * kh * kw)
                return float(2 * oc * kc * kh * kw)
        elif lt == "innerproduct":
            p = layer_raw.innerProduct
            return float(2 * p.inputChannels * p.outputChannels)
        elif lt == "pooling":
            p = layer_raw.pooling
            kh = p.kernelSize[0] if len(p.kernelSize) >= 1 else 1
            kw = p.kernelSize[1] if len(p.kernelSize) >= 2 else 1
            return float(kh * kw)
        elif lt == "batchnorm":
            return float(4 * layer_raw.batchnorm.channels)
        elif lt in ("activation", "relu", "sigmoid", "tanh", "softmax",
                    "add", "multiply", "unary"):
            return 1.0
    except Exception:
        pass
    return 0.0


def _estimate_bytes_coreml(layer_raw) -> float:
    total = 0
    for field_name in ("weights", "bias"):
        try:
            arr = getattr(layer_raw, field_name, None)
            if arr is not None:
                total += len(arr.floatValue) * 4
        except Exception:
            pass
    for attr in ("convolution", "innerProduct", "batchnorm"):
        try:
            sub = getattr(layer_raw, attr, None)
            if sub is not None:
                for wfield in ("weights", "bias"):
                    arr = getattr(sub, wfield, None)
                    if arr is not None:
                        total += len(arr.floatValue) * 4
        except Exception:
            pass
    return float(max(total, 1))


def _estimate_activation_mb(layer_raw, layer_type: str) -> float:
    lt = layer_type.lower()
    try:
        if lt == "convolution":
            oc = layer_raw.convolution.outputChannels
            if oc > 0:
                # Conservative 7x7 spatial assumption for deep layers
                return oc * 7 * 7 * 4 / (1024 * 1024)
        elif lt == "innerproduct":
            return layer_raw.innerProduct.outputChannels * 4 / (1024 * 1024)
        elif lt == "batchnorm":
            return layer_raw.batchnorm.channels * 7 * 7 * 4 / (1024 * 1024)
    except Exception:
        pass
    return 0.0


# ============================================================
# Fusion pattern registry
# ============================================================

_FUSION_PATTERNS: Dict[Tuple[str, str], str] = {
    ("convolution",  "activation"):   "Conv+Activation",
    ("convolution",  "batchnorm"):    "Conv+BN",
    ("batchnorm",    "activation"):   "BN+Activation",
    ("batchnorm",    "scale"):        "BN+Scale",
    ("innerproduct", "activation"):   "FC+Activation",
    ("innerproduct", "softmax"):      "FC+Softmax",
    ("add",          "activation"):   "Add+Activation",
    ("multiply",     "activation"):   "Mul+Activation",
}

_UNFUSED_HINTS: Dict[Tuple[str, str], str] = {
    ("convolution", "batchnorm"):
        "Conv+BN can be folded at export — use ct.optimize or fold_batch_norm",
    ("batchnorm", "scale"):
        "BN+Scale is fusible — consider folding before export",
}


# ============================================================
# Main profiler
# ============================================================

class CoreMLDeepProfiler:
    """
    Deep profiler for CoreML models.

    NeuralNetwork: all 5 features.
    MLProgram: cold start only (ANE-compiled, cannot be sliced).
    """

    RIDGE_POINT = 38.0   # Apple M1: ~2.6 TFLOPS / ~68 GB/s

    def __init__(self, model_path: Path, compute_unit: str = "all"):
        self.model_path = Path(model_path)
        self.compute_unit = compute_unit

        from ...profiling import CumulativeLayerProfiler
        self._layer_profiler = CumulativeLayerProfiler(model_path, compute_unit=compute_unit)
        self.model_type = self._layer_profiler.model_type

    def profile(
        self,
        num_runs: int = 50,
        warmup_runs: int = 10,
        cold_start_runs: int = 60,
    ) -> CoreMLDeepProfileResult:

        cold_start = self._profile_cold_start(cold_start_runs)

        if self.model_type == "mlProgram":
            return CoreMLDeepProfileResult(
                model_path=str(self.model_path),
                model_type="mlProgram",
                total_layers=len(self._layer_profiler.layers),
                total_time_ms=cold_start.stable_p50_ms,
                layer_timing=[], memory_flow=[],
                peak_memory_mb=0.0, peak_memory_layer="N/A",
                bottlenecks=[], compute_bound_layers=0, memory_bound_layers=0,
                fusion_groups=[], unfused_opportunities=[],
                cold_start=cold_start,
            )

        raw = self._layer_profiler.profile(num_runs=num_runs, warmup_runs=warmup_runs)
        layers_data = raw["layers"]
        total_ms = layers_data[-1]["cumulative_time"]["p50_ms"] if layers_data else 0.0

        layer_timing  = self._build_timing_rows(layers_data, total_ms)
        memory_flow   = self._build_memory_rows(layers_data)
        peak_mb       = max((r.activation_mb for r in memory_flow), default=0.0)
        peak_layer    = next((r.name for r in memory_flow if r.is_memory_peak), "unknown")
        bottlenecks   = self._build_bottleneck_rows(layers_data)
        compute_bound = sum(1 for b in bottlenecks if b.classification == "COMPUTE")
        memory_bound  = sum(1 for b in bottlenecks if b.classification == "MEMORY")
        fusion_groups, unfused = self._detect_fusion()

        return CoreMLDeepProfileResult(
            model_path=str(self.model_path),
            model_type="neuralNetwork",
            total_layers=len(layers_data),
            total_time_ms=total_ms,
            layer_timing=layer_timing,
            memory_flow=memory_flow,
            peak_memory_mb=peak_mb,
            peak_memory_layer=peak_layer,
            bottlenecks=bottlenecks,
            compute_bound_layers=compute_bound,
            memory_bound_layers=memory_bound,
            fusion_groups=fusion_groups,
            unfused_opportunities=unfused,
            cold_start=cold_start,
        )

    def _build_timing_rows(self, layers_data, total_ms):
        rows = []
        for ld in layers_data:
            inc = ld["incremental_time"]["p50_ms"]
            rows.append(LayerTimingRow(
                index=ld["index"], name=ld["name"], layer_type=ld["type"],
                incremental_p50_ms=inc,
                cumulative_p50_ms=ld["cumulative_time"]["p50_ms"],
                pct_total=(inc / total_ms * 100) if total_ms > 0 else 0.0,
                param_mb=ld["param_memory_mb"],
            ))
        return sorted(rows, key=lambda r: r.incremental_p50_ms, reverse=True)

    def _build_memory_rows(self, layers_data):
        rows = []
        for ld in layers_data:
            raw = self._layer_profiler.layers[ld["index"]]["raw"]
            rows.append(LayerMemoryRow(
                index=ld["index"], name=ld["name"], layer_type=ld["type"],
                param_mb=ld["param_memory_mb"],
                activation_mb=_estimate_activation_mb(raw, ld["type"]),
                is_memory_peak=False,
            ))
        if rows:
            max(rows, key=lambda r: r.activation_mb).is_memory_peak = True
        return sorted(rows, key=lambda r: r.activation_mb, reverse=True)

    def _build_bottleneck_rows(self, layers_data):
        rows = []
        for ld in layers_data:
            raw   = self._layer_profiler.layers[ld["index"]]["raw"]
            flops = _estimate_flops_coreml(ld["type"], raw)
            bm    = _estimate_bytes_coreml(raw)
            ai    = flops / bm if bm > 0 else 0.0
            cls   = ("UNKNOWN" if flops == 0
                     else "COMPUTE" if ai >= self.RIDGE_POINT
                     else "MEMORY")
            rows.append(LayerBottleneckRow(
                index=ld["index"], name=ld["name"], layer_type=ld["type"],
                flops=flops, bytes_moved=bm, arithmetic_intensity=ai,
                classification=cls,
                incremental_p50_ms=ld["incremental_time"]["p50_ms"],
            ))
        return sorted(rows, key=lambda r: r.incremental_p50_ms, reverse=True)

    def _profile_cold_start(self, runs: int):
        from ...profiling.cold_start import ColdStartProfiler
        return ColdStartProfiler(
            model_path=self.model_path, backend="coreml", num_runs=runs,
        ).profile()

    def _detect_fusion(self):
        layers = self._layer_profiler.layers
        groups, unfused, visited = [], [], set()
        group_id = 0

        for i, layer in enumerate(layers):
            if i in visited or i + 1 >= len(layers):
                continue
            lt  = (layer["type"] or "").lower()
            nlt = (layers[i + 1]["type"] or "").lower()
            pattern = (lt, nlt)

            if pattern in _FUSION_PATTERNS:
                groups.append(FusionGroup(
                    group_id=group_id,
                    layer_indices=[i, i + 1],
                    layer_names=[layer["name"], layers[i + 1]["name"]],
                    layer_types=[lt, nlt],
                    note=f"{_FUSION_PATTERNS[pattern]} — CoreML may fuse into single ANE kernel",
                ))
                visited.update([i, i + 1])
                group_id += 1
            elif pattern in _UNFUSED_HINTS:
                unfused.append(
                    f"layer_{i} ({lt}) -> layer_{i+1} ({nlt}): {_UNFUSED_HINTS[pattern]}"
                )

        return groups, unfused