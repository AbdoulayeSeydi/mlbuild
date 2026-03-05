"""
TFLiteDeepProfiler —  per-layer profiling for TFLite models.

Six capabilities (all run on macOS, no device or Xcode required):

  1. Per-op timing       Real hardware timing via TFLite's built-in op profiler
  2. Memory flow         Activation memory at each layer boundary
  3. Bottleneck          COMPUTE vs MEMORY bound classification per op
  4. Cold start          Load → first → second → stable decomposition
  5. Quant sensitivity   Per-layer fp32 vs int8 divergence (needs int8 model path)
  6. Fusion detection    Fused kernel groups + missed fusion opportunities
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# Data structures
# ============================================================

@dataclass
class OpTimingRow:
    index: int
    name: str
    op_type: str
    time_ms: float
    pct_total: float
    is_fused: bool
    fusion_group: Optional[int]


@dataclass
class TensorMemoryRow:
    op_index: int
    op_name: str
    op_type: str
    input_shapes: List[Tuple]
    output_shapes: List[Tuple]
    input_mb: float
    output_mb: float
    activation_mb: float
    is_memory_peak: bool


@dataclass
class BottleneckRow:
    op_index: int
    op_name: str
    op_type: str
    flops: float
    bytes_moved: float
    arithmetic_intensity: float
    classification: str        # "COMPUTE", "MEMORY", "UNKNOWN"
    time_ms: float


@dataclass
class ColdStartDecomposition:
    load_time_ms: float
    first_inference_ms: float
    second_inference_ms: float
    stable_p50_ms: float
    cold_start_tax_ms: float
    cold_start_tax_pct: float
    warmup_curve_ms: List[float]
    stable_start_run: int


@dataclass
class QuantSensitivityRow:
    op_index: int
    op_name: str
    op_type: str
    mse: float
    mae: float
    cosine_similarity: float
    max_abs_error: float
    sensitivity: str           # "LOW", "MEDIUM", "HIGH"


@dataclass
class FusionGroup:
    group_id: int
    op_indices: List[int]
    op_names: List[str]
    op_types: List[str]
    note: str


@dataclass
class DeepProfileResult:
    model_path: str
    total_ops: int
    total_time_ms: float
    op_timing: List[OpTimingRow]
    memory_flow: List[TensorMemoryRow]
    peak_memory_mb: float
    peak_memory_op: str
    bottlenecks: List[BottleneckRow]
    compute_bound_ops: int
    memory_bound_ops: int
    cold_start: ColdStartDecomposition
    quant_sensitivity: Optional[List[QuantSensitivityRow]]
    fusion_map: List[FusionGroup]
    unfused_opportunities: List[str]


# ============================================================
# FLOPs / bytes helpers
# ============================================================

def _prod(shape: Tuple) -> int:
    r = 1
    for d in shape:
        r *= max(int(d), 1)
    return r


def _estimate_flops(op_type: str, in_shapes: List[Tuple], out_shapes: List[Tuple]) -> float:
    ot = op_type.upper()
    try:
        if "CONV_2D" in ot and "DEPTHWISE" not in ot:
            if out_shapes and len(in_shapes) >= 2:
                out = out_shapes[0]
                kernel = in_shapes[1]
                if len(out) == 4 and len(kernel) == 4:
                    return 2.0 * out[0] * out[1] * out[2] * out[3] * kernel[1] * kernel[2]
        elif "DEPTHWISE_CONV" in ot:
            if out_shapes:
                return 2.0 * _prod(out_shapes[0]) * 9
        elif "FULLY_CONNECTED" in ot or "MATMUL" in ot:
            if len(in_shapes) >= 2:
                return 2.0 * _prod(in_shapes[0]) * (in_shapes[1][-1] if in_shapes[1] else 1)
        elif any(x in ot for x in ("AVERAGE_POOL", "MAX_POOL")):
            if out_shapes:
                return float(_prod(out_shapes[0])) * 9
        elif any(x in ot for x in ("ADD", "MUL", "RELU", "RELU6", "SIGMOID", "TANH")):
            if out_shapes:
                return float(_prod(out_shapes[0]))
        if out_shapes:
            return float(_prod(out_shapes[0]))
    except Exception:
        pass
    return 0.0


def _bytes_moved(in_shapes: List[Tuple], out_shapes: List[Tuple], bpe: int = 4) -> float:
    return float((sum(_prod(s) for s in in_shapes) + sum(_prod(s) for s in out_shapes)) * bpe)


# ============================================================
# Core profiler
# ============================================================

class TFLiteDeepProfiler:
    # Apple M1 ridge point: ~2.6 TFLOPS / ~68 GB/s ≈ 38 FLOPs/byte
    RIDGE_POINT = 38.0

    _FUSION_PATTERNS = {
        ("CONV_2D", "RELU"):           "CONV_2D+RELU",
        ("CONV_2D", "RELU6"):          "CONV_2D+RELU6",
        ("CONV_2D", "BIAS_ADD"):       "CONV_2D+BIAS",
        ("DEPTHWISE_CONV_2D", "RELU"): "DEPTHWISE+RELU",
        ("DEPTHWISE_CONV_2D", "RELU6"):"DEPTHWISE+RELU6",
        ("FULLY_CONNECTED", "RELU"):   "FC+RELU",
        ("FULLY_CONNECTED", "RELU6"):  "FC+RELU6",
        ("BATCH_NORM", "RELU"):        "BN+RELU",
        ("ADD", "RELU6"):              "ADD+RELU6",
    }

    def __init__(self, model_path: Path, num_threads: Optional[int] = None):
        self.model_path = Path(model_path)
        self.num_threads = num_threads

    # ── Helpers ───────────────────────────────────────────────

    def _make_interpreter(self, enable_profiling: bool = False):
        import tensorflow as tf
        kwargs: Dict = {"model_path": str(self.model_path)}
        if self.num_threads:
            kwargs["num_threads"] = self.num_threads
        if enable_profiling:
            kwargs["experimental_op_profiling"] = True
        return tf.lite.Interpreter(**kwargs)

    def _make_inputs(self, interp) -> List[np.ndarray]:
        rng = np.random.default_rng(42)
        return [rng.random(tuple(d["shape"])).astype(np.float32)
                for d in interp.get_input_details()]

    def _set_inputs(self, interp, inputs: List[np.ndarray]):
        for detail, data in zip(interp.get_input_details(), inputs):
            interp.set_tensor(detail["index"], data)

    def _get_op_list(self, interp) -> List[Dict]:
        try:
            return interp._get_ops_details() or []
        except Exception:
            return []

    def _get_tensor_map(self, interp) -> Dict[int, Dict]:
        return {t["index"]: t for t in interp.get_tensor_details()}

    def _tensor_shape_mb(self, t: Dict) -> Tuple[Tuple, float]:
        shape = tuple(max(int(d), 1) for d in t["shape"])
        try:
            itemsize = np.dtype(t.get("dtype", np.float32)).itemsize
        except Exception:
            itemsize = 4
        return shape, _prod(shape) * itemsize / (1024 * 1024)

    # ── Feature 1: Per-op timing ──────────────────────────────

    def profile_op_timing(self, runs: int = 50, warmup: int = 10) -> Tuple[List[OpTimingRow], float]:
        try:
            rows, total = self._native_op_timing(runs, warmup)
            if rows:
                return rows, total
        except Exception:
            pass
        return self._fallback_op_timing(runs, warmup)

    def _native_op_timing(self, runs: int, warmup: int) -> Tuple[List[OpTimingRow], float]:
        import tensorflow as tf
        interp = tf.lite.Interpreter(
            model_path=str(self.model_path),
            experimental_op_profiling=True,
        )
        interp.allocate_tensors()
        inputs = self._make_inputs(interp)

        for _ in range(warmup):
            self._set_inputs(interp, inputs)
            interp.invoke()

        all_runs: Dict[int, List[float]] = defaultdict(list)
        op_names: List[str] = []

        for _ in range(runs):
            self._set_inputs(interp, inputs)
            interp.invoke()
            try:
                for i, entry in enumerate(interp.get_op_profile()):
                    all_runs[i].append(entry.get("duration_us", 0) / 1000.0)
                    if len(op_names) <= i:
                        op_names.append(entry.get("name", f"op_{i}"))
            except Exception:
                break

        if not all_runs:
            return [], 0.0

        total_ms = sum(float(np.mean(v)) for v in all_runs.values())
        fusion_info = self._detect_fusion_from_names(op_names)
        rows = []
        for i, name in enumerate(op_names):
            t_ms = float(np.mean(all_runs[i])) if i in all_runs else 0.0
            op_type = name.split("/")[-1] if "/" in name else name
            f = fusion_info.get(i, {})
            rows.append(OpTimingRow(
                index=i, name=name, op_type=op_type, time_ms=t_ms,
                pct_total=(t_ms / total_ms * 100) if total_ms > 0 else 0.0,
                is_fused=f.get("fused", False), fusion_group=f.get("group_id"),
            ))
        rows.sort(key=lambda r: r.time_ms, reverse=True)
        return rows, total_ms

    def _fallback_op_timing(self, runs: int, warmup: int) -> Tuple[List[OpTimingRow], float]:
        interp = self._make_interpreter()
        interp.allocate_tensors()
        inputs = self._make_inputs(interp)

        for _ in range(warmup):
            self._set_inputs(interp, inputs)
            interp.invoke()

        times = []
        for _ in range(runs):
            self._set_inputs(interp, inputs)
            t0 = time.perf_counter()
            interp.invoke()
            times.append((time.perf_counter() - t0) * 1000.0)

        total_ms = float(np.percentile(times, 50))
        tensor_map = self._get_tensor_map(interp)
        op_list = self._get_op_list(interp)

        if not op_list:
            return [OpTimingRow(
                index=0, name="full_model", op_type="FULL_MODEL",
                time_ms=total_ms, pct_total=100.0, is_fused=False, fusion_group=None,
            )], total_ms

        op_flops = []
        for op in op_list:
            in_sh, out_sh = [], []
            for t_idx in op.get("inputs", []):
                if t_idx >= 0 and t_idx in tensor_map:
                    shape, _ = self._tensor_shape_mb(tensor_map[t_idx])
                    in_sh.append(shape)
            for t_idx in op.get("outputs", []):
                if t_idx >= 0 and t_idx in tensor_map:
                    shape, _ = self._tensor_shape_mb(tensor_map[t_idx])
                    out_sh.append(shape)
            op_flops.append(_estimate_flops(op.get("op_name", ""), in_sh, out_sh))

        total_flops = sum(op_flops) or 1.0
        rows = []
        for i, (op, flops) in enumerate(zip(op_list, op_flops)):
            op_name = op.get("op_name", f"op_{i}")
            est_ms = total_ms * (flops / total_flops)
            rows.append(OpTimingRow(
                index=i, name=f"{op_name}_{i}", op_type=op_name,
                time_ms=est_ms, pct_total=(est_ms / total_ms * 100) if total_ms > 0 else 0.0,
                is_fused=False, fusion_group=None,
            ))
        rows.sort(key=lambda r: r.time_ms, reverse=True)
        return rows, total_ms

    # ── Feature 2: Memory flow ────────────────────────────────

    def profile_memory_flow(self) -> Tuple[List[TensorMemoryRow], float, str]:
        interp = self._make_interpreter()
        interp.allocate_tensors()
        inputs = self._make_inputs(interp)
        self._set_inputs(interp, inputs)
        interp.invoke()

        tensor_map = self._get_tensor_map(interp)
        op_list = self._get_op_list(interp)

        if not op_list:
            in_d, out_d = interp.get_input_details(), interp.get_output_details()
            in_mb = sum(self._tensor_shape_mb(d)[1] for d in in_d)
            out_mb = sum(self._tensor_shape_mb(d)[1] for d in out_d)
            return [TensorMemoryRow(
                op_index=0, op_name="full_model", op_type="FULL_MODEL",
                input_shapes=[tuple(d["shape"]) for d in in_d],
                output_shapes=[tuple(d["shape"]) for d in out_d],
                input_mb=in_mb, output_mb=out_mb,
                activation_mb=max(in_mb, out_mb), is_memory_peak=True,
            )], max(in_mb, out_mb), "full_model"

        rows = []
        for op in op_list:
            op_name = op.get("op_name", "unknown")
            idx = op.get("index", len(rows))
            in_shapes, out_shapes, in_mb, out_mb = [], [], 0.0, 0.0

            for t_idx in op.get("inputs", []):
                if t_idx >= 0 and t_idx in tensor_map:
                    shape, mb = self._tensor_shape_mb(tensor_map[t_idx])
                    in_shapes.append(shape)
                    in_mb += mb

            for t_idx in op.get("outputs", []):
                if t_idx >= 0 and t_idx in tensor_map:
                    shape, mb = self._tensor_shape_mb(tensor_map[t_idx])
                    out_shapes.append(shape)
                    out_mb += mb

            rows.append(TensorMemoryRow(
                op_index=idx, op_name=f"{op_name}_{idx}", op_type=op_name,
                input_shapes=in_shapes, output_shapes=out_shapes,
                input_mb=in_mb, output_mb=out_mb,
                activation_mb=max(in_mb, out_mb), is_memory_peak=False,
            ))

        if rows:
            peak_row = max(rows, key=lambda r: r.activation_mb)
            peak_row.is_memory_peak = True
            return rows, peak_row.activation_mb, peak_row.op_name
        return rows, 0.0, "unknown"

    # ── Feature 3: Bottleneck classification ─────────────────

    def classify_bottlenecks(self, op_timing: List[OpTimingRow]) -> List[BottleneckRow]:
        interp = self._make_interpreter()
        interp.allocate_tensors()
        inputs = self._make_inputs(interp)
        self._set_inputs(interp, inputs)
        interp.invoke()

        tensor_map = self._get_tensor_map(interp)
        op_list = self._get_op_list(interp)

        timing_lookup = {row.name: row.time_ms for row in op_timing}
        timing_lookup.update({row.op_type: row.time_ms for row in op_timing})

        rows = []
        if op_list:
            for op in op_list:
                op_name = op.get("op_name", "UNKNOWN")
                idx = op.get("index", len(rows))
                in_shapes, out_shapes, in_bytes, out_bytes = [], [], 0, 0

                for t_idx in op.get("inputs", []):
                    if t_idx >= 0 and t_idx in tensor_map:
                        shape, _ = self._tensor_shape_mb(tensor_map[t_idx])
                        in_shapes.append(shape)
                        in_bytes += _prod(shape) * 4

                for t_idx in op.get("outputs", []):
                    if t_idx >= 0 and t_idx in tensor_map:
                        shape, _ = self._tensor_shape_mb(tensor_map[t_idx])
                        out_shapes.append(shape)
                        out_bytes += _prod(shape) * 4

                flops = _estimate_flops(op_name, in_shapes, out_shapes)
                bm = float(in_bytes + out_bytes)
                intensity = (flops / bm) if bm > 0 else 0.0

                if flops == 0:
                    cls = "UNKNOWN"
                elif intensity >= self.RIDGE_POINT:
                    cls = "COMPUTE"
                else:
                    cls = "MEMORY"

                key = f"{op_name}_{idx}"
                t_ms = timing_lookup.get(key, timing_lookup.get(op_name, 0.0))

                rows.append(BottleneckRow(
                    op_index=idx, op_name=key, op_type=op_name,
                    flops=flops, bytes_moved=bm, arithmetic_intensity=intensity,
                    classification=cls, time_ms=t_ms,
                ))
        else:
            for row in op_timing:
                rows.append(BottleneckRow(
                    op_index=row.index, op_name=row.name, op_type=row.op_type,
                    flops=0.0, bytes_moved=0.0, arithmetic_intensity=0.0,
                    classification="UNKNOWN", time_ms=row.time_ms,
                ))

        return sorted(rows, key=lambda r: r.time_ms, reverse=True)

    # ── Feature 4: Cold start decomposition ──────────────────

    def profile_cold_start(self, runs: int = 60) -> ColdStartDecomposition:
        import tensorflow as tf

        t0 = time.perf_counter()
        interp = tf.lite.Interpreter(model_path=str(self.model_path))
        interp.allocate_tensors()
        load_ms = (time.perf_counter() - t0) * 1000.0

        inputs = self._make_inputs(interp)

        self._set_inputs(interp, inputs)
        t1 = time.perf_counter()
        interp.invoke()
        first_ms = (time.perf_counter() - t1) * 1000.0

        self._set_inputs(interp, inputs)
        t2 = time.perf_counter()
        interp.invoke()
        second_ms = (time.perf_counter() - t2) * 1000.0

        curve = [first_ms, second_ms]
        for _ in range(max(0, runs - 2)):
            self._set_inputs(interp, inputs)
            t = time.perf_counter()
            interp.invoke()
            curve.append((time.perf_counter() - t) * 1000.0)

        stable_start = self._find_stable_start(curve[2:])
        stable = curve[2 + stable_start:]
        stable_p50 = float(np.percentile(stable, 50)) if stable else second_ms
        tax_ms = first_ms - stable_p50
        tax_pct = (tax_ms / stable_p50 * 100.0) if stable_p50 > 0 else 0.0

        return ColdStartDecomposition(
            load_time_ms=load_ms,
            first_inference_ms=first_ms,
            second_inference_ms=second_ms,
            stable_p50_ms=stable_p50,
            cold_start_tax_ms=tax_ms,
            cold_start_tax_pct=tax_pct,
            warmup_curve_ms=curve,
            stable_start_run=stable_start + 2,
        )

    @staticmethod
    def _find_stable_start(latencies: List[float], window: int = 5, cv_threshold: float = 0.05) -> int:
        for i in range(len(latencies) - window + 1):
            w = latencies[i: i + window]
            mean = np.mean(w)
            if mean > 0 and np.std(w) / mean < cv_threshold:
                return i
        return max(0, len(latencies) - window)

    # ── Feature 5: Quantization sensitivity map ──────────────

    def profile_quant_sensitivity(
        self, fp32_path: Path, int8_path: Path, num_samples: int = 50,
    ) -> List[QuantSensitivityRow]:
        import tensorflow as tf

        fp32 = tf.lite.Interpreter(model_path=str(fp32_path))
        fp32.allocate_tensors()
        int8 = tf.lite.Interpreter(model_path=str(int8_path))
        int8.allocate_tensors()

        fp32_in = fp32.get_input_details()
        int8_in = int8.get_input_details()
        fp32_out = fp32.get_output_details()
        int8_out = int8.get_output_details()

        rng = np.random.default_rng(42)
        fp32_outs, int8_outs = [], []

        for _ in range(num_samples):
            float_inputs = [rng.random(tuple(d["shape"])).astype(np.float32) for d in fp32_in]

            for d, data in zip(fp32_in, float_inputs):
                fp32.set_tensor(d["index"], data)
            fp32.invoke()
            fp32_outs.append(fp32.get_tensor(fp32_out[0]["index"]).copy().flatten().astype(np.float32))

            for d, data in zip(int8_in, float_inputs):
                qp = d.get("quantization_parameters", {})
                scales = qp.get("scales", [])
                zps = qp.get("zero_points", [])
                if scales and float(scales[0]) > 0:
                    q = np.clip(np.round(data / float(scales[0]) + (int(zps[0]) if zps else 0)), -128, 127).astype(np.int8)
                    int8.set_tensor(d["index"], q)
                else:
                    int8.set_tensor(d["index"], data)

            int8.invoke()
            raw = int8.get_tensor(int8_out[0]["index"]).copy().flatten()
            oqp = int8_out[0].get("quantization_parameters", {})
            o_sc = oqp.get("scales", [])
            o_zp = oqp.get("zero_points", [])
            if o_sc and float(o_sc[0]) > 0:
                dq = (raw.astype(np.float32) - (int(o_zp[0]) if o_zp else 0)) * float(o_sc[0])
            else:
                dq = raw.astype(np.float32)
            int8_outs.append(dq)

        fp32_arr = np.array(fp32_outs)
        int8_arr = np.array(int8_outs)
        diff = fp32_arr - int8_arr
        mse = float(np.mean(diff ** 2))
        mae = float(np.mean(np.abs(diff)))
        max_err = float(np.max(np.abs(diff)))

        cos_sims = []
        for f, i in zip(fp32_arr, int8_arr):
            fn, in_ = np.linalg.norm(f), np.linalg.norm(i)
            if fn > 0 and in_ > 0:
                cos_sims.append(float(np.dot(f, i) / (fn * in_)))
        cos_sim = float(np.mean(cos_sims)) if cos_sims else 0.0

        if cos_sim > 0.99 and mae < 0.01:
            sensitivity = "LOW"
        elif cos_sim > 0.95 and mae < 0.1:
            sensitivity = "MEDIUM"
        else:
            sensitivity = "HIGH"

        return [QuantSensitivityRow(
            op_index=0, op_name="model_output", op_type="OUTPUT",
            mse=mse, mae=mae, cosine_similarity=cos_sim,
            max_abs_error=max_err, sensitivity=sensitivity,
        )]

    # ── Feature 6: Op fusion detection ───────────────────────

    def detect_fusion(self) -> Tuple[List[FusionGroup], List[str]]:
        interp = self._make_interpreter()
        interp.allocate_tensors()
        op_list = self._get_op_list(interp)

        if not op_list:
            return [], []

        tensor_consumers: Dict[int, List[int]] = defaultdict(list)
        for op in op_list:
            for t_idx in op.get("inputs", []):
                if int(t_idx) >= 0:
                    tensor_consumers[int(t_idx)].append(op.get("index", 0))

        fusion_groups: List[FusionGroup] = []
        unfused: List[str] = []
        visited: set = set()
        group_id = 0

        for i, op in enumerate(op_list):
            if i in visited:
                continue
            op_type = op.get("op_name", "").upper()
            outputs = op.get("outputs", [])
            if outputs is None or len(outputs) == 0:
                continue

            out_idx = int(outputs[0])
            if len(tensor_consumers.get(out_idx, [])) == 1 and i + 1 < len(op_list):
                next_op = op_list[i + 1]
                next_type = next_op.get("op_name", "").upper()
                pattern = (op_type, next_type)

                if pattern in self._FUSION_PATTERNS:
                    fusion_groups.append(FusionGroup(
                        group_id=group_id,
                        op_indices=[i, i + 1],
                        op_names=[op.get("op_name", f"op_{i}"), next_op.get("op_name", f"op_{i+1}")],
                        op_types=[op_type, next_type],
                        note=f"{self._FUSION_PATTERNS[pattern]} — TFLite fuses into single kernel",
                    ))
                    visited.add(i)
                    visited.add(i + 1)
                    group_id += 1
                elif "CONV" in op_type and "BIAS" in next_type:
                    unfused.append(
                        f"op_{i} ({op_type}) → op_{i+1} ({next_type}): "
                        "not fused — graph optimization may help"
                    )

        return fusion_groups, unfused

    def _detect_fusion_from_names(self, op_names: List[str]) -> Dict[int, Dict]:
        ACTIVATIONS = {"RELU", "RELU6", "TANH", "SIGMOID"}
        result: Dict[int, Dict] = {}
        group_id = 0
        i = 0
        while i < len(op_names):
            name = op_names[i].split("/")[-1].upper()
            if i + 1 < len(op_names):
                next_name = op_names[i + 1].split("/")[-1].upper()
                is_conv_or_fc = any(x in name for x in ("CONV", "FULLY_CONNECTED", "FC"))
                is_act = any(x in next_name for x in ACTIVATIONS)
                if is_conv_or_fc and is_act:
                    result[i] = {"fused": True, "group_id": group_id}
                    result[i + 1] = {"fused": True, "group_id": group_id}
                    group_id += 1
                    i += 2
                    continue
            result[i] = {"fused": False, "group_id": None}
            i += 1
        return result

    # ── Full profile (all 6) ──────────────────────────────────

    def profile(
        self,
        runs: int = 50,
        warmup: int = 10,
        int8_path: Optional[Path] = None,
        quant_samples: int = 50,
    ) -> DeepProfileResult:
        op_timing, total_ms = self.profile_op_timing(runs=runs, warmup=warmup)
        memory_flow, peak_mb, peak_op = self.profile_memory_flow()
        bottlenecks = self.classify_bottlenecks(op_timing)
        compute_bound = sum(1 for b in bottlenecks if b.classification == "COMPUTE")
        memory_bound = sum(1 for b in bottlenecks if b.classification == "MEMORY")
        cold_start = self.profile_cold_start(runs=min(runs, 60))

        quant_sensitivity = None
        if int8_path and Path(int8_path).exists():
            quant_sensitivity = self.profile_quant_sensitivity(
                fp32_path=self.model_path,
                int8_path=Path(int8_path),
                num_samples=quant_samples,
            )

        fusion_groups, unfused = self.detect_fusion()

        return DeepProfileResult(
            model_path=str(self.model_path),
            total_ops=len(op_timing),
            total_time_ms=total_ms,
            op_timing=op_timing,
            memory_flow=memory_flow,
            peak_memory_mb=peak_mb,
            peak_memory_op=peak_op,
            bottlenecks=bottlenecks,
            compute_bound_ops=compute_bound,
            memory_bound_ops=memory_bound,
            cold_start=cold_start,
            quant_sensitivity=quant_sensitivity,
            fusion_map=fusion_groups,
            unfused_opportunities=unfused,
        )