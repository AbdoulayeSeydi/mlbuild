"""
Three-tier task detection for MLBuild.

Tier 1 — Op graph analysis       (ONNX only, highest confidence)
Tier 2 — Tensor name heuristics  (all formats, medium confidence)
Tier 3 — Shape heuristics        (all formats, lowest confidence)

Detection result feeds into:
  - InputSchema inference  (task_inputs.py)
  - Dynamic shape resolution
  - Registry storage (task_type column)
  - CLI warnings

Public API
----------
detect_task(info: ModelInfo, forced: str | None) -> DetectionResult
resolve_shape(shape, task, seq_len, defaults)    -> tuple[int, ...]
resolve_coreml_range_dim(range_dim, task, ...)   -> int
detection_warning(result: DetectionResult)       -> str | None
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import re


# ---------------------------------------------------------------------------
# TaskType
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    VISION     = "vision"
    NLP        = "nlp"
    AUDIO      = "audio"
    MULTIMODAL = "multimodal"
    UNKNOWN    = "unknown"

    @classmethod
    def from_str(cls, value: str) -> "TaskType":
        if not value:
            return cls.UNKNOWN
        return cls.__members__.get(value.upper(), cls.UNKNOWN)

    def is_known(self) -> bool:
        return self != TaskType.UNKNOWN


class DetectionTier(str, Enum):
    GRAPH    = "graph"      # Tier 1 — op/layer analysis
    NAME     = "name"       # Tier 2 — tensor name heuristics
    SHAPE    = "shape"      # Tier 3 — shape/dtype heuristics
    METADATA = "metadata"   # model-level metadata keys
    UNKNOWN  = "unknown"    # no signal found


# ---------------------------------------------------------------------------
# Core model descriptors
# ---------------------------------------------------------------------------

def _normalize_dtype(dtype: Optional[object]) -> Optional[np.dtype]:
    """
    Normalize dtype from any exporter format to numpy dtype.

    Handles: 'float32', 'fp32', np.float32, torch.float32, None.
    Never raises — returns None for unrecognised inputs.
    """
    if dtype is None:
        return None
    try:
        return np.dtype(dtype)
    except Exception:
        s = str(dtype).lower()
        if "float" in s or "fp" in s:
            return np.dtype("float32")
        if "int64" in s:
            return np.dtype("int64")
        if "int32" in s:
            return np.dtype("int32")
        if "bool" in s:
            return np.dtype("bool")
    return None


@dataclass
class TensorInfo:
    """
    Format-agnostic descriptor for a single input or output tensor.

    All fields are optional — callers populate what the format exposes.
    Missing fields degrade detection confidence but never crash.
    """
    name:      Optional[str]             = None  # may be stripped on import
    shape:     Optional[Tuple]           = None  # may contain -1/None for dynamic dims
    dtype:     Optional[np.dtype]        = None  # normalised via _normalize_dtype
    range_dim: Optional[Tuple[int, int]] = None  # CoreML RangeDim (min, max)

    def __post_init__(self):
        self.dtype = _normalize_dtype(self.dtype)


@dataclass
class ModelInfo:
    """
    Everything detection can observe about a model artifact.

    Populated by format-specific extractors (ONNX / TFLite / CoreML).
    Fields unavailable for a given format are left as empty lists / None.
    """
    # 'onnx' | 'tflite' | 'coreml_nn' | 'coreml_mlprogram'
    format: str = "unknown"

    inputs:      List[TensorInfo] = field(default_factory=list)
    outputs:     List[TensorInfo] = field(default_factory=list)

    # ONNX only
    op_types:    Set[str]         = field(default_factory=set)
    node_count:  Optional[int]    = None
    param_count: Optional[int]    = None

    # CoreML NeuralNetwork only
    layer_types: Set[str]         = field(default_factory=set)

    # Any format — model-level metadata key/value pairs
    metadata:    Dict[str, str]   = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Detection signal + result
# ---------------------------------------------------------------------------

@dataclass
class DetectionSignal:
    task:   TaskType
    tier:   DetectionTier
    score:  float
    reason: str


@dataclass
class DetectionResult:
    """
    Aggregated result of all detection tiers.

    primary — the single best task (used for input generation)
    tasks   — full set (>1 means multimodal candidate)
    signals — all raw signals for explain() / debugging
    tier    — highest-reliability tier that contributed
    """
    primary: TaskType
    tasks:   Set[TaskType]
    signals: List[DetectionSignal]
    tier:    DetectionTier = DetectionTier.UNKNOWN

    def explain(self) -> str:
        lines = ["Task detection signals:"]
        for s in sorted(self.signals, key=lambda x: -x.score):
            lines.append(
                f"  {s.task.value:<12} tier={s.tier.value:<8} "
                f"score={s.score:.2f}  {s.reason}"
            )
        lines.append(
            f"\n  → primary={self.primary.value}  "
            f"confidence_tier={self.tier.value}"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dynamic shape defaults
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DynamicDefaults:
    batch:                 int             = 1
    image_size_candidates: Tuple[int, ...] = (224, 256, 299, 384)
    seq_len_default:       int             = 128
    waveform_samples:      int             = 16_000
    mel_bins:              int             = 80
    generic_dim:           int             = 64


DYNAMIC_DEFAULTS = DynamicDefaults()


# ---------------------------------------------------------------------------
# Tier 1 — Op / layer graph analysis
# ---------------------------------------------------------------------------

_VISION_OPS = {
    "Conv", "ConvTranspose", "MaxPool", "AveragePool",
    "GlobalAveragePool", "GlobalMaxPool", "BatchNormalization",
    "InstanceNormalization", "Resize", "RoiAlign",
    "NonMaxSuppression", "DepthwiseConv",
}
_NLP_OPS = {
    "Attention", "MultiHeadAttention", "EmbedLayerNormalization",
    "SkipLayerNormalization", "RelativePositionBias",
    "Gather", "GatherElements",
    "LayerNormalization",
    "FastGelu", "Gelu", "BiasGelu",
}
_AUDIO_OPS = {
    "STFT", "MelWeightMatrix", "DFT",
    "BlackmanWindow", "HannWindow", "AudioDecoder",
}
_COREML_VISION_LAYERS = {
    "convolution", "pooling", "batchnorm",
    "upsample", "crop", "padding", "reorganize_data",
}
_COREML_NLP_LAYERS = {
    "innerProduct", "lstm", "gru", "simpleRecurrent",
    "embedding", "softmax", "layerNorm",
}
_POOL_OPS = {"AveragePool", "MaxPool", "GlobalAveragePool", "GlobalMaxPool"}
_NMS_OPS  = {"NonMaxSuppression"}
_TRANSFORMER_OPS = {"Attention", "LayerNormalization", "Softmax", "Gelu"}


def _graph_signals(info: ModelInfo) -> List[DetectionSignal]:
    signals: List[DetectionSignal] = []

    if info.format == "onnx" and info.op_types:
        vision_hits = info.op_types & _VISION_OPS
        nlp_hits    = info.op_types & _NLP_OPS
        audio_hits  = info.op_types & _AUDIO_OPS

        # Pooling / NMS → strong vision signal
        pool_nms = info.op_types & (_POOL_OPS | _NMS_OPS)
        if pool_nms:
            signals.append(DetectionSignal(
                TaskType.VISION, DetectionTier.GRAPH, 3.0,
                f"pooling/NMS ops: {', '.join(sorted(pool_nms))}",
            ))
        elif vision_hits:
            signals.append(DetectionSignal(
                TaskType.VISION, DetectionTier.GRAPH,
                min(3.0, len(vision_hits) / 3 * 3),
                f"vision ops: {', '.join(sorted(vision_hits)[:5])}",
            ))

        # Transformer ops + integer input → NLP
        if info.op_types & _TRANSFORMER_OPS:
            int_inputs = [
                t for t in info.inputs
                if t.dtype is not None and np.issubdtype(t.dtype, np.integer)
            ]
            if int_inputs:
                signals.append(DetectionSignal(
                    TaskType.NLP, DetectionTier.GRAPH, 3.0,
                    "transformer ops + integer token input",
                ))
            else:
                signals.append(DetectionSignal(
                    TaskType.NLP, DetectionTier.GRAPH, 1.5,
                    f"transformer ops: {', '.join(sorted(nlp_hits)[:5])}",
                ))

        if audio_hits:
            signals.append(DetectionSignal(
                TaskType.AUDIO, DetectionTier.GRAPH, 3.0,
                f"audio ops: {', '.join(sorted(audio_hits))}",
            ))

        # Conv-heavy without NLP ops → vision
        conv_ops = {o for o in info.op_types if "conv" in o.lower()}
        if len(conv_ops) >= 2 and not nlp_hits:
            signals.append(DetectionSignal(
                TaskType.VISION, DetectionTier.GRAPH, 2.4,
                f"conv-heavy graph ({len(conv_ops)} conv ops)",
            ))

        # Graph size heuristics
        if info.node_count:
            if info.node_count > 800:
                signals.append(DetectionSignal(
                    TaskType.NLP, DetectionTier.GRAPH, 1.5,
                    f"large graph ({info.node_count} nodes) — likely transformer",
                ))
            elif info.node_count < 60:
                signals.append(DetectionSignal(
                    TaskType.VISION, DetectionTier.GRAPH, 0.5,
                    f"small graph ({info.node_count} nodes) — possible lightweight CNN",
                ))

        # Long temporal dim → audio
        for inp in info.inputs:
            if inp.shape:
                concrete = [d for d in inp.shape if d and d > 0]
                if concrete and max(concrete) > 4000:
                    signals.append(DetectionSignal(
                        TaskType.AUDIO, DetectionTier.GRAPH, 2.5,
                        f"long temporal dimension: {max(concrete)}",
                    ))

    if info.format == "coreml_nn" and info.layer_types:
        vision_hits = info.layer_types & _COREML_VISION_LAYERS
        nlp_hits    = info.layer_types & _COREML_NLP_LAYERS
        if vision_hits:
            signals.append(DetectionSignal(
                TaskType.VISION, DetectionTier.GRAPH,
                min(3.0, len(vision_hits) / 2 * 3),
                f"coreml layers: {', '.join(sorted(vision_hits)[:4])}",
            ))
        if nlp_hits:
            signals.append(DetectionSignal(
                TaskType.NLP, DetectionTier.GRAPH,
                min(3.0, len(nlp_hits) / 2 * 3),
                f"coreml layers: {', '.join(sorted(nlp_hits)[:4])}",
            ))

    return signals


# ---------------------------------------------------------------------------
# Tier 2 — Tensor name heuristics + metadata signals
# ---------------------------------------------------------------------------

_NAME_PATTERNS: List[Tuple[re.Pattern, TaskType]] = [
    (re.compile(r"input_ids|token_ids|input_id",         re.I), TaskType.NLP),
    (re.compile(r"attention_mask|attn_mask",             re.I), TaskType.NLP),
    (re.compile(r"token_type_ids|segment_ids",           re.I), TaskType.NLP),
    (re.compile(r"\btoken\b|\bvocab\b|\bembedding\b",    re.I), TaskType.NLP),
    (re.compile(r"pixel_values|image|img|frame|rgb|bgr", re.I), TaskType.VISION),
    (re.compile(r"input_image|visual",                   re.I), TaskType.VISION),
    (re.compile(r"input_features|mel|spectrogram|stft",  re.I), TaskType.AUDIO),
    (re.compile(r"waveform|audio|pcm|speech",            re.I), TaskType.AUDIO),
]

_METADATA_KEYWORDS: Dict[TaskType, List[str]] = {
    TaskType.NLP:    ["nlp", "bert", "gpt", "transformer", "text", "roberta", "t5", "llm"],
    TaskType.VISION: ["vision", "image", "resnet", "mobilenet", "yolo", "vit", "efficientnet"],
    TaskType.AUDIO:  ["audio", "speech", "whisper", "wav2vec", "asr", "hubert"],
}


def _name_signals(info: ModelInfo) -> List[DetectionSignal]:
    signals: List[DetectionSignal] = []
    for tensor in info.inputs + info.outputs:
        if not tensor.name:
            continue
        for pattern, task in _NAME_PATTERNS:
            if pattern.search(tensor.name):
                signals.append(DetectionSignal(
                    task, DetectionTier.NAME, 2.0,
                    f"tensor name '{tensor.name}' matches /{pattern.pattern}/",
                ))
                break  # one signal per tensor
    return signals


def _metadata_signals(info: ModelInfo) -> List[DetectionSignal]:
    signals: List[DetectionSignal] = []
    meta = " ".join(str(v).lower() for v in info.metadata.values())
    if not meta.strip():
        return signals
    for task, keywords in _METADATA_KEYWORDS.items():
        hits = [kw for kw in keywords if kw in meta]
        if hits:
            signals.append(DetectionSignal(
                task, DetectionTier.METADATA, 4.0,
                f"metadata keywords: {', '.join(hits)}",
            ))
    return signals


# ---------------------------------------------------------------------------
# Tier 3 — Shape heuristics
# ---------------------------------------------------------------------------

def _rank(shape: Optional[Tuple]) -> Optional[int]:
    return len(shape) if shape is not None else None

def _resolved_dims(shape: Tuple) -> List[int]:
    return [d for d in shape if d is not None and d > 0]


def _shape_signals(info: ModelInfo) -> List[DetectionSignal]:
    signals: List[DetectionSignal] = []
    inputs = info.inputs

    if not inputs:
        return signals

    # Vision: rank-4 float, spatial dims >= 32
    for t in inputs:
        if _rank(t.shape) == 4 and t.dtype is not None and np.issubdtype(t.dtype, np.floating):
            spatial = [d for d in _resolved_dims(t.shape) if d >= 32]
            if len(spatial) >= 2:
                signals.append(DetectionSignal(
                    TaskType.VISION, DetectionTier.SHAPE, 1.5,
                    f"rank-4 float tensor, spatial dims {spatial}",
                ))

    # NLP: 1-3 integer rank-2 tensors
    int_inputs = [
        t for t in inputs
        if t.dtype is not None
        and np.issubdtype(t.dtype, np.integer)
        and _rank(t.shape) == 2
    ]
    if 1 <= len(int_inputs) <= 3:
        signals.append(DetectionSignal(
            TaskType.NLP, DetectionTier.SHAPE, 1.5,
            f"{len(int_inputs)} rank-2 integer tensor(s) — likely token sequence(s)",
        ))

    # Audio: rank 2-3 float, largest dim >= 1000
    for t in inputs:
        r = _rank(t.shape)
        if r in (2, 3) and t.dtype is not None and np.issubdtype(t.dtype, np.floating):
            dims = _resolved_dims(t.shape)
            if dims and max(dims) >= 1000:
                signals.append(DetectionSignal(
                    TaskType.AUDIO, DetectionTier.SHAPE, 0.8,
                    f"rank-{r} float tensor with large time dim {max(dims)}",
                ))

    return signals


# ---------------------------------------------------------------------------
# Arbitration engine
# ---------------------------------------------------------------------------

_MULTIMODAL_THRESHOLD = 2.5  # both tasks need this score to trigger multimodal


def _aggregate(signals: List[DetectionSignal]) -> Dict[TaskType, float]:
    scores: Dict[TaskType, float] = {}
    for s in signals:
        scores[s.task] = scores.get(s.task, 0.0) + s.score
    return scores


def _best_tier(signals: List[DetectionSignal]) -> DetectionTier:
    order = [
        DetectionTier.GRAPH, DetectionTier.METADATA,
        DetectionTier.NAME, DetectionTier.SHAPE, DetectionTier.UNKNOWN,
    ]
    present = {s.tier for s in signals}
    for tier in order:
        if tier in present:
            return tier
    return DetectionTier.UNKNOWN


def _arbitrate(signals: List[DetectionSignal]) -> DetectionResult:
    if not signals:
        return DetectionResult(
            primary=TaskType.UNKNOWN,
            tasks={TaskType.UNKNOWN},
            signals=[],
            tier=DetectionTier.UNKNOWN,
        )

    scores = _aggregate(signals)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    primary = ranked[0][0]

    high_scoring = {t for t, s in scores.items() if s > _MULTIMODAL_THRESHOLD}
    tasks = high_scoring if len(high_scoring) > 1 else {primary}
    if len(tasks) > 1:
        tasks.add(TaskType.MULTIMODAL)

    return DetectionResult(
        primary=primary,
        tasks=tasks,
        signals=signals,
        tier=_best_tier(signals),
    )


# ---------------------------------------------------------------------------
# Dynamic shape resolution  (task-aware)
# ---------------------------------------------------------------------------

def resolve_shape(
    shape: Tuple,
    task: TaskType = TaskType.UNKNOWN,
    seq_len: Optional[int] = None,
    defaults: DynamicDefaults = DYNAMIC_DEFAULTS,
) -> Tuple[int, ...]:
    """
    Replace dynamic dimensions (-1, None) with task-appropriate defaults.

    Parameters
    ----------
    shape    : raw shape tuple, may contain -1 or None
    task     : detected TaskType — drives which defaults are used
    seq_len  : explicit NLP sequence length override (from --seq-lens)
    defaults : DynamicDefaults instance

    Returns
    -------
    Fully concrete shape tuple with no dynamic dims.

    Examples
    --------
    >>> resolve_shape((1, -1), TaskType.NLP)
    (1, 128)
    >>> resolve_shape((1, 3, -1, -1), TaskType.VISION)
    (1, 3, 224, 224)
    >>> resolve_shape((1, -1, 80), TaskType.AUDIO)
    (1, 16000, 80)
    >>> resolve_shape((1, -1), TaskType.UNKNOWN)
    (1, 64)
    """
    resolved = []
    for i, dim in enumerate(shape):
        if dim is not None and dim > 0:
            resolved.append(int(dim))
            continue

        if task == TaskType.VISION:
            if i == 0:
                resolved.append(defaults.batch)
            elif i == 1:
                resolved.append(3)                              # channel (NCHW)
            else:
                resolved.append(defaults.image_size_candidates[0])  # 224

        elif task == TaskType.NLP:
            if i == 0:
                resolved.append(defaults.batch)
            else:
                resolved.append(seq_len or defaults.seq_len_default)

        elif task == TaskType.AUDIO:
            if i == 0:
                resolved.append(defaults.batch)
            elif i == 1:
                resolved.append(defaults.waveform_samples)     # 16000
            else:
                resolved.append(defaults.mel_bins)             # 80

        else:
            resolved.append(defaults.batch if i == 0 else defaults.generic_dim)

    return tuple(resolved)


def resolve_coreml_range_dim(
    range_dim: Tuple[int, int],
    task: TaskType = TaskType.UNKNOWN,
    seq_len: Optional[int] = None,
    defaults: DynamicDefaults = DYNAMIC_DEFAULTS,
) -> int:
    """
    Resolve a CoreML RangeDim (min, max) to a concrete integer.
    Clamps the task-appropriate default into [min, max].
    """
    lo, hi = range_dim
    if task == TaskType.NLP:
        target = seq_len or defaults.seq_len_default
    elif task == TaskType.VISION:
        target = defaults.image_size_candidates[0]
    elif task == TaskType.AUDIO:
        target = defaults.waveform_samples
    else:
        target = defaults.generic_dim
    return max(lo, min(hi, target))


# ---------------------------------------------------------------------------
# Public API — detect_task
# ---------------------------------------------------------------------------

def detect_task(
    info: ModelInfo,
    forced: Optional[str] = None,
) -> DetectionResult:
    """
    Run three-tier task detection on a model artifact.

    If `forced` is provided (from --task flag), it short-circuits all
    detection and returns a GRAPH-tier result immediately.

    Parameters
    ----------
    info   : ModelInfo populated by the format-specific extractor
    forced : string from --task CLI flag, or None

    Returns
    -------
    DetectionResult with .primary, .tasks, .signals, .tier, .explain()
    """
    if forced:
        task = TaskType.from_str(forced)
        return DetectionResult(
            primary=task,
            tasks={task},
            signals=[DetectionSignal(
                task, DetectionTier.GRAPH, 10.0,
                f"explicit --task {forced}",
            )],
            tier=DetectionTier.GRAPH,
        )

    signals: List[DetectionSignal] = []
    signals.extend(_graph_signals(info))
    signals.extend(_metadata_signals(info))
    signals.extend(_name_signals(info))
    signals.extend(_shape_signals(info))

    return _arbitrate(signals)


# ---------------------------------------------------------------------------
# CLI warning helper  (tiered — matches design plan)
# ---------------------------------------------------------------------------

def detection_warning(result: DetectionResult) -> Optional[str]:
    """
    Return a warning string based on detection confidence, or None if silent.

    Tier mapping:
      GRAPH / METADATA → None (high confidence, proceed silently)
      NAME             → medium confidence warning
      SHAPE / UNKNOWN  → low confidence warning + zeros fallback notice
    """
    task = result.primary
    tier = result.tier

    if tier in (DetectionTier.GRAPH, DetectionTier.METADATA):
        return None  # high confidence — silent

    if tier == DetectionTier.NAME:
        return (
            f"⚠  Task auto-detected as '{task.value}' (medium confidence)\n"
            f"   If incorrect, re-run with: --task vision|nlp|audio"
        )

    # SHAPE or UNKNOWN
    if task == TaskType.UNKNOWN:
        return (
            "⚠  Task could not be detected — running with zero tensors\n"
            "   Specify task explicitly: --task vision|nlp|audio"
        )

    return (
        f"⚠  Task auto-detected as '{task.value}' (low confidence) "
        f"— running with zeros as fallback\n"
        f"   If incorrect, re-run with: --task vision|nlp|audio"
    )