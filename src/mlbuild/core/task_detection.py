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

Public API (v1 — backward compat, unchanged)
--------------------------------------------
detect_task(info: ModelInfo, forced: str | None) -> DetectionResult
resolve_shape(shape, task, seq_len, defaults)    -> tuple[int, ...]
resolve_coreml_range_dim(range_dim, task, ...)   -> int
detection_warning(result: DetectionResult)       -> str | None

Public API (v2 — new, additive)
--------------------------------
build_profile(info: ModelInfo, result: DetectionResult) -> ModelProfile
resolve_shape_for_domain(shape, domain, subtype, ...) -> tuple[int, ...]
_assert_profile_consistency(task_type, profile) -> None  [migration guard]

Migration notes
---------------
- detect_task() signature and return type are UNCHANGED.
- Call build_profile(info, result) after detect_task() to get ModelProfile.
- _assert_profile_consistency() logs mismatches during the migration window.
  Delete in Step 7 when TaskType is fully phased out.
- resolve_shape() is unchanged. New consumers should use resolve_shape_for_domain().
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# TaskType  (preserved — backward compat, do not modify)
# ============================================================

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


# ============================================================
# NEW: Domain, Subtype, ExecutionMode, ModelProfile
# ============================================================

class Domain(str, Enum):
    """Structural domain — drives input generation."""
    VISION  = "vision"
    NLP     = "nlp"
    AUDIO   = "audio"
    TABULAR = "tabular"


class Subtype(str, Enum):
    """
    Behavioral subtype — drives output validation and execution caveats.

    SEGMENTATION and GENERATIVE_STATEFUL are recognized subtypes.
    SEGMENTATION has no functional builder yet — guarded in task_inputs.py.
    """
    DETECTION           = "detection"
    SEGMENTATION        = "segmentation"        # placeholder — guarded, no builder
    TIMESERIES          = "timeseries"
    RECOMMENDATION      = "recommendation"
    GENERATIVE_STATEFUL = "generative_stateful"
    MULTIMODAL          = "multimodal"
    NONE                = "none"


class ExecutionMode(str, Enum):
    """
    Runtime execution semantics — drives inference path and benchmark caveats.

    STREAMING has no functional implementation yet — guarded in task_inputs.py.
    """
    STANDARD           = "standard"
    STATEFUL           = "stateful"
    PARTIALLY_STATEFUL = "partially_stateful"
    KV_CACHE           = "kv_cache"
    MULTI_INPUT        = "multi_input"
    STREAMING          = "streaming"            # placeholder — guarded, not implemented


@dataclass
class ModelProfile:
    """
    Three-layer model characterization.

    domain    — structural (drives input generation)
    subtype   — behavioral (drives output validation)
    execution — runtime semantics (drives inference path and benchmark caveats)

    nms_inside   — detection-specific: NMS is baked into the graph.
                   Validation uses final-detection rules when True,
                   raw-prediction rules when False.
    state_optional — stateful-specific: state inputs exist but are not
                     required on every call (PARTIALLY_STATEFUL models).
    """
    domain:          Domain
    subtype:         Subtype
    execution:       ExecutionMode
    confidence:      float   # 0.0–1.0
    confidence_tier: str     # mirrors DetectionTier.value

    nms_inside:     bool = False
    state_optional: bool = False


# ============================================================
# Named threshold constants
# All subtype scoring thresholds live here. Adjust a constant,
# not the scoring logic, when a model misclassifies.
# ============================================================

# ── Detection ─────────────────────────────────────────────────
DETECT_NMS_SCORE           = 3.0   # NonMaxSuppression op present
DETECT_SIGMOID_CLASS_SCORE = 2.5   # Sigmoid/Softmax on class-count output
DETECT_BBOX_DIM_SCORE      = 2.0   # output last dim == 4 (bbox coords)
DETECT_ANCHOR_SHAPE_SCORE  = 2.0   # anchor-shaped output (classes+5 or +4)
DETECT_MULTI_OUTPUT_SCORE  = 0.8   # multiple float outputs, batch dims agree
DETECT_HARD_THRESHOLD      = 4.5   # score >= this → DETECTION unconditionally
DETECT_NMS_THRESHOLD       = 3.0   # score >= this AND nms_inside → DETECTION

# ── KV-cache ──────────────────────────────────────────────────
KV_MIN_HIGH_RANK_INPUTS      = 4                  # min float rank-4+ inputs to trigger
KV_COMMON_NUM_HEADS: Set[int] = {8, 12, 16, 32, 64}
KV_CONFIRMED_CONFIDENCE      = 0.85
KV_DOWNGRADED_CONFIDENCE     = 0.40   # repeated shapes but no secondary guardrail

# ── Time-series ───────────────────────────────────────────────
TS_LSTM_GRU_SCORE     = 3.0   # LSTM/GRU/RNN op present
TS_DILATED_CONV_SCORE = 2.5   # dilated Conv1d (requires node attrs — unused here)
TS_NAME_PATTERN_SCORE = 2.0   # temporal name pattern on input
TS_SHAPE_SCORE        = 1.5   # [B, T, F] float, no integer inputs
TS_HARD_THRESHOLD     = 3.0   # score >= this → TIMESERIES unconditionally
TS_SOFT_THRESHOLD     = 2.0   # score >= this AND no_integer_inputs → TIMESERIES

# ── Recommendation ────────────────────────────────────────────
REC_GATHER_NO_ATTN_SCORE   = 3.0   # Gather present, no Attention → rec
REC_GATHER_WITH_ATTN_SCORE = 1.0   # Gather + Attention → prefer NLP
REC_EMBEDDING_PARAM_SCORE  = 1.5   # embedding-heavy parameter count
REC_SCALAR_INT_INPUT_SCORE = 2.0   # scalar integer inputs (user_id, item_id)
REC_HARD_THRESHOLD         = 3.5   # score >= this → RECOMMENDATION
REC_SOFT_THRESHOLD         = 2.5   # score >= this AND not attn_present → RECOMMENDATION

# ── Multimodal arbitration ────────────────────────────────────
_MULTIMODAL_THRESHOLD = 2.5   # both tasks need this score to trigger multimodal


# ============================================================
# Core model descriptors
# ============================================================

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

    is_optional: set by the ONNX extractor in build.py when the input
                 has a corresponding initializer (default value present).
                 Used by _detect_stateful to identify PARTIALLY_STATEFUL models.
    """
    name:        Optional[str]             = None
    shape:       Optional[Tuple]           = None
    dtype:       Optional[np.dtype]        = None
    range_dim:   Optional[Tuple[int, int]] = None
    is_optional: bool                      = False  # NEW — populated by build.py extractor

    def __post_init__(self):
        self.dtype = _normalize_dtype(self.dtype)


@dataclass
class ModelInfo:
    """
    Everything detection can observe about a model artifact.
    Populated by format-specific extractors (ONNX / TFLite / CoreML).
    Fields unavailable for a given format are left as empty lists / None.
    """
    format:      str              = "unknown"
    inputs:      List[TensorInfo] = field(default_factory=list)
    outputs:     List[TensorInfo] = field(default_factory=list)
    op_types:    Set[str]         = field(default_factory=set)
    node_count:  Optional[int]    = None
    param_count: Optional[int]    = None
    layer_types: Set[str]         = field(default_factory=set)
    metadata:    Dict[str, str]   = field(default_factory=dict)


# ============================================================
# Detection signals + result  (unchanged — backward compat)
# ============================================================

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

    This dataclass is not modified. To get ModelProfile, call
    build_profile(info, result) after detect_task().
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


# ============================================================
# Dynamic shape defaults
# ============================================================

@dataclass(frozen=True)
class DynamicDefaults:
    batch:                 int             = 1
    image_size_candidates: Tuple[int, ...] = (224, 256, 299, 384)
    seq_len_default:       int             = 128
    waveform_samples:      int             = 16_000
    mel_bins:              int             = 80
    generic_dim:           int             = 64
    timeseries_seq_len:    int             = 96   # standard forecasting lookback (ETTh1 etc.)


DYNAMIC_DEFAULTS = DynamicDefaults()


# ============================================================
# Op set constants
# ============================================================

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
    "lstm", "gru", "simpleRecurrent",
    "embedding", "layerNorm",
}
_POOL_OPS        = {"AveragePool", "MaxPool", "GlobalAveragePool", "GlobalMaxPool"}
_NMS_OPS         = {"NonMaxSuppression"}
_TRANSFORMER_OPS = {"Attention", "LayerNormalization", "Softmax", "Gelu"}
_GATHER_OPS      = {"Gather", "GatherElements", "GatherND"}
_TS_OPS          = {"LSTM", "GRU", "RNN", "LSTM2"}
_ATTENTION_OPS   = {"Attention", "MultiHeadAttention"}


# ============================================================
# Tier 1 — Op / layer graph analysis  (unchanged)
# ============================================================

def _graph_signals(info: ModelInfo) -> List[DetectionSignal]:
    signals: List[DetectionSignal] = []

    if info.format == "onnx" and info.op_types:
        vision_hits = info.op_types & _VISION_OPS
        nlp_hits    = info.op_types & _NLP_OPS
        audio_hits  = info.op_types & _AUDIO_OPS

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

        conv_ops = {o for o in info.op_types if "conv" in o.lower()}
        if len(conv_ops) >= 2 and not nlp_hits:
            signals.append(DetectionSignal(
                TaskType.VISION, DetectionTier.GRAPH, 2.4,
                f"conv-heavy graph ({len(conv_ops)} conv ops)",
            ))

        if info.node_count:
            if info.node_count > 800:
                signals.append(DetectionSignal(
                    TaskType.NLP, DetectionTier.GRAPH, 1.5,
                    f"large graph ({info.node_count} nodes) — likely transformer",
                ))


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


# ============================================================
# Tier 2 — Tensor name heuristics + metadata  (unchanged)
# ============================================================

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
                score = 3.5 if task == TaskType.AUDIO else 2.0
                signals.append(DetectionSignal(
                    task, DetectionTier.NAME, score,
                    f"tensor name '{tensor.name}' matches /{pattern.pattern}/",
                ))
                break
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


# ============================================================
# Tier 3 — Shape heuristics  (unchanged)
# ============================================================

def _rank(shape: Optional[Tuple]) -> Optional[int]:
    return len(shape) if shape is not None else None


def _resolved_dims(shape: Tuple) -> List[int]:
    return [d for d in shape if d is not None and d > 0]


def _shape_signals(info: ModelInfo) -> List[DetectionSignal]:
    signals: List[DetectionSignal] = []
    inputs = info.inputs

    if not inputs:
        return signals

    for t in inputs:
        if _rank(t.shape) == 4 and t.dtype is not None and np.issubdtype(t.dtype, np.floating):
            spatial = [d for d in _resolved_dims(t.shape) if d >= 32]
            if len(spatial) >= 2:
                signals.append(DetectionSignal(
                    TaskType.VISION, DetectionTier.SHAPE, 1.5,
                    f"rank-4 float tensor, spatial dims {spatial}",
                ))

        if (_rank(t.shape) == 4 and t.shape[-1] in (1, 3) and
                t.shape is not None and
                all(d and d >= 32 for d in t.shape[1:3])):
            signals.append(DetectionSignal(
                TaskType.VISION, DetectionTier.SHAPE, 1.5,
                f"rank-4 NHWC tensor, spatial dims {t.shape[1:3]}",
            ))

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


# ============================================================
# Arbitration  (unchanged)
# ============================================================

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


# ============================================================
# NEW: Subtype and execution scoring helpers
# ============================================================

# ── KV-cache detection ────────────────────────────────────────

_KV_NAME_RE = re.compile(
    r"past_key|past_value|past_kv|key_cache|value_cache|"
    r"kv_cache|k_cache|v_cache|past_k|past_v|"
    r"key_states|value_states|present_key|present_value|"
    r"\bpk\d+\b|\bpv\d+\b",
    re.I,
)


def _detect_kv_cache(info: ModelInfo) -> Tuple[bool, float]:
    """
    Returns (is_kv_cache, confidence).

    Requires repeated high-rank (4D+) float inputs AND at least one of:
      - KV name pattern in any input name  (strongest — name is explicit)
      - a dimension matching common num_heads values (8, 12, 16, 32, 64)
      - paired tensors with key/value symmetry (even count, same shapes)

    Without a secondary guardrail, returns (False, KV_DOWNGRADED_CONFIDENCE)
    rather than True — prevents misclassifying vision transformers or
    batched feature extractors as generative models.
    """
    # Name patterns — check unconditionally first (dtype not required)
    if any(t.name and _KV_NAME_RE.search(t.name) for t in info.inputs):
        logger.debug("kv_cache_detected: name patterns matched")
        return True, KV_CONFIRMED_CONFIDENCE

    high_rank_floats = [
        t for t in info.inputs
        if t.dtype is not None
        and np.issubdtype(t.dtype, np.floating)
        and t.shape is not None
        and len(t.shape) >= 4
    ]

    if len(high_rank_floats) < KV_MIN_HIGH_RANK_INPUTS:
        return False, 0.0

    # Check for repeated shapes (key/value pairs share the same shape)
    shape_counts: Dict[Tuple, int] = {}
    for t in high_rank_floats:
        key = tuple(d if (d is not None and d > 0) else -1 for d in t.shape)
        shape_counts[key] = shape_counts.get(key, 0) + 1

    has_repeated = any(c >= 2 for c in shape_counts.values())
    if not has_repeated:
        return False, 0.0

    # Secondary guardrail 1: a concrete dim matches a common num_heads value
    has_heads_dim = any(
        d in KV_COMMON_NUM_HEADS
        for shape in shape_counts
        for d in shape
        if isinstance(d, int) and d > 0
    )

    # Secondary guardrail 2: all repeated shapes have even counts (key + value pairs)
    repeated = {s: c for s, c in shape_counts.items() if c >= 2}
    has_pairs = bool(repeated) and all(c % 2 == 0 for c in repeated.values())

    if has_heads_dim or has_pairs:
        logger.debug(
            "kv_cache_detected: repeated shapes + %s",
            "num_heads dim" if has_heads_dim else "paired key/value tensors",
        )
        return True, KV_CONFIRMED_CONFIDENCE

    # Repeated shapes but no secondary guardrail — downgrade, do not classify
    logger.debug(
        "kv_cache_candidate: %d repeated high-rank float inputs found but no "
        "num_heads dim or paired symmetry — confidence downgraded to %.2f, "
        "falling back to MULTI_INPUT",
        len(high_rank_floats),
        KV_DOWNGRADED_CONFIDENCE,
    )
    return False, KV_DOWNGRADED_CONFIDENCE


# ── Stateful (h0/c0) detection ────────────────────────────────

_STATEFUL_NAME_RE = re.compile(
    r"\bh0\b|\bc0\b|hidden_state|cell_state|\bhidden\b|\bcell\b|"
    r"\bh_n\b|\bc_n\b|initial_hidden|initial_cell|\bhx\b|\bcx\b",
    re.I,
)


def _is_hidden_state_shape(t: TensorInfo) -> bool:
    """
    Heuristic: float rank-3 tensor shaped [num_layers, batch, hidden_size].
    num_layers is typically 1–8; hidden_size is typically >= 32.
    """
    if t.shape is None or len(t.shape) != 3:
        return False
    if t.dtype is None or not np.issubdtype(t.dtype, np.floating):
        return False
    dims = [d for d in t.shape if d is not None and d > 0]
    return len(dims) == 3 and dims[0] <= 8 and dims[2] >= 32


def _detect_stateful(info: ModelInfo) -> Tuple[bool, bool]:
    """
    Returns (is_stateful, state_optional).

    is_stateful=True when any input matches h0/c0 name patterns or
    hidden-state shape heuristic.

    state_optional=True when ALL matched stateful inputs have is_optional=True
    (set by build.py extractor when the ONNX input has a default initializer).
    When optional: ExecutionMode.PARTIALLY_STATEFUL.
    When required: ExecutionMode.STATEFUL.
    """
    stateful_inputs = [
        t for t in info.inputs
        if (t.name and _STATEFUL_NAME_RE.search(t.name))
        or _is_hidden_state_shape(t)
    ]

    if not stateful_inputs:
        return False, False

    state_optional = all(t.is_optional for t in stateful_inputs)
    return True, state_optional


# ── Detection subtype scoring ─────────────────────────────────

def _score_detection(info: ModelInfo) -> Tuple[float, bool]:
    """
    Returns (detection_score, nms_inside).
    Uses DETECT_* constants only — no overlap with _graph_signals.
    """
    score      = 0.0
    nms_inside = False

    # Output names — strongest signal when shapes are missing (dynamic/unknown)
    output_names = [o.name.lower() for o in info.outputs if o.name]
    if any("box" in n or "bbox" in n for n in output_names):
        score += DETECT_BBOX_DIM_SCORE
    if any("score" in n or "conf" in n or "class" in n for n in output_names):
        score += DETECT_SIGMOID_CLASS_SCORE

    # NonMaxSuppression — strongest single signal
    if "NonMaxSuppression" in info.op_types:
        score     += DETECT_NMS_SCORE
        nms_inside = True

    # Sigmoid or Softmax on a class-count-like output dimension (20–1000)
    # Distinguishes detection from multi-head classifiers and regression models
    if "Sigmoid" in info.op_types or "Softmax" in info.op_types:
        for out in info.outputs:
            if out.shape:
                dims = [d for d in out.shape if d is not None and d > 0]
                if dims and 20 <= dims[-1] <= 1000:
                    score += DETECT_SIGMOID_CLASS_SCORE
                    break

    # Output with last concrete dim == 4  (bounding box coordinates)
    for out in info.outputs:
        if out.shape:
            dims = [d for d in out.shape if d is not None and d > 0]
            if dims and dims[-1] == 4:
                score += DETECT_BBOX_DIM_SCORE
                break

    # Anchor-shaped output: last dim = num_classes + 5 (YOLO) or + 4 (DETR-style)
    anchor_scored = False
    for out in info.outputs:
        if anchor_scored:
            break
        if out.shape and len(out.shape) == 3:
            last = out.shape[-1]
            if last and last > 0:
                for offset in (5, 4):
                    candidate = last - offset
                    if 10 <= candidate <= 1000:
                        score        += DETECT_ANCHOR_SHAPE_SCORE
                        anchor_scored = True
                        break

    # Multiple float outputs with agreeing batch dimensions
    float_outputs = [o for o in info.outputs if o.shape and len(o.shape) >= 2]
    if len(float_outputs) >= 2:
        batch_vals = {
            o.shape[0] for o in float_outputs
            if o.shape[0] is not None and o.shape[0] > 0
        }
        if len(batch_vals) == 1:
            score += DETECT_MULTI_OUTPUT_SCORE

    return score, nms_inside


# ── Time-series subtype scoring ───────────────────────────────

_TS_NAME_RE = re.compile(
    r"sequence|timestep|time_step|series|temporal|forecast|lookback|horizon",
    re.I,
)


def _score_timeseries(info: ModelInfo) -> float:
    """
    Score time-series likelihood using TS_* constants.

    Note: TS_DILATED_CONV_SCORE (2.5) requires dilation node attributes
    which are not available in the op_types set. It is defined for
    future use when node-level attribute extraction is added.
    """
    score = 0.0

    # Tier 1: recurrent ops
    if info.op_types & _TS_OPS:
        score += TS_LSTM_GRU_SCORE

    # Tier 2: temporal name patterns on inputs
    for t in info.inputs:
        if t.name and _TS_NAME_RE.search(t.name):
            score += TS_NAME_PATTERN_SCORE
            break

    # Tier 3: [B, T, F] rank-3 float input with no integer inputs
    has_int = any(
        t.dtype is not None and np.issubdtype(t.dtype, np.integer)
        for t in info.inputs
    )
    if not has_int:
        for t in info.inputs:
            if (
                t.shape and len(t.shape) == 3
                and t.dtype is not None
                and np.issubdtype(t.dtype, np.floating)
            ):
                score += TS_SHAPE_SCORE
                break

    return score


# ── Recommendation subtype scoring ───────────────────────────

def _score_recommendation(info: ModelInfo) -> Tuple[float, bool]:
    """
    Returns (recommendation_score, attention_present).

    Gather + Attention → low score (1.0) — prefer NLP classification.
    Gather without Attention → high score (3.0) — likely pure embedding rec.
    """
    attention_present = bool(info.op_types & _ATTENTION_OPS)
    gather_present    = bool(info.op_types & _GATHER_OPS)

    if not gather_present:
        return 0.0, attention_present

    score  = REC_GATHER_WITH_ATTN_SCORE if attention_present else REC_GATHER_NO_ATTN_SCORE

    # Scalar integer inputs (user_id / item_id pattern: shape [1] or [1, 1])
    scalar_int = [
        t for t in info.inputs
        if t.dtype is not None
        and np.issubdtype(t.dtype, np.integer)
        and t.shape is not None
        and (
            len(t.shape) == 1
            or (len(t.shape) == 2 and t.shape[-1] == 1)
        )
    ]
    if scalar_int:
        score += REC_SCALAR_INT_INPUT_SCORE

    return score, attention_present


# ── Shared helpers ────────────────────────────────────────────

def _has_integer_inputs(info: ModelInfo) -> bool:
    return any(
        t.dtype is not None and np.issubdtype(t.dtype, np.integer)
        for t in info.inputs
    )


def _tier_to_confidence(tier: DetectionTier, signals: List[DetectionSignal]) -> float:
    """Map detection tier + signal count to a 0.0–1.0 confidence score."""
    base = {
        DetectionTier.GRAPH:    0.90,
        DetectionTier.METADATA: 0.85,
        DetectionTier.NAME:     0.65,
        DetectionTier.SHAPE:    0.40,
        DetectionTier.UNKNOWN:  0.10,
    }.get(tier, 0.10)
    # Small boost for multiple corroborating signals
    if len(signals) >= 3:
        base = min(1.0, base + 0.05)
    return round(base, 2)


# ============================================================
# NEW: ModelProfile builder  (public API)
# ============================================================

_TASK_TO_DOMAIN: Dict[TaskType, Domain] = {
    TaskType.VISION:     Domain.VISION,
    TaskType.NLP:        Domain.NLP,
    TaskType.AUDIO:      Domain.AUDIO,
    TaskType.MULTIMODAL: Domain.VISION,   # refined by subtype below
    TaskType.UNKNOWN:    Domain.TABULAR,
}


def build_profile(info: ModelInfo, result: DetectionResult) -> ModelProfile:
    """
    Build a ModelProfile from a ModelInfo and DetectionResult.

    Call this after detect_task(). DetectionResult is not modified.

    Subtype priority order:
      KV_CACHE > DETECTION > TIMESERIES > RECOMMENDATION > MULTIMODAL > NONE

    Domain corrections applied when subtype overrides the primary task:
      - TIMESERIES: NLP domain corrected to TABULAR (LSTM is not token-based)
      - RECOMMENDATION: always TABULAR
      - KV_CACHE/GENERATIVE_STATEFUL: always NLP
    """
    # ── Domain from primary task ────────────────────────────
    domain = _TASK_TO_DOMAIN.get(result.primary, Domain.TABULAR)

    # ── Name-based domain override (beats op/layer scoring) ────
    _audio_name_re = re.compile(r"spectrogram|mel|stft|waveform|audio|pcm|speech", re.I)
    if any(t.name and _audio_name_re.search(t.name) for t in info.inputs):
        domain = Domain.AUDIO

    # ── Score all subtypes ──────────────────────────────────
    detect_score, nms_inside = _score_detection(info)
    ts_score                 = _score_timeseries(info)
    rec_score, attn_present  = _score_recommendation(info)
    is_kv,  _kv_conf         = _detect_kv_cache(info)
    is_st,  st_optional      = _detect_stateful(info)

    # ── Subtype resolution  (priority order matters) ────────
    subtype          = Subtype.NONE
    nms_inside_final = False

    # Name-based recommendation — fires before scoring (works without op_types)
    _inames_rec = [t.name.lower() for t in info.inputs if t.name]
    if any("user" in n for n in _inames_rec) and any("item" in n for n in _inames_rec):
        subtype = Subtype.RECOMMENDATION
        domain  = Domain.TABULAR

    elif is_kv:
        subtype = Subtype.GENERATIVE_STATEFUL
        domain  = Domain.NLP

    elif detect_score >= DETECT_HARD_THRESHOLD:
        subtype          = Subtype.DETECTION
        nms_inside_final = nms_inside

    elif detect_score >= DETECT_NMS_THRESHOLD and nms_inside:
        subtype          = Subtype.DETECTION
        nms_inside_final = True

    elif ts_score >= TS_HARD_THRESHOLD:
        subtype = Subtype.TIMESERIES
        if domain in (Domain.NLP, Domain.VISION):
            domain = Domain.TABULAR

    elif ts_score >= TS_SOFT_THRESHOLD and not _has_integer_inputs(info):
        subtype = Subtype.TIMESERIES
        if domain in (Domain.NLP, Domain.VISION):
            domain = Domain.TABULAR

    elif rec_score >= REC_HARD_THRESHOLD:
        subtype = Subtype.RECOMMENDATION
        domain  = Domain.TABULAR
        logger.debug(
            "recommendation_classified score=%.2f gather=True attn=%s",
            rec_score, attn_present,
        )

    elif (rec_score >= REC_SOFT_THRESHOLD and not attn_present and (
        any("user" in n or "item" in n or "entity" in n
            for n in [t.name.lower() for t in info.inputs if t.name])
        or _has_integer_inputs(info)
    )):
        subtype = Subtype.RECOMMENDATION
        domain  = Domain.TABULAR

    elif (TaskType.MULTIMODAL in result.tasks or (
        any(t.shape and len(t.shape) == 4 for t in info.inputs) and
        any("input_ids" in (t.name or "").lower() for t in info.inputs)
    )) and result.primary != TaskType.AUDIO and domain != Domain.AUDIO:
        subtype = Subtype.MULTIMODAL
        domain  = Domain.VISION

    elif "convolution" in info.layer_types and len(info.outputs) >= 2 and subtype == Subtype.NONE:
        subtype = Subtype.DETECTION

    # Generative: input_ids + logits pattern without KV cache (single-pass)
    elif not is_kv:
        input_names  = [t.name.lower() for t in info.inputs  if t.name]
        output_names = [t.name.lower() for t in info.outputs if t.name]
        if (any("input_ids" in n for n in input_names) and
                (any("logit" in n for n in output_names) or
                 any(t.shape and len(t.shape) == 3 and t.shape[-1] and t.shape[-1] > 1000
                     for t in info.outputs)) and
                len(info.inputs) == 1):
            subtype = Subtype.GENERATIVE_STATEFUL
            domain  = Domain.NLP

    # ── Execution mode ──────────────────────────────────────
    if is_kv:
        execution = ExecutionMode.KV_CACHE
    elif is_st:
        execution = (
            ExecutionMode.PARTIALLY_STATEFUL if st_optional
            else ExecutionMode.STATEFUL
        )
    elif len(info.inputs) > 1:
        execution = ExecutionMode.MULTI_INPUT
    else:
        execution = ExecutionMode.STANDARD

    # ── Confidence ──────────────────────────────────────────
    confidence = _tier_to_confidence(result.tier, result.signals)

    return ModelProfile(
        domain          = domain,
        subtype         = subtype,
        execution       = execution,
        confidence      = confidence,
        confidence_tier = result.tier.value,
        nms_inside      = nms_inside_final,
        state_optional  = st_optional,
    )


# ============================================================
# NEW: Migration consistency guard
# ============================================================

def _assert_profile_consistency(task_type: TaskType, profile: ModelProfile) -> None:
    """
    Log a warning if the legacy TaskType and ModelProfile.domain disagree.

    Active during the migration window only — remove in Step 7 when
    TaskType is fully phased out and this becomes dead code.

    Domain.TABULAR has no TaskType equivalent; mismatches there are
    expected (recommendation, time-series) and not logged.
    """
    try:
        derived = TaskType(profile.domain.value)
    except ValueError:
        # Domain.TABULAR → no TaskType equivalent, mismatch is expected
        return

    if derived != task_type and task_type != TaskType.UNKNOWN:
        logger.warning(
            "migration_consistency_mismatch  "
            "TaskType=%s  derived_from_profile=%s  "
            "profile.domain=%s  profile.subtype=%s  "
            "— check consumer routing",
            task_type.value,
            derived.value,
            profile.domain.value,
            profile.subtype.value,
        )


# ============================================================
# Dynamic shape resolution
# ============================================================

def resolve_shape(
    shape: Tuple,
    task: TaskType = TaskType.UNKNOWN,
    seq_len: Optional[int] = None,
    defaults: DynamicDefaults = DYNAMIC_DEFAULTS,
) -> Tuple[int, ...]:
    """
    Replace dynamic dimensions (-1, None) with task-appropriate defaults.
    Signature and behavior unchanged — backward compat for all v1 consumers.
    New consumers (task_inputs.py Step 2) should use resolve_shape_for_domain().

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
                resolved.append(3)
            else:
                resolved.append(defaults.image_size_candidates[0])

        elif task == TaskType.NLP:
            if i == 0:
                resolved.append(defaults.batch)
            else:
                resolved.append(seq_len or defaults.seq_len_default)

        elif task == TaskType.AUDIO:
            if i == 0:
                resolved.append(defaults.batch)
            elif i == 1:
                resolved.append(defaults.waveform_samples)
            else:
                resolved.append(defaults.mel_bins)

        else:
            resolved.append(defaults.batch if i == 0 else defaults.generic_dim)

    return tuple(resolved)


def resolve_shape_for_domain(
    shape: Tuple,
    domain: Domain,
    subtype: Subtype = Subtype.NONE,
    seq_len: Optional[int] = None,
    defaults: DynamicDefaults = DYNAMIC_DEFAULTS,
) -> Tuple[int, ...]:
    """
    Domain-aware shape resolution for use with ModelProfile.
    Used by task_inputs.py (Step 2) — replaces resolve_shape for new consumers.

    TABULAR + TIMESERIES uses timeseries_seq_len (96) for dim 1.
    TABULAR + other subtypes uses generic_dim for all non-batch dims.

    Examples
    --------
    >>> resolve_shape_for_domain((1, -1, 7), Domain.TABULAR, Subtype.TIMESERIES)
    (1, 96, 7)
    >>> resolve_shape_for_domain((1, -1), Domain.TABULAR, Subtype.RECOMMENDATION)
    (1, 64)
    >>> resolve_shape_for_domain((1, 3, -1, -1), Domain.VISION)
    (1, 3, 224, 224)
    """
    resolved = []
    for i, dim in enumerate(shape):
        if dim is not None and dim > 0:
            resolved.append(int(dim))
            continue

        if domain == Domain.VISION:
            if i == 0:
                resolved.append(defaults.batch)
            elif i == 1:
                resolved.append(3)
            else:
                resolved.append(defaults.image_size_candidates[0])

        elif domain == Domain.NLP:
            if i == 0:
                resolved.append(defaults.batch)
            else:
                resolved.append(seq_len or defaults.seq_len_default)

        elif domain == Domain.AUDIO:
            if i == 0:
                resolved.append(defaults.batch)
            elif i == 1:
                resolved.append(defaults.waveform_samples)
            else:
                resolved.append(defaults.mel_bins)

        elif domain == Domain.TABULAR:
            if i == 0:
                resolved.append(defaults.batch)
            elif i == 1 and subtype == Subtype.TIMESERIES:
                resolved.append(seq_len or defaults.timeseries_seq_len)  # 96
            else:
                resolved.append(defaults.generic_dim)

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
    Clamps the task-appropriate default into [min, max]. Unchanged from v1.
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


# ============================================================
# Format-specific Tier 1 extractors  (unchanged)
# ============================================================

def extract_tflite_info(path) -> ModelInfo:
    from pathlib import Path
    path = Path(path)
    info = ModelInfo(format="tflite")
    try:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=str(path))
        interp.allocate_tensors()
        for t in interp.get_input_details():
            info.inputs.append(TensorInfo(
                name=t["name"],
                shape=tuple(int(x) for x in t["shape"].tolist()),
            ))
        for t in interp.get_output_details():
            info.outputs.append(TensorInfo(
                name=t["name"],
                shape=tuple(int(x) for x in t["shape"].tolist()),
            ))
    except Exception:
        pass
    return info


def extract_coreml_info(path) -> ModelInfo:
    """
    Extract layer types from a CoreML model for Tier 1 detection.
    Uses coremltools which is already installed.
    Falls back to empty ModelInfo on any error.
    """
    from pathlib import Path
    path = Path(path)

    info = ModelInfo(format="coreml_nn")

    try:
        import coremltools as ct

        model = ct.models.MLModel(str(path))
        spec  = model.get_spec()

        layer_names: Set[str] = set()

        if spec.HasField("neuralNetwork"):
            for layer in spec.neuralNetwork.layers:
                layer_type = layer.WhichOneof("layer")
                if layer_type:
                    layer_names.add(layer_type)
            info.format = "coreml_nn"

        elif spec.HasField("mlProgram"):
            for func in spec.mlProgram.functions.values():
                for block in func.block_specializations.values():
                    for op in block.operations:
                        if op.type:
                            layer_names.add(op.type)

        info.layer_types = layer_names

        for inp in spec.description.input:
            shape = None
            dtype = None
            if inp.type.HasField("multiArrayType"):
                arr   = inp.type.multiArrayType
                shape = tuple(arr.shape) if arr.shape else None
                dtype = "float32"
            info.inputs.append(TensorInfo(
                name  = inp.name,
                shape = shape,
                dtype = dtype,
            ))

        for out in spec.description.output:
            shape = None
            if out.type.HasField("multiArrayType"):
                arr   = out.type.multiArrayType
                shape = tuple(arr.shape) if arr.shape else None
            info.outputs.append(TensorInfo(
                name  = out.name,
                shape = shape,
            ))

    except Exception:
        pass  # graceful degradation to Tier 2/3

    return info


# ============================================================
# Public API
# ============================================================

def detect_task(
    info: ModelInfo,
    forced: Optional[str] = None,
) -> DetectionResult:
    """
    Run three-tier task detection. Signature and return type unchanged.

    After calling this, call build_profile(info, result) to get ModelProfile.

    Parameters
    ----------
    info   : ModelInfo populated by the format-specific extractor
    forced : legacy --task flag value. Deprecated in Step 6 — use
             --force-domain / --force-subtype / --force-execution instead.
             Kept for backward compat with existing build.py callers.

    Returns
    -------
    DetectionResult — unchanged from v1.
    """
    if forced:
        task = TaskType.from_str(forced)
        return DetectionResult(
            primary = task,
            tasks   = {task},
            signals = [DetectionSignal(
                task, DetectionTier.GRAPH, 10.0,
                f"explicit --task {forced}",
            )],
            tier    = DetectionTier.GRAPH,
        )

    signals: List[DetectionSignal] = []
    signals.extend(_graph_signals(info))
    signals.extend(_metadata_signals(info))
    signals.extend(_name_signals(info))
    signals.extend(_shape_signals(info))

    return _arbitrate(signals)


def detection_warning(result: DetectionResult) -> Optional[str]:
    """
    Return a warning string based on detection confidence, or None if silent.

    Tier mapping:
      GRAPH / METADATA → None  (high confidence, proceed silently)
      NAME             → medium confidence warning
      SHAPE / UNKNOWN  → low confidence warning + fallback notice

    Updated to reference --force-domain / --force-subtype flags.
    """
    task = result.primary
    tier = result.tier

    if tier in (DetectionTier.GRAPH, DetectionTier.METADATA):
        return None

    if tier == DetectionTier.NAME:
        return (
            f"⚠  Task auto-detected as '{task.value}' (medium confidence)\n"
            f"   If incorrect, re-run with: --force-domain vision|nlp|audio|tabular\n"
            f"   For behavioral subtype:    --force-subtype detection|timeseries|multimodal"
        )

    if task == TaskType.UNKNOWN:
        return (
            "⚠  Task could not be detected — running with zero tensors\n"
            "   Specify explicitly: --force-domain vision|nlp|audio|tabular"
        )

    return (
        f"⚠  Task auto-detected as '{task.value}' (low confidence) "
        f"— running with zeros as fallback\n"
        f"   If incorrect, re-run with: --force-domain vision|nlp|audio|tabular\n"
        f"   For behavioral subtype:    --force-subtype detection|timeseries|recommendation|generative|multimodal"
    )