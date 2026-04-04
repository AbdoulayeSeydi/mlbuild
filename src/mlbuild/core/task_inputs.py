"""
MLBuild Task Input System
Synthetic input generation.

Design goals
------------
• strict separation of responsibilities
• deterministic synthetic generation
• multimodal-safe (per-input role classification)
• task-agnostic shape resolution
• backend-agnostic numpy output
• safe fallback behaviour
• explicit logging on every unknown/unimplemented case

Logical modules contained in this file
--------------------------------------

guards
    Unimplemented subtype/execution mode runtime guards

core.input_schema
    InputSchema

core.input_roles
    Role constants, ROLE_PRIORITY, InputRole classification

detection.input_roles
    NLP/audio role inference (v1 — unchanged)

runtime.shape_resolution
    dynamic dimension resolution with substitution warnings

builders
    Per-domain/subtype schema builders

synthetic.generator
    TensorGenerator (extended for new roles)

reporting.input_description
    human-readable descriptions

Public API (v1 — unchanged)
----------------------------
InputSchema, Role, get_nlp_seq_lens, infer_nlp_roles, infer_audio_role
resolve_shape, build_input_schemas, TensorGenerator, NLPBatch
TaskInputFactory, describe_inputs

Public API (v2 — additive)
---------------------------
ROLE_PRIORITY, build_input_schemas_with_roles
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np

from .task_detection import (
    # v1 — backward compat
    TaskType,
    DetectionResult,
    ModelInfo,
    TensorInfo,
    DynamicDefaults,
    DYNAMIC_DEFAULTS,
    resolve_shape as _td_resolve_shape,       # task_detection.resolve_shape (v1)
    # v2 — new
    Domain,
    Subtype,
    ExecutionMode,
    ModelProfile,
    resolve_shape_for_domain as _resolve_for_domain,
    _assert_profile_consistency,
)

logger = logging.getLogger(__name__)


# ============================================================
# UNIMPLEMENTED GUARDS
# These are the single source of truth for what's not yet built.
# When you implement a subtype/execution mode, remove it from the set.
# The warning and fallback disappear automatically.
# ============================================================

_UNIMPLEMENTED_SUBTYPES: set = {
    Subtype.SEGMENTATION,        # no builder — falls back to domain builder
}

_UNIMPLEMENTED_EXECUTION: set = {
    ExecutionMode.STREAMING,     # not yet implemented — falls back to STANDARD
}


# ============================================================
# InputSchema  (unchanged — backward compat)
# ============================================================

@dataclass
class InputSchema:
    """
    Canonical description of a model input.

    name   : tensor name
    shape  : fully resolved shape (no dynamic dims)
    dtype  : numpy dtype
    role   : semantic role string (see Role constants)
    task   : owning TaskType (v1 compat — UNKNOWN for new task types)
    """
    name:  str
    shape: Tuple[int, ...]
    dtype: np.dtype
    role:  str
    task:  TaskType


# ============================================================
# Role constants  (v1 preserved, new constants added)
# ============================================================

class Role:
    # ── v1 (unchanged) ─────────────────────────────────────
    IMAGE           = "image"
    TOKEN_IDS       = "token_ids"
    ATTENTION_MASK  = "attention_mask"
    TOKEN_TYPE_IDS  = "token_type_ids"
    SPECTROGRAM     = "spectrogram"
    WAVEFORM        = "waveform"
    ZEROS           = "zeros"

    # ── v2 (new) ────────────────────────────────────────────
    UNKNOWN_FLOAT   = "unknown_float"   # float, no pattern match
    UNKNOWN_INT     = "unknown_int"     # integer, no pattern match
    KV_STATE        = "kv_state"        # KV-cache state tensor (zero-initialized)
    TIMESERIES      = "timeseries"      # [B, T, F] time-series feature tensor
    TABULAR         = "tabular"         # generic tabular float tensor


# Deterministic tie-breaking priority (highest priority first).
# When two roles score within _TIE_THRESHOLD of each other,
# pick the higher-priority role, not the higher-scoring one.
ROLE_PRIORITY: List[str] = [
    Role.IMAGE,
    Role.TOKEN_IDS,
    Role.ATTENTION_MASK,
    Role.TOKEN_TYPE_IDS,
    Role.SPECTROGRAM,
    Role.WAVEFORM,
    Role.UNKNOWN_FLOAT,
    Role.UNKNOWN_INT,
]

_ROLE_PRIORITY_MAP: Dict[str, int] = {r: i for i, r in enumerate(ROLE_PRIORITY)}

# Classification thresholds
_CONFIDENCE_MIN  = 0.30   # below this → UNKNOWN_FLOAT / UNKNOWN_INT
_TIE_THRESHOLD   = 0.10   # scores within this → use ROLE_PRIORITY to break tie


# ============================================================
# NLP SEQUENCE LENGTH LADDER  (v1 unchanged)
# ============================================================

DEFAULT_NLP_SEQ_LENS: List[int] = [16, 64, 128, 256]


def get_nlp_seq_lens(model_max: Optional[int]) -> List[int]:
    """
    Return the list of sequence lengths to benchmark for an NLP model.
    Clamps DEFAULT_NLP_SEQ_LENS to model_max. Falls back to [model_max]
    when nothing fits so at least one benchmark always runs.

    Examples
    --------
    >>> get_nlp_seq_lens(None)
    [16, 64, 128, 256]
    >>> get_nlp_seq_lens(128)
    [16, 64, 128]
    >>> get_nlp_seq_lens(8)
    [8]
    """
    if model_max is None:
        return list(DEFAULT_NLP_SEQ_LENS)

    clamped = [s for s in DEFAULT_NLP_SEQ_LENS if s <= model_max]

    if not clamped:
        return [model_max]

    ladder_max = DEFAULT_NLP_SEQ_LENS[-1]
    if model_max < ladder_max and model_max not in clamped:
        clamped.append(model_max)

    return sorted(set(clamped))


# ============================================================
# ROLE INFERENCE  (v1 — unchanged)
# ============================================================

_NLP_PATTERNS = {
    re.compile(r"input_ids?",      re.I): Role.TOKEN_IDS,
    re.compile(r"token_ids?",      re.I): Role.TOKEN_IDS,
    re.compile(r"attention_mask",  re.I): Role.ATTENTION_MASK,
    re.compile(r"attn_mask",       re.I): Role.ATTENTION_MASK,
    re.compile(r"token_type_ids",  re.I): Role.TOKEN_TYPE_IDS,
    re.compile(r"segment_ids",     re.I): Role.TOKEN_TYPE_IDS,
}

_AUDIO_PATTERNS = {
    re.compile(r"mel|spectrogram|features", re.I): Role.SPECTROGRAM,
    re.compile(r"audio|wave|pcm|speech",    re.I): Role.WAVEFORM,
}


def infer_nlp_roles(tensors: List[TensorInfo]) -> List[str]:
    """
    Robust NLP role inference. Rules:
    1. Name matches always win.
    2. Prefer token_ids and attention_mask.
    3. Remaining tensors default to zeros.
    """
    roles: List[Optional[str]] = [None] * len(tensors)

    for i, t in enumerate(tensors):
        if not t.name:
            continue
        for pattern, role in _NLP_PATTERNS.items():
            if pattern.search(t.name):
                roles[i] = role
                break

    if Role.TOKEN_IDS not in roles and tensors:
        roles[0] = Role.TOKEN_IDS

    if len(tensors) > 1 and Role.ATTENTION_MASK not in roles:
        for i, r in enumerate(roles):
            if r is None:
                roles[i] = Role.ATTENTION_MASK
                break

    return [r if r else Role.ZEROS for r in roles]


def infer_audio_role(tensor: TensorInfo) -> str:
    """Robust audio role inference."""
    if tensor.name:
        for pattern, role in _AUDIO_PATTERNS.items():
            if pattern.search(tensor.name):
                return role

    if tensor.shape and len(tensor.shape) == 3:
        dims = [d for d in tensor.shape if isinstance(d, int)]
        if dims and min(dims) < 256:
            return Role.SPECTROGRAM

    return Role.WAVEFORM


# ============================================================
# PER-INPUT ROLE CLASSIFICATION  (v2 — multimodal support)
# ============================================================

_ROLE_IMAGE_PATTERNS    = re.compile(r"pixel_values|image|img|frame|rgb|bgr|visual", re.I)
_ROLE_ATTN_PATTERNS     = re.compile(r"attention_mask|attn_mask",                    re.I)
_ROLE_TOK_TYPE_PATTERNS = re.compile(r"token_type_ids|segment_ids",                  re.I)
_ROLE_TOKEN_PATTERNS    = re.compile(r"input_ids?|token_ids?|\btoken\b",              re.I)
_ROLE_SPEC_PATTERNS     = re.compile(r"mel|spectrogram|stft|features",               re.I)
_ROLE_WAVE_PATTERNS     = re.compile(r"waveform|audio|pcm|speech|wave",              re.I)


def _classify_input_role(t: TensorInfo) -> Tuple[str, float]:
    """
    Classify a single input tensor into a role with confidence score.

    Returns (role, confidence) where confidence is 0.0–1.0.
    Deterministic: when two candidates score within _TIE_THRESHOLD,
    ROLE_PRIORITY breaks the tie rather than score.

    Inputs below _CONFIDENCE_MIN → UNKNOWN_FLOAT or UNKNOWN_INT.
    """
    name     = t.name or ""
    rank     = len(t.shape) if t.shape else None
    is_float = t.dtype is not None and np.issubdtype(t.dtype, np.floating)
    is_int   = t.dtype is not None and np.issubdtype(t.dtype, np.integer)
    dims     = [d for d in (t.shape or []) if d is not None and d > 0]

    candidates: Dict[str, float] = {}

    # IMAGE: rank 4, float, ≥2 spatial dims ≥ 32
    if rank == 4 and is_float:
        spatial = [d for d in dims if d >= 32]
        if len(spatial) >= 2:
            candidates[Role.IMAGE] = 0.90 + (0.05 if _ROLE_IMAGE_PATTERNS.search(name) else 0.0)

    # ATTENTION_MASK / TOKEN_TYPE_IDS / TOKEN_IDS — integer signals
    if is_int:
        if _ROLE_ATTN_PATTERNS.search(name):
            candidates[Role.ATTENTION_MASK] = 0.95
        if _ROLE_TOK_TYPE_PATTERNS.search(name):
            candidates[Role.TOKEN_TYPE_IDS] = 0.92
        if _ROLE_TOKEN_PATTERNS.search(name):
            candidates[Role.TOKEN_IDS] = 0.90
        # Integer rank 1-2 without a name match → likely token input
        if not candidates and rank in (1, 2):
            candidates[Role.TOKEN_IDS] = 0.70

    # SPECTROGRAM: rank 3, float, name or feature-dim heuristic
    if rank == 3 and is_float:
        if _ROLE_SPEC_PATTERNS.search(name):
            candidates[Role.SPECTROGRAM] = 0.85
        elif dims and min(dims) < 256:
            candidates[Role.SPECTROGRAM] = 0.65

    # WAVEFORM: rank 1-2, float, large temporal dim or name
    if is_float and rank in (1, 2):
        if _ROLE_WAVE_PATTERNS.search(name):
            candidates[Role.WAVEFORM] = 0.85
        elif dims and max(dims) >= 1000:
            candidates[Role.WAVEFORM] = 0.72

    # No candidates — return unknown
    if not candidates:
        return (Role.UNKNOWN_FLOAT if is_float else Role.UNKNOWN_INT), 0.0

    # Sort by score
    ranked = sorted(candidates.items(), key=lambda x: -x[1])
    best_role, best_score = ranked[0]

    # Below minimum threshold → unknown
    if best_score < _CONFIDENCE_MIN:
        return (Role.UNKNOWN_FLOAT if is_float else Role.UNKNOWN_INT), best_score

    # Tie-breaking: within _TIE_THRESHOLD → use ROLE_PRIORITY
    if len(ranked) >= 2:
        _, second_score = ranked[1]
        if best_score - second_score <= _TIE_THRESHOLD:
            best_role = min(
                candidates,
                key=lambda r: _ROLE_PRIORITY_MAP.get(r, len(ROLE_PRIORITY)),
            )

    return best_role, best_score


# ============================================================
# SHAPE RESOLUTION
# ============================================================

DEFAULT_BATCH = 1
DEFAULT_DIM   = 16


def resolve_shape(
    shape: Optional[Iterable[Optional[int]]],
    task: TaskType = TaskType.UNKNOWN,
    seq_len: Optional[int] = None,
    defaults: DynamicDefaults = DYNAMIC_DEFAULTS,
) -> Tuple[int, ...]:
    """
    Resolve dynamic tensor shapes using task-appropriate defaults.
    v1 API — unchanged. New consumers use _resolve_with_warnings().

    Rules
    -----
    • Concrete dims (> 0)  → kept as-is
    • None / -1 / 0        → resolved by task and dim position
    • Empty / None shape   → (1, DEFAULT_DIM)

    Examples
    --------
    >>> resolve_shape((1, -1), TaskType.NLP)
    (1, 128)
    >>> resolve_shape((1, 3, -1, -1), TaskType.VISION)
    (1, 3, 224, 224)
    >>> resolve_shape((1, -1, 80), TaskType.AUDIO)
    (1, 16000, 80)
    >>> resolve_shape((1, -1), TaskType.UNKNOWN)
    (1, 16)
    """
    if not shape:
        return (DEFAULT_BATCH, DEFAULT_DIM)

    shape_tuple = tuple(shape)

    if task != TaskType.UNKNOWN:
        return _td_resolve_shape(shape_tuple, task, seq_len, defaults)

    resolved = []
    for i, d in enumerate(shape_tuple):
        if d is None or d <= 0:
            resolved.append(DEFAULT_BATCH if i == 0 else DEFAULT_DIM)
        else:
            resolved.append(int(d))
    return tuple(resolved)


def _resolve_simple(
    shape: Optional[Tuple],
    defaults: DynamicDefaults,
) -> Tuple[int, ...]:
    """Safe fallback resolution — batch=1, all others → generic_dim."""
    if not shape:
        return (defaults.batch, defaults.generic_dim)
    resolved = []
    for i, d in enumerate(shape):
        if d is not None and isinstance(d, int) and d > 0:
            resolved.append(d)
        else:
            resolved.append(defaults.batch if i == 0 else defaults.generic_dim)
    return tuple(resolved)


def _resolve_with_warnings(
    shape:       Optional[Tuple],
    domain:      Domain,
    subtype:     Subtype,
    seq_len:     Optional[int],
    defaults:    DynamicDefaults,
    tensor_name: str,
) -> Tuple[int, ...]:
    """
    Resolve shape via resolve_shape_for_domain and log every substitution.

    Three warning classes:
      dim 0 (batch)    → "benchmark run with batch=1, throughput not captured"
      dim 1 (seq/time) → context-appropriate sequence length note
      other dims       → "feature dimension — sensitivity may be high"
    """
    if not shape:
        return (defaults.batch, defaults.generic_dim)

    dynamic_indices = {
        i for i, d in enumerate(shape)
        if d is None or not isinstance(d, int) or d <= 0
    }

    resolved = _resolve_for_domain(shape, domain, subtype, seq_len, defaults)

    for i in dynamic_indices:
        val = resolved[i]
        if i == 0:
            logger.warning(
                "dynamic_shape  tensor='%s'  dim=0 (batch)  resolved_to=%d  "
                "— benchmark run with batch=1, throughput scaling not captured",
                tensor_name, val,
            )
        elif i == 1 and subtype == Subtype.TIMESERIES:
            logger.warning(
                "dynamic_shape  tensor='%s'  dim=1 (sequence)  resolved_to=%d  "
                "— time-series window T=%d",
                tensor_name, val, val,
            )
        elif i == 1 and domain == Domain.NLP:
            logger.warning(
                "dynamic_shape  tensor='%s'  dim=1 (sequence)  resolved_to=%d  "
                "— NLP seq_len=%d",
                tensor_name, val, val,
            )
        else:
            logger.warning(
                "dynamic_shape  tensor='%s'  dim=%d (feature)  resolved_to=%d  "
                "— benchmark sensitivity to this dimension may be high",
                tensor_name, i, val,
            )

    return resolved


def _resolve_kv_shape(
    shape:    Optional[Tuple],
    defaults: DynamicDefaults,
) -> Tuple[int, ...]:
    """
    Resolve KV-cache state tensor shape.
    Dynamic sequence dim → 0 (empty cache is the correct initial state).
    All other dynamic dims → batch=1 or generic_dim.
    """
    if not shape:
        return (defaults.batch, defaults.generic_dim)
    resolved = []
    for i, d in enumerate(shape):
        if d is not None and isinstance(d, int) and d > 0:
            resolved.append(d)
        else:
            # Sequence dim in KV tensors → 0 (empty cache)
            resolved.append(0)
    return tuple(resolved)


# ============================================================
# SCHEMA BUILDERS
# ============================================================

def _safe_dtype(t: TensorInfo, fallback: str) -> np.dtype:
    if t.dtype is None:
        return np.dtype(fallback)
    return np.dtype(t.dtype)


# ── v1 builders (unchanged) ──────────────────────────────────

def _build_nlp(
    inputs:   List[TensorInfo],
    seq_len:  Optional[int]   = None,
    defaults: DynamicDefaults = DYNAMIC_DEFAULTS,
) -> List[InputSchema]:
    int_inputs = [
        t for t in inputs
        if t.dtype is not None and np.issubdtype(t.dtype, np.integer)
    ]
    roles   = infer_nlp_roles(int_inputs)
    schemas = []
    for t, role in zip(int_inputs, roles):
        schemas.append(InputSchema(
            name  = t.name or "input",
            shape = resolve_shape(t.shape, TaskType.NLP, seq_len, defaults),
            dtype = _safe_dtype(t, "int64"),
            role  = role,
            task  = TaskType.NLP,
        ))
    return schemas


def _build_vision(
    inputs:   List[TensorInfo],
    defaults: DynamicDefaults = DYNAMIC_DEFAULTS,
) -> List[InputSchema]:
    return [
        InputSchema(
            name  = t.name or "input",
            shape = resolve_shape(t.shape, TaskType.VISION, None, defaults),
            dtype = _safe_dtype(t, "float32"),
            role  = Role.IMAGE,
            task  = TaskType.VISION,
        )
        for t in inputs
    ]


def _build_audio(
    inputs:   List[TensorInfo],
    defaults: DynamicDefaults = DYNAMIC_DEFAULTS,
) -> List[InputSchema]:
    return [
        InputSchema(
            name  = t.name or "input",
            shape = resolve_shape(t.shape, TaskType.AUDIO, None, defaults),
            dtype = _safe_dtype(t, "float32"),
            role  = infer_audio_role(t),
            task  = TaskType.AUDIO,
        )
        for t in inputs
    ]


def _build_zero(
    inputs:   List[TensorInfo],
    defaults: DynamicDefaults = DYNAMIC_DEFAULTS,
) -> List[InputSchema]:
    return [
        InputSchema(
            name  = t.name or "input",
            shape = resolve_shape(t.shape, TaskType.UNKNOWN, None, defaults),
            dtype = _safe_dtype(t, "float32"),
            role  = Role.ZEROS,
            task  = TaskType.UNKNOWN,
        )
        for t in inputs
    ]


# ── v2 builders (new) ────────────────────────────────────────

def _build_detection(
    inputs:   List[TensorInfo],
    defaults: DynamicDefaults = DYNAMIC_DEFAULTS,
) -> List[InputSchema]:
    """
    Detection uses vision-style inputs — rank-4 float image tensors.
    All detection-specific logic lives in output validation (task_validation.py).
    """
    return _build_vision(inputs, defaults)


def _build_timeseries(
    info:      ModelInfo,
    execution: ExecutionMode,
    seq_len:   Optional[int]   = None,
    defaults:  DynamicDefaults = DYNAMIC_DEFAULTS,
) -> List[InputSchema]:
    """
    Time-series input builder.

    Main inputs:   [B, T, F] float tensors, T resolved to timeseries_seq_len (96).
    State inputs:  h0/c0 style tensors.
      STATEFUL          → include as zeros (cold-start initialization).
      PARTIALLY_STATEFUL → omit (state is optional).

    Logs a sliding-window caveat regardless of execution mode.
    """
    _STATEFUL_NAME_RE = re.compile(
        r"\bh0\b|\bc0\b|hidden_state|cell_state|\bhidden\b|\bcell\b|"
        r"\bh_n\b|\bc_n\b|initial_hidden|initial_cell|\bhx\b|\bcx\b",
        re.I,
    )

    def _is_state_shape(t: TensorInfo) -> bool:
        if not (t.shape and len(t.shape) == 3):
            return False
        if t.dtype is None or not np.issubdtype(t.dtype, np.floating):
            return False
        dims = [d for d in t.shape if d and d > 0]
        return len(dims) == 3 and dims[0] <= 8 and dims[2] >= 32

    main_inputs:  List[TensorInfo] = []
    state_inputs: List[TensorInfo] = []

    for t in info.inputs:
        if (t.name and _STATEFUL_NAME_RE.search(t.name)) or _is_state_shape(t):
            state_inputs.append(t)
        else:
            main_inputs.append(t)

    ts_seq_len = seq_len or defaults.timeseries_seq_len  # 96
    schemas: List[InputSchema] = []

    for t in main_inputs:
        resolved = _resolve_with_warnings(
            t.shape, Domain.TABULAR, Subtype.TIMESERIES,
            ts_seq_len, defaults, t.name or "input",
        )
        schemas.append(InputSchema(
            name  = t.name or "input",
            shape = resolved,
            dtype = _safe_dtype(t, "float32"),
            role  = Role.TIMESERIES,
            task  = TaskType.UNKNOWN,
        ))

    if execution == ExecutionMode.STATEFUL and state_inputs:
        for t in state_inputs:
            resolved = _resolve_simple(t.shape, defaults)
            schemas.append(InputSchema(
                name  = t.name or "state",
                shape = resolved,
                dtype = _safe_dtype(t, "float32"),
                role  = Role.ZEROS,
                task  = TaskType.UNKNOWN,
            ))
        logger.info(
            "timeseries_stateful_init  %d state tensors initialized to zeros "
            "— benchmark reflects cold-start latency only. "
            "Real stateful latency will differ.",
            len(state_inputs),
        )

    elif execution == ExecutionMode.PARTIALLY_STATEFUL and state_inputs:
        logger.info(
            "timeseries_partially_stateful  %d optional state inputs omitted "
            "— running without state, cold-start latency only.",
            len(state_inputs),
        )

    logger.info(
        "timeseries_model  assumed fixed window input (T=%d). "
        "Model may expect rolling context; benchmark reflects single-window latency only.",
        ts_seq_len,
    )

    return schemas


def _build_kv_cache(
    info:     ModelInfo,
    seq_len:  Optional[int]   = None,
    defaults: DynamicDefaults = DYNAMIC_DEFAULTS,
) -> List[InputSchema]:
    """
    KV-cache generative model input builder.

    Integer inputs (token_ids, typically [1×1]) → NLP schema.
    Float high-rank inputs (past_key_values) → Role.KV_STATE zeros,
    seq_len dim resolved to 0 (empty cache = correct initial state).

    Logs the number of state tensors initialized and the seq_len used.
    """
    int_inputs   = [t for t in info.inputs if t.dtype is not None and np.issubdtype(t.dtype, np.integer)]
    float_inputs = [t for t in info.inputs if t.dtype is not None and np.issubdtype(t.dtype, np.floating)]

    schemas: List[InputSchema] = []

    # Token IDs → standard NLP builder
    schemas.extend(_build_nlp(int_inputs, seq_len, defaults))

    # KV state tensors → zeros with seq_len=0 (empty cache)
    for t in float_inputs:
        resolved = _resolve_kv_shape(t.shape, defaults)
        schemas.append(InputSchema(
            name  = t.name or "kv_state",
            shape = resolved,
            dtype = _safe_dtype(t, "float32"),
            role  = Role.KV_STATE,
            task  = TaskType.NLP,
        ))

    logger.info(
        "kv_cache_inputs_initialized  token_inputs=%d  state_tensors=%d  "
        "seq_len=0 (empty cache). If runtime rejects zero-length tensors, "
        "re-run with seq_len=1.",
        len(int_inputs),
        len(float_inputs),
    )

    return schemas


def _build_schema_for_role(
    t:        TensorInfo,
    role:     str,
    seq_len:  Optional[int],
    defaults: DynamicDefaults,
) -> InputSchema:
    """Build a single InputSchema from a TensorInfo and its assigned role."""
    name = t.name or "input"

    if role == Role.IMAGE:
        return InputSchema(
            name  = name,
            shape = _resolve_with_warnings(t.shape, Domain.VISION, Subtype.NONE, None, defaults, name),
            dtype = _safe_dtype(t, "float32"),
            role  = Role.IMAGE,
            task  = TaskType.VISION,
        )

    if role in (Role.TOKEN_IDS, Role.ATTENTION_MASK, Role.TOKEN_TYPE_IDS):
        return InputSchema(
            name  = name,
            shape = _resolve_with_warnings(t.shape, Domain.NLP, Subtype.NONE, seq_len, defaults, name),
            dtype = _safe_dtype(t, "int64"),
            role  = role,
            task  = TaskType.NLP,
        )

    if role == Role.SPECTROGRAM:
        return InputSchema(
            name  = name,
            shape = _resolve_with_warnings(t.shape, Domain.AUDIO, Subtype.NONE, None, defaults, name),
            dtype = _safe_dtype(t, "float32"),
            role  = Role.SPECTROGRAM,
            task  = TaskType.AUDIO,
        )

    if role == Role.WAVEFORM:
        return InputSchema(
            name  = name,
            shape = _resolve_with_warnings(t.shape, Domain.AUDIO, Subtype.NONE, None, defaults, name),
            dtype = _safe_dtype(t, "float32"),
            role  = Role.WAVEFORM,
            task  = TaskType.AUDIO,
        )

    # UNKNOWN_FLOAT, UNKNOWN_INT — generic fallback
    is_int   = t.dtype is not None and np.issubdtype(t.dtype, np.integer)
    resolved = _resolve_simple(t.shape, defaults)
    return InputSchema(
        name  = name,
        shape = resolved,
        dtype = _safe_dtype(t, "int64" if is_int else "float32"),
        role  = role,
        task  = TaskType.UNKNOWN,
    )


def _build_multimodal(
    inputs:   List[TensorInfo],
    seq_len:  Optional[int]   = None,
    defaults: DynamicDefaults = DYNAMIC_DEFAULTS,
) -> List[InputSchema]:
    """
    Per-input role classification for multimodal models.

    Each input is classified independently using _classify_input_role().
    UNKNOWN_FLOAT / UNKNOWN_INT assignments are logged explicitly.
    Set subtype=MULTIMODAL only when inputs span different domains.
    """
    schemas: List[InputSchema] = []

    for t in inputs:
        role, confidence = _classify_input_role(t)

        if role in (Role.UNKNOWN_FLOAT, Role.UNKNOWN_INT):
            logger.warning(
                "multimodal_unknown_input  name='%s'  shape=%s  dtype=%s  "
                "classified_as=%s  confidence=%.2f  "
                "— using generic %s schema",
                t.name or "unnamed",
                t.shape,
                t.dtype,
                role,
                confidence,
                "float" if role == Role.UNKNOWN_FLOAT else "int",
            )

        schemas.append(_build_schema_for_role(t, role, seq_len, defaults))

    return schemas


def _build_recommendation(
    inputs:   List[TensorInfo],
    defaults: DynamicDefaults = DYNAMIC_DEFAULTS,
) -> List[InputSchema]:
    """
    Recommendation model input builder.

    Integer inputs (user_id, item_id) → kept as integer tensors.
    Float inputs (feature vectors, embeddings) → Role.TABULAR float tensors.
    No fake NLP token sequences are generated.
    """
    schemas: List[InputSchema] = []

    for t in inputs:
        resolved = _resolve_simple(t.shape, defaults)
        is_int   = t.dtype is not None and np.issubdtype(t.dtype, np.integer)

        schemas.append(InputSchema(
            name  = t.name or "input",
            shape = resolved,
            dtype = _safe_dtype(t, "int64" if is_int else "float32"),
            role  = Role.TOKEN_IDS if is_int else Role.TABULAR,
            task  = TaskType.UNKNOWN,
        ))

    return schemas


def _build_tabular(
    inputs:   List[TensorInfo],
    defaults: DynamicDefaults = DYNAMIC_DEFAULTS,
) -> List[InputSchema]:
    """Generic tabular (TABULAR domain, no specific subtype)."""
    return [
        InputSchema(
            name  = t.name or "input",
            shape = _resolve_simple(t.shape, defaults),
            dtype = _safe_dtype(t, "float32"),
            role  = Role.TABULAR,
            task  = TaskType.UNKNOWN,
        )
        for t in inputs
    ]


# ============================================================
# INTERNAL ROUTING
# ============================================================

def _route_by_profile(
    info:     ModelInfo,
    profile:  ModelProfile,
    seq_len:  Optional[int],
    defaults: DynamicDefaults,
) -> List[InputSchema]:
    """Profile-aware routing. Called when profile is available."""
    domain    = profile.domain
    subtype   = profile.subtype
    execution = profile.execution

    # Resolve unimplemented placeholders to safe fallbacks
    effective_subtype = Subtype.NONE if subtype in _UNIMPLEMENTED_SUBTYPES else subtype
    effective_exec    = ExecutionMode.STANDARD if execution in _UNIMPLEMENTED_EXECUTION else execution

    # Execution-mode routing takes priority over subtype
    if effective_exec == ExecutionMode.KV_CACHE:
        return _build_kv_cache(info, seq_len, defaults)

    # Subtype routing
    if effective_subtype == Subtype.DETECTION:
        return _build_detection(info.inputs, defaults)

    if effective_subtype == Subtype.TIMESERIES:
        return _build_timeseries(info, effective_exec, seq_len, defaults)

    if effective_subtype in (Subtype.GENERATIVE_STATEFUL,):
        return _build_kv_cache(info, seq_len, defaults)

    if effective_subtype == Subtype.RECOMMENDATION:
        return _build_recommendation(info.inputs, defaults)

    if effective_subtype == Subtype.MULTIMODAL:
        return _build_multimodal(info.inputs, seq_len, defaults)

    # Fall through to domain-based routing
    if domain == Domain.VISION:
        return _build_vision(info.inputs, defaults)
    if domain == Domain.NLP:
        return _build_nlp(info.inputs, seq_len, defaults)
    if domain == Domain.AUDIO:
        return _build_audio(info.inputs, defaults)
    if domain == Domain.TABULAR:
        return _build_tabular(info.inputs, defaults)

    return []


def _route_by_result(
    info:     ModelInfo,
    result:   DetectionResult,
    seq_len:  Optional[int],
    defaults: DynamicDefaults,
) -> List[InputSchema]:
    """Legacy routing via DetectionResult — v1 behavior preserved."""
    schemas: List[InputSchema] = []
    tasks = result.tasks or {result.primary}

    for task in tasks:
        if task == TaskType.NLP:
            schemas.extend(_build_nlp(info.inputs, seq_len, defaults))
        elif task == TaskType.VISION:
            schemas.extend(_build_vision(info.inputs, defaults))
        elif task == TaskType.AUDIO:
            schemas.extend(_build_audio(info.inputs, defaults))

    return schemas


# ============================================================
# PUBLIC API — SCHEMA BUILDING
# ============================================================

def build_input_schemas(
    info:     ModelInfo,
    result:   DetectionResult,
    seq_len:  Optional[int]        = None,
    defaults: DynamicDefaults      = DYNAMIC_DEFAULTS,
    profile:  Optional[ModelProfile] = None,   # NEW — keyword-only, backward compat
) -> List[InputSchema]:
    """
    Build input schemas for all detected tasks.

    v1 behavior preserved when profile=None.
    When profile is provided, routes via ModelProfile (domain + subtype + execution).

    Unimplemented subtypes/execution modes log a warning and fall back
    to the domain-level builder.

    Parameters
    ----------
    info     : ModelInfo from the format-specific extractor
    result   : DetectionResult from detect_task()
    seq_len  : NLP/time-series sequence length override
    defaults : DynamicDefaults (uses DYNAMIC_DEFAULTS if omitted)
    profile  : ModelProfile from build_profile() — enables v2 routing

    Returns
    -------
    List[InputSchema] — one schema per model input tensor.
    Use build_input_schemas_with_roles() to also get the input_roles dict.
    """
    if not info.inputs:
        return []

    # Emit guard warnings for unimplemented placeholders
    if profile is not None:
        if profile.subtype in _UNIMPLEMENTED_SUBTYPES:
            logger.warning(
                "unimplemented_subtype  subtype=%s  "
                "— no functional input builder, falling back to domain builder. "
                "Results may not reflect true model semantics.",
                profile.subtype.value,
            )
        if profile.execution in _UNIMPLEMENTED_EXECUTION:
            logger.warning(
                "unimplemented_execution  execution=%s  "
                "— falling back to STANDARD. "
                "Benchmark results will not reflect %s semantics.",
                profile.execution.value,
                profile.execution.value,
            )

    # Route
    if profile is not None:
        schemas = _route_by_profile(info, profile, seq_len, defaults)
    else:
        schemas = _route_by_result(info, result, seq_len, defaults)

    # Final fallback
    if not schemas:
        schemas = _build_zero(info.inputs, defaults)

    return schemas


def build_input_schemas_with_roles(
    info:     ModelInfo,
    result:   DetectionResult,
    profile:  Optional[ModelProfile] = None,
    seq_len:  Optional[int]          = None,
    defaults: DynamicDefaults        = DYNAMIC_DEFAULTS,
) -> Tuple[List[InputSchema], Dict[str, str]]:
    """
    v2 API — returns (schemas, input_roles).

    input_roles maps tensor name → assigned role string.
    Used by build.py to populate BuildView.input_roles so that
    UNKNOWN_FLOAT / UNKNOWN_INT classifications are visible in JSON/CSV
    without requiring log access.

    Any role that is UNKNOWN_FLOAT or UNKNOWN_INT signals that the
    benchmark inputs for that tensor may not reflect real-world values.
    """
    schemas    = build_input_schemas(info, result, seq_len, defaults, profile)
    input_roles = {s.name: s.role for s in schemas}
    return schemas, input_roles


# ============================================================
# TENSOR GENERATION  (extended for new roles)
# ============================================================

class TensorGenerator:
    """Deterministic synthetic tensor generator."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def generate(self, schema: InputSchema) -> np.ndarray:
        shape = schema.shape
        dtype = schema.dtype

        if schema.role == Role.IMAGE:
            return self.rng.normal(0.0, 1.0, size=shape).astype(dtype)

        if schema.role == Role.TOKEN_IDS:
            if dtype.itemsize < 4:
                dtype = np.int32
            return self.rng.integers(0, 30000, size=shape, dtype=dtype)

        if schema.role == Role.ATTENTION_MASK:
            return np.ones(shape, dtype=dtype)

        if schema.role == Role.TOKEN_TYPE_IDS:
            return np.zeros(shape, dtype=dtype)

        if schema.role == Role.SPECTROGRAM:
            return self.rng.normal(-5.0, 3.0, size=shape).astype(dtype)

        if schema.role == Role.WAVEFORM:
            return self.rng.uniform(-1.0, 1.0, size=shape).astype(dtype)

        # ── v2 roles ─────────────────────────────────────────

        if schema.role == Role.KV_STATE:
            # Empty KV cache — seq_len=0 may create a zero-size dim, which
            # is valid numpy. Runtime rejection handled by the benchmark runner.
            return np.zeros(shape, dtype=dtype)

        if schema.role == Role.TIMESERIES:
            # Realistic-ish time-series features (mean 0, std 1)
            return self.rng.normal(0.0, 1.0, size=shape).astype(dtype)

        if schema.role == Role.TABULAR:
            return self.rng.normal(0.0, 1.0, size=shape).astype(dtype)

        if schema.role == Role.UNKNOWN_FLOAT:
            # Safe generic float tensor — zeros are less likely to cause
            # numerical issues than random values for an unknown model.
            return np.zeros(shape, dtype=dtype)

        if schema.role == Role.UNKNOWN_INT:
            return np.zeros(shape, dtype=dtype)

        # Fallback (ZEROS + any unrecognized role)
        return np.zeros(shape, dtype=dtype)


# ============================================================
# NLP BATCH DESCRIPTOR  (unchanged)
# ============================================================

@dataclass
class NLPBatch:
    """
    A single NLP input batch for one sequence length.

    seq_len : the sequence length used for this batch
    inputs  : name → tensor dict ready for backend inference
    schemas : the InputSchema list this batch was generated from
    """
    seq_len: int
    inputs:  Dict[str, np.ndarray]
    schemas: List[InputSchema]


# ============================================================
# FACTORY  (v1 preserved, v2 additions)
# ============================================================

class TaskInputFactory:
    """
    Public interface for MLBuild synthetic input generation.

    Usage (v1 — unchanged)
    -----------------------
    factory = TaskInputFactory()
    inputs  = factory.generate(schemas)

    batches = factory.generate_nlp_batches(info, result, model_max_seq_len=512)
    for batch in batches:
        runner.run(batch.inputs)

    Usage (v2 — with ModelProfile)
    --------------------------------
    inputs, roles = factory.generate_with_roles(info, result, profile)
    """

    def __init__(self, seed: Optional[int] = None):
        self.generator = TensorGenerator(seed)

    def generate(
        self,
        schemas: List[InputSchema],
    ) -> Dict[str, np.ndarray]:
        if not schemas:
            return {}
        return {s.name: self.generator.generate(s) for s in schemas}

    def generate_for_model(
        self,
        info:     ModelInfo,
        result:   DetectionResult,
        seq_len:  Optional[int]          = None,
        defaults: DynamicDefaults        = DYNAMIC_DEFAULTS,
        profile:  Optional[ModelProfile] = None,   # NEW — keyword, backward compat
    ) -> Dict[str, np.ndarray]:
        schemas = build_input_schemas(info, result, seq_len, defaults, profile)
        return self.generate(schemas)

    def generate_with_roles(
        self,
        info:     ModelInfo,
        result:   DetectionResult,
        profile:  Optional[ModelProfile] = None,
        seq_len:  Optional[int]          = None,
        defaults: DynamicDefaults        = DYNAMIC_DEFAULTS,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
        """
        v2 API — returns (inputs_dict, input_roles_dict).
        input_roles maps tensor name → role string for build.py storage.
        """
        schemas, roles = build_input_schemas_with_roles(
            info, result, profile, seq_len, defaults,
        )
        return self.generate(schemas), roles

    def generate_nlp_batches(
        self,
        info:              ModelInfo,
        result:            DetectionResult,
        model_max_seq_len: Optional[int]   = None,
        defaults:          DynamicDefaults = DYNAMIC_DEFAULTS,
    ) -> List[NLPBatch]:
        """
        Generate one input batch per NLP sequence length.

        Returns List[NLPBatch], one per seq length in the ladder.

        Example
        -------
        [
          NLPBatch(seq_len=16,  inputs={'input_ids': ..., 'attention_mask': ...}),
          NLPBatch(seq_len=64,  inputs={...}),
          NLPBatch(seq_len=128, inputs={...}),
          NLPBatch(seq_len=256, inputs={...}),
        ]
        """
        seq_lens = get_nlp_seq_lens(model_max_seq_len)
        batches: List[NLPBatch] = []

        for seq_len in seq_lens:
            schemas = build_input_schemas(info, result, seq_len=seq_len, defaults=defaults)
            inputs  = self.generate(schemas)
            batches.append(NLPBatch(seq_len=seq_len, inputs=inputs, schemas=schemas))

        return batches


# ============================================================
# REPORTING  (updated for new task types)
# ============================================================

def describe_inputs(schemas: List[InputSchema]) -> str:
    """
    Human-readable input description.
    Honest — does not pretend tensors are real data.
    """
    if not schemas:
        return "model has no inputs"

    roles = {s.role for s in schemas}

    if Role.KV_STATE in roles:
        return "synthetic token input + zero-initialized KV-cache state"

    if Role.TIMESERIES in roles:
        return "synthetic time-series feature tensor"

    if Role.TABULAR in roles:
        return "synthetic tabular feature tensor"

    if Role.UNKNOWN_FLOAT in roles or Role.UNKNOWN_INT in roles:
        return "synthetic tensors (role unknown — shape from metadata)"

    task = schemas[0].task

    if task == TaskType.VISION:
        return "synthetic image tensor"
    if task == TaskType.NLP:
        return "synthetic token sequence"
    if task == TaskType.AUDIO:
        return "synthetic audio tensor"

    # Mixed tasks (multimodal)
    if len({s.task for s in schemas}) > 1:
        return "synthetic multimodal inputs"

    return "synthetic tensors"