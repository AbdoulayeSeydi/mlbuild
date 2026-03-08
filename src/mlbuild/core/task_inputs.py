"""
MLBuild Task Input System
Enterprise-grade synthetic input generation.

Design goals
------------
• strict separation of responsibilities
• deterministic synthetic generation
• multimodal-safe
• task-agnostic shape resolution
• backend-agnostic numpy output
• safe fallback behaviour

Logical modules contained in this file
--------------------------------------

core.input_schema
    InputSchema

detection.input_roles
    NLP/audio role inference

runtime.shape_resolution
    dynamic dimension resolution

synthetic.generator
    tensor generation

reporting.input_description
    human readable descriptions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable
import numpy as np
import re

from .task_detection import (
    TaskType,
    DetectionResult,
    ModelInfo,
    TensorInfo,
    DynamicDefaults,
    DYNAMIC_DEFAULTS,
    resolve_shape as _task_aware_resolve_shape,
)


# ============================================================
# CORE SCHEMA
# ============================================================


@dataclass
class InputSchema:
    """
    Canonical description of a model input.

    name   : tensor name
    shape  : fully resolved shape
    dtype  : numpy dtype
    role   : semantic role
    task   : owning task
    """

    name: str
    shape: Tuple[int, ...]
    dtype: np.dtype
    role: str
    task: TaskType


# ============================================================
# ROLE CONSTANTS
# ============================================================


class Role:
    IMAGE = "image"
    TOKEN_IDS = "token_ids"
    ATTENTION_MASK = "attention_mask"
    TOKEN_TYPE_IDS = "token_type_ids"
    SPECTROGRAM = "spectrogram"
    WAVEFORM = "waveform"
    ZEROS = "zeros"


# ============================================================
# PATCH 1 — NLP SEQUENCE LENGTH LADDER
# ============================================================

DEFAULT_NLP_SEQ_LENS: List[int] = [16, 64, 128, 256]


def get_nlp_seq_lens(model_max: Optional[int]) -> List[int]:
    """
    Return the list of sequence lengths to benchmark for an NLP model.

    Clamps DEFAULT_NLP_SEQ_LENS to model_max. If nothing fits, falls
    back to [model_max] so at least one benchmark always runs.

    Examples
    --------
    >>> get_nlp_seq_lens(None)
    [16, 64, 128, 256]
    >>> get_nlp_seq_lens(512)
    [16, 64, 128, 256]
    >>> get_nlp_seq_lens(128)
    [16, 64, 128]
    >>> get_nlp_seq_lens(64)
    [16, 64]
    >>> get_nlp_seq_lens(8)
    [8]
    """
    if model_max is None:
        return list(DEFAULT_NLP_SEQ_LENS)

    # AFTER (fixed):
    clamped = [s for s in DEFAULT_NLP_SEQ_LENS if s <= model_max]

    if not clamped:
        return [model_max]

    # Only append model_max if it's an intermediate point not already
    # covered by the ladder — i.e. between two ladder values.
    ladder_max = DEFAULT_NLP_SEQ_LENS[-1]
    if model_max < ladder_max and model_max not in clamped:
        clamped.append(model_max)

    return sorted(set(clamped))


# ============================================================
# ROLE INFERENCE
# ============================================================


_NLP_PATTERNS = {
    re.compile(r"input_ids?", re.I): Role.TOKEN_IDS,
    re.compile(r"token_ids?", re.I): Role.TOKEN_IDS,
    re.compile(r"attention_mask", re.I): Role.ATTENTION_MASK,
    re.compile(r"attn_mask", re.I): Role.ATTENTION_MASK,
    re.compile(r"token_type_ids", re.I): Role.TOKEN_TYPE_IDS,
    re.compile(r"segment_ids", re.I): Role.TOKEN_TYPE_IDS,
}


_AUDIO_PATTERNS = {
    re.compile(r"mel|spectrogram|features", re.I): Role.SPECTROGRAM,
    re.compile(r"audio|wave|pcm|speech", re.I): Role.WAVEFORM,
}


def infer_nlp_roles(tensors: List[TensorInfo]) -> List[str]:
    """
    Robust NLP role inference.

    Rules
    -----
    1. Name matches always win
    2. Prefer token_ids and attention_mask
    3. Remaining tensors default to zeros
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

    roles = [r if r else Role.ZEROS for r in roles]

    return roles


def infer_audio_role(tensor: TensorInfo) -> str:
    """
    Robust audio role inference.
    """

    if tensor.name:
        for pattern, role in _AUDIO_PATTERNS.items():
            if pattern.search(tensor.name):
                return role

    if tensor.shape and len(tensor.shape) == 3:

        dims = [d for d in tensor.shape if isinstance(d, int)]

        if dims:

            feature_dim = min(dims)

            if feature_dim < 256:
                return Role.SPECTROGRAM

    return Role.WAVEFORM


# ============================================================
# PATCH 2 — SHAPE RESOLUTION (task-aware)
# ============================================================

DEFAULT_BATCH = 1
DEFAULT_DIM = 16


def resolve_shape(
    shape: Optional[Iterable[Optional[int]]],
    task: TaskType = TaskType.UNKNOWN,
    seq_len: Optional[int] = None,
    defaults: DynamicDefaults = DYNAMIC_DEFAULTS,
) -> Tuple[int, ...]:
    """
    Resolve dynamic tensor shapes using task-appropriate defaults.

    Rules
    -----
    • Concrete dims (> 0)  → kept as-is
    • None / -1 / 0        → resolved by task and dim position:
        NLP    dim 0 → 1 (batch), dim 1+ → seq_len or 128
        VISION dim 0 → 1, dim 1 → 3 (channel), dim 2+ → 224
        AUDIO  dim 0 → 1, dim 1 → 16000 (T), dim 2+ → 80 (mel)
        UNKNOWN           → DEFAULT_DIM (16) for safety
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

    # For known tasks delegate to the task-aware resolver in task_detection
    if task != TaskType.UNKNOWN:
        return _task_aware_resolve_shape(shape_tuple, task, seq_len, defaults)

    # UNKNOWN — safe fallback
    resolved = []
    for i, d in enumerate(shape_tuple):
        if d is None or d <= 0:
            resolved.append(DEFAULT_BATCH if i == 0 else DEFAULT_DIM)
        else:
            resolved.append(int(d))
    return tuple(resolved)


# ============================================================
# SCHEMA BUILDING
# ============================================================


def build_input_schemas(
    info: ModelInfo,
    result: DetectionResult,
    seq_len: Optional[int] = None,
    defaults: DynamicDefaults = DYNAMIC_DEFAULTS,
) -> List[InputSchema]:
    """
    Build input schemas for all detected tasks.

    Multimodal models are fully supported.
    seq_len is threaded through to NLP builders for per-seq-len
    shape resolution during benchmark ladder runs.
    """

    if not info.inputs:
        return []

    schemas: List[InputSchema] = []

    tasks = result.tasks or {result.primary}

    for task in tasks:

        if task == TaskType.NLP:
            schemas.extend(_build_nlp(info.inputs, seq_len, defaults))

        elif task == TaskType.VISION:
            schemas.extend(_build_vision(info.inputs, defaults))

        elif task == TaskType.AUDIO:
            schemas.extend(_build_audio(info.inputs, defaults))

    if not schemas:
        schemas.extend(_build_zero(info.inputs, defaults))

    return schemas


def _safe_dtype(t: TensorInfo, fallback: str) -> np.dtype:

    if t.dtype is None:
        return np.dtype(fallback)

    return np.dtype(t.dtype)


def _build_nlp(
    inputs: List[TensorInfo],
    seq_len: Optional[int] = None,
    defaults: DynamicDefaults = DYNAMIC_DEFAULTS,
) -> List[InputSchema]:

    int_inputs = [
        t for t in inputs
        if t.dtype is not None and np.issubdtype(t.dtype, np.integer)
    ]

    roles = infer_nlp_roles(int_inputs)

    schemas = []

    for t, role in zip(int_inputs, roles):

        schemas.append(
            InputSchema(
                name=t.name or "input",
                shape=resolve_shape(t.shape, TaskType.NLP, seq_len, defaults),
                dtype=_safe_dtype(t, "int64"),
                role=role,
                task=TaskType.NLP,
            )
        )

    return schemas


def _build_vision(
    inputs: List[TensorInfo],
    defaults: DynamicDefaults = DYNAMIC_DEFAULTS,
) -> List[InputSchema]:

    schemas = []

    for t in inputs:

        schemas.append(
            InputSchema(
                name=t.name or "input",
                shape=resolve_shape(t.shape, TaskType.VISION, None, defaults),
                dtype=_safe_dtype(t, "float32"),
                role=Role.IMAGE,
                task=TaskType.VISION,
            )
        )

    return schemas


def _build_audio(
    inputs: List[TensorInfo],
    defaults: DynamicDefaults = DYNAMIC_DEFAULTS,
) -> List[InputSchema]:

    schemas = []

    for t in inputs:

        schemas.append(
            InputSchema(
                name=t.name or "input",
                shape=resolve_shape(t.shape, TaskType.AUDIO, None, defaults),
                dtype=_safe_dtype(t, "float32"),
                role=infer_audio_role(t),
                task=TaskType.AUDIO,
            )
        )

    return schemas


def _build_zero(
    inputs: List[TensorInfo],
    defaults: DynamicDefaults = DYNAMIC_DEFAULTS,
) -> List[InputSchema]:

    schemas = []

    for t in inputs:

        schemas.append(
            InputSchema(
                name=t.name or "input",
                shape=resolve_shape(t.shape, TaskType.UNKNOWN, None, defaults),
                dtype=_safe_dtype(t, "float32"),
                role=Role.ZEROS,
                task=TaskType.UNKNOWN,
            )
        )

    return schemas


# ============================================================
# TENSOR GENERATION
# ============================================================


class TensorGenerator:
    """
    Deterministic synthetic tensor generator.
    """

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

            return self.rng.integers(
                0, 30000, size=shape, dtype=dtype
            )

        if schema.role == Role.ATTENTION_MASK:
            return np.ones(shape, dtype=dtype)

        if schema.role == Role.TOKEN_TYPE_IDS:
            return np.zeros(shape, dtype=dtype)

        if schema.role == Role.SPECTROGRAM:
            return self.rng.normal(-5.0, 3.0, size=shape).astype(dtype)

        if schema.role == Role.WAVEFORM:
            return self.rng.uniform(-1.0, 1.0, size=shape).astype(dtype)

        return np.zeros(shape, dtype=dtype)


# ============================================================
# PATCH 3 — NLP BATCH DESCRIPTOR + generate_nlp_batches
# ============================================================


@dataclass
class NLPBatch:
    """
    A single NLP input batch for one sequence length.

    Attributes
    ----------
    seq_len : the sequence length used for this batch
    inputs  : name → tensor dict ready for backend inference
    schemas : the InputSchema list this batch was generated from
    """
    seq_len: int
    inputs:  Dict[str, np.ndarray]
    schemas: List[InputSchema]


# ============================================================
# FACTORY
# ============================================================


class TaskInputFactory:
    """
    Public interface for MLBuild synthetic input generation.

    Usage
    -----
    # Single inference pass (vision / audio / unknown)
    factory = TaskInputFactory()
    inputs = factory.generate(schemas)

    # Multi-seq-len NLP benchmarking
    batches = factory.generate_nlp_batches(info, result, model_max_seq_len=512)
    for batch in batches:
        runner.run(batch.inputs)
    """

    def __init__(self, seed: Optional[int] = None):

        self.generator = TensorGenerator(seed)

    def generate(
        self,
        schemas: List[InputSchema],
    ) -> Dict[str, np.ndarray]:

        if not schemas:
            return {}

        return {
            schema.name: self.generator.generate(schema)
            for schema in schemas
        }

    def generate_for_model(
        self,
        info: ModelInfo,
        result: DetectionResult,
        seq_len: Optional[int] = None,
        defaults: DynamicDefaults = DYNAMIC_DEFAULTS,
    ) -> Dict[str, np.ndarray]:

        schemas = build_input_schemas(info, result, seq_len=seq_len, defaults=defaults)

        return self.generate(schemas)

    def generate_nlp_batches(
        self,
        info: ModelInfo,
        result: DetectionResult,
        model_max_seq_len: Optional[int] = None,
        defaults: DynamicDefaults = DYNAMIC_DEFAULTS,
    ) -> List[NLPBatch]:
        """
        Generate one input batch per NLP sequence length.

        Sequence lengths determined by get_nlp_seq_lens(), clamped
        to model_max_seq_len if provided.

        Returns
        -------
        List[NLPBatch], one per seq length in the ladder.

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
            schemas = build_input_schemas(
                info, result, seq_len=seq_len, defaults=defaults
            )
            inputs = self.generate(schemas)
            batches.append(NLPBatch(
                seq_len=seq_len,
                inputs=inputs,
                schemas=schemas,
            ))

        return batches


# ============================================================
# REPORTING
# ============================================================


def describe_inputs(schemas: List[InputSchema]) -> str:
    """
    Human-readable input description.

    Honest description — does not pretend
    tensors are real images or spectrograms.
    """

    if not schemas:
        return "model has no inputs"

    task = schemas[0].task

    if task == TaskType.VISION:
        return "synthetic image tensor"

    if task == TaskType.NLP:
        return "synthetic token sequence"

    if task == TaskType.AUDIO:
        return "synthetic audio tensor"

    return "synthetic tensors"