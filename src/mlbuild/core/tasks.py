"""
MLBuild Task System

Task abstraction for:

- task typing
- task detection arbitration
- synthetic input generation
- benchmarking policies
- dynamic shape resolution
- output validation

The internal layout mirrors future module splits:

core/tasks
detection/signals
detection/arbitration
benchmark/policies
runtime/dynamic_shapes
synthetic/templates
synthetic/descriptions
validation/output_schema
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Iterable
import numpy as np


# =============================================================================
# core/tasks
# =============================================================================


class TaskType(str, Enum):
    """Top-level ML task domains."""

    VISION = "vision"
    NLP = "nlp"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"
    UNKNOWN = "unknown"

    @classmethod
    def from_str(cls, value: str) -> "TaskType":
        if not value:
            return cls.UNKNOWN
        return cls.__members__.get(value.upper(), cls.UNKNOWN)


class DetectionTier(str, Enum):
    """
    Reliability tier of task detection.
    """

    GRAPH_MATCH = "graph_match"
    NAME_HEURISTIC = "name_heuristic"
    SHAPE_HEURISTIC = "shape_heuristic"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TaskDetection:
    """
    Result of task detection.
    """

    tasks: set[TaskType]
    tier: DetectionTier

    def primary(self) -> TaskType:
        if not self.tasks:
            return TaskType.UNKNOWN
        return next(iter(self.tasks))

    def is_multitask(self) -> bool:
        return len(self.tasks) > 1


# =============================================================================
# detection/signals
# =============================================================================


class DetectionSignalType(str, Enum):
    """
    Sources of task detection evidence.
    """

    GRAPH_PATTERN = "graph_pattern"
    OP_DISTRIBUTION = "op_distribution"
    SHAPE_PATTERN = "shape_pattern"
    INPUT_NAME = "input_name"
    OUTPUT_NAME = "output_name"
    MODEL_METADATA = "model_metadata"


SIGNAL_WEIGHTS: dict[DetectionSignalType, float] = {

    DetectionSignalType.GRAPH_PATTERN: 1.0,
    DetectionSignalType.OP_DISTRIBUTION: 0.7,
    DetectionSignalType.MODEL_METADATA: 0.6,
    DetectionSignalType.SHAPE_PATTERN: 0.5,
    DetectionSignalType.INPUT_NAME: 0.4,
    DetectionSignalType.OUTPUT_NAME: 0.4,
}


@dataclass(frozen=True)
class DetectionSignal:
    """
    A single signal indicating a possible ML task.
    """

    task: TaskType
    signal_type: DetectionSignalType
    score: float
    evidence: str = ""

    def weighted_score(self) -> float:
        return self.score * SIGNAL_WEIGHTS[self.signal_type]


# =============================================================================
# detection/arbitration
# =============================================================================


@dataclass
class TaskArbitrationResult:
    """
    Result of multi-signal task arbitration.
    """

    tasks: set[TaskType]
    task_scores: dict[TaskType, float]
    tier: DetectionTier
    signals_used: list[DetectionSignal]

    def primary_task(self) -> TaskType:

        if not self.task_scores:
            return TaskType.UNKNOWN

        return max(self.task_scores.items(), key=lambda x: x[1])[0]


def determine_detection_tier(signals: list[DetectionSignal]) -> DetectionTier:

    types = {s.signal_type for s in signals}

    if DetectionSignalType.GRAPH_PATTERN in types:
        return DetectionTier.GRAPH_MATCH

    if DetectionSignalType.OP_DISTRIBUTION in types:
        return DetectionTier.NAME_HEURISTIC

    if DetectionSignalType.SHAPE_PATTERN in types:
        return DetectionTier.SHAPE_HEURISTIC

    return DetectionTier.UNKNOWN


def arbitrate_tasks(
    signals: list[DetectionSignal],
    threshold: float = 0.6,
) -> TaskArbitrationResult:
    """
    Determine model tasks from detection signals.
    """

    if not signals:
        return TaskArbitrationResult(
            tasks={TaskType.UNKNOWN},
            task_scores={},
            tier=DetectionTier.UNKNOWN,
            signals_used=[],
        )

    scores: dict[TaskType, float] = {}

    for signal in signals:

        weighted = signal.weighted_score()

        scores[signal.task] = scores.get(signal.task, 0.0) + weighted

    total = sum(scores.values())

    if total > 0:
        scores = {k: v / total for k, v in scores.items()}

    selected = {task for task, score in scores.items() if score >= threshold}

    if not selected:
        selected = {max(scores.items(), key=lambda x: x[1])[0]}

    tier = determine_detection_tier(signals)

    return TaskArbitrationResult(
        tasks=selected,
        task_scores=scores,
        tier=tier,
        signals_used=signals,
    )


# =============================================================================
# runtime/dynamic_shapes
# =============================================================================


@dataclass(frozen=True)
class DynamicDefaults:

    batch: int = 1

    image_size_candidates: tuple[int, ...] = (224, 256, 299, 384)

    seq_len_default: int = 128

    waveform_samples: int = 16000

    mel_bins: int = 80

    generic_dim: int = 64


DYNAMIC_DEFAULTS = DynamicDefaults()


def resolve_dynamic_dim(dim: int | None, fallback: int) -> int:

    if dim in (None, -1):
        return fallback

    return int(dim)


# =============================================================================
# benchmark/policies
# =============================================================================


DEFAULT_NLP_SEQ_LENS: list[int] = [16, 64, 128, 256]


def get_nlp_seq_lens(model_max: Optional[int]) -> list[int]:
    if model_max is None:
        return list(DEFAULT_NLP_SEQ_LENS)
    if model_max < DEFAULT_NLP_SEQ_LENS[0]:
        return [model_max]
    ladder = [s for s in DEFAULT_NLP_SEQ_LENS if s <= model_max]
    # Only append model_max if it's an intermediate point not already in the ladder
    if model_max not in ladder and model_max < DEFAULT_NLP_SEQ_LENS[-1]:
        ladder.append(model_max)
    return sorted(set(ladder))


# =============================================================================
# synthetic/templates
# =============================================================================


@dataclass(frozen=True)
class TaskInputTemplate:

    dtype: np.dtype
    value_range: tuple[float, float] = (0.0, 1.0)
    fill_value: Optional[float | int] = None
    description: str = ""


INPUT_TEMPLATES: dict[str, TaskInputTemplate] = {

    "image": TaskInputTemplate(
        dtype=np.float32,
        value_range=(0.0, 1.0),
        description="Normalized image tensor",
    ),

    "token_ids": TaskInputTemplate(
        dtype=np.int64,
        value_range=(0, 30000),
        description="Random token IDs",
    ),

    "attention_mask": TaskInputTemplate(
        dtype=np.int64,
        fill_value=1,
        description="Full attention mask",
    ),

    "token_type_ids": TaskInputTemplate(
        dtype=np.int64,
        fill_value=0,
        description="Single segment token types",
    ),

    "waveform": TaskInputTemplate(
        dtype=np.float32,
        value_range=(-1.0, 1.0),
        description="Normalized audio waveform",
    ),

    "spectrogram": TaskInputTemplate(
        dtype=np.float32,
        value_range=(-10.0, 0.0),
        description="Log-mel spectrogram",
    ),

    "zeros": TaskInputTemplate(
        dtype=np.float32,
        fill_value=0,
        description="Zero tensor fallback",
    ),
}


def generate_tensor(shape: tuple[int, ...], template: TaskInputTemplate) -> np.ndarray:

    if template.fill_value is not None:
        return np.full(shape, template.fill_value, dtype=template.dtype)

    low, high = template.value_range

    data = np.random.uniform(low, high, size=shape)

    return data.astype(template.dtype)


# =============================================================================
# synthetic/descriptions
# =============================================================================


def describe_tensor(shape: tuple[int, ...], dtype: np.dtype) -> str:

    dims = "×".join(map(str, shape))

    return f"{dims} tensor ({dtype})"


# =============================================================================
# validation/output_schema
# =============================================================================


Validator = Callable[[np.ndarray], bool]


def _softmax_like(x: np.ndarray) -> bool:

    if x.ndim < 2:
        return False

    probs = x.sum(axis=-1)

    return np.allclose(probs, 1.0, atol=1e-2)


def _unit_norm(x: np.ndarray) -> bool:

    norms = np.linalg.norm(x, axis=-1)

    return np.allclose(norms, 1.0, atol=1e-2)


def _bbox_format(x: np.ndarray) -> bool:

    return x.ndim >= 3 and x.shape[-1] >= 4


def _ctc_matrix(x: np.ndarray) -> bool:

    return x.ndim == 3


@dataclass(frozen=True)
class OutputSchema:

    name: str
    expected_rank: Optional[int]
    validator: Optional[Validator]
    description: str = ""

    def validate(self, output: np.ndarray) -> bool:

        if self.expected_rank is not None:
            if output.ndim != self.expected_rank:
                return False

        if self.validator:
            return self.validator(output)

        return True


OUTPUT_SCHEMAS: dict[TaskType, list[OutputSchema]] = {

    TaskType.VISION: [

        OutputSchema(
            name="classification_logits",
            expected_rank=2,
            validator=None,
            description="[B, num_classes]",
        ),

        OutputSchema(
            name="embedding",
            expected_rank=2,
            validator=_unit_norm,
            description="[B, hidden_dim]",
        ),

        OutputSchema(
            name="detection_boxes",
            expected_rank=3,
            validator=_bbox_format,
            description="[B, N, 4+] boxes",
        ),
    ],

    TaskType.NLP: [

        OutputSchema(
            name="token_logits",
            expected_rank=3,
            validator=None,
            description="[B, seq_len, vocab]",
        ),

        OutputSchema(
            name="sentence_embedding",
            expected_rank=2,
            validator=_unit_norm,
            description="[B, hidden_dim]",
        ),
    ],

    TaskType.AUDIO: [

        OutputSchema(
            name="ctc_logits",
            expected_rank=3,
            validator=_ctc_matrix,
            description="[B, T, vocab]",
        ),

        OutputSchema(
            name="classification_logits",
            expected_rank=2,
            validator=None,
            description="[B, num_classes]",
        ),
    ],

    TaskType.MULTIMODAL: [],
    TaskType.UNKNOWN: [],
}


def validate_output(output: np.ndarray, task: TaskType) -> bool:

    schemas = OUTPUT_SCHEMAS.get(task, [])

    for schema in schemas:
        if schema.validate(output):
            return True

    return False


# =============================================================================
# helper utilities
# =============================================================================


def infer_primary_task(tasks: Iterable[TaskType]) -> TaskType:

    tasks = set(tasks)

    if not tasks:
        return TaskType.UNKNOWN

    if len(tasks) == 1:
        return next(iter(tasks))

    if TaskType.VISION in tasks:
        return TaskType.VISION

    if TaskType.NLP in tasks:
        return TaskType.NLP

    return next(iter(tasks))