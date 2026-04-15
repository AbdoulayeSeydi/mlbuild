"""
Dtype-aware synthetic input generation for accuracy checks.

Design principles
-----------------
- Deterministic: caller provides seeded numpy Generator
- Strict validation: invalid shapes / dtypes fail early
- Vectorized generation for performance
- Clear semantics: "samples" means batch dimension

Task-aware sampling
-------------------
generate_batch_task_aware() selects a sampling strategy based on task_type:

    vision   — float inputs near decision boundary [0.4, 0.6]
    nlp      — integer tokens from vocabulary extremes (top/bottom 10%)
    audio    — float inputs from Laplace distribution (heavy-tailed)
    other    — falls back to generate_batch() uniform sampling

Dataset loading
---------------
load_dataset_batch() loads real inputs from .npz or .npy files.
Keys must match InputSpec names. Shapes must be compatible.
Sliced to `samples` rows — errors if file has fewer rows.

Notes
-----
Callers should generate inputs once and reuse them across
model variants during explore runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import numpy as np


# ============================================================
# Constants
# ============================================================

DEFAULT_INT_RANGE = 10_000

# Fraction of int_range used for vocabulary boundary sampling (nlp)
_NLP_BOUNDARY_FRACTION = 0.10


# ============================================================
# InputSpec — descriptor for one model input tensor
# ============================================================

@dataclass(frozen=True)
class InputSpec:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype

    def __post_init__(self) -> None:

        if not self.name:
            raise ValueError("InputSpec.name must not be empty")

        if not self.shape:
            raise ValueError(f"InputSpec '{self.name}': shape must not be empty")

        for dim in self.shape:
            if not isinstance(dim, int):
                raise TypeError(
                    f"InputSpec '{self.name}': shape dims must be ints, got {type(dim)}"
                )
            if dim <= 0:
                raise ValueError(
                    f"InputSpec '{self.name}': shape dims must be ≥ 1, got {dim}"
                )

        object.__setattr__(self, "dtype", np.dtype(self.dtype))


# ============================================================
# Spec validation
# ============================================================

def _validate_unique_names(specs: Iterable[InputSpec]) -> None:

    seen: set[str] = set()

    for s in specs:
        if s.name in seen:
            raise ValueError(f"Duplicate InputSpec name detected: '{s.name}'")
        seen.add(s.name)


# ============================================================
# Single batch generation (uniform — unchanged)
# ============================================================

def generate_batch(
    specs: list[InputSpec],
    rng: np.random.Generator,
    samples: int,
    *,
    int_range: int = DEFAULT_INT_RANGE,
) -> dict[str, np.ndarray]:
    """
    Generate a batch of synthetic inputs using uniform sampling.

    Parameters
    ----------
    specs
        InputSpec descriptors.
    rng
        Seeded numpy random generator.
    samples
        Number of batch samples.
    int_range
        Upper bound for integer sampling [0, int_range).

    Returns
    -------
    dict[str, np.ndarray]

        Example:
        {
            "input_ids": (samples, seq_len),
            "attention_mask": (samples, seq_len)
        }
    """

    if samples <= 0:
        raise ValueError("samples must be > 0")

    if int_range <= 0:
        raise ValueError("int_range must be > 0")

    _validate_unique_names(specs)

    result: dict[str, np.ndarray] = {}

    for s in specs:

        batch_shape = (samples, *s.shape)
        dtype = s.dtype

        if np.issubdtype(dtype, np.floating):
            arr = rng.uniform(-1.0, 1.0, size=batch_shape).astype(dtype)

        elif np.issubdtype(dtype, np.integer):
            arr = rng.integers(0, int_range, size=batch_shape, dtype=dtype)

        elif np.issubdtype(dtype, np.bool_):
            arr = rng.integers(0, 2, size=batch_shape, dtype=np.bool_)

        else:
            raise ValueError(
                f"Input '{s.name}': unsupported dtype {dtype}. "
                "Supported: floating, integer, bool."
            )

        result[s.name] = arr

    return result


# ============================================================
# Task-aware batch generation
# ============================================================

def generate_batch_task_aware(
    specs: list[InputSpec],
    rng: np.random.Generator,
    samples: int,
    task_type: Optional[str],
    *,
    int_range: int = DEFAULT_INT_RANGE,
) -> dict[str, np.ndarray]:
    """
    Generate a batch of synthetic inputs using a task-appropriate distribution.

    Sampling strategy per task_type
    --------------------------------
    vision
        Float inputs sampled from [0.4, 0.6] — near decision boundary.
        Exercises classifier sensitivity where small differences matter most.

    nlp
        Integer tokens sampled from the top and bottom 10% of int_range.
        Exercises vocabulary boundary tokens (special tokens, rare tokens)
        which often expose quantization sensitivity.

    audio
        Float inputs sampled from Laplace(0, 0.3) — heavy-tailed.
        Exercises model behaviour on signal spikes and silence boundaries.

    None / "unknown" / anything else
        Falls back to generate_batch() uniform sampling.

    Parameters
    ----------
    specs
        InputSpec descriptors.
    rng
        Seeded numpy random generator.
    samples
        Number of batch samples.
    task_type
        Task type string. See above for valid values.
    int_range
        Upper bound for integer sampling [0, int_range). Used by nlp path.
    """

    if task_type == "vision":
        return _generate_vision_batch(specs, rng, samples)

    if task_type == "nlp":
        return _generate_nlp_batch(specs, rng, samples, int_range=int_range)

    if task_type == "audio":
        return _generate_audio_batch(specs, rng, samples)

    # multimodal, unknown, None — uniform fallback
    return generate_batch(specs, rng, samples, int_range=int_range)


def _generate_vision_batch(
    specs: list[InputSpec],
    rng: np.random.Generator,
    samples: int,
) -> dict[str, np.ndarray]:
    """
    Float inputs near decision boundary [0.4, 0.6].
    Integer / bool inputs use uniform fallback — boundary semantics
    don't apply to discrete token inputs.
    """
    _validate_unique_names(specs)
    result: dict[str, np.ndarray] = {}

    for s in specs:
        batch_shape = (samples, *s.shape)
        dtype = s.dtype

        if np.issubdtype(dtype, np.floating):
            arr = rng.uniform(0.4, 0.6, size=batch_shape).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            arr = rng.integers(0, DEFAULT_INT_RANGE, size=batch_shape, dtype=dtype)
        elif np.issubdtype(dtype, np.bool_):
            arr = rng.integers(0, 2, size=batch_shape, dtype=np.bool_)
        else:
            raise ValueError(
                f"Input '{s.name}': unsupported dtype {dtype}."
            )

        result[s.name] = arr

    return result


def _generate_nlp_batch(
    specs: list[InputSpec],
    rng: np.random.Generator,
    samples: int,
    *,
    int_range: int = DEFAULT_INT_RANGE,
) -> dict[str, np.ndarray]:
    """
    Integer tokens sampled from vocabulary boundaries.

    For each sample, randomly selects either the bottom 10% or top 10%
    of [0, int_range) — exercising special tokens and rare high-index tokens.
    Float inputs fall back to uniform sampling.
    """
    _validate_unique_names(specs)
    result: dict[str, np.ndarray] = {}

    boundary = max(1, int(int_range * _NLP_BOUNDARY_FRACTION))
    # low band:  [0, boundary)
    # high band: [int_range - boundary, int_range)

    for s in specs:
        batch_shape = (samples, *s.shape)
        dtype = s.dtype

        if np.issubdtype(dtype, np.integer):
            # For each element, flip a coin: low band or high band
            use_high = rng.integers(0, 2, size=batch_shape).astype(bool)
            low  = rng.integers(0, boundary, size=batch_shape, dtype=dtype)
            high = rng.integers(int_range - boundary, int_range, size=batch_shape, dtype=dtype)
            arr  = np.where(use_high, high, low)
        elif np.issubdtype(dtype, np.floating):
            arr = rng.uniform(-1.0, 1.0, size=batch_shape).astype(dtype)
        elif np.issubdtype(dtype, np.bool_):
            arr = rng.integers(0, 2, size=batch_shape, dtype=np.bool_)
        else:
            raise ValueError(
                f"Input '{s.name}': unsupported dtype {dtype}."
            )

        result[s.name] = arr

    return result


def _generate_audio_batch(
    specs: list[InputSpec],
    rng: np.random.Generator,
    samples: int,
) -> dict[str, np.ndarray]:
    """
    Float inputs from Laplace(0, 0.3) — heavy-tailed distribution.
    Exercises model behaviour on signal spikes and silence boundaries.
    Integer / bool inputs use uniform fallback.
    """
    _validate_unique_names(specs)
    result: dict[str, np.ndarray] = {}

    for s in specs:
        batch_shape = (samples, *s.shape)
        dtype = s.dtype

        if np.issubdtype(dtype, np.floating):
            arr = rng.laplace(loc=0.0, scale=0.3, size=batch_shape).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            arr = rng.integers(0, DEFAULT_INT_RANGE, size=batch_shape, dtype=dtype)
        elif np.issubdtype(dtype, np.bool_):
            arr = rng.integers(0, 2, size=batch_shape, dtype=np.bool_)
        else:
            raise ValueError(
                f"Input '{s.name}': unsupported dtype {dtype}."
            )

        result[s.name] = arr

    return result


# ============================================================
# Dataset loader
# ============================================================

def load_dataset_batch(
    path: str | Path,
    specs: list[InputSpec],
    samples: int,
) -> dict[str, np.ndarray]:
    """
    Load real inputs from a .npz or .npy file.

    .npz (multi-input models)
        Keys must match InputSpec names exactly.
        Each array must have shape (N, *spec.shape) where N >= samples.

    .npy (single-input models only)
        Loaded as a single array assigned to the only spec's name.
        Requires exactly one InputSpec.

    Parameters
    ----------
    path
        Path to .npz or .npy file.
    specs
        InputSpec descriptors. Used to validate keys and shapes.
    samples
        Number of rows to slice from the dataset.
        Raises ValueError if the file has fewer rows than requested.

    Returns
    -------
    dict[str, np.ndarray]
        Same format as generate_batch(). Shape: (samples, *spec.shape).
    """

    path = Path(path)

    if not path.exists():
        raise ValueError(f"Dataset file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".npz":
        return _load_npz(path, specs, samples)
    elif suffix == ".npy":
        return _load_npy(path, specs, samples)
    else:
        raise ValueError(
            f"Unsupported dataset format '{suffix}'. "
            "Supported: .npz (multi-input), .npy (single-input)."
        )


def _load_npz(
    path: Path,
    specs: list[InputSpec],
    samples: int,
) -> dict[str, np.ndarray]:

    data = np.load(path, allow_pickle=False)
    spec_names = {s.name for s in specs}
    file_keys  = set(data.files)

    missing = spec_names - file_keys
    if missing:
        raise ValueError(
            f"Dataset missing keys for inputs: {sorted(missing)}. "
            f"File contains: {sorted(file_keys)}"
        )

    result: dict[str, np.ndarray] = {}

    for s in specs:
        arr = data[s.name]

        if arr.ndim < 1:
            raise ValueError(
                f"Dataset key '{s.name}': expected at least 1 dimension, got scalar."
            )

        n_rows = arr.shape[0]
        if n_rows < samples:
            raise ValueError(
                f"Dataset key '{s.name}' has {n_rows} rows "
                f"but {samples} samples requested."
            )

        sliced = arr[:samples]

        # Validate trailing shape matches spec
        expected_shape = (samples, *s.shape)
        if sliced.shape != expected_shape:
            raise ValueError(
                f"Dataset key '{s.name}': expected shape {expected_shape}, "
                f"got {sliced.shape} after slicing."
            )

        result[s.name] = sliced.astype(s.dtype)

    return result


def _load_npy(
    path: Path,
    specs: list[InputSpec],
    samples: int,
) -> dict[str, np.ndarray]:

    if len(specs) != 1:
        raise ValueError(
            f".npy format supports single-input models only. "
            f"Got {len(specs)} InputSpecs. Use .npz for multi-input models."
        )

    s   = specs[0]
    arr = np.load(path, allow_pickle=False)

    if arr.ndim < 1:
        raise ValueError(
            f"Dataset file '{path}': expected at least 1 dimension, got scalar."
        )

    n_rows = arr.shape[0]
    if n_rows < samples:
        raise ValueError(
            f"Dataset file '{path}' has {n_rows} rows "
            f"but {samples} samples requested."
        )

    sliced = arr[:samples]

    expected_shape = (samples, *s.shape)
    if sliced.shape != expected_shape:
        raise ValueError(
            f"Dataset file '{path}': expected shape {expected_shape}, "
            f"got {sliced.shape} after slicing."
        )

    return {s.name: sliced.astype(s.dtype)}


# ============================================================
# Sample generator (iterator form — unchanged)
# ============================================================

def generate_samples(
    specs: list[InputSpec],
    rng: np.random.Generator,
    n: int,
    *,
    int_range: int = DEFAULT_INT_RANGE,
):
    """
    Yield n individual input dictionaries.

    This is useful when models do not support batched inference.
    """

    batch = generate_batch(specs, rng, n, int_range=int_range)

    for i in range(n):
        yield {k: v[i] for k, v in batch.items()}