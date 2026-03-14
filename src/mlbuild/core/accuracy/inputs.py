"""
Dtype-aware synthetic input generation for accuracy checks.

Design principles
-----------------
• Deterministic: caller provides seeded numpy Generator
• Strict validation: invalid shapes / dtypes fail early
• Vectorized generation for performance
• Clear semantics: "samples" means batch dimension

Notes
-----
Callers should generate inputs once and reuse them across
model variants during explore runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np


# ============================================================
# Constants
# ============================================================

DEFAULT_INT_RANGE = 10_000


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

        # normalize dtype
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
# Single batch generation
# ============================================================

def generate_batch(
    specs: list[InputSpec],
    rng: np.random.Generator,
    samples: int,
    *,
    int_range: int = DEFAULT_INT_RANGE,
) -> dict[str, np.ndarray]:
    """
    Generate a batch of synthetic inputs.

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

        # Floating types
        if np.issubdtype(dtype, np.floating):

            arr = rng.uniform(
                -1.0,
                1.0,
                size=batch_shape,
            ).astype(dtype)

        # Integer types
        elif np.issubdtype(dtype, np.integer):

            arr = rng.integers(
                0,
                int_range,
                size=batch_shape,
                dtype=dtype,
            )

        # Boolean
        elif np.issubdtype(dtype, np.bool_):

            arr = rng.integers(
                0,
                2,
                size=batch_shape,
                dtype=np.bool_,
            )

        else:
            raise ValueError(
                f"Input '{s.name}': unsupported dtype {dtype}. "
                "Supported: floating, integer, bool."
            )

        result[s.name] = arr

    return result


# ============================================================
# Sample generator (iterator form)
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