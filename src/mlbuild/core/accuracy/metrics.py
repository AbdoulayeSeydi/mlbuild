"""
Accuracy metrics for output divergence detection.

Metrics
-------
cosine_similarity   Primary gate. Scale-invariant direction preservation.
mean_abs_error      Diagnostic. Distributed error signal.
max_abs_error       Diagnostic. Outlier detector. Never gates.
top1_agreement      Conditional gate. For classification logits [batch, N], N > 1.

Design principles
-----------------
• Deterministic output ordering
• Strict structural validation
• Flatten outputs once
• Vectorized metric computation
• NaN / Inf protection
"""

from __future__ import annotations

from typing import Optional, Iterable
import numpy as np


# ============================================================
# Validation
# ============================================================

def _validate_outputs(
    baseline: list[dict[str, np.ndarray]],
    candidate: list[dict[str, np.ndarray]],
) -> list[str]:
    """
    Validate output structure and return sorted output keys.
    """

    if not baseline or not candidate:
        raise ValueError("Outputs must not be empty.")

    if len(baseline) != len(candidate):
        raise ValueError(
            f"Sample count mismatch: baseline={len(baseline)} candidate={len(candidate)}"
        )

    base_keys = set(baseline[0].keys())
    cand_keys = set(candidate[0].keys())

    if base_keys != cand_keys:
        raise ValueError(
            f"Output keys differ. baseline={base_keys}, candidate={cand_keys}"
        )

    keys = sorted(base_keys)

    for i, (b, c) in enumerate(zip(baseline, candidate)):
        if set(b.keys()) != base_keys:
            raise ValueError(f"Baseline sample {i} keys mismatch.")
        if set(c.keys()) != cand_keys:
            raise ValueError(f"Candidate sample {i} keys mismatch.")

        for k in keys:
            if b[k].shape != c[k].shape:
                raise ValueError(
                    f"Shape mismatch for key '{k}' at sample {i}: "
                    f"{b[k].shape} vs {c[k].shape}"
                )

    return keys


# ============================================================
# Flatten outputs once
# ============================================================

def _flatten_outputs(
    baseline: list[dict[str, np.ndarray]],
    candidate: list[dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Flatten outputs into 2D matrices [samples, values].
    """

    keys = _validate_outputs(baseline, candidate)

    n_samples = len(baseline)

    # compute total feature width
    width = sum(int(np.prod(baseline[0][k].shape)) for k in keys)

    dtype = np.result_type(
        *[baseline[0][k].dtype for k in keys],
        *[candidate[0][k].dtype for k in keys],
        np.float32,
    )

    base = np.empty((n_samples, width), dtype=dtype)
    cand = np.empty((n_samples, width), dtype=dtype)

    for i, (b_out, c_out) in enumerate(zip(baseline, candidate)):
        offset = 0

        for k in keys:
            b = b_out[k].reshape(-1)
            c = c_out[k].reshape(-1)

            n = b.size

            base[i, offset : offset + n] = b
            cand[i, offset : offset + n] = c

            offset += n

    if not np.isfinite(base).all() or not np.isfinite(cand).all():
        raise ValueError("Outputs contain NaN or Inf values.")

    return base, cand, keys


# ============================================================
# Cosine similarity
# ============================================================

def cosine_similarity(
    baseline_outputs: list[dict[str, np.ndarray]],
    candidate_outputs: list[dict[str, np.ndarray]],
) -> float:

    baseline, candidate, _ = _flatten_outputs(baseline_outputs, candidate_outputs)

    dot = np.sum(baseline * candidate, axis=1)

    norm_b = np.linalg.norm(baseline, axis=1)
    norm_c = np.linalg.norm(candidate, axis=1)

    denom = norm_b * norm_c

    valid = denom > 1e-12

    if not np.any(valid):
        # both vectors zero → perfect agreement
        return 1.0

    sims = dot[valid] / denom[valid]

    return float(np.mean(sims))


# ============================================================
# Mean absolute error
# ============================================================

def mean_abs_error(
    baseline_outputs: list[dict[str, np.ndarray]],
    candidate_outputs: list[dict[str, np.ndarray]],
) -> float:

    baseline, candidate, _ = _flatten_outputs(baseline_outputs, candidate_outputs)

    return float(np.mean(np.abs(baseline - candidate)))


# ============================================================
# Max absolute error
# ============================================================

def max_abs_error(
    baseline_outputs: list[dict[str, np.ndarray]],
    candidate_outputs: list[dict[str, np.ndarray]],
) -> float:

    baseline, candidate, _ = _flatten_outputs(baseline_outputs, candidate_outputs)

    return float(np.max(np.abs(baseline - candidate)))


# ============================================================
# Top-1 agreement
# ============================================================

def _detect_logits(
    baseline: list[dict[str, np.ndarray]],
    candidate: list[dict[str, np.ndarray]],
) -> Optional[str]:
    """
    Detect classification logits output key.

    Returns key name if logits detected.
    """

    if not baseline:
        return None

    if len(baseline[0]) != 1:
        return None

    key = next(iter(baseline[0]))

    arr = baseline[0][key]

    if arr.ndim != 2:
        return None

    if arr.shape[1] <= 1:
        return None

    return key


def top1_agreement(
    baseline_outputs: list[dict[str, np.ndarray]],
    candidate_outputs: list[dict[str, np.ndarray]],
) -> Optional[float]:
    """
    Fraction of samples where argmax matches.
    Returns value in [0,1].
    """

    key = _detect_logits(baseline_outputs, candidate_outputs)

    if key is None:
        return None

    b = np.stack([o[key] for o in baseline_outputs])
    c = np.stack([o[key] for o in candidate_outputs])

    b = b.reshape(b.shape[0], -1)
    c = c.reshape(c.shape[0], -1)

    matches = np.argmax(b, axis=1) == np.argmax(c, axis=1)

    return float(np.mean(matches))


# ============================================================
# Streaming accumulators (used by checker.py)
# ============================================================

class CosineAccumulator:
    def __init__(self):
        self._dots = []
        self._norm_b = []
        self._norm_c = []

    def update(self, b: np.ndarray, c: np.ndarray) -> None:
        self._dots.append(float(np.dot(b, c)))
        self._norm_b.append(float(np.linalg.norm(b)))
        self._norm_c.append(float(np.linalg.norm(c)))

    def compute(self) -> float:
        dots   = np.array(self._dots)
        norm_b = np.array(self._norm_b)
        norm_c = np.array(self._norm_c)
        denom  = norm_b * norm_c
        valid  = denom > 1e-12
        if not np.any(valid):
            return 1.0
        return float(np.mean(dots[valid] / denom[valid]))


class MAEAccumulator:
    def __init__(self):
        self._total = 0.0
        self._count = 0

    def update(self, b: np.ndarray, c: np.ndarray) -> None:
        self._total += float(np.sum(np.abs(b - c)))
        self._count += b.size

    def compute(self) -> float:
        return self._total / self._count if self._count > 0 else 0.0


class MaxAEAccumulator:
    def __init__(self):
        self._max = 0.0

    def update(self, b: np.ndarray, c: np.ndarray) -> None:
        self._max = max(self._max, float(np.max(np.abs(b - c))))

    def compute(self) -> float:
        return self._max


class Top1Accumulator:
    """
    Only active when output is 2D [1, N] with N > 1.
    Returns None if never activated.
    """
    def __init__(self):
        self._matches = 0
        self._total   = 0
        self._active  = False

    def update(self, b: np.ndarray, c: np.ndarray) -> None:
        # b/c here are the raw per-output arrays (already batch-stripped by _validate_outputs)
        if b.ndim == 1 and b.shape[0] > 1:
            self._active = True
            self._matches += int(np.argmax(b) == np.argmax(c))
            self._total   += 1

    def compute(self) -> float | None:
        if not self._active or self._total == 0:
            return None
        return self._matches / self._total