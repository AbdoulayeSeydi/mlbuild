"""
Accuracy metrics for output divergence detection.

Metrics
-------
cosine_similarity   Primary gate. Scale-invariant direction preservation.
mean_abs_error      Diagnostic. Distributed error signal.
max_abs_error       Diagnostic. Outlier detector. Never gates.
rmse                Optional gate. Penalizes outliers more than MAE.
top1_agreement      Conditional gate. For classification logits [batch, N], N > 1.
kl_divergence       Diagnostic. KL(baseline || candidate). Classifiers only. Never gates.
js_divergence       Diagnostic. Jensen-Shannon divergence. Bounded [0, ln2]. Never gates.
error_p50/p95/p99   Diagnostic. Percentile breakdown of absolute error. Never gates.

Design principles
-----------------
- Deterministic output ordering
- Strict structural validation
- Flatten outputs once
- Vectorized metric computation
- NaN / Inf protection

KL/JS stability contract
------------------------
- Inputs are softmax-normalized via log-sum-exp before KL/JS computation.
- Distributions are epsilon-clipped to [1e-10, 1] after normalization.
- KL direction is fixed: KL(baseline || candidate). Never reversed.
- JS divergence is symmetric: 0.5*KL(p||m) + 0.5*KL(q||m), m = 0.5*(p+q).
- Both return None if never activated (non-classifier outputs).
"""

from __future__ import annotations

from typing import Optional
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
# KL/JS shared helpers
# ============================================================

_KL_EPS = 1e-10


def _log_softmax_to_prob(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax via log-sum-exp, then epsilon-clipped.
    Input x is a 1D float array (logits or raw scores).
    Returns a valid probability distribution summing to ~1.
    """
    x = x.astype(np.float64)
    shifted = x - np.max(x)                        # log-sum-exp stability
    probs = np.exp(shifted) / np.sum(np.exp(shifted))
    return np.clip(probs, _KL_EPS, 1.0)


def _kl_div(p: np.ndarray, q: np.ndarray) -> float:
    """
    KL(p || q). Both inputs must already be valid probability distributions.
    Direction is fixed: baseline=p, candidate=q. Never reversed.
    """
    return float(np.sum(p * np.log(p / q)))


def _js_div(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen-Shannon divergence. Symmetric. Bounded [0, ln2].
    """
    m = 0.5 * (p + q)
    return float(0.5 * _kl_div(p, m) + 0.5 * _kl_div(q, m))


# ============================================================
# Cosine similarity
# ============================================================

def cosine_similarity(
    baseline_outputs: list[dict[str, np.ndarray]],
    candidate_outputs: list[dict[str, np.ndarray]],
) -> float:

    baseline, candidate, _ = _flatten_outputs(baseline_outputs, candidate_outputs)

    dot    = np.sum(baseline * candidate, axis=1)
    norm_b = np.linalg.norm(baseline, axis=1)
    norm_c = np.linalg.norm(candidate, axis=1)
    denom  = norm_b * norm_c
    valid  = denom > 1e-12

    if not np.any(valid):
        return 1.0

    return float(np.mean(dot[valid] / denom[valid]))


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
    Returns key name if logits detected, None otherwise.
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
    Returns value in [0, 1], or None if not a classifier output.
    """
    key = _detect_logits(baseline_outputs, candidate_outputs)
    if key is None:
        return None

    b = np.stack([o[key] for o in baseline_outputs])
    c = np.stack([o[key] for o in candidate_outputs])

    b = b.reshape(b.shape[0], -1)
    c = c.reshape(c.shape[0], -1)

    return float(np.mean(np.argmax(b, axis=1) == np.argmax(c, axis=1)))


# ============================================================
# Streaming accumulators (used by checker.py)
# ============================================================

class CosineAccumulator:
    def __init__(self):
        self._dots   = []
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
    Only active when output is 1D with more than 1 element.
    Returns None if never activated.
    """
    def __init__(self):
        self._matches = 0
        self._total   = 0
        self._active  = False

    def update(self, b: np.ndarray, c: np.ndarray) -> None:
        if b.ndim == 1 and b.shape[0] > 1:
            self._active   = True
            self._matches += int(np.argmax(b) == np.argmax(c))
            self._total   += 1

    def compute(self) -> float | None:
        if not self._active or self._total == 0:
            return None
        return self._matches / self._total


class RMSEAccumulator:
    """
    Root mean squared error across all output elements.
    Optional gate via config.rmse_threshold.
    """
    def __init__(self):
        self._sq_total = 0.0
        self._count    = 0

    def update(self, b: np.ndarray, c: np.ndarray) -> None:
        diff = (b.astype(np.float64) - c.astype(np.float64))
        self._sq_total += float(np.sum(diff ** 2))
        self._count    += b.size

    def compute(self) -> float:
        if self._count == 0:
            return 0.0
        return float(np.sqrt(self._sq_total / self._count))


class KLDivAccumulator:
    """
    KL divergence: KL(baseline || candidate). Direction fixed, never reversed.

    Only meaningful for classifier outputs (probability vectors).
    Caller is responsible for activating only when task_type is a classifier.

    Numerical stability contract:
    - Inputs softmax-normalized via log-sum-exp internally.
    - Distributions epsilon-clipped to [1e-10, 1] after normalization.
    - Operates in float64 throughout.

    Returns None if never updated.
    """
    def __init__(self):
        self._values: list[float] = []

    def update(self, b: np.ndarray, c: np.ndarray) -> None:
        p = _log_softmax_to_prob(b.reshape(-1))
        q = _log_softmax_to_prob(c.reshape(-1))
        self._values.append(_kl_div(p, q))

    def compute(self) -> float | None:
        if not self._values:
            return None
        return float(np.mean(self._values))


class JSDivAccumulator:
    """
    Jensen-Shannon divergence. Symmetric. Bounded [0, ln2 ≈ 0.693].

    Same activation contract as KLDivAccumulator — caller guards on task_type.
    Same numerical stability contract as KLDivAccumulator.

    Returns None if never updated.
    """
    def __init__(self):
        self._values: list[float] = []

    def update(self, b: np.ndarray, c: np.ndarray) -> None:
        p = _log_softmax_to_prob(b.reshape(-1))
        q = _log_softmax_to_prob(c.reshape(-1))
        self._values.append(_js_div(p, q))

    def compute(self) -> float | None:
        if not self._values:
            return None
        return float(np.mean(self._values))


class PercentileAEAccumulator:
    """
    Percentile breakdown of absolute error: p50, p95, p99.

    Stores all absolute error values across all updates.
    Memory justified: 32 samples * typical output width stays well
    under 1MB. Not intended for sample counts above ~10k.

    Returns (p50, p95, p99) tuple. Never gates CI.
    """
    def __init__(self):
        self._errors: list[np.ndarray] = []

    def update(self, b: np.ndarray, c: np.ndarray) -> None:
        self._errors.append(np.abs(b.astype(np.float64) - c.astype(np.float64)).reshape(-1))

    def compute(self) -> tuple[float, float, float]:
        if not self._errors:
            return (0.0, 0.0, 0.0)
        all_errors = np.concatenate(self._errors)
        return (
            float(np.percentile(all_errors, 50)),
            float(np.percentile(all_errors, 95)),
            float(np.percentile(all_errors, 99)),
        )