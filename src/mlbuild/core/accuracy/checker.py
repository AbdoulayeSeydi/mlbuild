"""
Output divergence checker for MLBuild accuracy verification.

Pipeline
--------
run_accuracy_check()
    ↓
validate_runner_compatibility()
    ↓
input generation (shared RNG)
    ↓
runner.predict()
    ↓
streaming metrics accumulators
    ↓
AccuracyResult

Design Principles
-----------------
• deterministic sampling
• strict structural validation
• streaming metric computation (no output accumulation)
• backend-agnostic accuracy layer
• consistent logging + failure formatting
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable

import numpy as np

from .config import AccuracyConfig, AccuracyResult
from .inputs import generate_batch
from .metrics import (
    CosineAccumulator,
    MAEAccumulator,
    MaxAEAccumulator,
    Top1Accumulator,
)

logger = logging.getLogger(__name__)


# ============================================================
# Runner validation
# ============================================================

def _validate_runner_compatibility(baseline_runner, candidate_runner) -> list:
    """
    Ensure both runners expose identical input specifications.
    """

    base_spec = baseline_runner.get_input_spec()
    cand_spec = candidate_runner.get_input_spec()

    if len(base_spec) != len(cand_spec):
        raise ValueError(
            "Runner input count mismatch: "
            f"{len(base_spec)} vs {len(cand_spec)}"
        )

    for b, c in zip(base_spec, cand_spec):

        if b.name != c.name:
            raise ValueError(
                f"Input name mismatch: {b.name} vs {c.name}"
            )

        if tuple(b.shape) != tuple(c.shape):
            raise ValueError(
                f"Shape mismatch for input '{b.name}': "
                f"{b.shape} vs {c.shape}"
            )

        if np.dtype(b.dtype) != np.dtype(c.dtype):
            raise ValueError(
                f"Dtype mismatch for input '{b.name}': "
                f"{b.dtype} vs {c.dtype}"
            )

        if len(b.shape) == 0:
            raise ValueError(
                f"Input '{b.name}' has invalid scalar shape"
            )

    return base_spec


# ============================================================
# Output validation
# ============================================================

def _validate_outputs(
    outputs: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Ensure runner outputs are valid and normalized.
    """

    if not isinstance(outputs, dict):
        raise ValueError("Runner.predict() must return dict[str, ndarray]")

    if not outputs:
        raise ValueError("Runner returned empty outputs")

    normalized = {}

    for k, v in outputs.items():

        if not isinstance(v, np.ndarray):
            raise ValueError(f"Output '{k}' is not ndarray")

        if v.ndim == 0:
            raise ValueError(f"Output '{k}' is scalar")

        # remove batch dim if present
        if v.shape[0] == 1:
            v = v[0]

        normalized[k] = v

    return normalized


# ============================================================
# Output structure validation
# ============================================================

def _validate_output_structure(
    baseline: Dict[str, np.ndarray],
    candidate: Dict[str, np.ndarray],
) -> None:
    """
    Ensure both runners return identical output structures.
    """

    if set(baseline.keys()) != set(candidate.keys()):
        raise ValueError(
            f"Output key mismatch: "
            f"{set(baseline.keys())} vs {set(candidate.keys())}"
        )

    for k in baseline:

        if baseline[k].shape != candidate[k].shape:
            raise ValueError(
                f"Output shape mismatch for '{k}': "
                f"{baseline[k].shape} vs {candidate[k].shape}"
            )


# ============================================================
# Accuracy check
# ============================================================

def run_accuracy_check(
    baseline_runner,
    candidate_runner,
    config: AccuracyConfig | None = None,
    baseline_build_id: str = "",
    candidate_build_id: str = "",
    precomputed_batch: dict | None = None,
) -> AccuracyResult:
    """
    Compare outputs between two runners.
    """

    if config is None:
        config = AccuracyConfig()

    spec = _validate_runner_compatibility(
        baseline_runner,
        candidate_runner,
    )

    logger.info(
        "accuracy_check start | baseline=%s candidate=%s samples=%d seed=%d",
        baseline_build_id,
        candidate_build_id,
        config.samples,
        config.seed,
    )

    logger.debug(
        "input_spec=%s",
        [(s.name, s.shape, s.dtype) for s in spec],
    )

    if precomputed_batch is not None:
        batch = precomputed_batch
    else:
        rng = np.random.default_rng(config.seed)
        batch = generate_batch(
            spec,
            rng,
            samples=config.samples,
        )

    cosine = CosineAccumulator()
    mae = MAEAccumulator()
    maxae = MaxAEAccumulator()
    top1 = Top1Accumulator()

    first_structure_validated = False

    # ============================================================
    # streaming inference loop
    # ============================================================

    for i in range(config.samples):

        sample = {k: v[i] for k, v in batch.items()}

        base_out = _validate_outputs(
            baseline_runner.predict(sample)
        )

        cand_out = _validate_outputs(
            candidate_runner.predict(sample)
        )

        if not first_structure_validated:

            _validate_output_structure(base_out, cand_out)

            first_structure_validated = True

        for k in base_out:

            b = base_out[k].reshape(-1)
            c = cand_out[k].reshape(-1)

            cosine.update(b, c)
            mae.update(b, c)
            maxae.update(b, c)
            top1.update(base_out[k], cand_out[k])

    # ============================================================
    # finalize metrics
    # ============================================================

    cos_sim = cosine.compute()
    mae_val = mae.compute()
    maxae_val = maxae.compute()
    top1_val = top1.compute()

    logger.debug(
        "metrics | cosine=%.6f mae=%.6f maxae=%.6f top1=%s",
        cos_sim,
        mae_val,
        maxae_val,
        None if top1_val is None else f"{top1_val:.4f}",
    )

    # ============================================================
    # gating
    # ============================================================

    failures: list[str] = []

    if cos_sim < config.cosine_threshold:

        failures.append(
            f"cosine_similarity {cos_sim:.6f} "
            f"< threshold {config.cosine_threshold:.6f}"
        )

    if config.mae_threshold is not None and mae_val > config.mae_threshold:

        failures.append(
            f"mean_abs_error {mae_val:.6f} "
            f"> threshold {config.mae_threshold:.6f}"
        )

    if top1_val is not None and top1_val < config.top1_threshold:

        failures.append(
            f"top1_agreement {top1_val:.4f} "
            f"< threshold {config.top1_threshold:.4f}"
        )

    logger.info(
        "accuracy_check complete | failures=%d",
        len(failures),
    )

    return AccuracyResult(
        baseline_build_id=baseline_build_id,
        candidate_build_id=candidate_build_id,
        cosine_similarity=cos_sim,
        mean_abs_error=mae_val,
        max_abs_error=maxae_val,
        top1_agreement=top1_val,
        num_samples=config.samples,
        seed=config.seed,
        failure_reasons=tuple(failures),
    )