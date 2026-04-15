"""
Output divergence checker for MLBuild accuracy verification.

Pipeline
--------
run_accuracy_check()
    ↓
validate_runner_compatibility()
    ↓
input generation (precomputed → dataset → task-aware synthetic)
    ↓
runner.predict()
    ↓
streaming metrics accumulators
    ↓
AccuracyResult

Design Principles
-----------------
- deterministic sampling
- strict structural validation
- streaming metric computation (no output accumulation)
- backend-agnostic accuracy layer — cross-format comparison supported
- consistent logging + failure formatting

Input generation precedence
---------------------------
1. precomputed_batch  — caller-supplied, used as-is
2. context.dataset_path — real inputs loaded from .npz / .npy
3. generate_batch_task_aware — task-specific synthetic sampling
   task_type precedence: CLI --task > build.task_type > None

KL/JS activation
----------------
KL and JS accumulators are only updated when task_type is a classifier
type ("vision" or "nlp"). For all other task types they remain inactive
and return None. This is a caller-level guard, not metric-level.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

from .config import AccuracyConfig, AccuracyResult, EvaluationContext
from .inputs import generate_batch_task_aware, load_dataset_batch
from .metrics import (
    CosineAccumulator,
    MAEAccumulator,
    MaxAEAccumulator,
    Top1Accumulator,
    RMSEAccumulator,
    KLDivAccumulator,
    JSDivAccumulator,
    PercentileAEAccumulator,
)

logger = logging.getLogger(__name__)

# Task types for which KL/JS divergence is meaningful
_CLASSIFIER_TASK_TYPES = frozenset({"vision", "nlp"})


# ============================================================
# Runner validation
# ============================================================

def _validate_runner_compatibility(baseline_runner, candidate_runner) -> list:
    """
    Ensure both runners expose compatible input specifications.

    For cross-format comparisons (e.g. CoreML vs TFLite), input names
    and shapes must still match — they represent the same model.
    Dtype mismatches are warned but not fatal, as CoreML and TFLite
    may report slightly different numeric types for the same logical input.
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
            logger.warning(
                "dtype mismatch for input '%s': %s vs %s — "
                "continuing (cross-format comparison)",
                b.name, b.dtype, c.dtype,
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

    Shape mismatch is a hard error even in cross-format mode —
    if the output shapes differ the models are not comparable.
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
    config: AccuracyConfig,
    context: EvaluationContext,
    precomputed_batch: Optional[dict] = None,
) -> AccuracyResult:
    """
    Compare outputs between two runners.

    Parameters
    ----------
    baseline_runner
        Runner exposing get_input_spec() and predict().
    candidate_runner
        Runner exposing get_input_spec() and predict().
    config
        Policy: thresholds, profile, sampling parameters.
    context
        Runtime metadata: build IDs, task_type, dataset_path, cross_format.
    precomputed_batch
        Optional pre-generated input batch. If provided, skips all
        input generation logic and uses this batch directly.
    """

    spec = _validate_runner_compatibility(
        baseline_runner,
        candidate_runner,
    )

    logger.info(
        "accuracy_check start | baseline=%s candidate=%s samples=%d seed=%d "
        "task_type=%s cross_format=%s",
        context.baseline_build_id,
        context.candidate_build_id,
        config.samples,
        config.seed,
        context.task_type,
        context.cross_format,
    )

    logger.debug(
        "input_spec=%s",
        [(s.name, s.shape, s.dtype) for s in spec],
    )

    # ============================================================
    # Input generation — precedence: precomputed > dataset > synthetic
    # ============================================================

    if precomputed_batch is not None:
        batch = precomputed_batch
        logger.debug("using precomputed_batch")

    elif context.dataset_path is not None:
        batch = load_dataset_batch(
            context.dataset_path,
            spec,
            config.samples,
        )
        logger.debug("loaded dataset from %s", context.dataset_path)

    else:
        rng = np.random.default_rng(config.seed)
        batch = generate_batch_task_aware(
            spec,
            rng,
            config.samples,
            context.task_type,
        )
        logger.debug(
            "generated task-aware batch | task_type=%s", context.task_type
        )

    # ============================================================
    # Accumulator setup
    # ============================================================

    cosine    = CosineAccumulator()
    mae       = MAEAccumulator()
    maxae     = MaxAEAccumulator()
    top1      = Top1Accumulator()
    rmse      = RMSEAccumulator()
    percentile = PercentileAEAccumulator()

    # KL/JS only active for classifier task types — set once, not per sample
    use_distribution_metrics = context.task_type in _CLASSIFIER_TASK_TYPES
    kl = KLDivAccumulator()
    js = JSDivAccumulator()

    first_structure_validated = False

    # ============================================================
    # Streaming inference loop
    # ============================================================

    for i in range(config.samples):

        sample = {k: v[i] for k, v in batch.items()}

        base_out = _validate_outputs(baseline_runner.predict(sample))
        cand_out = _validate_outputs(candidate_runner.predict(sample))

        if not first_structure_validated:
            _validate_output_structure(base_out, cand_out)
            first_structure_validated = True

        for k in base_out:

            b = base_out[k].reshape(-1)
            c = cand_out[k].reshape(-1)

            cosine.update(b, c)
            mae.update(b, c)
            maxae.update(b, c)
            rmse.update(b, c)
            percentile.update(b, c)
            top1.update(base_out[k], cand_out[k])

            if use_distribution_metrics:
                kl.update(b, c)
                js.update(b, c)

    # ============================================================
    # Finalize metrics
    # ============================================================

    cos_sim   = cosine.compute()
    mae_val   = mae.compute()
    maxae_val = maxae.compute()
    top1_val  = top1.compute()
    rmse_val  = rmse.compute()
    kl_val    = kl.compute()
    js_val    = js.compute()
    p50, p95, p99 = percentile.compute()

    logger.debug(
        "metrics | cosine=%.6f mae=%.6f maxae=%.6f rmse=%.6f "
        "top1=%s kl=%s js=%s p50=%.6f p95=%.6f p99=%.6f",
        cos_sim, mae_val, maxae_val, rmse_val,
        None if top1_val is None else f"{top1_val:.4f}",
        None if kl_val   is None else f"{kl_val:.6f}",
        None if js_val   is None else f"{js_val:.6f}",
        p50, p95, p99,
    )

    # ============================================================
    # Gating
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

    if config.rmse_threshold is not None and rmse_val > config.rmse_threshold:
        failures.append(
            f"rmse {rmse_val:.6f} "
            f"> threshold {config.rmse_threshold:.6f}"
        )

    if top1_val is not None and top1_val < config.top1_threshold:
        failures.append(
            f"top1_agreement {top1_val:.4f} "
            f"< threshold {config.top1_threshold:.4f}"
        )

    logger.info(
        "accuracy_check complete | baseline=%s candidate=%s failures=%d",
        context.baseline_build_id,
        context.candidate_build_id,
        len(failures),
    )

    return AccuracyResult(
        baseline_build_id=context.baseline_build_id,
        candidate_build_id=context.candidate_build_id,
        cosine_similarity=cos_sim,
        mean_abs_error=mae_val,
        max_abs_error=maxae_val,
        rmse=rmse_val,
        top1_agreement=top1_val,
        kl_divergence=kl_val,
        js_divergence=js_val,
        error_p50=p50,
        error_p95=p95,
        error_p99=p99,
        num_samples=config.samples,
        seed=config.seed,
        failure_reasons=tuple(failures),
    )