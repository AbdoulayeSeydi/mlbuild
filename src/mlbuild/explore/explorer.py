"""
MLBuild explore — sweep backends and methods from a raw ONNX file.

Architecture
------------
explore()
    ↓
resolve_or_build()       ← Step 4
    ↓
resolve_or_optimize()    ← Step 5
    ↓
benchmark_all()          ← Step 6
    ↓
assign_verdicts()        ← Step 7
    ↓
ExploreResult
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from ..core.hash import compute_source_hash
from ..core.types import Build
from ..registry.local import LocalRegistry
from ..cli.commands.build import run_build
from ..core.types import Build, VariantResult, BackendResult, ExploreResult

logger = logging.getLogger(__name__)


# ============================================================
# Public types
# ============================================================

Backend = Literal["coreml", "tflite"]


# ============================================================
# Backend registry
# ============================================================

SUPPORTED_BACKENDS: set[str] = {"coreml", "tflite"}


# ============================================================
# resolve_or_build
# ============================================================

def resolve_or_build(
    onnx_path: Path,
    target: str,
    name: str,
    backend: Backend,
    registry: LocalRegistry,
) -> Build:
    """
    Resolve an existing baseline build or create one.

    Identity key:
        (source_hash, format, target_device, parent_build_id IS NULL)

    Name is NOT part of identity.

    Parameters
    ----------
    onnx_path
        Path to source ONNX model.
    target
        Target device identifier.
    name
        Logical build name.
    backend
        Model runtime format (coreml, tflite).
    registry
        Build registry.

    Returns
    -------
    Build
        Baseline build artifact.
    """

    _validate_inputs(onnx_path, backend)

    source_hash = compute_source_hash(onnx_path)

    existing = registry.find_baseline(
        source_hash=source_hash,
        format=backend,
        target_device=target,
    )

    if existing is not None:
        logger.info(
            "reusing baseline build=%s backend=%s target=%s",
            existing.build_id[:16],
            backend,
            target,
        )
        return existing

    logger.info(
        "building baseline backend=%s target=%s source=%s",
        backend,
        target,
        onnx_path.name,
    )

    build = run_build(
        model_path=onnx_path,
        target=target,
        name=name,
        quantization="fp32",
        format=backend,
        registry=registry,
    )

    return build


from ..optimize.optimizer import optimize, OptimizeError


def resolve_or_optimize(
    baseline: Build,
    methods: list[str],
    registry: LocalRegistry,
    calibration_data: "Path | None" = None,
) -> list[Build]:
    """
    Return existing variants or create them via the optimize pass.

    Reuses any variant whose candidate_id already exists in the registry.
    Never re-optimizes — cache hit = skip.

    Parameters
    ----------
    baseline : root Build to optimize from
    methods  : list of methods to apply, e.g. ["fp16"] or ["fp16", "int8"]
    registry : build registry

    Returns
    -------
    List of variant Builds (one per method, in order).
    """
    variants = []

    for method in methods:
        try:
            results = optimize(
                build_id=baseline.build_id,
                registry=registry,
                method=method,
                benchmark=False,     # benchmarking handled by benchmark_all()
                calibration_data=calibration_data if method == "int8" else None,
            )
            if results:
                variants.append(results[0])
                logger.info(
                    "variant ready method=%s build=%s",
                    method,
                    results[0].build_id[:16],
                )
        except OptimizeError as e:
            logger.warning("optimize skipped method=%s reason=%s", method, e)
            continue
        except Exception as e:
            logger.exception("optimize failed method=%s: %s", method, e)
            continue

    return variants

from ..benchmark.runner import BENCHMARK_RUNNERS


def benchmark_all(
    builds: list[Build],
    backend: str,
    registry: LocalRegistry,
    benchmark_runs: int = 100,
    warmup_runs: int = 20,
) -> None:
    """
    Benchmark all builds in-place, writing cached_latency_p50_ms to DB.

    Cache rule: if cached_latency_p50_ms is already set, skip.

    Parameters
    ----------
    builds         : list of builds to benchmark (baseline + variants)
    backend        : "coreml" | "tflite"
    registry       : build registry
    benchmark_runs : number of inference runs
    warmup_runs    : number of warmup runs
    """
    runner_cls = BENCHMARK_RUNNERS.get(backend)
    if runner_cls is None:
        logger.warning("no benchmark runner for backend=%s, skipping", backend)
        return

    for build in builds:
        if build.cached_latency_p50_ms is not None:
            logger.info(
                "benchmark cache hit build=%s p50=%.2fms",
                build.build_id[:16],
                build.cached_latency_p50_ms,
            )
            continue

        logger.info("benchmarking build=%s backend=%s", build.build_id[:16], backend)

        try:
            runner = runner_cls(
                model_path=build.artifact_path,
                warmup_runs=warmup_runs,
                benchmark_runs=benchmark_runs,
            )
            result = runner.run(build_id=build.build_id)

            registry.update_cached_benchmark(
                build_id=build.build_id,
                latency_p50_ms=result.latency_p50,
                latency_p95_ms=result.latency_p95,
                memory_peak_mb=result.memory_peak_mb or 0.0,
            )

            # Patch in-memory so assign_verdicts sees the value
            object.__setattr__(build, "cached_latency_p50_ms", result.latency_p50)
            object.__setattr__(build, "cached_latency_p95_ms", result.latency_p95)

            logger.info(
                "benchmark complete build=%s p50=%.2fms",
                build.build_id[:16],
                result.latency_p50,
            )

        except Exception as e:
            logger.warning(
                "benchmark failed build=%s: %s",
                build.build_id[:16],
                e,
            )

from ..core.types import VariantResult, BackendResult


def assign_verdicts(
    builds: list[Build],
    backend: str,
) -> BackendResult:
        """
        Score all builds for a backend and assign verdicts.

        Verdict logic (data-driven):
        - baseline  : always fp32
        - skip      : slower or equal to baseline
        - recommended : highest efficiency score among non-skip variants
        - aggressive  : smallest size among remaining (if different from recommended)

        Score formula:
            score = 0.6 * (baseline_latency / variant_latency)
                + 0.4 * (baseline_size / variant_size)
        Higher = better.
        """
        baseline_build = next(
            (b for b in builds if b.optimization_method is None),
            None,
        )

        if baseline_build is None:
            logger.warning("no baseline build found for backend=%s", backend)
            return BackendResult(backend=backend, variants=[])

        baseline_latency = baseline_build.cached_latency_p50_ms
        baseline_size = float(baseline_build.size_mb)

        results: list[VariantResult] = []

        # Baseline entry
        results.append(VariantResult(
            build_id=baseline_build.build_id,
            method="fp32",
            size_mb=baseline_size,
            latency_p50_ms=baseline_latency,
            verdict="baseline",
            latency_delta_pct=None,
            size_delta_pct=None,
        ))

        # Score non-baseline builds
        for b in builds:
            if b.optimization_method is None:
                continue

            method = b.optimization_method
            latency = b.cached_latency_p50_ms
            size = float(b.size_mb)

            # Compute deltas (negative = improvement)
            latency_delta_pct = (
                ((latency - baseline_latency) / baseline_latency * 100)
                if baseline_latency and latency
                else None
            )
            size_delta_pct = (
                ((size - baseline_size) / baseline_size * 100)
                if baseline_size
                else None
            )

            # No latency data — can't score
            if latency is None or baseline_latency is None:
                results.append(VariantResult(
                    build_id=b.build_id,
                    method=method,
                    size_mb=size,
                    latency_p50_ms=latency,
                    verdict="skip",
                    latency_delta_pct=latency_delta_pct,
                    size_delta_pct=size_delta_pct,
                ))
                continue

            # Compute composite score (higher = better)
            # 0.6 weight on latency speedup, 0.4 on size reduction
            # Skip only if score <= 1.0 (strictly worse on both axes)
            score = (
                0.6 * (baseline_latency / latency)
                + 0.4 * (baseline_size / size)
            )

            if score <= 1.0:
                results.append(VariantResult(
                    build_id=b.build_id,
                    method=method,
                    size_mb=size,
                    latency_p50_ms=latency,
                    verdict="skip",
                    latency_delta_pct=latency_delta_pct,
                    size_delta_pct=size_delta_pct,
                ))
                continue

            results.append(VariantResult(
                build_id=b.build_id,
                method=method,
                size_mb=size,
                latency_p50_ms=latency,
                verdict=None,       # assigned below
                latency_delta_pct=latency_delta_pct,
                size_delta_pct=size_delta_pct,
                _score=score,
            ))

        # Second pass: assign recommended / aggressive
        scoreable = [r for r in results if r.verdict is None]

        if scoreable:
            recommended = max(scoreable, key=lambda r: r._score)
            recommended.verdict = "recommended"

            remaining = [r for r in scoreable if r is not recommended]
            if remaining:
                aggressive = min(remaining, key=lambda r: r.size_mb)
                aggressive.verdict = "aggressive"

        # Any still-None verdicts (shouldn't happen, safety net)
        for r in results:
            if r.verdict is None:
                r.verdict = "skip"

        return BackendResult(backend=backend, variants=results)


from ..core.accuracy.config import AccuracyConfig
from ..core.accuracy.inputs import generate_batch
from ..core.accuracy.checker import run_accuracy_check


def _run_accuracy_checks(
    baseline,
    backend_result,
    config,
    registry,
) -> None:
    """
    Run accuracy checks for all non-baseline variants in backend_result.
    Inputs are generated once from the baseline runner and reused.
    Variants that fail accuracy have their verdict overridden to skip.
    Results are saved to the registry.
    """
    from ..benchmark.runner import CoreMLBenchmarkRunner, TFLiteBenchmarkRunner

    def _make_runner(build):
        if build.format == "coreml":
            return CoreMLBenchmarkRunner(build.artifact_path)
        elif build.format == "tflite":
            return TFLiteBenchmarkRunner(build.artifact_path)
        else:
            raise ValueError(f"Unsupported format for accuracy: {build.format}")

    try:
        baseline_runner = _make_runner(baseline)
        spec = baseline_runner.get_input_spec()
    except Exception as e:
        logger.warning("accuracy: failed to create baseline runner: %s", e)
        return

    import numpy as np
    rng = np.random.default_rng(config.seed)
    try:
        batch = generate_batch(spec, rng, samples=config.samples)
    except Exception as e:
        logger.warning("accuracy: failed to generate input batch: %s", e)
        return

    for variant in backend_result.variants:
        if variant.verdict == "baseline":
            continue

        try:
            variant_build = registry.resolve_build(variant.build_id)
        except Exception as e:
            logger.warning("accuracy: could not resolve build %s: %s", variant.build_id[:16], e)
            continue

        if variant_build is None:
            continue

        try:
            candidate_runner = _make_runner(variant_build)
            result = run_accuracy_check(
                baseline_runner,
                candidate_runner,
                config=config,
                baseline_build_id=baseline.build_id,
                candidate_build_id=variant.build_id,
                precomputed_batch=batch,
            )
            try:
                registry.save_accuracy_check(result)
            except Exception as save_exc:
                logger.warning("accuracy: save failed for %s: %s", variant.build_id[:16], save_exc)

            variant.accuracy_passed = result.passed
            variant.accuracy_cosine = result.cosine_similarity
            variant.accuracy_top1 = result.top1_agreement
            variant.accuracy_failure = result.failure_reasons[0] if result.failure_reasons else None

            if not result.passed and variant.verdict != "skip":
                logger.info(
                    "accuracy: overriding verdict to skip for build=%s reasons=%s",
                    variant.build_id[:16],
                    result.failure_reasons,
                )
                variant.verdict = "skip"

        except Exception as e:
            logger.warning("accuracy: check failed for build=%s: %s", variant.build_id[:16], e)

def explore(
    onnx_path: Path,
    target: str,
    name: str,
    backends: list[str],
    fast: bool,
    registry: LocalRegistry,
    calibration_data: "Path | None" = None,
    check_accuracy: bool = False,
    accuracy_samples: int = 32,
    accuracy_seed: int = 42,
    accuracy_cosine_threshold: float = 0.99,
    accuracy_top1_threshold: float = 0.99,
) -> ExploreResult:
    """
    Full explore sweep — builds, optimizes, benchmarks, assigns verdicts.
    """
    from ..core.types import ExploreResult

    onnx_path = Path(onnx_path)
    methods = ["fp16"] if fast else ["fp16", "int8"]
    benchmark_runs = 20 if fast else 100

    result = ExploreResult(
        name=name,
        source_path=str(onnx_path),
        target=target,
        fast_mode=fast,
    )

    for backend in backends:
        logger.info("explore backend=%s target=%s", backend, target)

        try:
            baseline = resolve_or_build(onnx_path, target, name, backend, registry)
            variants = resolve_or_optimize(baseline, methods, registry, calibration_data=calibration_data)
            all_builds = [baseline] + variants

            benchmark_all(
                builds=all_builds,
                backend=backend,
                registry=registry,
                benchmark_runs=benchmark_runs,
            )

            backend_result = assign_verdicts(all_builds, backend)

            if check_accuracy:
                acc_config = AccuracyConfig(
                    samples=accuracy_samples,
                    seed=accuracy_seed,
                    cosine_threshold=accuracy_cosine_threshold,
                    top1_threshold=accuracy_top1_threshold,
                )
                _run_accuracy_checks(
                    baseline=baseline,
                    backend_result=backend_result,
                    config=acc_config,
                    registry=registry,
                )

            result.backends.append(backend_result)

        except Exception as e:
            logger.exception("explore failed backend=%s: %s", backend, e)
            continue

    return result

# ============================================================
# Validation
# ============================================================

def _validate_inputs(
    onnx_path: Path,
    backend: str,
) -> None:
    """
    Validate explore inputs early to prevent downstream failures.
    """

    if not onnx_path.exists():
        raise FileNotFoundError(f"Model not found: {onnx_path}")

    if not onnx_path.is_file():
        raise ValueError(f"Model path is not a file: {onnx_path}")

    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported backend '{backend}'. "
            f"Supported backends: {sorted(SUPPORTED_BACKENDS)}"
        )