"""
MLBuild CI Runner

CI orchestration and validation.

Pipeline:
    1. Resolve baseline
    2. Ensure baseline benchmarked
    3. Produce candidate builds
    4. Benchmark candidates
    5. Deterministically select candidate
    6. Run regression checks
    7. Optional accuracy validation
    8. Emit CIReport
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List, Tuple, Literal

from .thresholds import ThresholdConfig, ThresholdViolation
from .reporter import CIReport, EnvironmentInfo

logger = logging.getLogger(__name__)

CIResult = Literal["pass", "fail", "skip"]


class CIError(RuntimeError):
    """Hard failure in CI pipeline."""


# ---------------------------------------------------------------------
# Typed Build Record
# ---------------------------------------------------------------------

class BuildRecord:
    """
    Minimal typed interface expected from registry builds.
    """

    build_id: str
    format: str
    artifact_path: Path
    size_mb: float
    cached_latency_p50_ms: Optional[float]
    cached_latency_p95_ms: Optional[float]
    optimization_method: Optional[str]
    parent_build_id: Optional[str]
    model_name: Optional[str]


# ---------------------------------------------------------------------
# CI Runner
# ---------------------------------------------------------------------

class CIRunner:

    def __init__(
        self,
        registry,
        thresholds: ThresholdConfig,
        workspace_root: Path,
    ):
        self.registry = registry
        self.thresholds = thresholds
        self.workspace_root = workspace_root

    # --------------------------------------------------
    # Public entrypoint
    # --------------------------------------------------

    def run(
        self,
        baseline_ref: str,
        model_path: Optional[Path] = None,
        build_id: Optional[str] = None,
        target: str = "auto",
        dataset: Optional[Path] = None,
        fail_on_missing_baseline: bool = False,
    ) -> CIReport:

        baseline, baseline_tag = self._resolve_baseline(
            baseline_ref, fail_on_missing_baseline
        )

        if baseline is None:
            return self._skipped_report(baseline_ref)

        baseline = self._ensure_benchmarked(baseline)

        candidates = self._produce_candidates(
            model_path=model_path,
            build_id=build_id,
            target=target,
            baseline=baseline,
        )

        if not candidates:
            raise CIError("No candidate builds produced")

        candidates = [self._ensure_benchmarked(c) for c in candidates]

        candidate = self._select_candidate(candidates)

        violations: list = []

        latency_delta, size_delta = self._compute_deltas(baseline, candidate)

        violations.extend(
            self.thresholds.check_regression(
                baseline.cached_latency_p50_ms,
                candidate.cached_latency_p50_ms,
                float(baseline.size_mb),
                float(candidate.size_mb),
            )
        )

        accuracy_cosine = None
        accuracy_top1 = None
        accuracy_passed = None

        if dataset is not None:
            accuracy_cosine, accuracy_top1, accuracy_passed = self._run_accuracy(
                baseline, candidate, dataset
            )

            if accuracy_passed is False and accuracy_cosine is not None:
                violations.append(
                    ThresholdViolation(
                        metric="accuracy_cosine",
                        rule="accuracy",
                        actual=accuracy_cosine,
                        threshold=self.thresholds.cosine_threshold,
                        unit="ratio",
                    )
                )

        result: CIResult
        if violations:
            result = "fail"
        else:
            result = "pass"

        model_name = self._resolve_model_name(
            baseline,
            model_path,
            build_id,
        )

        report = CIReport(
            model=model_name,
            environment=EnvironmentInfo.now(
                device=getattr(baseline, "target_device", None),
                backend=getattr(baseline, "format", None),
            ),
            baseline_tag=baseline_tag,
            baseline_build_id=baseline.build_id,
            baseline_latency_ms=baseline.cached_latency_p50_ms,
            baseline_size_mb=float(baseline.size_mb),
            candidate_build_id=candidate.build_id,
            candidate_variant=self._variant_name(candidate),
            candidate_parent_build_id=candidate.parent_build_id,
            candidate_latency_ms=candidate.cached_latency_p50_ms,
            candidate_size_mb=float(candidate.size_mb),
            latency_delta_pct=latency_delta,
            size_delta_pct=size_delta,
            latency_regression_pct=self.thresholds.latency_regression_pct,
            size_regression_pct=self.thresholds.size_regression_pct,
            latency_budget_ms=self.thresholds.latency_budget_ms,
            size_budget_mb=self.thresholds.size_budget_mb,
            accuracy_cosine=accuracy_cosine,
            accuracy_top1=accuracy_top1,
            accuracy_passed=accuracy_passed,
            result=result,
            violations=violations,
        )

        report.save(self.workspace_root)

        logger.info(
            "ci result=%s baseline=%s candidate=%s violations=%d",
            result,
            baseline.build_id[:16],
            candidate.build_id[:16],
            len(violations),
        )

        return report

    # --------------------------------------------------
    # Baseline
    # --------------------------------------------------

    def _resolve_baseline(
        self,
        ref: str,
        fail_on_missing: bool,
    ) -> Tuple[Optional[BuildRecord], Optional[str]]:

        build, tag = self.registry.resolve_tag(ref)

        if build is None:
            msg = f"Baseline not found: '{ref}'"

            if fail_on_missing:
                raise CIError(msg)

            logger.warning("%s — CI skipped", msg)
            return None, None

        logger.info(
            "baseline resolved build=%s tag=%s",
            build.build_id[:16],
            tag,
        )

        return build, tag

    def _skipped_report(self, ref: str) -> CIReport:

        return CIReport(
            model="unknown",
            environment=EnvironmentInfo.now(),
            baseline_tag=ref,
            baseline_build_id="",
            baseline_latency_ms=None,
            baseline_size_mb=0.0,
            candidate_build_id="",
            candidate_variant="",
            candidate_parent_build_id=None,
            candidate_latency_ms=None,
            candidate_size_mb=0.0,
            latency_delta_pct=None,
            size_delta_pct=0.0,
            latency_regression_pct=self.thresholds.latency_regression_pct,
            size_regression_pct=self.thresholds.size_regression_pct,
            latency_budget_ms=self.thresholds.latency_budget_ms,
            size_budget_mb=self.thresholds.size_budget_mb,
            result="skip",
            violations=[],
        )

    # --------------------------------------------------
    # Candidate production
    # --------------------------------------------------

    def _produce_candidates(
        self,
        model_path: Optional[Path],
        build_id: Optional[str],
        target: str,
        baseline=None,
    ) -> List[BuildRecord]:

        if build_id is not None:

            build = self.registry.resolve_build(build_id)

            if build is None:
                raise CIError(f"Build not found: {build_id}")

            return [build]

        if model_path is None:
            raise CIError("Either model_path or build_id must be provided")

        from ...explore.explorer import explore
        from ...core.device import detect_device

        if target == "auto":
            target = detect_device().target

        result = explore(
            onnx_path=model_path,
            target=target,
            name=model_path.stem,
            backends=[baseline.format],
            fast=False,
            registry=self.registry,
        )

        candidates: List[BuildRecord] = []

        for backend in result.backends:
            for variant in backend.variants:

                if variant.verdict == "baseline":
                    continue

                build = self.registry.resolve_build(variant.build_id)

                if build is not None:
                    candidates.append(build)

        logger.info("explore produced %d candidates", len(candidates))

        return candidates

    # --------------------------------------------------
    # Benchmark guard
    # --------------------------------------------------

    def _ensure_benchmarked(self, build: BuildRecord) -> BuildRecord:

        if build.cached_latency_p50_ms is not None:
            return build

        from ...benchmark.runner import BENCHMARK_RUNNERS

        runner_cls = BENCHMARK_RUNNERS.get(build.format)

        if runner_cls is None:
            logger.warning("no benchmark runner for format=%s", build.format)
            return build

        runner = runner_cls(model_path=build.artifact_path)

        result = runner.run(build_id=build.build_id)

        self.registry.update_cached_benchmark(
            build_id=build.build_id,
            latency_p50_ms=result.latency_p50,
            latency_p95_ms=result.latency_p95,
            memory_peak_mb=result.memory_peak_mb or 0.0,
        )

        # reload updated build
        return self.registry.resolve_build(build.build_id)

    # --------------------------------------------------
    # Candidate selection
    # --------------------------------------------------

    def _select_candidate(
        self,
        candidates: List[BuildRecord],
    ) -> BuildRecord:

        recommended = [
            c for c in candidates
            if (c.optimization_method or "").lower() == "fp16"
        ]

        pool = recommended if recommended else candidates

        pool.sort(
            key=lambda b: (
                b.cached_latency_p50_ms
                if b.cached_latency_p50_ms is not None
                else float("inf")
            )
        )

        selected = pool[0]

        logger.info(
            "candidate selected build=%s method=%s latency=%.2fms",
            selected.build_id[:16],
            selected.optimization_method,
            selected.cached_latency_p50_ms or -1,
        )

        return selected

    # --------------------------------------------------
    # Delta computation
    # --------------------------------------------------

    def _compute_deltas(
        self,
        baseline: BuildRecord,
        candidate: BuildRecord,
    ):

        b_lat = baseline.cached_latency_p50_ms
        c_lat = candidate.cached_latency_p50_ms

        latency_delta = None

        if b_lat is not None and c_lat is not None and b_lat != 0:
            latency_delta = ((c_lat - b_lat) / b_lat) * 100

        b_size = float(baseline.size_mb)
        c_size = float(candidate.size_mb)

        size_delta = None

        if b_size != 0:
            size_delta = ((c_size - b_size) / b_size) * 100

        return latency_delta, size_delta

    # --------------------------------------------------
    # Accuracy
    # --------------------------------------------------

    def _run_accuracy(
        self,
        baseline: BuildRecord,
        candidate: BuildRecord,
        dataset: Path,
    ):

        from ...validation.accuracy_validator import AccuracyValidator

        validator = AccuracyValidator(
            build=candidate,
            baseline=baseline,
            dataset=dataset,
            cosine_threshold=self.thresholds.cosine_threshold,
            top1_threshold=self.thresholds.top1_threshold,
            registry=self.registry,
        )

        result = validator.validate()

        if result.skipped:
            logger.warning("accuracy check skipped: %s", result.skip_reason)
            return None, None, None

        logger.info(
            "accuracy cosine=%.4f top1=%.4f passed=%s",
            result.cosine_similarity,
            result.top1_agreement,
            result.passed,
        )

        return (
            result.cosine_similarity,
            result.top1_agreement,
            result.passed,
        )

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _variant_name(self, build: BuildRecord) -> str:

        if build.optimization_method:
            return build.optimization_method

        return "unknown"

    def _resolve_model_name(
        self,
        baseline,
        model_path: Optional[Path],
        build_id: Optional[str],
    ) -> str:

        name = getattr(baseline, "name", None)
        if name:
            return name

        if model_path is not None:
            return model_path.name

        if build_id is not None:
            return f"build:{build_id[:16]}"

        return "unknown"