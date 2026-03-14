"""
Accuracy validator for mlbuild validate --dataset.

Isolated from validate.py so the command stays thin.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AccuracyViolation:
    metric: str
    threshold: float
    actual: float
    passed: bool

    def as_violation_dict(self) -> dict:
        return {
            "constraint": f"accuracy_{self.metric}",
            "limit": self.threshold,
            "actual": self.actual,
            "unit": "",
        }


@dataclass
class AccuracyValidationResult:
    skipped: bool = False
    skip_reason: str = ""
    cosine_similarity: Optional[float] = None
    top1_agreement: Optional[float] = None
    passed: bool = True
    violations: list[AccuracyViolation] = None

    def __post_init__(self):
        if self.violations is None:
            self.violations = []


class AccuracyValidator:
    """
    Runs output divergence check between a build and its baseline.

    Parameters
    ----------
    build           : Build to validate
    baseline        : Reference build to compare against
    dataset         : Path to calibration data (images, .npy dir, .npz)
    cosine_threshold: Minimum cosine similarity
    top1_threshold  : Minimum top-1 agreement
    max_samples     : Cap on samples loaded
    registry        : LocalRegistry (for saving results)
    """

    def __init__(
        self,
        build,
        baseline,
        dataset: Path,
        cosine_threshold: float = 0.99,
        top1_threshold: float = 0.99,
        max_samples: int = 200,
        registry=None,
    ):
        self.build = build
        self.baseline = baseline
        self.dataset = Path(dataset)
        self.cosine_threshold = cosine_threshold
        self.top1_threshold = top1_threshold
        self.max_samples = max_samples
        self.registry = registry

    def validate(self) -> AccuracyValidationResult:
        """
        Run accuracy check. Returns AccuracyValidationResult.
        Never raises — errors are returned as skipped with reason.
        """
        from ..core.accuracy.calibration import CalibrationLoader, CalibrationError
        from ..core.accuracy.config import AccuracyConfig
        from ..core.accuracy.checker import run_accuracy_check
        from ..benchmark.runner import CoreMLBenchmarkRunner, TFLiteBenchmarkRunner

        # Build runners
        try:
            baseline_runner = self._make_runner(self.baseline)
            candidate_runner = self._make_runner(self.build)
        except Exception as e:
            logger.warning("accuracy_validator: runner creation failed: %s", e)
            return AccuracyValidationResult(skipped=True, skip_reason=f"runner error: {e}")

        # Read input spec from baseline
        try:
            spec = baseline_runner.get_input_spec()
            # spec is list[InputSpec(name, shape, dtype)]
            first = spec[0]
            input_name = first.name
            input_shape = (1,) + tuple(first.shape)  # add batch dim
            layout = self._detect_layout(input_shape)
        except Exception as e:
            logger.warning("accuracy_validator: input spec failed: %s", e)
            return AccuracyValidationResult(skipped=True, skip_reason=f"input spec error: {e}")

        # Load calibration samples
        try:
            loader = CalibrationLoader(
                source=self.dataset,
                input_name=input_name,
                input_shape=input_shape,
                layout=layout,
                max_samples=self.max_samples,
            )
            samples = loader.as_list()
        except CalibrationError as e:
            return AccuracyValidationResult(skipped=True, skip_reason=str(e))
        except Exception as e:
            logger.warning("accuracy_validator: dataset load failed: %s", e)
            return AccuracyValidationResult(skipped=True, skip_reason=f"dataset error: {e}")

        if not samples:
            return AccuracyValidationResult(skipped=True, skip_reason="no samples loaded")

        logger.info(
            "accuracy_validator samples=%d input=%s shape=%s",
            len(samples),
            input_name,
            input_shape,
        )

        # Convert list[dict] to precomputed_batch format expected by checker
        # Each sample has shape (1, ...) from CalibrationLoader — squeeze batch dim before stacking
        # so precomputed_batch has shape (N, C, H, W) as checker expects
        import numpy as np
        precomputed_batch = {input_name: np.stack([s[input_name].squeeze(0) for s in samples])}

        # Run accuracy check
        config = AccuracyConfig(
            samples=len(samples),
            seed=42,
            cosine_threshold=self.cosine_threshold,
            top1_threshold=self.top1_threshold,
        )

        try:
            result = run_accuracy_check(
                baseline_runner=baseline_runner,
                candidate_runner=candidate_runner,
                config=config,
                baseline_build_id=self.baseline.build_id,
                candidate_build_id=self.build.build_id,
                precomputed_batch=precomputed_batch,
            )
        except Exception as e:
            logger.warning("accuracy_validator: check failed: %s", e)
            return AccuracyValidationResult(skipped=True, skip_reason=f"check error: {e}")

        # Persist if registry provided
        if self.registry is not None:
            try:
                self.registry.save_accuracy_check(result)
            except Exception as e:
                logger.warning("accuracy_validator: save failed: %s", e)

        # Build violations
        violations = []
        if result.cosine_similarity is not None and result.cosine_similarity < self.cosine_threshold:
            violations.append(AccuracyViolation(
                metric="cosine",
                threshold=self.cosine_threshold,
                actual=result.cosine_similarity,
                passed=False,
            ))
        if result.top1_agreement is not None and result.top1_agreement < self.top1_threshold:
            violations.append(AccuracyViolation(
                metric="top1",
                threshold=self.top1_threshold,
                actual=result.top1_agreement,
                passed=False,
            ))

        return AccuracyValidationResult(
            skipped=False,
            cosine_similarity=result.cosine_similarity,
            top1_agreement=result.top1_agreement,
            passed=result.passed,
            violations=violations,
        )

    def _make_runner(self, build):
        from ..benchmark.runner import CoreMLBenchmarkRunner, TFLiteBenchmarkRunner
        if build.format == "coreml":
            return CoreMLBenchmarkRunner(build.artifact_path)
        elif build.format == "tflite":
            return TFLiteBenchmarkRunner(build.artifact_path)
        raise ValueError(f"Unsupported format: {build.format}")

    def _detect_layout(self, shape: tuple) -> Optional[str]:
        if len(shape) == 4:
            return "nchw" if shape[1] in (1, 3) else "nhwc"
        return None
