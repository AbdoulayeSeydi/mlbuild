"""
MLBuild Output Validation System
================================

Validation architecture in a single file.

Logical modules contained in this file:

1.  Result Model
2.  Configuration
3.  Output Schema
4.  Check Infrastructure
5.  Structural Checks          (universal)
6.  Vision Checks              (classification)
7.  Detection Checks           (NMS-aware — v2)
8.  NLP Checks
9.  Audio Checks
10. Time-Series Checks         (v2)
11. Cross-Output Checks
12. Validator
13. CLI helpers

Design goals
------------
• No architecture boundary leaks
• Supports multi-output models
• Supports cross-output validation
• Strict mode only escalates structural issues
• Sampling-based checks (no large tensor scans)
• Extensible check registry
• Output schema validation
• Deterministic validation scoring
• NMS-aware detection validation (v2)
• Unimplemented subtype guards (v2)

Public API (v1 — unchanged)
----------------------------
Status, CheckResult, ValidationResult
StrictOutputConfig, OutputSchema, Check, sample_tensor
NaNCheck, AllZeroCheck, BoundingBoxCheck, LogitVarianceCheck
UnitNormEmbeddingCheck, CTCShapeCheck, DetectionConsistency
TaskOutputValidator
format_validation_output, should_exit_on_validation, format_validation_warning

Public API (v2 — additive)
---------------------------
DetectionFinalCheck, DetectionRawCheck
TimeSeriesShapeCheck, TimeSeriesValueCheck
validate_with_profile
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

import numpy as np

from .task_detection import (
    # v1
    TaskType,
    # v2
    Subtype,
    ExecutionMode,
    ModelProfile,
)

logger = logging.getLogger(__name__)


# ── Unimplemented subtype guard ─────────────────────────────────────────────
# Single source of truth — remove a subtype from this set when you implement it.
# The warning and fallback disappear automatically.

_UNIMPLEMENTED_SUBTYPES: set = {
    Subtype.SEGMENTATION,
}


# =============================================================================
# Result Model  (v1 — unchanged)
# =============================================================================

class Status(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class CheckResult:
    name:       str
    status:     Status
    message:    str
    structural: bool  = False
    score:      float = 1.0


@dataclass
class ValidationResult:
    checks:  List[CheckResult]
    task:    TaskType = TaskType.UNKNOWN
    subtype: Subtype  = Subtype.NONE       # v2 — carried for downstream routing

    def summary(self) -> Dict[str, int]:
        final = [c.status for c in self.checks]
        return {
            "pass": sum(s == Status.PASS for s in final),
            "warn": sum(s == Status.WARN for s in final),
            "fail": sum(s == Status.FAIL for s in final),
            "skip": sum(s == Status.SKIP for s in final),
        }

    def score(self) -> float:
        if not self.checks:
            return 1.0
        return float(sum(c.score for c in self.checks) / len(self.checks))

    def failed(self) -> bool:
        return any(c.status == Status.FAIL for c in self.checks)

    def explain(self) -> str:
        lines = [f"Output validation — task={self.task.value}  subtype={self.subtype.value}"]
        for c in self.checks:
            icon = {"pass": "✓", "warn": "⚠", "fail": "✗", "skip": "—"}[c.status.value]
            lines.append(f"  {icon} [{c.name}] {c.message}  (score={c.score:.2f})")
        lines.append(
            f"\n  → score={self.score():.2f}  "
            f"{'FAILED' if self.failed() else 'PASSED'}"
        )
        return "\n".join(lines)


# =============================================================================
# Configuration  (v1 — unchanged)
# =============================================================================

@dataclass
class StrictOutputConfig:
    """
    Controls whether output validation failures are hard errors.

    global_strict  : read from .mlbuild/config.toml [validation] strict_output
    command_strict : passed via --strict-output flag on individual commands

    Resolution rule: command_strict always overrides global_strict when set.
    Neither set → soft mode (warn only, never blocks pipeline).
    """
    global_strict:  bool           = False
    command_strict: Optional[bool] = None

    @property
    def is_strict(self) -> bool:
        if self.command_strict is not None:
            return self.command_strict
        return self.global_strict

    @classmethod
    def from_command(
        cls,
        strict_flag:   bool,
        global_strict: bool = False,
    ) -> "StrictOutputConfig":
        return cls(
            global_strict  = global_strict,
            command_strict = strict_flag if strict_flag else None,
        )

    @classmethod
    def soft(cls) -> "StrictOutputConfig":
        return cls(global_strict=False, command_strict=None)

    @classmethod
    def hard(cls) -> "StrictOutputConfig":
        return cls(global_strict=False, command_strict=True)


# =============================================================================
# Output Schema  (v1 — unchanged)
# =============================================================================

@dataclass
class OutputSchema:
    name:         str
    rank:         Optional[int] = None
    last_dim_min: Optional[int] = None
    last_dim_max: Optional[int] = None
    description:  str           = ""


# =============================================================================
# Check Infrastructure  (v1 — unchanged)
# =============================================================================

MAX_SAMPLE = 1_000_000


class Check:
    name:          str           = "check"
    task_affinity: Set[TaskType] = set()   # empty = universal

    def run(self, outputs: Dict[str, np.ndarray]) -> CheckResult:
        raise NotImplementedError

    def relevant_for(self, task: TaskType) -> bool:
        return not self.task_affinity or task in self.task_affinity


def sample_tensor(t: np.ndarray) -> np.ndarray:
    flat = t.ravel()
    if flat.size > MAX_SAMPLE:
        flat = flat[:MAX_SAMPLE]
    return flat


# =============================================================================
# Structural Checks  (universal — v1 unchanged)
# =============================================================================

class NaNCheck(Check):
    name          = "nan_check"
    task_affinity: Set[TaskType] = set()

    def run(self, outputs):
        for name, t in outputs.items():
            s = sample_tensor(t)
            if np.isnan(s).any() or np.isinf(s).any():
                return CheckResult(
                    self.name, Status.FAIL,
                    f"{name} contains NaN/Inf",
                    structural=True, score=0.0,
                )
        return CheckResult(self.name, Status.PASS, "no NaN/Inf detected")


class AllZeroCheck(Check):
    name          = "all_zero"
    task_affinity: Set[TaskType] = set()

    def run(self, outputs):
        for name, t in outputs.items():
            s = sample_tensor(t)
            if np.all(s == 0):
                return CheckResult(
                    self.name, Status.WARN,
                    f"{name} appears all zeros",
                    structural=True, score=0.5,
                )
        return CheckResult(self.name, Status.PASS, "outputs non-zero")


# =============================================================================
# Vision Checks  (v1 — unchanged)
# =============================================================================

class BoundingBoxCheck(Check):
    name          = "bbox_shape"
    task_affinity: Set[TaskType] = {TaskType.VISION}

    def run(self, outputs):
        for name, t in outputs.items():
            if t.ndim >= 2 and 4 <= t.shape[-1] <= 16:
                return CheckResult(
                    self.name, Status.PASS,
                    f"{name} resembles bounding box tensor",
                    score=1.0,
                )
        return CheckResult(self.name, Status.SKIP, "no bbox tensor detected")


# =============================================================================
# Detection Checks  (v2 — NMS-aware)
# =============================================================================

class DetectionFinalCheck(Check):
    """
    Validation for detection models with NMS baked into the graph.
    Outputs are final detections — validate value ranges, not just shapes.

    Expects outputs containing at least one of:
      - A float tensor with last dim 4 (boxes in [0,1] or pixel coords)
      - A float tensor with values in [0, 1] (scores/class probabilities)

    Used when ModelProfile.nms_inside=True.
    """
    name          = "detection_final"
    task_affinity: Set[TaskType] = {TaskType.VISION}

    def run(self, outputs):
        issues: List[str] = []

        # Find box tensor (last dim == 4)
        box_tensor = None
        for name, t in outputs.items():
            dims = [d for d in t.shape if d > 0]
            if dims and dims[-1] == 4:
                box_tensor = (name, t)
                break

        # Find score tensor (float values in plausible class count range)
        score_tensor = None
        for name, t in outputs.items():
            if t.dtype.kind == 'f' and t.ndim >= 2:
                dims = [d for d in t.shape if d > 0]
                if dims and 10 <= dims[-1] <= 1000:
                    score_tensor = (name, t)
                    break

        if box_tensor is not None:
            bname, bt = box_tensor
            sample = sample_tensor(bt)
            if sample.size > 0:
                bmin, bmax = float(sample.min()), float(sample.max())
                # Accept both normalized [0,1] and pixel coords (up to ~8192)
                if bmax > 8192 or bmin < -1:
                    issues.append(
                        f"{bname} box coordinates out of expected range "
                        f"[{bmin:.2f}, {bmax:.2f}]"
                    )

        if score_tensor is not None:
            sname, st = score_tensor
            sample = sample_tensor(st)
            if sample.size > 0:
                smin, smax = float(sample.min()), float(sample.max())
                if smin < -0.05 or smax > 1.05:
                    issues.append(
                        f"{sname} scores out of [0, 1] range "
                        f"[{smin:.3f}, {smax:.3f}]"
                    )

        # Count consistency: if multiple outputs, check batch dims agree
        if len(outputs) >= 2:
            batch_dims = set()
            for t in outputs.values():
                if t.ndim >= 2 and t.shape[0] > 0:
                    batch_dims.add(t.shape[0])
            if len(batch_dims) > 1:
                issues.append(f"batch dim mismatch across outputs: {sorted(batch_dims)}")

        if issues:
            return CheckResult(
                self.name, Status.WARN,
                "detection output issues: " + "; ".join(issues),
                structural=False, score=0.7,
            )
        return CheckResult(
            self.name, Status.PASS,
            "detection outputs within expected ranges",
            score=1.0,
        )


class DetectionRawCheck(Check):
    """
    Validation for detection models without NMS (raw anchor predictions).
    Only validates shape consistency and finiteness — never value ranges.

    Raw anchor logits can take any float value; validating [0,1] ranges
    would produce false WARNs on every raw YOLO/DETR output.

    Used when ModelProfile.nms_inside=False.
    """
    name          = "detection_raw"
    task_affinity: Set[TaskType] = {TaskType.VISION}

    def run(self, outputs):
        issues: List[str] = []

        # Shape consistency: batch dims must agree across outputs
        batch_dims = set()
        for t in outputs.values():
            if t.ndim >= 2 and t.shape[0] > 0:
                batch_dims.add(t.shape[0])

        if len(batch_dims) > 1:
            issues.append(f"batch dim mismatch across raw outputs: {sorted(batch_dims)}")

        # Finiteness only — no range check
        for name, t in outputs.items():
            s = sample_tensor(t)
            if np.isnan(s).any() or np.isinf(s).any():
                issues.append(f"{name} contains NaN/Inf in raw predictions")

        if issues:
            return CheckResult(
                self.name, Status.FAIL,
                "raw detection output issues: " + "; ".join(issues),
                structural=True, score=0.0,
            )
        return CheckResult(
            self.name, Status.PASS,
            "raw detection outputs finite and shape-consistent",
            score=1.0,
        )


# =============================================================================
# NLP Checks  (v1 — unchanged)
# =============================================================================

class LogitVarianceCheck(Check):
    name          = "logit_variance"
    task_affinity: Set[TaskType] = {TaskType.NLP, TaskType.VISION, TaskType.AUDIO}

    def run(self, outputs):
        for name, t in outputs.items():
            if t.ndim >= 2:
                reshaped = t.reshape(-1, t.shape[-1])
                sample   = reshaped[:256]
                variance = np.var(sample, axis=-1).mean()
                if variance > 0.5:
                    return CheckResult(
                        self.name, Status.PASS,
                        f"{name} appears logit-like (var={variance:.2f})",
                        score=1.0,
                    )
        return CheckResult(self.name, Status.WARN, "no strong logits detected", score=0.7)


class UnitNormEmbeddingCheck(Check):
    name          = "embedding_norm"
    task_affinity: Set[TaskType] = {TaskType.NLP, TaskType.VISION}

    def run(self, outputs):
        for name, t in outputs.items():
            if t.ndim == 2:
                sample    = t[:64]
                norms     = np.linalg.norm(sample, axis=-1)
                mean_norm = norms.mean()
                if 0.8 <= mean_norm <= 1.2:
                    return CheckResult(
                        self.name, Status.PASS,
                        f"{name} embeddings unit norm",
                        score=1.0,
                    )
                return CheckResult(
                    self.name, Status.WARN,
                    f"{name} embedding norm {mean_norm:.2f}",
                    score=0.8,
                )
        return CheckResult(self.name, Status.SKIP, "no embedding tensor")


# =============================================================================
# Audio Checks  (v1 — unchanged)
# =============================================================================

class CTCShapeCheck(Check):
    name          = "ctc_shape"
    task_affinity: Set[TaskType] = {TaskType.AUDIO}

    def run(self, outputs):
        for name, t in outputs.items():
            if t.ndim == 3:
                vocab = t.shape[-1]
                if 20 <= vocab <= 2000:
                    return CheckResult(
                        self.name, Status.PASS,
                        f"{name} resembles CTC logits",
                        score=1.0,
                    )
        return CheckResult(self.name, Status.SKIP, "no CTC output")


# =============================================================================
# Time-Series Checks  (v2)
# =============================================================================

class TimeSeriesShapeCheck(Check):
    """
    Validate that time-series outputs have a plausible forecast shape.

    Expected: rank 2 or 3 float tensor.
      [B, horizon]         → univariate forecast
      [B, horizon, vars]   → multivariate forecast

    Does not validate the forecast horizon value — that's model-specific.
    """
    name          = "timeseries_shape"
    task_affinity: Set[TaskType] = set()   # TABULAR has no TaskType equivalent

    def run(self, outputs):
        for name, t in outputs.items():
            if t.ndim in (2, 3) and t.dtype.kind == 'f':
                return CheckResult(
                    self.name, Status.PASS,
                    f"{name} has plausible forecast shape {t.shape}",
                    score=1.0,
                )
        return CheckResult(
            self.name, Status.WARN,
            "no float rank-2/3 output found — unexpected shape for time-series forecast",
            score=0.6,
        )


class TimeSeriesValueCheck(Check):
    """
    Validate that time-series forecast values are finite and not all-zero.
    A cold-start zero-initialized stateful model will naturally produce
    near-zero outputs — this is a WARN, not a FAIL.
    """
    name          = "timeseries_values"
    task_affinity: Set[TaskType] = set()

    def run(self, outputs):
        for name, t in outputs.items():
            if t.ndim in (2, 3) and t.dtype.kind == 'f':
                s = sample_tensor(t)
                if np.isnan(s).any() or np.isinf(s).any():
                    return CheckResult(
                        self.name, Status.FAIL,
                        f"{name} forecast contains NaN/Inf",
                        structural=True, score=0.0,
                    )
                if np.all(np.abs(s) < 1e-7):
                    return CheckResult(
                        self.name, Status.WARN,
                        f"{name} forecast is effectively zero "
                        "(expected for cold-start stateful models)",
                        structural=False, score=0.7,
                    )
                return CheckResult(
                    self.name, Status.PASS,
                    f"{name} forecast values finite and non-zero",
                    score=1.0,
                )
        return CheckResult(self.name, Status.SKIP, "no float forecast output found")


# =============================================================================
# Cross-Output Checks  (v1 — unchanged)
# =============================================================================

class DetectionConsistency(Check):
    """
    Legacy cross-output check — name-based 'boxes'/'scores' lookup.
    Works for models that name their outputs explicitly.
    For structural NMS-aware validation use DetectionFinalCheck / DetectionRawCheck.
    """
    name          = "detection_consistency"
    task_affinity: Set[TaskType] = {TaskType.VISION}

    def run(self, outputs):
        if "boxes" in outputs and "scores" in outputs:
            if len(outputs["boxes"]) != len(outputs["scores"]):
                return CheckResult(
                    self.name, Status.FAIL,
                    "boxes and scores length mismatch",
                    structural=True, score=0.0,
                )
            return CheckResult(
                self.name, Status.PASS,
                "boxes and scores aligned",
                score=1.0,
            )
        return CheckResult(self.name, Status.SKIP, "not a detection model")


# =============================================================================
# Validator  (v1 core preserved, v2 routing added)
# =============================================================================

class TaskOutputValidator:

    def __init__(
        self,
        config:  Optional[StrictOutputConfig] = None,
        schemas: Optional[List[OutputSchema]]  = None,
    ):
        self.config  = config or StrictOutputConfig()
        self.schemas = schemas or []

        # v1 default check registry (unchanged)
        self.checks: List[Check] = [
            NaNCheck(),
            AllZeroCheck(),
            LogitVarianceCheck(),
            UnitNormEmbeddingCheck(),
            BoundingBoxCheck(),
            CTCShapeCheck(),
            DetectionConsistency(),
        ]

    def _apply_schema(self, outputs: Dict[str, np.ndarray]) -> List[CheckResult]:
        results: List[CheckResult] = []
        for schema in self.schemas:
            if schema.name not in outputs:
                continue
            t = outputs[schema.name]
            if schema.rank and t.ndim != schema.rank:
                results.append(CheckResult(
                    "schema_rank", Status.FAIL,
                    f"{schema.name} rank mismatch",
                    structural=True, score=0,
                ))
            if schema.last_dim_min and t.shape[-1] < schema.last_dim_min:
                results.append(CheckResult(
                    "schema_last_dim_min", Status.FAIL,
                    f"{schema.name} last dim too small",
                    structural=True, score=0,
                ))
        return results

    def _promote_structural_warnings(self, results: List[CheckResult]) -> List[CheckResult]:
        """Strict mode — promote structural WARNs to FAILs."""
        if not self.config.is_strict:
            return results
        promoted = []
        for r in results:
            if r.status == Status.WARN and r.structural:
                promoted.append(CheckResult(
                    r.name, Status.FAIL, r.message,
                    structural=True, score=r.score,
                ))
            else:
                promoted.append(r)
        return promoted

    def validate(
        self,
        outputs: Dict[str, np.ndarray],
        task:    TaskType = TaskType.UNKNOWN,
    ) -> ValidationResult:
        """
        v1 API — validate outputs with task-level granularity.
        Use validate_with_profile() for subtype-aware (NMS-aware) validation.
        """
        checks: List[CheckResult] = []
        checks.extend(self._apply_schema(outputs))
        for check in self.checks:
            checks.append(check.run(outputs))
        checks = self._promote_structural_warnings(checks)
        return ValidationResult(checks=checks, task=task, subtype=Subtype.NONE)

    def validate_with_profile(
        self,
        outputs: Dict[str, np.ndarray],
        profile: ModelProfile,
    ) -> ValidationResult:
        """
        v2 API — validate outputs using the full ModelProfile.

        Routes to subtype-specific check sets:
          DETECTION + nms_inside=True  → DetectionFinalCheck (value ranges)
          DETECTION + nms_inside=False → DetectionRawCheck (shape/finite only)
          TIMESERIES                   → TimeSeriesShapeCheck + TimeSeriesValueCheck
          MULTIMODAL                   → structural checks only (pass-through)
          UNKNOWN_STRUCTURED / NONE    → structural checks only (pass-through)
          UNIMPLEMENTED subtypes       → logs warning, falls back to structural

        Parameters
        ----------
        outputs : name → tensor dict returned by the backend runner
        profile : ModelProfile from build_profile()

        Returns
        -------
        ValidationResult — always returns, never raises.
        """
        subtype = profile.subtype
        task_type = TaskType.VISION if profile.domain.value == "vision" else \
                    TaskType.NLP    if profile.domain.value == "nlp"    else \
                    TaskType.AUDIO  if profile.domain.value == "audio"  else \
                    TaskType.UNKNOWN

        # Unimplemented subtype guard
        if subtype in _UNIMPLEMENTED_SUBTYPES:
            logger.warning(
                "validation_unimplemented_subtype  subtype=%s  "
                "— no subtype-specific checks available, running structural only. "
                "Results may not reflect true model correctness.",
                subtype.value,
            )
            return self._structural_only(outputs, task_type, subtype)

        checks: List[CheckResult] = []
        checks.extend(self._apply_schema(outputs))

        # ── Detection (NMS-aware) ─────────────────────────
        if subtype == Subtype.DETECTION:
            # Always run structural checks
            checks.append(NaNCheck().run(outputs))
            checks.append(AllZeroCheck().run(outputs))
            # Route based on whether NMS is inside the graph
            if profile.nms_inside:
                checks.append(DetectionFinalCheck().run(outputs))
            else:
                checks.append(DetectionRawCheck().run(outputs))
            # Legacy name-based check still runs — harmlessly SKIPs when names differ
            checks.append(DetectionConsistency().run(outputs))

        # ── Time-series ───────────────────────────────────
        elif subtype == Subtype.TIMESERIES:
            checks.append(NaNCheck().run(outputs))
            checks.append(TimeSeriesShapeCheck().run(outputs))
            checks.append(TimeSeriesValueCheck().run(outputs))

        # ── Generative (KV-cache) ─────────────────────────
        elif subtype == Subtype.GENERATIVE_STATEFUL:
            # Single-pass KV-cache output is logit-shaped — use NLP checks
            checks.append(NaNCheck().run(outputs))
            checks.append(AllZeroCheck().run(outputs))
            checks.append(LogitVarianceCheck().run(outputs))

        # ── Recommendation ────────────────────────────────
        elif subtype == Subtype.RECOMMENDATION:
            # Score output should be finite and scalar-ish — structural only
            checks.extend(self._structural_checks(outputs))

        # ── Multimodal ────────────────────────────────────
        elif subtype == Subtype.MULTIMODAL:
            # Structural checks only — per-modality validation requires knowing
            # which output belongs to which modality, which we don't have yet.
            checks.extend(self._structural_checks(outputs))

        # ── Default: domain-level routing ─────────────────
        else:
            # NONE, UNKNOWN_STRUCTURED, or any unrecognized subtype
            # Structural checks + task-level domain checks
            checks.append(NaNCheck().run(outputs))
            checks.append(AllZeroCheck().run(outputs))
            domain_checks = self._domain_checks(task_type)
            for check in domain_checks:
                checks.append(check.run(outputs))

        checks = self._promote_structural_warnings(checks)
        return ValidationResult(checks=checks, task=task_type, subtype=subtype)

    def _structural_checks(self, outputs: Dict[str, np.ndarray]) -> List[CheckResult]:
        """Run universal structural checks only."""
        return [NaNCheck().run(outputs), AllZeroCheck().run(outputs)]

    def _structural_only(
        self,
        outputs:  Dict[str, np.ndarray],
        task:     TaskType,
        subtype:  Subtype,
    ) -> ValidationResult:
        """Return a ValidationResult with structural checks only."""
        checks = self._structural_checks(outputs)
        checks = self._promote_structural_warnings(checks)
        return ValidationResult(checks=checks, task=task, subtype=subtype)

    def _domain_checks(self, task: TaskType) -> List[Check]:
        """Return the standard domain-level checks for a TaskType."""
        if task == TaskType.VISION:
            return [LogitVarianceCheck(), BoundingBoxCheck(), DetectionConsistency()]
        if task == TaskType.NLP:
            return [LogitVarianceCheck(), UnitNormEmbeddingCheck()]
        if task == TaskType.AUDIO:
            return [LogitVarianceCheck(), CTCShapeCheck()]
        return []

    def relevant_checks(self, task: TaskType) -> List[Check]:
        """Return only the checks relevant for a given task."""
        return [c for c in self.checks if c.relevant_for(task)]


# =============================================================================
# CLI Helpers  (v1 — unchanged)
# =============================================================================

def format_validation_output(result: ValidationResult) -> str:
    lines = []
    for c in result.checks:
        prefix = {
            Status.PASS: "✓",
            Status.WARN: "⚠",
            Status.FAIL: "✗",
            Status.SKIP: "—",
        }[c.status.value]
        lines.append(f"{prefix} {c.name}: {c.message}")
    lines.append("")
    lines.append(f"validation score: {result.score():.2f}")
    return "\n".join(lines)


def should_exit_on_validation(
    result: ValidationResult,
    config: StrictOutputConfig,
) -> bool:
    """
    Return True if the CLI should exit with a non-zero code.

    Hard fail conditions:
    - Any FAIL status (structural or schema violation)
    - Strict mode active AND any WARN present
    """
    if result.failed():
        return True
    if config.is_strict and result.summary()["warn"] > 0:
        return True
    return False


def format_validation_warning(result: ValidationResult) -> Optional[str]:
    """
    Return a short CLI warning string if there are issues, else None.
    Used by benchmark.py, validate.py, compare.py after each run.
    """
    summary = result.summary()
    if not result.failed() and summary["warn"] == 0:
        return None

    lines = [f"⚠  Output validation [{result.task.value}/{result.subtype.value}]:"]
    for c in result.checks:
        if c.status in (Status.WARN, Status.FAIL):
            icon = "✗" if c.status == Status.FAIL else "⚠"
            lines.append(f"   {icon} {c.message}")
    return "\n".join(lines)


# =============================================================================
# Convenience: build a validator pre-configured for a ModelProfile  (v2)
# =============================================================================

def validate_with_profile(
    outputs:  Dict[str, np.ndarray],
    profile:  ModelProfile,
    config:   Optional[StrictOutputConfig] = None,
    schemas:  Optional[List[OutputSchema]]  = None,
) -> ValidationResult:
    """
    Convenience function: create a validator and run validate_with_profile()
    in one call. Suitable for use in benchmark.py and other consumers that
    have a ModelProfile available.

    Parameters
    ----------
    outputs : name → tensor dict from the backend runner
    profile : ModelProfile from build_profile()
    config  : StrictOutputConfig (defaults to soft mode)
    schemas : optional custom OutputSchema list

    Returns
    -------
    ValidationResult with task, subtype, and all check results.
    """
    validator = TaskOutputValidator(config=config, schemas=schemas)
    return validator.validate_with_profile(outputs, profile)