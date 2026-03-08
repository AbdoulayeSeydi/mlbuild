"""
MLBuild Output Validation System
================================

Enterprise-grade validation architecture in a single file.

Logical modules contained in this file:

1. Result Model
2. Configuration
3. Output Schema
4. Check Infrastructure
5. Structural Checks
6. Vision Checks
7. NLP Checks
8. Audio Checks
9. Cross-Output Checks
10. Validator
11. CLI helpers

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

PATCH NOTES
-----------
Fix 1 — StrictOutputConfig: restored command_strict / global_strict
         override logic. command_strict (--strict-output flag) always
         wins over global_strict (.mlbuild/config.toml). Flagged for
         wiring in Step 6 (build.py / import_cmd.py).

Fix 2 — TaskType: imported and threaded through validate() so
         benchmark.py (Step 7) can route task-relevant checks into
         the output table. task_affinity added to Check base class
         so each check declares which tasks it is relevant for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set
import numpy as np

from .task_detection import TaskType


# =============================================================================
# Result Model
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

    checks: List[CheckResult]
    task:   TaskType = TaskType.UNKNOWN   # FIX 2 — carries task for Step 7

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
        """Full explanation for --verbose or report output."""
        lines = [f"Output validation — task={self.task.value}"]
        for c in self.checks:
            icon = {"pass": "✓", "warn": "⚠", "fail": "✗", "skip": "—"}[c.status.value]
            lines.append(f"  {icon} [{c.name}] {c.message}  (score={c.score:.2f})")
        lines.append(f"\n  → score={self.score():.2f}  "
                     f"{'FAILED' if self.failed() else 'PASSED'}")
        return "\n".join(lines)


# =============================================================================
# Configuration  — FIX 1: command_strict overrides global_strict
# =============================================================================

@dataclass
class StrictOutputConfig:
    """
    Controls whether output validation failures are hard errors.

    Scope
    -----
    global_strict  : read from .mlbuild/config.toml [validation] strict_output
    command_strict : passed via --strict-output flag on individual commands

    Resolution rule: command_strict always overrides global_strict when set.
    Neither set → soft mode (warn only, never blocks pipeline).

    TODO (Step 6) — wire global_strict from config.toml in build.py /
                    import_cmd.py and pass command_strict from the
                    --strict-output CLI flag.
    """
    global_strict:  bool           = False
    command_strict: Optional[bool] = None   # None = not set by command

    @property
    def is_strict(self) -> bool:
        """command_strict always wins when explicitly set."""
        if self.command_strict is not None:
            return self.command_strict
        return self.global_strict

    @classmethod
    def from_command(
        cls,
        strict_flag: bool,
        global_strict: bool = False,
    ) -> "StrictOutputConfig":
        """
        Build config from a CLI --strict-output flag value.

        Parameters
        ----------
        strict_flag   : True if --strict-output was passed
        global_strict : value read from .mlbuild/config.toml
        """
        return cls(
            global_strict=global_strict,
            command_strict=strict_flag if strict_flag else None,
        )

    @classmethod
    def soft(cls) -> "StrictOutputConfig":
        """Convenience — explicit soft mode (default behaviour)."""
        return cls(global_strict=False, command_strict=None)

    @classmethod
    def hard(cls) -> "StrictOutputConfig":
        """Convenience — explicit strict mode."""
        return cls(global_strict=False, command_strict=True)


# =============================================================================
# Output Schema
# =============================================================================

@dataclass
class OutputSchema:

    name:         str
    rank:         Optional[int] = None
    last_dim_min: Optional[int] = None
    last_dim_max: Optional[int] = None
    description:  str           = ""


# =============================================================================
# Check Infrastructure  — FIX 2: task_affinity added to base class
# =============================================================================

MAX_SAMPLE = 1_000_000


class Check:

    name: str = "check"

    # Which tasks this check is relevant for.
    # Empty set = universal (runs for all tasks).
    # benchmark.py uses this to filter the output table per task.
    task_affinity: Set[TaskType] = set()

    def run(self, outputs: Dict[str, np.ndarray]) -> CheckResult:
        raise NotImplementedError

    def relevant_for(self, task: TaskType) -> bool:
        """Return True if this check should appear in task's output table."""
        return not self.task_affinity or task in self.task_affinity


def sample_tensor(t: np.ndarray) -> np.ndarray:

    flat = t.ravel()

    if flat.size > MAX_SAMPLE:
        flat = flat[:MAX_SAMPLE]

    return flat


# =============================================================================
# Structural Checks  (universal — all tasks)
# =============================================================================

class NaNCheck(Check):

    name = "nan_check"
    task_affinity: Set[TaskType] = set()   # universal

    def run(self, outputs):

        for name, t in outputs.items():

            s = sample_tensor(t)

            if np.isnan(s).any() or np.isinf(s).any():

                return CheckResult(
                    self.name,
                    Status.FAIL,
                    f"{name} contains NaN/Inf",
                    structural=True,
                    score=0.0,
                )

        return CheckResult(self.name, Status.PASS, "no NaN/Inf detected")


class AllZeroCheck(Check):

    name = "all_zero"
    task_affinity: Set[TaskType] = set()   # universal

    def run(self, outputs):

        for name, t in outputs.items():

            s = sample_tensor(t)

            if np.all(s == 0):

                return CheckResult(
                    self.name,
                    Status.WARN,
                    f"{name} appears all zeros",
                    structural=True,
                    score=0.5,
                )

        return CheckResult(self.name, Status.PASS, "outputs non-zero")


# =============================================================================
# Vision Checks
# =============================================================================

class BoundingBoxCheck(Check):

    name = "bbox_shape"
    task_affinity: Set[TaskType] = {TaskType.VISION}

    def run(self, outputs):

        for name, t in outputs.items():

            if t.ndim >= 2:

                last = t.shape[-1]

                if 4 <= last <= 16:

                    return CheckResult(
                        self.name,
                        Status.PASS,
                        f"{name} resembles bounding box tensor",
                        score=1.0,
                    )

        return CheckResult(self.name, Status.SKIP, "no bbox tensor detected")


# =============================================================================
# NLP Checks
# =============================================================================

class LogitVarianceCheck(Check):

    name = "logit_variance"
    task_affinity: Set[TaskType] = {TaskType.NLP, TaskType.VISION, TaskType.AUDIO}

    def run(self, outputs):

        for name, t in outputs.items():

            if t.ndim >= 2:

                reshaped = t.reshape(-1, t.shape[-1])

                sample = reshaped[:256]

                variance = np.var(sample, axis=-1).mean()

                if variance > 0.5:

                    return CheckResult(
                        self.name,
                        Status.PASS,
                        f"{name} appears logit-like (var={variance:.2f})",
                        score=1.0,
                    )

        return CheckResult(
            self.name,
            Status.WARN,
            "no strong logits detected",
            score=0.7,
        )


class UnitNormEmbeddingCheck(Check):

    name = "embedding_norm"
    task_affinity: Set[TaskType] = {TaskType.NLP, TaskType.VISION}

    def run(self, outputs):

        for name, t in outputs.items():

            if t.ndim == 2:

                sample = t[:64]

                norms = np.linalg.norm(sample, axis=-1)

                mean_norm = norms.mean()

                if 0.8 <= mean_norm <= 1.2:

                    return CheckResult(
                        self.name,
                        Status.PASS,
                        f"{name} embeddings unit norm",
                        score=1.0,
                    )

                return CheckResult(
                    self.name,
                    Status.WARN,
                    f"{name} embedding norm {mean_norm:.2f}",
                    score=0.8,
                )

        return CheckResult(self.name, Status.SKIP, "no embedding tensor")


# =============================================================================
# Audio Checks
# =============================================================================

class CTCShapeCheck(Check):

    name = "ctc_shape"
    task_affinity: Set[TaskType] = {TaskType.AUDIO}

    def run(self, outputs):

        for name, t in outputs.items():

            if t.ndim == 3:

                vocab = t.shape[-1]

                if 20 <= vocab <= 2000:

                    return CheckResult(
                        self.name,
                        Status.PASS,
                        f"{name} resembles CTC logits",
                        score=1.0,
                    )

        return CheckResult(self.name, Status.SKIP, "no CTC output")


# =============================================================================
# Cross-Output Checks
# =============================================================================

class DetectionConsistency(Check):

    name = "detection_consistency"
    task_affinity: Set[TaskType] = {TaskType.VISION}

    def run(self, outputs):

        if "boxes" in outputs and "scores" in outputs:

            if len(outputs["boxes"]) != len(outputs["scores"]):

                return CheckResult(
                    self.name,
                    Status.FAIL,
                    "boxes and scores length mismatch",
                    structural=True,
                    score=0.0,
                )

            return CheckResult(
                self.name,
                Status.PASS,
                "boxes and scores aligned",
                score=1.0,
            )

        return CheckResult(self.name, Status.SKIP, "not a detection model")


# =============================================================================
# Validator
# =============================================================================

class TaskOutputValidator:

    def __init__(
        self,
        config:  Optional[StrictOutputConfig] = None,
        schemas: Optional[List[OutputSchema]]  = None,
    ):

        self.config  = config or StrictOutputConfig()
        self.schemas = schemas or []

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
                    "schema_rank",
                    Status.FAIL,
                    f"{schema.name} rank mismatch",
                    structural=True,
                    score=0,
                ))

            if schema.last_dim_min and t.shape[-1] < schema.last_dim_min:

                results.append(CheckResult(
                    "schema_last_dim_min",
                    Status.FAIL,
                    f"{schema.name} last dim too small",
                    structural=True,
                    score=0,
                ))

        return results

    def validate(
        self,
        outputs: Dict[str, np.ndarray],
        task: TaskType = TaskType.UNKNOWN,   # FIX 2 — task param
    ) -> ValidationResult:
        """
        Validate model outputs against task-appropriate expectations.

        Parameters
        ----------
        outputs : name → tensor dict returned by the backend runner
        task    : TaskType for this inference pass — carried in the
                  ValidationResult for benchmark.py (Step 7) to use
                  when filtering task-relevant checks for the output table

        Returns
        -------
        ValidationResult — always returns, never raises.
        Soft mode  : WARN results don't block pipeline.
        Strict mode: structural WARNs promoted to FAIL.
        """
        checks: List[CheckResult] = []

        checks.extend(self._apply_schema(outputs))

        for check in self.checks:

            result = check.run(outputs)

            # Strict mode — promote structural warnings to failures
            if (
                self.config.is_strict          # FIX 1 — use is_strict property
                and result.status == Status.WARN
                and result.structural
            ):
                result = CheckResult(
                    result.name,
                    Status.FAIL,
                    result.message,
                    structural=True,
                    score=result.score,
                )

            checks.append(result)

        return ValidationResult(checks=checks, task=task)   # FIX 2 — pass task

    def relevant_checks(self, task: TaskType) -> List[Check]:
        """
        Return only the checks relevant for a given task.
        Used by benchmark.py to build the task-specific output table.
        """
        return [c for c in self.checks if c.relevant_for(task)]


# =============================================================================
# CLI Helpers
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
    config: StrictOutputConfig,   # FIX 1 — full config, not bare bool
) -> bool:
    """
    Return True if the CLI should exit with a non-zero code.

    Hard fail conditions:
    - Any FAIL status (structural or schema violation)
    - Strict mode active AND any WARN present
    """
    if result.failed():
        return True

    if config.is_strict and result.summary()["warn"] > 0:   # FIX 1
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

    lines = [f"⚠  Output validation [{result.task.value}]:"]

    for c in result.checks:
        if c.status in (Status.WARN, Status.FAIL):
            icon = "✗" if c.status == Status.FAIL else "⚠"
            lines.append(f"   {icon} {c.message}")

    return "\n".join(lines)