"""
MLBuild CI Threshold System

Provides:
- Immutable threshold configuration
- Strict config loading from .mlbuild/config.toml
- CLI override support
- Deterministic regression and budget evaluation
- Structured violation reporting for CI systems
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, List, Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# TOML IMPORT (resolved once at module load)
# ---------------------------------------------------------------------

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]


# ---------------------------------------------------------------------
# TYPES
# ---------------------------------------------------------------------

Metric = Literal["latency", "latency_p95", "memory", "size", "accuracy_cosine", "accuracy_top1"]
Rule = Literal["regression", "budget", "accuracy"]


@dataclass(frozen=True)
class ThresholdViolation:
    """
    Structured violation used by CI systems.

    Fields are intentionally explicit so JSON serialization
    produces predictable output for dashboards and GitHub actions.
    """

    metric: Metric
    rule: Rule
    actual: float
    threshold: float
    unit: str

    @property
    def message(self) -> str:
        if self.rule == "regression":
            return f"{self.metric} regressed {self.actual:.2f}{self.unit} (threshold: {self.threshold}{self.unit})"
        elif self.rule == "budget":
            return f"{self.metric} {self.actual:.2f}{self.unit} exceeds budget {self.threshold}{self.unit}"
        elif self.rule == "accuracy":
            return f"accuracy {self.metric} {self.actual:.4f} below threshold {self.threshold:.4f}"
        return f"{self.metric} violation: {self.actual} > {self.threshold}"

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "rule": self.rule,
            "actual": self.actual,
            "threshold": self.threshold,
            "unit": self.unit,
            "message": self.message,
        }


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ThresholdConfig:
    """
    Immutable configuration used for CI evaluation.
    """

    # Relative regressions (% change from baseline)
    latency_regression_pct: float = 10.0
    size_regression_pct: float = 5.0

    # Absolute budgets
    latency_budget_ms: Optional[float] = None
    p95_budget_ms:     Optional[float] = None
    memory_budget_mb:  Optional[float] = None
    size_budget_mb: Optional[float] = None

    # Accuracy thresholds
    cosine_threshold: float = 0.99
    top1_threshold: float = 0.99

    # -----------------------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------------------

    def __post_init__(self):
        if self.latency_regression_pct < 0:
            raise ValueError("latency_regression_pct must be >= 0")

        if self.size_regression_pct < 0:
            raise ValueError("size_regression_pct must be >= 0")

        if self.cosine_threshold < 0 or self.cosine_threshold > 1:
            raise ValueError("cosine_threshold must be between 0 and 1")

        if self.top1_threshold < 0 or self.top1_threshold > 1:
            raise ValueError("top1_threshold must be between 0 and 1")

        if self.latency_budget_ms is not None and self.latency_budget_ms < 0:
            raise ValueError("latency_budget_ms must be >= 0")

        if self.p95_budget_ms is not None and self.p95_budget_ms < 0:
            raise ValueError("p95_budget_ms must be >= 0")

        if self.memory_budget_mb is not None and self.memory_budget_mb < 0:
            raise ValueError("memory_budget_mb must be >= 0")

        if self.size_budget_mb is not None and self.size_budget_mb < 0:
            raise ValueError("size_budget_mb must be >= 0")

    # -----------------------------------------------------------------
    # CONFIG LOADING
    # -----------------------------------------------------------------

    @classmethod
    def from_workspace(cls, workspace_root: Path) -> "ThresholdConfig":
        """
        Load configuration from `.mlbuild/config.toml`.

        Behavior:
        - Missing file → defaults
        - Parse error → fail CI
        """

        config_path = workspace_root / ".mlbuild" / "config.toml"

        if not config_path.exists():
            logger.debug("No MLBuild config found, using defaults.")
            return cls()

        if tomllib is None:
            logger.warning("tomllib/tomli not available — using default thresholds")
            return cls()

        try:
            with config_path.open("rb") as f:
                data = tomllib.load(f)
        except Exception as e:
            raise RuntimeError(f"Invalid .mlbuild/config.toml: {e}") from e

        ci = data.get("ci", {})
        acc = ci.get("accuracy", {})

        return cls(
            latency_regression_pct=float(ci.get("latency_regression_pct", 10.0)),
            size_regression_pct=float(ci.get("size_regression_pct", 5.0)),
            latency_budget_ms=(
                float(ci["latency_budget_ms"]) if "latency_budget_ms" in ci else None
            ),
            p95_budget_ms=(
                float(ci["p95_budget_ms"]) if "p95_budget_ms" in ci else None
            ),
            memory_budget_mb=(
                float(ci["memory_budget_mb"]) if "memory_budget_mb" in ci else None
            ),
            size_budget_mb=(
                float(ci["size_budget_mb"]) if "size_budget_mb" in ci else None
            ),
            cosine_threshold=float(acc.get("cosine_threshold", 0.99)),
            top1_threshold=float(acc.get("top1_threshold", 0.99)),
        )

    # -----------------------------------------------------------------
    # CLI OVERRIDES
    # -----------------------------------------------------------------

    def apply_overrides(
        self,
        latency_regression_pct: Optional[float] = None,
        size_regression_pct: Optional[float] = None,
        latency_budget_ms: Optional[float] = None,
        p95_budget_ms: Optional[float] = None,
        memory_budget_mb: Optional[float] = None,
        size_budget_mb: Optional[float] = None,
        cosine_threshold: Optional[float] = None,
        top1_threshold: Optional[float] = None,
    ) -> "ThresholdConfig":
        """
        Return new config with CLI overrides applied.
        """

        overrides = {}

        if latency_regression_pct is not None:
            overrides["latency_regression_pct"] = float(latency_regression_pct)

        if size_regression_pct is not None:
            overrides["size_regression_pct"] = float(size_regression_pct)

        if latency_budget_ms is not None:
            overrides["latency_budget_ms"] = float(latency_budget_ms)

        if p95_budget_ms is not None:
            overrides["p95_budget_ms"] = float(p95_budget_ms)

        if memory_budget_mb is not None:
            overrides["memory_budget_mb"] = float(memory_budget_mb)

        if size_budget_mb is not None:
            overrides["size_budget_mb"] = float(size_budget_mb)

        if cosine_threshold is not None:
            overrides["cosine_threshold"] = float(cosine_threshold)

        if top1_threshold is not None:
            overrides["top1_threshold"] = float(top1_threshold)

        return replace(self, **overrides)

    # -----------------------------------------------------------------
    # EVALUATION
    # -----------------------------------------------------------------

    def evaluate(
        self,
        baseline_latency: Optional[float],
        candidate_latency: Optional[float],
        baseline_size: float,
        candidate_size: float,
        candidate_p95_latency: Optional[float] = None,
        candidate_memory_mb: Optional[float] = None,
        cosine_similarity: Optional[float] = None,
        top1_agreement: Optional[float] = None,
    ) -> List[ThresholdViolation]:
        """
        Evaluate all regression and budget constraints.

        Returns list of violations. Empty list = pass.
        """

        violations: List[ThresholdViolation] = []

        if baseline_size <= 0:
            raise ValueError("baseline_size must be > 0")

        # -------------------------------------------------------------
        # LATENCY REGRESSION
        # -------------------------------------------------------------

        if baseline_latency is not None and candidate_latency is not None:
            delta_pct = ((candidate_latency - baseline_latency) / baseline_latency) * 100
            delta_pct = round(delta_pct, 4)

            if delta_pct > self.latency_regression_pct:
                violations.append(
                    ThresholdViolation(
                        metric="latency",
                        rule="regression",
                        actual=delta_pct,
                        threshold=self.latency_regression_pct,
                        unit="percent",
                    )
                )

        # -------------------------------------------------------------
        # SIZE REGRESSION
        # -------------------------------------------------------------

        delta_size_pct = ((candidate_size - baseline_size) / baseline_size) * 100
        delta_size_pct = round(delta_size_pct, 4)

        if delta_size_pct > self.size_regression_pct:
            violations.append(
                ThresholdViolation(
                    metric="size",
                    rule="regression",
                    actual=delta_size_pct,
                    threshold=self.size_regression_pct,
                    unit="percent",
                )
            )

        # -------------------------------------------------------------
        # LATENCY BUDGET
        # -------------------------------------------------------------

        if self.latency_budget_ms is not None and candidate_latency is not None:
            if candidate_latency > self.latency_budget_ms:
                violations.append(
                    ThresholdViolation(
                        metric="latency",
                        rule="budget",
                        actual=round(candidate_latency, 4),
                        threshold=self.latency_budget_ms,
                        unit="ms",
                    )
                )

        # -------------------------------------------------------------
        # P95 LATENCY BUDGET
        # -------------------------------------------------------------

        if self.p95_budget_ms is not None and candidate_p95_latency is not None:
            if candidate_p95_latency > self.p95_budget_ms:
                violations.append(
                    ThresholdViolation(
                        metric="latency_p95",
                        rule="budget",
                        actual=round(candidate_p95_latency, 4),
                        threshold=self.p95_budget_ms,
                        unit="ms",
                    )
                )

        # -------------------------------------------------------------
        # MEMORY BUDGET
        # -------------------------------------------------------------

        if self.memory_budget_mb is not None and candidate_memory_mb is not None:
            if candidate_memory_mb > self.memory_budget_mb:
                violations.append(
                    ThresholdViolation(
                        metric="memory",
                        rule="budget",
                        actual=round(candidate_memory_mb, 4),
                        threshold=self.memory_budget_mb,
                        unit="MB",
                    )
                )

        # -------------------------------------------------------------
        # SIZE BUDGET
        # -------------------------------------------------------------

        if self.size_budget_mb is not None:
            if candidate_size > self.size_budget_mb:
                violations.append(
                    ThresholdViolation(
                        metric="size",
                        rule="budget",
                        actual=round(candidate_size, 4),
                        threshold=self.size_budget_mb,
                        unit="MB",
                    )
                )

        # -------------------------------------------------------------
        # ACCURACY VALIDATION
        # -------------------------------------------------------------

        if cosine_similarity is not None:
            if cosine_similarity < self.cosine_threshold:
                violations.append(
                    ThresholdViolation(
                        metric="accuracy_cosine",
                        rule="accuracy",
                        actual=round(cosine_similarity, 6),
                        threshold=self.cosine_threshold,
                        unit="ratio",
                    )
                )

        if top1_agreement is not None:
            if top1_agreement < self.top1_threshold:
                violations.append(
                    ThresholdViolation(
                        metric="accuracy_top1",
                        rule="accuracy",
                        actual=round(top1_agreement, 6),
                        threshold=self.top1_threshold,
                        unit="ratio",
                    )
                )

        return violations

    def check_regression(
        self,
        baseline_latency,
        candidate_latency,
        baseline_size,
        candidate_size,
        candidate_p95_latency=None,
        candidate_memory_mb=None,
        cosine_similarity=None,
        top1_agreement=None,
    ):
        """Alias for evaluate() — used by CIRunner."""
        return self.evaluate(
            baseline_latency=baseline_latency,
            candidate_latency=candidate_latency,
            baseline_size=baseline_size,
            candidate_size=candidate_size,
            candidate_p95_latency=candidate_p95_latency,
            candidate_memory_mb=candidate_memory_mb,
            cosine_similarity=cosine_similarity,
            top1_agreement=top1_agreement,
        )