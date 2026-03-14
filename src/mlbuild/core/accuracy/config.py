"""
Accuracy check configuration and result types.

Accuracy gates (default):
    cosine_similarity  ≥ 0.99   (primary gate)
    top1_agreement     ≥ 0.99   (classifiers only)

Diagnostics (never gate by default):
    mean_abs_error
    max_abs_error

MAE becomes a gate only if mae_threshold is explicitly provided.

Design principles
-----------------
• Configuration is strictly validated.
• Results are immutable once produced.
• Domain objects contain no persistence logic.
• Pass/fail state is derived, not stored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


# ============================================================
# Default thresholds
# ============================================================

DEFAULT_THRESHOLDS: dict[str, float] = {
    "cosine_similarity": 0.99,
    "top1_agreement":    0.99,
}


# ============================================================
# AccuracyConfig
# ============================================================

@dataclass(frozen=True)
class AccuracyConfig:
    """
    Configuration for an accuracy check run.
    """

    cosine_threshold: float = 0.99
    top1_threshold: float = 0.99
    mae_threshold: Optional[float] = None

    samples: int = 32
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0 <= self.cosine_threshold <= 1:
            raise ValueError("cosine_threshold must be within [0, 1]")
        if not 0 <= self.top1_threshold <= 1:
            raise ValueError("top1_threshold must be within [0, 1]")
        if self.mae_threshold is not None and self.mae_threshold < 0:
            raise ValueError("mae_threshold must be ≥ 0")
        if self.samples <= 0:
            raise ValueError("samples must be > 0")
        if self.seed < 0:
            raise ValueError("seed must be ≥ 0")

    @classmethod
    def default(cls) -> "AccuracyConfig":
        return cls()

    @classmethod
    def from_cli(
        cls,
        cosine_threshold: float = 0.99,
        top1_threshold: float = 0.99,
        mae_threshold: Optional[float] = None,
        samples: int = 32,
        seed: int = 42,
    ) -> "AccuracyConfig":
        return cls(
            cosine_threshold=cosine_threshold,
            top1_threshold=top1_threshold,
            mae_threshold=mae_threshold,
            samples=samples,
            seed=seed,
        )


# ============================================================
# AccuracyResult
# ============================================================

@dataclass(frozen=True)
class AccuracyResult:
    """
    Result of an accuracy comparison between two builds.

    Instances represent immutable facts and should never be mutated.
    """

    baseline_build_id: str
    candidate_build_id: str

    cosine_similarity: float
    mean_abs_error: float
    max_abs_error: float
    top1_agreement: Optional[float]

    num_samples: int
    seed: int

    failure_reasons: Tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return len(self.failure_reasons) == 0

    def as_db_row(self) -> dict:
        from datetime import datetime, timezone
        return {
            "baseline_build_id":  self.baseline_build_id,
            "candidate_build_id": self.candidate_build_id,
            "cosine_similarity":  self.cosine_similarity,
            "mean_abs_error":     self.mean_abs_error,
            "max_abs_error":      self.max_abs_error,
            "top1_agreement":     self.top1_agreement,
            "num_samples":        self.num_samples,
            "seed":               self.seed,
            "passed":             1 if self.passed else 0,
            "created_at":         datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }