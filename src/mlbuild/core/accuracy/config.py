"""
Accuracy check configuration and result types.

Three frozen dataclasses — each with a strict role:

    AccuracyConfig     — policy (thresholds, profile, sampling)
    EvaluationContext  — runtime metadata (build IDs, task, dataset)
    AccuracyResult     — output artifact (metrics, pass/fail)

Accuracy gates (default profile):
    cosine_similarity  ≥ 0.99   (primary gate)
    top1_agreement     ≥ 0.99   (classifiers only)

Optional gates (never active unless explicitly set):
    mae_threshold
    rmse_threshold

Diagnostics (never gate):
    max_abs_error
    kl_divergence
    js_divergence
    error_p50 / p95 / p99

Design principles
-----------------
- All three types are frozen — no silent mutation in CI runs.
- Configuration is strictly validated at construction time.
- Results are immutable once produced.
- Domain objects contain no persistence logic.
- Pass/fail state is derived, never stored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


# ============================================================
# Profiles
# ============================================================

PROFILES: dict[str, dict] = {
    "strict": {
        "cosine_threshold": 0.999,
        "top1_threshold":   0.999,
        "mae_threshold":    0.001,
        "rmse_threshold":   0.005,
    },
    "default": {
        "cosine_threshold": 0.99,
        "top1_threshold":   0.99,
        "mae_threshold":    None,
        "rmse_threshold":   None,
    },
    "loose": {
        "cosine_threshold": 0.95,
        "top1_threshold":   0.95,
        "mae_threshold":    None,
        "rmse_threshold":   None,
    },
}

VALID_PROFILES = frozenset(PROFILES.keys())

# ============================================================
# Default thresholds (kept for backward compatibility)
# ============================================================

DEFAULT_THRESHOLDS: dict[str, float] = {
    "cosine_similarity": 0.99,
    "top1_agreement":    0.99,
}


# ============================================================
# AccuracyConfig — policy only
# ============================================================

@dataclass(frozen=True)
class AccuracyConfig:
    """
    Policy configuration for an accuracy check run.

    Thresholds, sampling parameters, and profile selection.
    Contains no runtime metadata — see EvaluationContext for that.
    """

    cosine_threshold: float        = 0.99
    top1_threshold:   float        = 0.99
    mae_threshold:    Optional[float] = None
    rmse_threshold:   Optional[float] = None

    samples: int = 32
    seed:    int = 42

    profile: Optional[str] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.cosine_threshold <= 1.0:
            raise ValueError("cosine_threshold must be within [0, 1]")
        if not 0.0 <= self.top1_threshold <= 1.0:
            raise ValueError("top1_threshold must be within [0, 1]")
        if self.mae_threshold is not None and self.mae_threshold < 0:
            raise ValueError("mae_threshold must be ≥ 0")
        if self.rmse_threshold is not None and self.rmse_threshold < 0:
            raise ValueError("rmse_threshold must be ≥ 0")
        if self.samples <= 0:
            raise ValueError("samples must be > 0")
        if self.seed < 0:
            raise ValueError("seed must be ≥ 0")
        if self.profile is not None and self.profile not in VALID_PROFILES:
            raise ValueError(
                f"profile must be one of {sorted(VALID_PROFILES)}, got '{self.profile}'"
            )

    @classmethod
    def default(cls) -> "AccuracyConfig":
        return cls()

    @classmethod
    def from_profile(cls, profile: str, **overrides) -> "AccuracyConfig":
        """
        Build a config from a named profile, with optional field overrides.

        Explicit overrides take precedence over profile defaults.

        Example
        -------
        # strict profile, but bump samples to 64
        cfg = AccuracyConfig.from_profile("strict", samples=64)
        """
        if profile not in VALID_PROFILES:
            raise ValueError(
                f"Unknown profile '{profile}'. "
                f"Valid options: {sorted(VALID_PROFILES)}"
            )
        base = dict(PROFILES[profile])
        base["profile"] = profile
        base.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**base)

    @classmethod
    def from_cli(
        cls,
        cosine_threshold: float | None  = None,
        top1_threshold:   float | None  = None,
        mae_threshold:    Optional[float] = None,
        rmse_threshold:   Optional[float] = None,
        samples:          int          = 32,
        seed:             int          = 42,
        profile:          Optional[str] = None,
    ) -> "AccuracyConfig":
        """
        Construct from CLI args.

        If profile is provided, it seeds defaults first.
        Any explicit non-None CLI flags override the profile.
        """
        if profile is not None:
            return cls.from_profile(
                profile,
                cosine_threshold=cosine_threshold,
                top1_threshold=top1_threshold,
                mae_threshold=mae_threshold,
                rmse_threshold=rmse_threshold,
                samples=samples,
                seed=seed,
            )
        return cls(
            cosine_threshold=cosine_threshold or 0.99,
            top1_threshold=top1_threshold     or 0.99,
            mae_threshold=mae_threshold,
            rmse_threshold=rmse_threshold,
            samples=samples,
            seed=seed,
        )


# ============================================================
# EvaluationContext — runtime metadata only
# ============================================================

VALID_TASK_TYPES = frozenset({"vision", "nlp", "audio", "multimodal", "unknown"})


@dataclass(frozen=True)
class EvaluationContext:
    """
    Runtime metadata for an accuracy check run.

    Carries identity and environment information.
    Contains no policy — see AccuracyConfig for that.

    task_type precedence (enforced by caller, not here):
        CLI --task  >  build.task_type  >  None
    """

    baseline_build_id:  str
    candidate_build_id: str

    task_type:    Optional[str] = None
    dataset_path: Optional[str] = None
    cross_format: bool          = False

    def __post_init__(self) -> None:
        if not self.baseline_build_id:
            raise ValueError("baseline_build_id must not be empty")
        if not self.candidate_build_id:
            raise ValueError("candidate_build_id must not be empty")
        if self.baseline_build_id == self.candidate_build_id:
            raise ValueError(
                "baseline_build_id and candidate_build_id must differ"
            )
        if (
            self.task_type is not None
            and self.task_type not in VALID_TASK_TYPES
        ):
            raise ValueError(
                f"task_type must be one of {sorted(VALID_TASK_TYPES)}, "
                f"got '{self.task_type}'"
            )


# ============================================================
# AccuracyResult — output artifact only
# ============================================================

@dataclass(frozen=True)
class AccuracyResult:
    """
    Result of an accuracy comparison between two builds.

    Immutable once produced. Never mutated after construction.

    New diagnostic fields (never gate CI by default):
        rmse          — root mean squared error across all outputs
        kl_divergence — KL(baseline || candidate), classifiers only
        js_divergence — Jensen-Shannon divergence, classifiers only, bounded [0, ln2]
        error_p50     — median absolute error
        error_p95     — 95th percentile absolute error
        error_p99     — 99th percentile absolute error
    """

    baseline_build_id:  str
    candidate_build_id: str

    cosine_similarity: float
    mean_abs_error:    float
    max_abs_error:     float
    rmse:              float
    top1_agreement:    Optional[float]

    # Classifier-only distribution metrics (None for non-classifiers)
    kl_divergence: Optional[float]
    js_divergence: Optional[float]

    # Percentile error breakdown
    error_p50: float
    error_p95: float
    error_p99: float

    num_samples: int
    seed:        int

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
            "rmse":               self.rmse,
            "top1_agreement":     self.top1_agreement,
            "kl_divergence":      self.kl_divergence,
            "js_divergence":      self.js_divergence,
            "error_p50":          self.error_p50,
            "error_p95":          self.error_p95,
            "error_p99":          self.error_p99,
            "num_samples":        self.num_samples,
            "seed":               self.seed,
            "passed":             1 if self.passed else 0,
            "created_at":         datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }