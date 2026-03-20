from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Artifact:
    format: str           # "coreml" | "tflite" | "onnx"
    target: str           # "apple_m1" | "android_arm64" etc.
    quantize: str         # "fp32" | "fp16" | "int8"
    size_mb: float
    sha256: str           # full 64-char hash
    priority: int = 0     # 0 = primary, higher = lower priority


@dataclass
class BenchmarkRow:
    id: str
    compute_unit: str
    device: str
    p50_ms: float | None
    p95_ms: float | None
    runs: int
    warmup: int | None
    batch_size: int | None
    input_shape: str | None
    backend: str
    ran_at: datetime


@dataclass
class AccuracyRow:
    compared_to: str       # short build ID (8 chars)
    compared_to_full: str  # full build ID for --json
    primary_metric: str    # "cosine" | "top1" | "mae"
    threshold: float | None
    cosine: float | None
    mae: float | None
    top1: float | None
    dataset: str | None
    passed: bool
    ran_at: datetime


@dataclass
class BuildView:
    # Build
    id: str                # full 64-char
    id_short: str          # first 8 chars
    name: str | None
    created_at: datetime
    source: str            # "built" | "imported"

    # Task
    task_type: str         # "vision" | "nlp" | "audio" | "unknown"
    detection_tier: str    # "tier-1 (explicit)" | "tier-2 (inferred)" | "tier-3 (default)"

    # Artifacts — sorted by priority ascending
    artifacts: list[Artifact]

    # Benchmarks — sorted by ran_at descending
    benchmarks: list[BenchmarkRow]

    # Accuracy — empty list = section renders dimmed
    accuracy_records: list[AccuracyRow]

    # Tags + Notes
    tags: list[str] = field(default_factory=list)
    notes: str | None = None


# ── Prune models ───────────────────────────────────────────────

DELETE_BATCH_SIZE = 100


@dataclass
class PruneCandidate:
    id: str
    id_short: str
    name: str | None
    created_at: datetime
    tags: list[str]
    size_mb: float             # sum of ALL artifact files
    artifact_paths: list[str]  # deduped at delete time in registry
    protected: bool            # set by registry, never by CLI
    primary_format: str        # set by registry, CLI never derives it


@dataclass
class PrunePlan:
    candidates: list[PruneCandidate]   # will be deleted
    skipped: list[PruneCandidate]      # protected — never touched


@dataclass
class PruneResult:
    builds_deleted: int
    files_deleted: int
    bytes_reclaimed: int    # actual os.stat bytes only, never estimated
    file_errors: int        # non-fatal, tracked separately