"""
Experiment and Run domain models.
Enterprise-grade lifecycle-enforced experiment tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, Mapping, Any
from types import MappingProxyType
from uuid import UUID
import json
import hashlib



# ============================================================
# Constants
# ============================================================

SCHEMA_VERSION = 1


# ============================================================
# Enums
# ============================================================

class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============================================================
# Structured Value Objects
# ============================================================

@dataclass(frozen=True)
class Metadata:
    """Validated metadata container (JSON-safe only)."""
    values: Mapping[str, Any]

    def __post_init__(self):
        self._validate_json_safe(self.values)
        object.__setattr__(
            self,
            "values",
            MappingProxyType(dict(sorted(self.values.items())))
        )

    @staticmethod
    def _validate_json_safe(obj):
        try:
            json.dumps(obj)
        except Exception:
            raise ValueError("Metadata must be JSON serializable.")


@dataclass(frozen=True)
class ParamSet:
    """Deterministic, JSON-safe parameters."""
    values: Mapping[str, Any]

    def __post_init__(self):
        Metadata._validate_json_safe(self.values)
        object.__setattr__(
            self,
            "values",
            MappingProxyType(dict(sorted(self.values.items())))
        )


@dataclass(frozen=True)
class MetricSet:
    """Scalar numeric metrics only (v1 constraint)."""
    scalars: Mapping[str, float]

    def __post_init__(self):
        normalized = {}
        for k, v in self.scalars.items():
            if not isinstance(v, (int, float)):
                raise ValueError("Metrics must be numeric scalars.")
            normalized[k] = float(v)

        object.__setattr__(
            self,
            "scalars",
            MappingProxyType(dict(sorted(normalized.items())))
        )


@dataclass(frozen=True)
class TagSet:
    """Strict string key-value tags."""
    labels: Mapping[str, str]

    def __post_init__(self):
        validated = {}
        for k, v in self.labels.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise ValueError("Tags must be string key-value pairs.")
            validated[k] = v

        object.__setattr__(
            self,
            "labels",
            MappingProxyType(dict(sorted(validated.items())))
        )


@dataclass(frozen=True)
class HardwareSnapshot:
    """Hardware environment capture for reproducibility."""
    device_name: str
    os_version: str
    cpu_cores: int
    accelerator: Optional[str]
    driver_version: Optional[str]


# ============================================================
# Experiment Model
# ============================================================

@dataclass
class Experiment:
    """
    Experiment container.

    Soft-delete semantics:
    - deleted_at != None means archived.
    - Archive is reversible.
    - Runs are not auto-deleted.
    """

    experiment_id: UUID
    name: str
    description: Optional[str]
    created_at: datetime
    metadata: Metadata
    deleted_at: Optional[datetime] = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if not isinstance(self.experiment_id, UUID):
            raise ValueError("experiment_id must be UUID.")

        if self.deleted_at and self.deleted_at < self.created_at:
            raise ValueError("deleted_at cannot precede created_at.")

    def archive(self, timestamp: datetime):
        if self.deleted_at is not None:
            raise ValueError("Experiment already archived.")
        self.deleted_at = timestamp

    def restore(self):
        if self.deleted_at is None:
            raise ValueError("Experiment is not archived.")
        self.deleted_at = None


# ============================================================
# Run Model
# ============================================================

@dataclass
class Run:
    """
    Lifecycle-enforced experimental run.
    Terminal states are immutable.
    """

    run_id: UUID
    experiment_id: UUID
    run_name: Optional[str]
    status: RunStatus

    started_at: Optional[datetime]
    ended_at: Optional[datetime]
    build_id: Optional[str]

    params: ParamSet
    metrics: Optional[MetricSet]
    tags: TagSet
    hardware: HardwareSnapshot

    schema_version: int = SCHEMA_VERSION

    _frozen: bool = field(default=False, init=False, repr=False)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def __post_init__(self):
        if not isinstance(self.run_id, UUID):
            raise ValueError("run_id must be UUID.")

        if not isinstance(self.experiment_id, UUID):
            raise ValueError("experiment_id must be UUID.")

        self._validate_state()

    def _validate_state(self):
        if self.status in {RunStatus.PENDING}:
            if self.started_at or self.ended_at:
                raise ValueError("Pending runs cannot have timestamps.")

        if self.status == RunStatus.RUNNING:
            if not self.started_at:
                raise ValueError("Running runs must have started_at.")
            if self.ended_at:
                raise ValueError("Running runs cannot have ended_at.")

        if self.status in {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }:
            if not self.started_at or not self.ended_at:
                raise ValueError("Terminal runs require timestamps.")
            if self.ended_at < self.started_at:
                raise ValueError("ended_at must be >= started_at.")

        if self.status == RunStatus.COMPLETED:
            if not self.metrics:
                raise ValueError("Completed runs must include metrics.")
            # build_id is optional for completed runs

    # --------------------------------------------------------
    # Lifecycle Transitions
    # --------------------------------------------------------

    def start(self, timestamp: datetime):
        self._assert_mutable()
        if self.status != RunStatus.PENDING:
            raise ValueError("Only pending runs can start.")
        self.status = RunStatus.RUNNING
        self.started_at = timestamp

    def complete(self, timestamp: datetime, metrics: MetricSet):
        self._assert_mutable()
        if self.status != RunStatus.RUNNING:
            raise ValueError("Only running runs can complete.")
        self.status = RunStatus.COMPLETED
        self.ended_at = timestamp
        self.metrics = metrics
        self._freeze()

    def fail(self, timestamp: datetime):
        self._assert_mutable()
        if self.status != RunStatus.RUNNING:
            raise ValueError("Only running runs can fail.")
        self.status = RunStatus.FAILED
        self.ended_at = timestamp
        self._freeze()

    def cancel(self, timestamp: datetime):
        self._assert_mutable()
        if self.status not in {RunStatus.PENDING, RunStatus.RUNNING}:
            raise ValueError("Cannot cancel completed run.")
        self.status = RunStatus.CANCELLED
        self.ended_at = timestamp
        self._freeze()

    # --------------------------------------------------------
    # Immutability Enforcement
    # --------------------------------------------------------

    def _freeze(self):
        self._frozen = True

    def _assert_mutable(self):
        if self._frozen:
            raise RuntimeError("Run is immutable after reaching terminal state.")

    # --------------------------------------------------------
    # Deterministic Serialization
    # --------------------------------------------------------

    def to_canonical_json(self) -> str:
        payload = {
            "run_id": str(self.run_id),
            "experiment_id": str(self.experiment_id),
            "run_name": self.run_name,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "build_id": self.build_id,
            "params": dict(self.params.values),
            "metrics": dict(self.metrics.scalars) if self.metrics else None,
            "tags": dict(self.tags.labels),
            "hardware": asdict(self.hardware),
            "schema_version": self.schema_version,
        }

        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def reproducibility_hash(self) -> str:
        canonical = self.to_canonical_json().encode()
        return hashlib.sha256(canonical).hexdigest()