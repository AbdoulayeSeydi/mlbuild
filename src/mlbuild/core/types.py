from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, Any, Optional, TypedDict


# -----------------------------
# Enums
# -----------------------------
class Runtime(str, Enum):
    COREML = "coreml"
    ONNX_RUNTIME = "onnx_runtime"
    TFLITE = "tflite"


class MeasurementType(str, Enum):
    DEVICE_LOCAL = "device_local"
    DEVICE_CONNECTED = "device_connected"


class ComputeUnit(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    ANE = "ane"


# -----------------------------
# Typed Configs
# -----------------------------
class QuantizationConfig(TypedDict):
    type: str  # 'fp32', 'fp16', 'int8'
    per_channel: bool


class OptimizerConfig(TypedDict):
    fuse_ops: bool
    constant_folding: bool
    # Extendable with future optimization flags


# -----------------------------
# Core Data Types
# -----------------------------
@dataclass(frozen=True)
class Build:
    """
    Represents a reproducible ML build artifact.
    
    Immutable and schema-versioned for tracking.
    Build ID now includes environment fingerprint for full reproducibility.
    """
    # Primary identifier (deterministic fingerprint)
    build_id: str                           # SHA256(source_hash + config_hash + artifact_hash + env_fingerprint + mlbuild_version)
    
    # Core hashes (first-class, never inferred)
    artifact_hash: str                      # SHA256 of normalized CoreML artifact
    source_hash: str                        # SHA256 of source model file
    config_hash: str                        # SHA256 of build configuration
    env_fingerprint: str                    # SHA256 of environment data
    
    # Human metadata
    name: Optional[str]                     # Human-friendly build name
    notes: Optional[str]                    # Build notes (string, not dict)
    created_at: datetime                    # Timezone-aware UTC datetime
    
    # Source info
    source_path: str                        # Path to source model
    target_device: str                      # E.g., 'apple_a17'
    format: str                             # E.g., 'coreml'
    
    # Configuration (JSON-serializable)
    quantization: Dict[str, Any]            # Changed from TypedDict to Dict for flexibility
    optimizer_config: Dict[str, Any]        # Changed from TypedDict to Dict for flexibility
    backend_versions: Dict[str, str]
    environment_data: Dict[str, Any]        # Full environment snapshot
    
    # System info
    mlbuild_version: str
    python_version: str
    platform: str
    os_version: str
    
    # Artifact metadata
    artifact_path: str                      # Path to .mlbuild/artifacts
    size_mb: Decimal

    # v7: Optimization lineage and graph storage
    variant_id: Optional[str] = None
    root_build_id: Optional[str] = None
    parent_build_id: Optional[str] = None
    optimization_pass: Optional[str] = None
    optimization_method: Optional[str] = None
    weight_precision: Optional[str] = None
    activation_precision: Optional[str] = None
    has_graph: bool = False
    graph_format: Optional[str] = None
    graph_path: Optional[str] = None
    cached_latency_p50_ms: Optional[float] = None
    cached_latency_p95_ms: Optional[float] = None
    cached_memory_peak_mb: Optional[float] = None
    
    task_type: Optional[str] = None 
    schema_version: int = 3                 # Version of dataclass schema
    
    @property
    def quantization_type(self) -> str:
        """Stable accessor for quantization type."""
        if not self.quantization:
            return "unknown"
        if isinstance(self.quantization, dict):
            return self.quantization.get("type", "unknown")
        return getattr(self.quantization, "type", "unknown")

    @property
    def size_bytes(self) -> int:
        """Stable accessor for size in bytes."""
        return int(self.size_mb * 1024 * 1024)

    @property
    def config(self) -> dict:
        """Stable accessor for full configuration."""
        return {
            "quantization": self.quantization,
            "optimizer": self.optimizer_config,
        }

    def to_public_dict(self, include_hashes: bool = False) -> dict:
        """
        Stable public projection for CLI/JSON output.
        
        Does NOT expose internal/private fields.
        """
        data = {
            "build_id": self.build_id,
            "name": self.name,
            "target_device": self.target_device,
            "format": self.format,
            "size_bytes": self.size_bytes,
            "size_mb": float(self.size_mb),
            "created_at": self.created_at,
            "notes": self.notes,
            "quantization_type": self.quantization_type,
        }
        
        if include_hashes:
            data.update({
                "artifact_hash": self.artifact_hash,
                "source_hash": self.source_hash,
                "config_hash": self.config_hash,
                "env_fingerprint": self.env_fingerprint,
            })
        
        return data


@dataclass(frozen=True)
class Benchmark:
    """Benchmark measurement result."""
    # Required fields first (no defaults)
    build_id: str
    device_chip: str
    runtime: str
    measurement_type: str
    num_runs: int
    measured_at: datetime
    
    # Optional fields last (with defaults)
    id: Optional[int] = None
    compute_unit: Optional[str] = None
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    memory_peak_mb: Optional[float] = None


@dataclass(frozen=True)
class Tag:
    """
    Represents a build tag for grouping or classification.
    """
    build_id: str
    tag: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: int = 3


# -----------------------------
# Explore Result Types
# -----------------------------

@dataclass
class VariantResult:
    """Result for a single build variant in an explore sweep."""
    build_id: str
    method: str                          # "fp32" | "fp16" | "int8"
    size_mb: float
    latency_p50_ms: Optional[float]
    verdict: str                         # "baseline" | "recommended" | "aggressive" | "skip"
    latency_delta_pct: Optional[float] = None   # vs baseline, None for baseline
    size_delta_pct: Optional[float] = None
    _score: float = field(default=0.0, repr=False, compare=False)
    # Accuracy check results (populated by explore --check-accuracy)
    accuracy_passed: Optional[bool] = None
    accuracy_cosine: Optional[float] = None
    accuracy_top1: Optional[float] = None
    accuracy_failure: Optional[str] = None  # first failure reason, if any


@dataclass
class BackendResult:
    """All variant results for a single backend."""
    backend: str                         # "coreml" | "tflite"
    variants: list[VariantResult] = field(default_factory=list)

    @property
    def recommended(self) -> Optional[VariantResult]:
        return next((v for v in self.variants if v.verdict == "recommended"), None)

    @property
    def baseline(self) -> Optional[VariantResult]:
        return next((v for v in self.variants if v.verdict == "baseline"), None)


@dataclass
class ExploreResult:
    """Full result of an explore sweep across backends."""
    name: str
    source_path: str
    target: str
    fast_mode: bool
    backends: list[BackendResult] = field(default_factory=list)

    @property
    def all_variants(self) -> list[VariantResult]:
        return [v for b in self.backends for v in b.variants]

    @property
    def total_registered(self) -> int:
        return len(self.all_variants)