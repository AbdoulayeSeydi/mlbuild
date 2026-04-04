from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, Any, List, Optional, TypedDict


# -----------------------------
# Enums  (unchanged)
# -----------------------------

class Runtime(str, Enum):
    COREML      = "coreml"
    ONNX_RUNTIME = "onnx_runtime"
    TFLITE      = "tflite"


class MeasurementType(str, Enum):
    DEVICE_LOCAL     = "device_local"
    DEVICE_CONNECTED = "device_connected"


class ComputeUnit(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    ANE = "ane"


# -----------------------------
# Typed Configs  (unchanged)
# -----------------------------

class QuantizationConfig(TypedDict):
    type: str           # 'fp32', 'fp16', 'int8'
    per_channel: bool


class OptimizerConfig(TypedDict):
    fuse_ops: bool
    constant_folding: bool


# -----------------------------
# Core Data Types
# -----------------------------

@dataclass(frozen=True)
class Build:
    """
    Represents a reproducible ML build artifact.

    Immutable and schema-versioned for tracking.
    Build ID includes environment fingerprint for full reproducibility.

    Schema v4 additions (Step 4 — ModelProfile storage):
      subtype              — behavioral subtype string
      execution_mode       — runtime execution mode string
      nms_inside           — detection NMS location flag
      state_optional       — partially-stateful flag
      model_profile_json   — full ModelProfile serialized as JSON
      benchmark_caveats    — machine-readable limitation notes
      input_roles          — tensor name → role string mapping
    """
    # Primary identifier
    build_id:        str   # SHA256(source_hash + config_hash + artifact_hash + env_fingerprint + version)

    # Core hashes
    artifact_hash:   str
    source_hash:     str
    config_hash:     str
    env_fingerprint: str

    # Human metadata
    name:       Optional[str]
    notes:      Optional[str]
    created_at: datetime

    # Source info
    source_path:   str
    target_device: str
    format:        str   # 'coreml' | 'tflite'

    # Configuration
    quantization:    Dict[str, Any]
    optimizer_config: Dict[str, Any]
    backend_versions: Dict[str, str]
    environment_data: Dict[str, Any]

    # System info
    mlbuild_version: str
    python_version:  str
    platform:        str
    os_version:      str

    # Artifact metadata
    artifact_path: str
    size_mb:       Decimal

    # v7: Optimization lineage and graph storage
    variant_id:          Optional[str] = None
    root_build_id:       Optional[str] = None
    parent_build_id:     Optional[str] = None
    optimization_pass:   Optional[str] = None
    optimization_method: Optional[str] = None
    weight_precision:    Optional[str] = None
    activation_precision: Optional[str] = None
    has_graph:    bool          = False
    graph_format: Optional[str] = None
    graph_path:   Optional[str] = None
    cached_latency_p50_ms:  Optional[float] = None
    cached_latency_p95_ms:  Optional[float] = None
    cached_memory_peak_mb:  Optional[float] = None

    task_type:      Optional[str] = None
    schema_version: int           = 4   # bumped from 3 → 4 for ModelProfile fields

    # ── Schema v4: ModelProfile fields ───────────────────────────────────────
    # All Optional/defaulted so records written before v4 remain readable.

    # Behavioral subtype (mirrors Subtype enum value).
    # "detection" | "timeseries" | "recommendation" | "generative_stateful"
    # | "multimodal" | "segmentation" | "none"
    subtype: Optional[str] = None

    # Runtime execution mode (mirrors ExecutionMode enum value).
    # "standard" | "stateful" | "partially_stateful" | "kv_cache" | "multi_input"
    execution_mode: Optional[str] = None

    # Detection-specific: True when NonMaxSuppression is baked into the graph.
    nms_inside: bool = False

    # Stateful-specific: True when state inputs are optional (PARTIALLY_STATEFUL).
    state_optional: bool = False

    # Full ModelProfile serialized as JSON for forward compatibility.
    # Includes domain, subtype, execution, confidence, confidence_tier,
    # nms_inside, state_optional.
    model_profile_json: Optional[str] = None

    # Machine-readable benchmark limitation notes.
    # Populated at build time; travels with benchmark records.
    # Pipelines should check this before presenting latency as authoritative.
    benchmark_caveats: List[str] = field(default_factory=list)

    # Input tensor role assignments: tensor_name → role_string.
    # Entries with value "unknown_float" or "unknown_int" signal that
    # the corresponding input may not reflect real-world values.
    input_roles: Dict[str, str] = field(default_factory=dict)

    # ── Stable accessors (unchanged) ─────────────────────────────────────────

    @property
    def quantization_type(self) -> str:
        if not self.quantization:
            return "unknown"
        if isinstance(self.quantization, dict):
            return self.quantization.get("type", "unknown")
        return getattr(self.quantization, "type", "unknown")

    @property
    def size_bytes(self) -> int:
        return int(self.size_mb * 1024 * 1024)

    @property
    def config(self) -> dict:
        return {
            "quantization": self.quantization,
            "optimizer":    self.optimizer_config,
        }

    def to_public_dict(self, include_hashes: bool = False) -> dict:
        data = {
            "build_id":         self.build_id,
            "name":             self.name,
            "target_device":    self.target_device,
            "format":           self.format,
            "size_bytes":       self.size_bytes,
            "size_mb":          float(self.size_mb),
            "created_at":       self.created_at,
            "notes":            self.notes,
            "quantization_type": self.quantization_type,
            "subtype":          self.subtype,
            "execution_mode":   self.execution_mode,
        }
        if include_hashes:
            data.update({
                "artifact_hash":  self.artifact_hash,
                "source_hash":    self.source_hash,
                "config_hash":    self.config_hash,
                "env_fingerprint": self.env_fingerprint,
            })
        return data


@dataclass(frozen=True)
class Benchmark:
    """Benchmark measurement result.  (unchanged)"""
    build_id:         str
    device_chip:      str
    runtime:          str
    measurement_type: str
    num_runs:         int
    measured_at:      datetime

    id:              Optional[int]   = None
    compute_unit:    Optional[str]   = None
    latency_p50_ms:  Optional[float] = None
    latency_p95_ms:  Optional[float] = None
    latency_p99_ms:  Optional[float] = None
    memory_peak_mb:  Optional[float] = None


@dataclass(frozen=True)
class Tag:
    """Build tag.  (unchanged)"""
    build_id:       str
    tag:            str
    created_at:     datetime       = field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: int            = 4


# -----------------------------
# Explore Result Types  (unchanged)
# -----------------------------

@dataclass
class VariantResult:
    build_id:            str
    method:              str
    size_mb:             float
    latency_p50_ms:      Optional[float]
    verdict:             str
    latency_delta_pct:   Optional[float] = None
    size_delta_pct:      Optional[float] = None
    _score:              float           = field(default=0.0, repr=False, compare=False)
    accuracy_passed:     Optional[bool]  = None
    accuracy_cosine:     Optional[float] = None
    accuracy_top1:       Optional[float] = None
    accuracy_failure:    Optional[str]   = None


@dataclass
class BackendResult:
    backend:  str
    variants: list[VariantResult] = field(default_factory=list)

    @property
    def recommended(self) -> Optional[VariantResult]:
        return next((v for v in self.variants if v.verdict == "recommended"), None)

    @property
    def baseline(self) -> Optional[VariantResult]:
        return next((v for v in self.variants if v.verdict == "baseline"), None)


@dataclass
class ExploreResult:
    name:        str
    source_path: str
    target:      str
    fast_mode:   bool
    backends:    list[BackendResult] = field(default_factory=list)

    @property
    def all_variants(self) -> list[VariantResult]:
        return [v for b in self.backends for v in b.variants]

    @property
    def total_registered(self) -> int:
        return len(self.all_variants)