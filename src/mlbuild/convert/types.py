from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal
from enum import Enum


# ---------------------------------------------------------------------
# Enums — eliminate all stringly-typed footguns
# ---------------------------------------------------------------------

class QuantizeMode(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"


class LoadMode(str, Enum):
    AUTO = "auto"
    JIT = "jit"
    EAGER = "eager"


class TargetDevice(str, Enum):
    APPLE_M1 = "apple_m1"
    APPLE_M2 = "apple_m2"
    APPLE_M3 = "apple_m3"
    APPLE_A15 = "apple_a15"
    APPLE_A16 = "apple_a16"
    APPLE_A17 = "apple_a17"
    APPLE_A18 = "apple_a18"


class ConvertStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------
# PathStep — graph edge with reasoning
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class PathStep:
    src: str
    dst: str
    reason: str


# ---------------------------------------------------------------------
# ConvertParams — strict, validated inputs
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ConvertParams:
    input_shape: Tuple[int, ...]
    quantize: QuantizeMode
    load_mode: LoadMode
    opset: Optional[int]
    target: Optional[TargetDevice]
    name: Optional[str]
    notes: Optional[str]


# ---------------------------------------------------------------------
# RunMetadata — environment + determinism context
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    env_versions: Dict[str, Optional[str]]  # torch, onnx, tf, etc.
    deterministic: bool
    seed: Optional[int]


# ---------------------------------------------------------------------
# ConvertContext — executor input contract
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ConvertContext:
    input_path: Path
    output_dir: Path

    src_format: str
    dst_format: str

    step_index: int
    total_steps: int

    params: ConvertParams
    run: RunMetadata

    previous_step: Optional["StepResult"] = None


# ---------------------------------------------------------------------
# ConvertOutput — executor output contract (source of truth)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ConvertOutput:
    path: Path
    converter_version: str

    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    duration_seconds: Optional[float] = None


# ---------------------------------------------------------------------
# ValidationResult — validator output ONLY
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    format: str

    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# CacheInfo — explicit cache traceability
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class CacheInfo:
    hit: bool
    cache_key: Optional[str] = None
    source_build_id: Optional[str] = None


# ---------------------------------------------------------------------
# StepResult — fully structured, no duplication
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class StepResult:
    src_format: str
    dst_format: str

    output_path: Path
    file_size_mb: float

    duration_seconds: float

    conversion: ConvertOutput
    validation: ValidationResult

    cache: CacheInfo

    build_id: Optional[str]
    parent_build_id: Optional[str]


# ---------------------------------------------------------------------
# ConvertResult — full pipeline result with explicit state
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ConvertResult:
    status: ConvertStatus

    final_path: Optional[Path]
    final_format: Optional[str]
    final_build_id: Optional[str]

    steps: Tuple[StepResult, ...]

    intermediate_build_ids: Tuple[str, ...]
    cache_hits: Tuple[str, ...]

    total_duration_seconds: float

    run: RunMetadata


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

__all__ = [
    "QuantizeMode",
    "LoadMode",
    "TargetDevice",
    "ConvertStatus",
    "PathStep",
    "ConvertParams",
    "RunMetadata",
    "ConvertContext",
    "ConvertOutput",
    "ValidationResult",
    "CacheInfo",
    "StepResult",
    "ConvertResult",
]