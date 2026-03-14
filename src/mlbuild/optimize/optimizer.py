"""
Variant registration system for MLBuild.

Architecture
------------

optimizer
   ↓
PassResult
   ↓
VariantBuilder
   ↓
ArtifactStore
   ↓
BuildHasher
   ↓
Registry
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, Optional

from ..core.types import Build
from ..core.hash import compute_artifact_hash, compute_source_hash
from ..registry.local import LocalRegistry

logger = logging.getLogger(__name__)


# ============================================================
# PassResult abstraction (NEW)
# ============================================================


class PassResult:
    """
    Standard result returned by optimization passes.

    Every optimizer pass returns this object so the optimizer
    never needs to know pass-specific metadata.

    This allows MLBuild to scale to many passes without turning
    the optimizer into a metadata router.
    """

    def __init__(
        self,
        artifact_path: Path,
        method: str,
        weight_precision: str,
        activation_precision: str,
        backend_versions: Optional[dict] = None,
    ):
        self.artifact_path = artifact_path
        self.method = method
        self.weight_precision = weight_precision
        self.activation_precision = activation_precision
        self.backend_versions = backend_versions or {}


# ============================================================
# Identity / hashing
# ============================================================


def _sha256_hex(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


class BuildHasher:
    """Centralized build identity generation."""

    @staticmethod
    def compute_config_hash(source_config: str, method: str, precision: str) -> str:
        return _sha256_hex(f"{source_config}:{method}:{precision}")

    @staticmethod
    def compute_build_id(source_hash: str, config_hash: str, artifact_hash: str) -> str:
        return _sha256_hex(f"{source_hash}{config_hash}{artifact_hash}")


# ============================================================
# Artifact storage
# ============================================================


class ArtifactStore:
    """
    Content-addressable artifact storage.

    Guarantees:
    • atomic writes
    • deduplication by artifact hash
    • size tracking
    """

    def __init__(self, artifacts_root: Path):
        self.root = artifacts_root

    def store(self, artifact_path: Path, artifact_hash: str) -> Tuple[Path, int]:

        final_dir = self.root / artifact_hash

        if final_dir.exists():
            return final_dir, self._compute_size(final_dir)

        tmp_dir = self.root / f".tmp_{uuid.uuid4().hex}"

        if artifact_path.is_dir():
            shutil.copytree(artifact_path, tmp_dir)
        else:
            tmp_dir.mkdir(parents=True)
            shutil.copy2(artifact_path, tmp_dir / artifact_path.name)

        tmp_dir.rename(final_dir)

        size_bytes = self._compute_size(final_dir)

        return final_dir, size_bytes

    def _compute_size(self, path: Path) -> int:
        size = 0
        for f in path.rglob("*"):
            if f.is_file():
                size += f.stat().st_size
        return size


# ============================================================
# Variant builder
# ============================================================


class VariantBuilder:

    def __init__(self, source: Build):
        self.source = source
        self.fields = {}

    @classmethod
    def from_source(cls, source: Build):
        return cls(source)

    def with_identity(self, build_id, artifact_hash, config_hash):
        self.fields.update(
            {
                "build_id": build_id,
                "artifact_hash": artifact_hash,
                "config_hash": config_hash,
            }
        )
        return self

    def with_artifact(self, artifact_dir: str, size_mb: float):
        self.fields.update(
            {
                "artifact_path": artifact_dir,
                "size_mb": size_mb,
            }
        )
        return self

    def with_quantization(self, method, weight_precision, activation_precision):
        self.fields.update(
            {
                "optimization_pass": "quantization",
                "optimization_method": method,
                "weight_precision": weight_precision,
                "activation_precision": activation_precision,
            }
        )
        return self

    def with_backend_versions(self, backend_versions: dict):
        self.fields["backend_versions"] = backend_versions
        return self

    def build(self) -> Build:
        source = self.source
        data = {**source.__dict__}
        data.update(self.fields)
        data.update(
            {
                "parent_build_id": source.build_id,
                "root_build_id": source.root_build_id or source.build_id,
                "created_at": datetime.now(timezone.utc),
                "notes": f"Optimized from {source.build_id[:16]}",
                "variant_id": f"{data.get('optimization_method', 'opt')}-{data['build_id'][:8]}",
            }
        )
        return Build(**data)


# ============================================================
# Optimizer entrypoints
# ============================================================


def register_variant_from_pass(
    source: Build,
    result: PassResult,
    registry: LocalRegistry,
) -> Build:
    """
    Register variant using PassResult.

    Optimizer only interacts with PassResult objects,
    making the optimizer independent from pass metadata.
    """

    logger.info(
        "variant_registration_start",
        extra={
            "parent_build": source.build_id,
            "method": result.method,
        },
    )

    artifact_hash = (
        compute_artifact_hash(result.artifact_path)
        if Path(result.artifact_path).is_dir()
        else compute_source_hash(result.artifact_path)
    )

    artifact_store = ArtifactStore(registry.artifacts_root)

    artifact_dir, size_bytes = artifact_store.store(
        result.artifact_path,
        artifact_hash,
    )

    size_mb = size_bytes / (1024 * 1024)

    config_hash = BuildHasher.compute_config_hash(
        source.config_hash,
        result.method,
        result.weight_precision,
    )

    build_id = BuildHasher.compute_build_id(
        source.source_hash,
        config_hash,
        artifact_hash,
    )

    variant = (
        VariantBuilder.from_source(source)
        .with_identity(build_id, artifact_hash, config_hash)
        .with_artifact(str(artifact_dir), size_mb)
        .with_quantization(
            result.method,
            result.weight_precision,
            result.activation_precision,
        )
        .with_backend_versions(result.backend_versions)
        .build()
    )

    registry.save_build(variant)

    logger.info(
        "variant_registered",
        extra={
            "build_id": build_id,
            "parent": source.build_id,
            "method": result.method,
        },
    )

    return variant

from .passes.quantization import QuantizationPass
from .passes.pruning import PruningPass, PruningPassError
from .backends.coreml_backend import CoreMLBackend
from .backends.tflite_backend import TFLiteBackend

BACKENDS = {
    "coreml": CoreMLBackend(),
    "tflite": TFLiteBackend(),
}

SUPPORTED_METHODS = ["fp16", "int8"]


class OptimizeError(RuntimeError):
    """Raised when optimize() cannot proceed."""


def optimize(
    build_id: str,
    registry: LocalRegistry,
    method: Optional[str] = None,
    benchmark: bool = True,
    graphs_root: Optional[Path] = None,
    calibration_data: Optional[Path] = None,
) -> list[Build]:
    """
    Entry point for mlbuild optimize command.

    Parameters
    ----------
    build_id         : source build to optimize (exact or prefix)
    registry         : LocalRegistry instance
    method           : 'fp16' or 'int8'. None = auto sweep (both methods)
    benchmark        : run benchmark after registration
    graphs_root      : override for .mlbuild/graphs/ location
    calibration_data : path to calibration data for static INT8

    Returns
    -------
    List of newly registered variant Builds (1 per method).
    """
    # Resolve build
    source = registry.resolve_build(build_id)
    if source is None:
        raise OptimizeError(f"Build not found: {build_id}")

    # Validate format
    if source.format not in BACKENDS:
        raise OptimizeError(
            f"No optimization backend for format '{source.format}'. "
            f"Supported: {list(BACKENDS)}"
        )

    # Validate precision direction
    _validate_precision_direction(source, method)

    # Warn if optimizing an already-optimized variant
    if source.parent_build_id is not None:
        logger.warning(
            f"Build {source.build_id[:16]} is already a variant "
            f"(parent={source.parent_build_id[:16]}). "
            f"Consider optimizing the root build instead: {source.root_build_id[:16]}"
        )
    
    # Resolve graphs root
    if graphs_root is None:
        graphs_root = registry.db_path.parent / "graphs"

    backend = BACKENDS[source.format]
    pass_ = QuantizationPass(backend)

    methods = [method] if method else SUPPORTED_METHODS
    variants = []

    for m in methods:
        # Use int8_static as the method name when calibration data is provided.
        # This ensures static and dynamic INT8 builds get distinct build IDs
        # because method flows into config_hash → build_id.
        effective_method = "int8_static" if (m == "int8" and calibration_data) else m

        logger.info(f"optimize start build={source.build_id[:16]} method={effective_method}")

        try:
            result_dict = pass_.apply(
                source, m, graphs_root,
                calibration_data=calibration_data if m == "int8" else None,
            )

            result = PassResult(
                artifact_path=result_dict["artifact_path"],
                method=effective_method,
                weight_precision=result_dict["weight_precision"],
                activation_precision=result_dict["activation_precision"],
            )

            # Check if this variant already exists before registering
            artifact_hash = (
                compute_artifact_hash(result.artifact_path)
                if Path(result.artifact_path).is_dir()
                else compute_source_hash(result.artifact_path)
            )
            config_hash = BuildHasher.compute_config_hash(
                source.config_hash, effective_method, result.weight_precision
            )
            candidate_id = BuildHasher.compute_build_id(
                source.source_hash, config_hash, artifact_hash
            )
            existing = registry.get_build(candidate_id)
            if existing:
                logger.info(f"variant already exists method={m} build={candidate_id[:16]}, skipping")
                variants.append(existing)
                continue

            variant = register_variant_from_pass(source, result, registry)
            variants.append(variant)

            if benchmark:
                try:
                    delta = run_benchmark_delta(source, variant, registry)
                    logger.info(
                        f"benchmark delta method={m} "
                        f"latency={delta['latency_delta_pct']:.1f}% "
                        f"size={delta['size_delta_pct']:.1f}%"
                    )
                except Exception as e:
                    logger.warning(f"benchmark delta failed: {e}")

        except Exception as e:
            logger.exception(f"optimization failed for method={m}: {e}")
            # skip this method and continue with others
            continue

    return variants


def _validate_precision_direction(source: Build, method: Optional[str]) -> None:
    """Block precision upgrades (INT8 → FP16 is not allowed)."""
    if source.weight_precision == "int8" and method == "fp16":
        raise OptimizeError(
            f"Cannot upgrade precision: source build is already INT8. "
            f"FP16 → INT8 is a lossy reduction; INT8 → FP16 is not supported."
        )
    # int8_static is a valid re-optimization of a dynamic int8 build — allow it
    if method in ("int8_static",):
        return
    if source.weight_precision is not None and method is not None:
        if source.weight_precision == METHOD_TO_PRECISION.get(method):
            raise OptimizeError(
                f"Build {source.build_id[:16]} is already {method}. "
                f"To re-optimize, use the root build: {source.root_build_id or source.build_id}"
            )


# Import needed for _validate_precision_direction
from .passes.quantization import METHOD_TO_PRECISION



def prune(
    build_id: str,
    registry: LocalRegistry,
    sparsity: float = 0.5,
    benchmark: bool = True,
    graphs_root: Optional[Path] = None,
) -> list[Build]:
    """
    Entry point for mlbuild optimize --pass prune.

    Parameters
    ----------
    build_id   : source build to prune (exact or prefix)
    registry   : LocalRegistry instance
    sparsity   : fraction of weights to zero (0.0-1.0)
    benchmark  : run benchmark after registration
    graphs_root: override for .mlbuild/graphs/ location

    Returns
    -------
    List containing the single pruned Build.
    """
    source = registry.resolve_build(build_id)
    if source is None:
        raise OptimizeError(f"Build not found: {build_id}")

    if source.format not in BACKENDS:
        raise OptimizeError(
            f"No pruning backend for format '{source.format}'. "
            f"Supported: {list(BACKENDS)}"
        )

    if source.parent_build_id is not None:
        logger.warning(
            f"Build {source.build_id[:16]} is already a variant. "
            f"Consider pruning the root build: {source.root_build_id[:16]}"
        )

    if graphs_root is None:
        graphs_root = registry.db_path.parent / "graphs"

    coreml_backend = CoreMLBackend()
    pass_ = PruningPass(coreml_backend=coreml_backend)

    method = f"prune_{sparsity:.2f}"

    logger.info(f"prune start build={source.build_id[:16]} sparsity={sparsity}")

    try:
        result_dict = pass_.apply(
            source=source,
            sparsity=sparsity,
            graphs_root=graphs_root,
            registry=registry,
        )
    except PruningPassError as e:
        raise OptimizeError(str(e)) from e

    result = PassResult(
        artifact_path=result_dict["artifact_path"],
        method=method,
        weight_precision=result_dict["weight_precision"],
        activation_precision=result_dict["activation_precision"],
    )

    # Check if variant already exists
    from ..core.hash import compute_artifact_hash, compute_source_hash
    artifact_hash = (
        compute_artifact_hash(result.artifact_path)
        if Path(result.artifact_path).is_dir()
        else compute_source_hash(result.artifact_path)
    )
    config_hash = BuildHasher.compute_config_hash(
        source.config_hash, method, result.weight_precision
    )
    candidate_id = BuildHasher.compute_build_id(
        source.source_hash, config_hash, artifact_hash
    )
    existing = registry.get_build(candidate_id)
    if existing:
        logger.info(f"pruned variant already exists build={candidate_id[:16]}, skipping")
        return [existing]

    variant = register_variant_from_pass(source, result, registry)

    if benchmark:
        try:
            delta = run_benchmark_delta(source, variant, registry)
            logger.info(
                f"benchmark delta sparsity={sparsity} "
                f"latency={delta['latency_delta_pct']:.1f}% "
                f"size={delta['size_delta_pct']:.1f}%"
            )
        except Exception as e:
            logger.warning(f"benchmark delta failed: {e}")

    return [variant]


def run_benchmark_delta(
    source: Build,
    variant: Build,
    registry: LocalRegistry,
    runs: int = 100,
    warmup: int = 20,
) -> dict:
    """
    Benchmark source and variant, store cached results, return delta.

    Returns
    -------
    dict with keys: source_p50, variant_p50, latency_delta_ms,
                    latency_delta_pct, size_delta_pct
    """
    from ..benchmark.runner import CoreMLBenchmarkRunner, ComputeUnit

    results = {}

    for label, build in [("source", source), ("variant", variant)]:
        try:
            runner = CoreMLBenchmarkRunner(
                model_path=build.artifact_path,
                compute_unit=ComputeUnit.ALL,
                warmup_runs=warmup,
                benchmark_runs=runs,
            )
            result = runner.run(build_id=build.build_id)

            registry.update_cached_benchmark(
                build_id=build.build_id,
                latency_p50_ms=result.latency_p50,
                latency_p95_ms=result.latency_p95,
                memory_peak_mb=result.memory_peak_mb or 0.0,
            )

            results[label] = result
            logger.info(
                f"benchmark complete build={build.build_id[:16]} "
                f"p50={result.latency_p50:.2f}ms"
            )

        except Exception as e:
            logger.warning(f"benchmark failed for {label} build={build.build_id[:16]}: {e}")
            results[label] = None

    source_result = results.get("source")
    variant_result = results.get("variant")

    if source_result and variant_result:
        latency_delta_ms = variant_result.latency_p50 - source_result.latency_p50
        latency_delta_pct = (latency_delta_ms / source_result.latency_p50) * 100
    else:
        latency_delta_ms = None
        latency_delta_pct = None

    source_size = float(source.size_mb)
    variant_size = float(variant.size_mb)
    size_delta_pct = ((variant_size - source_size) / source_size) * 100 if source_size > 0 else None

    return {
        "source_p50": source_result.latency_p50 if source_result else None,
        "variant_p50": variant_result.latency_p50 if variant_result else None,
        "latency_delta_ms": latency_delta_ms,
        "latency_delta_pct": latency_delta_pct,
        "size_delta_pct": size_delta_pct,
    }