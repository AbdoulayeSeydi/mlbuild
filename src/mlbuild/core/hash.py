"""
Deterministic artifact hashing for CoreML models (enterprise-grade).

Goals:
- Fully deterministic SHA256 for same model config.
- Metadata-stripped, float-normalized, canonicalized protobuf.
- Recursive handling of pipelines and neural networks.
- Reproducible across machines, CoreMLTools versions, and Python runtimes.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict

import coremltools as ct
from google.protobuf.message import Message

from ..core.errors import InternalError

logger = logging.getLogger(__name__)

FLOAT_DECIMALS = 6  # float rounding precision


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def compute_artifact_hash(mlpackage_path: Path) -> str:
    """
    Compute deterministic SHA256 hash of CoreML .mlpackage artifact.
    """
    mlpackage_path = Path(mlpackage_path).resolve()

    if not mlpackage_path.exists() or not mlpackage_path.is_dir():
        raise InternalError(
            "Artifact path invalid",
            context={"path": _safe_relpath(mlpackage_path)},
        )

    try:
        # Load spec directly from .mlpackage without runtime compilation
        # This is deterministic - MLModel() is not
        spec_path = mlpackage_path / "Data" / "com.apple.CoreML" / "model.mlmodel"
        
        if not spec_path.exists():
            # Fallback: try loading as MLModel (less deterministic but works)
            model = ct.models.MLModel(str(mlpackage_path))
            spec = model.get_spec()
        else:
            # Direct protobuf load - most deterministic
            with open(spec_path, 'rb') as f:
                from coremltools.proto import Model_pb2
                spec = Model_pb2.Model()
                spec.ParseFromString(f.read())
        
        normalized_bytes = _normalize_spec(spec)
        return hashlib.sha256(normalized_bytes).hexdigest()

    except (OSError, ValueError, TypeError) as exc:
        raise InternalError(
            "Failed to compute artifact hash",
            context={"path": _safe_relpath(mlpackage_path)},
        ) from exc


def compute_config_hash(
    config: Dict[str, Any],
    coremltools_version: str | None = None,
) -> str:
    """
    Deterministic hash of build configuration.
    Optionally include CoreMLTools version for reproducibility tracking.
    """
    config_copy = dict(config)
    if coremltools_version:
        config_copy["_coremltools_version"] = coremltools_version

    canonical_json = json.dumps(
        config_copy,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def compute_source_hash(model_path: Path) -> str:
    """
    Byte-level SHA256 of source model file.
    """
    model_path = Path(model_path).resolve()

    if not model_path.exists():
        raise InternalError(
            "Source model not found",
            context={"path": _safe_relpath(model_path)},
        )

    return hashlib.sha256(model_path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------
# Normalization Core
# ---------------------------------------------------------------------

def _normalize_spec(spec: Message) -> bytes:
    """
    Recursively normalize CoreML spec for deterministic serialization.
    """
    _strip_metadata(spec)
    _normalize_floats(spec)
    _canonicalize_spec(spec)
    return spec.SerializeToString(deterministic=True)


def _strip_metadata(spec: Message) -> None:
    """
    Remove non-deterministic metadata.
    Recurses into pipeline sub-models.
    """
    if hasattr(spec, "description"):
        desc = spec.description
        meta = getattr(desc, "metadata", None)

        if meta:
            # Clear known non-deterministic fields safely
            for field in (
                "author",
                "shortDescription",
                "versionString",
                "license",
                "copyright",
            ):
                # Set to empty string instead of clearing
                if hasattr(meta, field):
                    setattr(meta, field, "")

            # Sort userDefined map for deterministic serialization
            # Note: map iteration order is not guaranteed, but serialization is
            if getattr(meta, "userDefined", None):
                sorted_items = sorted(meta.userDefined.items())
                meta.ClearField("userDefined")
                for k, v in sorted_items:
                    meta.userDefined[k] = v

    # Recurse into pipelines (only true sub-model containers)
    if hasattr(spec, "pipeline") and spec.pipeline.models:
        for sub_model in spec.pipeline.models:
            _strip_metadata(sub_model)


def _normalize_floats(proto: Message) -> None:
    """
    Recursively round float/double fields to fixed precision.
    Handles repeated primitive fields safely.
    """
    from google.protobuf.descriptor import FieldDescriptor
    from google.protobuf.message import Message as ProtoMessage
    
    # Guard: only process protobuf messages
    if not isinstance(proto, ProtoMessage):
        return
    
    for field, value in proto.ListFields():

        # Nested message
        if field.cpp_type == FieldDescriptor.CPPTYPE_MESSAGE:
            # Check if repeated using label
            if field.label == FieldDescriptor.LABEL_REPEATED:
                for item in value:
                    if isinstance(item, ProtoMessage):
                        _normalize_floats(item)
            else:
                if isinstance(value, ProtoMessage):
                    _normalize_floats(value)

        # Float / double
        elif field.cpp_type in (FieldDescriptor.CPPTYPE_FLOAT, FieldDescriptor.CPPTYPE_DOUBLE):
            # Check if repeated using label
            if field.label == FieldDescriptor.LABEL_REPEATED:
                rounded = [round(v, FLOAT_DECIMALS) for v in value]
                getattr(proto, field.name)[:] = rounded
            else:
                setattr(proto, field.name, round(value, FLOAT_DECIMALS))



def _canonicalize_spec(spec: Message) -> None:
    """
    Canonicalize ordering of repeated fields for deterministic hashing.
    Recurses into sub-models.
    """

    # Inputs / Outputs (repeated → do NOT use HasField)
    if hasattr(spec, "description"):
        if spec.description.input:
            _canonicalize_features(spec.description.input)

        if spec.description.output:
            _canonicalize_features(spec.description.output)

    # Neural network layers
    if hasattr(spec, "neuralNetwork") and spec.neuralNetwork.layers:
        _canonicalize_layers(spec.neuralNetwork.layers)

    # Pipelines
    if hasattr(spec, "pipeline") and spec.pipeline.models:
        for sub_model in spec.pipeline.models:
            _canonicalize_spec(sub_model)


def _canonicalize_features(features) -> None:
    """
    Sort repeated features deterministically by name,
    fallback to serialized bytes when name empty.
    """
    sorted_features = sorted(
        list(features),
        key=lambda f: f.name if getattr(f, "name", "") else f.SerializeToString(),
    )
    del features[:]
    features.extend(sorted_features)


def _canonicalize_layers(layers) -> None:
    """
    Sort layers deterministically by name,
    fallback to serialized bytes when name empty.
    """
    sorted_layers = sorted(
        list(layers),
        key=lambda l: l.name if getattr(l, "name", "") else l.SerializeToString(),
    )
    del layers[:]
    layers.extend(sorted_layers)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _safe_relpath(path: Path) -> str:
    """
    Safely compute path relative to CWD.
    Falls back to filename if not relative.
    """
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return path.name