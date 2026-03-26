"""
Model format detection and capability validation.

Design principles:
- Capability-based target families (not chip SKUs)
- Registry-driven format support
- Strong structural validation
- Input normalization
- Extensible to new formats/backends
- Deterministic API outputs
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, FrozenSet, Literal, Optional, Set


# ================================================================
# Types
# ================================================================

Format = Literal["tflite", "coreml", "onnx"]

TargetFamily = Literal[
    "apple_silicon",
    "apple_ios",
    "android",
    "linux_arm",
    "linux_x86",
    "onnxruntime",
]

Capability = Literal[
    "npu",
    "gpu",
    "cpu",
    "fp16",
    "int8",
]


# ================================================================
# Exceptions
# ================================================================

class ModelFormatError(ValueError):
    pass


class TargetCompatibilityError(ValueError):
    pass


# ================================================================
# Format Registry
# ================================================================

@dataclass(frozen=True)
class FormatSpec:
    name: Format
    validator: Callable[[Path], None]
    supported_targets: FrozenSet[TargetFamily]
    capabilities: FrozenSet[Capability]


class FormatRegistry:
    """
    Central registry for supported model formats.

    New formats (ONNX, TensorRT, etc.) should be registered here.
    """

    _registry: Dict[Format, FormatSpec] = {}

    @classmethod
    def register(cls, spec: FormatSpec) -> None:
        cls._registry[spec.name] = spec

    @classmethod
    def get(cls, fmt: str) -> FormatSpec:
        fmt = fmt.strip().lower()
        if fmt not in cls._registry:
            raise ModelFormatError(
                f"Unknown format '{fmt}'. Supported formats: "
                f"{sorted(cls._registry.keys())}"
            )
        return cls._registry[fmt]

    @classmethod
    def supported_formats(cls) -> Set[str]:
        return set(cls._registry.keys())


# ================================================================
# SKU → Family Mapping
# ================================================================

# Maps CLI --target chip SKUs to the capability-based target families
# used in FormatRegistry. Add new chips here only — registry stays clean.
_TARGET_TO_FAMILY: Dict[str, TargetFamily] = {
    # Apple Silicon (Mac)
    "apple_m1":       "apple_silicon",
    "apple_m2":       "apple_silicon",
    "apple_m3":       "apple_silicon",
    # Apple iOS (A-series)
    "apple_a15":      "apple_ios",
    "apple_a16":      "apple_ios",
    "apple_a17":      "apple_ios",
    "apple_a18":      "apple_ios",
    # Android
    "android_arm64":  "android",
    "android_arm32":  "android",
    "android_x86":    "android",
    # Linux edge
    "raspberry_pi":   "linux_arm",
    "coral_tpu":      "linux_arm",
    "generic_linux":  "linux_x86",
    # ONNX Runtime
    "onnxruntime_cpu": "onnxruntime",
    "onnxruntime_gpu": "onnxruntime",
    "onnxruntime_ane": "onnxruntime",
}


def resolve_target_family(target: str) -> TargetFamily:
    """
    Resolve a CLI target SKU to its capability-based family.

    Args:
        target: CLI target string e.g. "android_arm64", "apple_m1".

    Returns:
        TargetFamily string e.g. "android", "apple_silicon".

    Raises:
        TargetCompatibilityError: If the target SKU is not recognised.
    """
    target = target.strip().lower()
    if target not in _TARGET_TO_FAMILY:
        raise TargetCompatibilityError(
            f"Unrecognised target '{target}'. "
            f"Known targets: {sorted(_TARGET_TO_FAMILY.keys())}"
        )
    return _TARGET_TO_FAMILY[target]


# ================================================================
# TFLite Validation
# ================================================================

_FLATBUFFER_MAGIC_OFFSET = 4
_TFLITE_IDENTIFIERS = {b"TFL3", b"TFL2"}


def _validate_tflite(path: Path) -> None:
    if not path.is_file():
        raise ModelFormatError(f"'{path}' is not a valid file.")

    try:
        with path.open("rb") as f:
            header = f.read(8)
    except OSError as e:
        raise ModelFormatError(f"Unable to read '{path}': {e}") from e

    if len(header) < 8:
        raise ModelFormatError(
            f"'{path.name}' is too small to be a valid TFLite model."
        )

    identifier = header[_FLATBUFFER_MAGIC_OFFSET:_FLATBUFFER_MAGIC_OFFSET + 4]
    if identifier not in _TFLITE_IDENTIFIERS:
        raise ModelFormatError(
            f"'{path.name}' is not a valid TFLite FlatBuffer "
            "(expected identifier TFL3 or TFL2)."
        )


# ================================================================
# CoreML Validation
# ================================================================

def _validate_mlmodel(path: Path) -> None:
    """
    Validate legacy .mlmodel file.
    Basic structural verification — not full protobuf parsing.
    """
    if not path.is_file():
        raise ModelFormatError(f"'{path}' is not a valid file.")

    if path.stat().st_size < 128:
        raise ModelFormatError(
            f"'{path.name}' is too small to be a valid CoreML model."
        )


def _validate_mlpackage(path: Path) -> None:
    if not path.is_dir():
        raise ModelFormatError(f"'{path}' is not a directory.")

    manifest = path / "Manifest.json"
    data_dir = path / "Data"

    if not manifest.exists():
        raise ModelFormatError(
            f"Invalid .mlpackage: missing Manifest.json."
        )

    if not data_dir.exists() or not data_dir.is_dir():
        raise ModelFormatError(
            f"Invalid .mlpackage: missing Data directory."
        )

    try:
        manifest_data = json.loads(manifest.read_text())
    except Exception as e:
        raise ModelFormatError(
            f"Manifest.json is not valid JSON: {e}"
        ) from e

    if "rootModelIdentifier" not in manifest_data:
        raise ModelFormatError(
            "Invalid .mlpackage: Manifest.json missing "
            "'rootModelIdentifier'."
        )

    if not any(data_dir.rglob("*")):
        raise ModelFormatError(
            "Invalid .mlpackage: Data directory is empty."
        )


def _validate_coreml(path: Path) -> None:
    if path.suffix.lower() == ".mlmodel":
        _validate_mlmodel(path)
    elif path.suffix.lower() == ".mlpackage":
        _validate_mlpackage(path)
    else:
        raise ModelFormatError(
            f"Unsupported CoreML file type: {path.suffix}"
        )
    
# ================================================================
# ONNX Validation
# ================================================================

def _validate_onnx(path: Path) -> None:
    """Validate ONNX file via protobuf check."""
    if not path.is_file():
        raise ModelFormatError(f"'{path}' is not a valid file.")
    try:
        import onnx
        model = onnx.load(str(path))
        onnx.checker.check_model(model)
    except ImportError:
        raise ModelFormatError("onnx package required for ONNX import: pip install onnx")
    except Exception as e:
        raise ModelFormatError(f"Invalid ONNX model '{path.name}': {e}")


# ================================================================
# Detection
# ================================================================

def detect_and_validate_format(path: Path) -> Format:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".tflite":
        _validate_tflite(path)
        return "tflite"

    if suffix in {".mlmodel", ".mlpackage"}:
        _validate_coreml(path)
        return "coreml"

    if suffix == ".onnx":
        _validate_onnx(path)
        return "onnx"

    raise ModelFormatError(
        f"Unsupported model file type '{suffix}'. "
        "Supported: .tflite, .mlmodel, .mlpackage"
    )


# ================================================================
# Compatibility Validation
# ================================================================

def validate_format_target_compat(fmt: str, target: str) -> None:
    """
    Validate that a model format is compatible with the requested target device.

    Resolves the CLI target SKU (e.g. "android_arm64") to its capability-based
    family (e.g. "android") before checking against the format registry.

    Args:
        fmt:    Model format — "tflite" or "coreml".
        target: CLI target SKU (e.g. "android_arm64", "apple_m1").

    Raises:
        TargetCompatibilityError: If the SKU is unknown or incompatible.
        ModelFormatError: If the format is unknown.
    """
    fmt = fmt.strip().lower()

    # Resolve SKU → family (raises TargetCompatibilityError if unknown)
    family = resolve_target_family(target)

    spec = FormatRegistry.get(fmt)

    if family not in spec.supported_targets:
        raise TargetCompatibilityError(
            f"Format '{fmt}' is not compatible with target '{target}' "
            f"(family: '{family}').\n"
            f"Supported target families for '{fmt}': "
            f"{sorted(spec.supported_targets)}"
        )


def valid_targets_for_format(fmt: str) -> list[str]:
    spec = FormatRegistry.get(fmt)
    return sorted(spec.supported_targets)


def capabilities_for_format(fmt: str) -> list[str]:
    spec = FormatRegistry.get(fmt)
    return sorted(spec.capabilities)


# ================================================================
# Registry Initialization
# ================================================================

FormatRegistry.register(
    FormatSpec(
        name="tflite",
        validator=_validate_tflite,
        supported_targets=frozenset({
            "android",
            "linux_arm",
            "linux_x86",
        }),
        capabilities=frozenset({
            "cpu",
            "gpu",
            "npu",
            "int8",
            "fp16",
        }),
    )
)

FormatRegistry.register(
    FormatSpec(
        name="coreml",
        validator=_validate_coreml,
        supported_targets=frozenset({
            "apple_silicon",
            "apple_ios",
        }),
        capabilities=frozenset({
            "cpu",
            "gpu",
            "npu",
            "fp16",
        }),
    )
)

FormatRegistry.register(
    FormatSpec(
        name="onnx",
        validator=_validate_onnx,
        supported_targets=frozenset({"onnxruntime"}),
        capabilities=frozenset({"cpu", "gpu", "npu", "fp16", "int8"}),
    )
)