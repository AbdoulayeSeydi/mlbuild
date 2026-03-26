"""
Conversion output validator — robust, extensible, and production-ready.

Features:
- Extensible validator registry (decorator-based)
- Format-safe via enum
- Fail-fast on broken/corrupted artifacts
- Includes validator version in metrics
- Accurate metrics and honest warnings
- Zero-byte file and path normalization checks
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Callable, Optional

from mlbuild.convert.types import ValidationResult
from mlbuild.core.errors import ConvertError, ErrorCode


# ---------------------------------------------------------------------
# Format enum
# ---------------------------------------------------------------------

class Format(str, Enum):
    ONNX = "onnx"
    COREML = "coreml"
    TFLITE = "tflite"


# ---------------------------------------------------------------------
# Validator registry
# ---------------------------------------------------------------------

VALIDATORS: Dict[Format, Callable[[Path], ValidationResult]] = {}

def register_validator(fmt: Format):
    def wrapper(fn: Callable[[Path], ValidationResult]):
        if fmt in VALIDATORS:
            raise RuntimeError(f"Validator already registered for {fmt}")
        VALIDATORS[fmt] = fn
        return fn
    return wrapper


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------

def validate_output(path: Path, fmt: Format) -> ValidationResult:
    """
    Validate a conversion output artifact.

    Args:
        path: Path to the produced artifact
        fmt:  Format enum — ONNX, COREML, TFLITE

    Returns:
        ValidationResult with passed, warnings, and metrics.

    Raises:
        ConvertError if artifact is broken or unreadable.
    """
    path = path.resolve()

    if not path.exists() or path.stat().st_size == 0:
        raise ConvertError(
            f"Artifact missing or empty: {path}",
            stage=f"validate:{fmt.value}",
            error_code=ErrorCode.CONVERSION_FAILED,
        )

    validator_fn = VALIDATORS.get(fmt)
    if validator_fn is None:
        raise ConvertError(
            f"No validator registered for format '{fmt.value}'. "
            f"Known formats: {sorted(f.value for f in VALIDATORS)}",
            stage=f"validate:{fmt.value}",
            error_code=ErrorCode.CONVERSION_FAILED,
        )

    return validator_fn(path)


# ---------------------------------------------------------------------
# ONNX Validator
# ---------------------------------------------------------------------

@register_validator(Format.ONNX)
def _validate_onnx(path: Path) -> ValidationResult:
    warnings: List[str] = []
    metrics: Dict[str, Any] = {}

    try:
        import onnx
        import onnx.checker
    except ImportError:
        raise ConvertError(
            "ONNX is required for validation. Install: pip install onnx",
            stage="validate:onnx",
            error_code=ErrorCode.DEPENDENCY_MISSING,
        )

    try:
        # Validate from file directly (avoids loading entire model)
        onnx.checker.check_model(str(path))
    except onnx.checker.ValidationError as e:
        raise ConvertError(
            f"ONNX validation failed: {e}",
            stage="validate:onnx",
            error_code=ErrorCode.CONVERSION_FAILED,
        )
    except Exception as e:
        raise ConvertError(
            f"ONNX validation error: {e}",
            stage="validate:onnx",
            error_code=ErrorCode.CONVERSION_FAILED,
        )

    # Optional: load model for metrics only (can skip for performance)
    try:
        model = onnx.load(str(path))
        metrics["num_nodes"] = len(model.graph.node)
        # Count non-standard domains (honest metric)
        standard_domains = {"", "ai.onnx", "ai.onnx.ml"}
        non_standard_ops = [
            n.op_type for n in model.graph.node
            if n.domain not in standard_domains
        ]
        metrics["non_standard_ops"] = len(non_standard_ops)
        if non_standard_ops:
            warnings.append(
                f"{len(non_standard_ops)} nodes use non-standard domains: "
                f"{', '.join(sorted(set(non_standard_ops)))}"
            )
        # Record opset version
        metrics["opset_version"] = next(
            (o.version for o in model.opset_import if o.domain in ("", "ai.onnx")),
            None
        )
    except Exception:
        warnings.append("Failed to collect ONNX graph metrics")
        metrics["num_nodes"] = None
        metrics["non_standard_ops"] = None
        metrics["opset_version"] = None

    metrics["validator_version"] = getattr(onnx, "__version__", "unknown")

    return ValidationResult(
        passed=True,
        format=Format.ONNX.value,
        warnings=warnings,
        metrics=metrics,
    )


# ---------------------------------------------------------------------
# CoreML Validator
# ---------------------------------------------------------------------

@register_validator(Format.COREML)
def _validate_coreml(path: Path) -> ValidationResult:
    warnings: List[str] = []
    metrics: Dict[str, Any] = {}

    try:
        import coremltools as ct
    except ImportError:
        raise ConvertError(
            "coremltools is required for validation. Install: pip install coremltools",
            stage="validate:coreml",
            error_code=ErrorCode.DEPENDENCY_MISSING,
        )

    try:
        model = ct.models.MLModel(str(path))
        spec = model.get_spec()
        metrics["spec_version"] = getattr(spec, "specificationVersion", None)
        metrics["configured_compute_units"] = (
            str(model.compute_unit) if hasattr(model, "compute_unit") else "unknown"
        )
    except Exception as e:
        raise ConvertError(
            f"CoreML validation failed: {e}",
            stage="validate:coreml",
            error_code=ErrorCode.CONVERSION_FAILED,
        )

    metrics["validator_version"] = getattr(ct, "__version__", "unknown")

    return ValidationResult(
        passed=True,
        format=Format.COREML.value,
        warnings=warnings,
        metrics=metrics,
    )


# ---------------------------------------------------------------------
# TFLite Validator
# ---------------------------------------------------------------------

@register_validator(Format.TFLITE)
def _validate_tflite(path: Path) -> ValidationResult:
    warnings: List[str] = []
    metrics: Dict[str, Any] = {}

    try:
        import tensorflow as tf
    except ImportError:
        raise ConvertError(
            "TensorFlow is required for TFLite validation. Install: pip install tensorflow",
            stage="validate:tflite",
            error_code=ErrorCode.DEPENDENCY_MISSING,
        )

    try:
        interpreter = tf.lite.Interpreter(model_path=str(path))
        interpreter.allocate_tensors()
    except Exception as e:
        raise ConvertError(
            f"TFLite validation failed: {e}",
            stage="validate:tflite",
            error_code=ErrorCode.CONVERSION_FAILED,
        )

    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        metrics["io_tensor_count"] = len(input_details) + len(output_details)
    except Exception:
        warnings.append("Failed to collect TFLite IO tensor count")
        metrics["io_tensor_count"] = None

    # No delegate fallback info without inference run
    metrics["unsupported_ops"] = None
    metrics["cpu_fallback_ops"] = None
    metrics["validator_version"] = getattr(tf, "__version__", "unknown")

    return ValidationResult(
        passed=True,
        format=Format.TFLITE.value,
        warnings=warnings,
        metrics=metrics,
    )