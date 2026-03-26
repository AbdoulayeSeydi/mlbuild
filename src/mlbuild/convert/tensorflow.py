"""
ONNX/SavedModel → TFLite conversion executors.

Two registered edges:
    ("onnx",       "tflite") — convert_onnx_to_tflite
    ("savedmodel", "tflite") — convert_savedmodel_to_tflite

ONNX → TFLite:
    Delegates to the existing TFLiteConverter (onnx2tf under the hood).

SavedModel → TFLite:
    Uses tf.lite.TFLiteConverter.from_saved_model() directly.

Quantization:
    fp32 — default, no compression
    fp16 — OPTIMIZE_FOR_SIZE
    int8 — hard ConvertError (requires representative dataset)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List

from mlbuild.convert.graph import register_conversion
from mlbuild.convert.types import ConvertContext, ConvertOutput
from mlbuild.core.errors import ConvertError, ErrorCode
from mlbuild.core.hash import compute_source_hash

logger = logging.getLogger("mlbuild.convert.tensorflow")

CONVERTER_VERSION_OX = "onnx_to_tflite_v2"
CONVERTER_VERSION_TF = "savedmodel_to_tflite_v2"


# ---------------------------------------------------------------------
# Quantization guard
# ---------------------------------------------------------------------

def _check_quantize(quantize: str, stage: str) -> None:
    if quantize == "int8":
        raise ConvertError(
            "int8 quantization requires a representative dataset "
            "and is not currently supported. Use --quantize fp16 instead.",
            stage=stage,
            error_code=ErrorCode.CONVERSION_FAILED,
        )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _resolve_single_tflite(output_dir: Path, stage: str) -> Path:
    candidates = sorted(output_dir.glob("*.tflite"))

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        raise ConvertError(
            f"Multiple TFLite outputs found: {[str(c) for c in candidates]}",
            stage=stage,
            error_code=ErrorCode.CONVERSION_FAILED,
        )

    raise ConvertError(
        f"No .tflite output found in {output_dir}",
        stage=stage,
        error_code=ErrorCode.CONVERSION_FAILED,
    )


def _validate_nonempty_file(path: Path, stage: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise ConvertError(
            f"Invalid or empty TFLite artifact: {path}",
            stage=stage,
            error_code=ErrorCode.CONVERSION_FAILED,
        )


# ---------------------------------------------------------------------
# Registered executors
# ---------------------------------------------------------------------

@register_conversion("onnx", "tflite")
def convert_onnx_to_tflite(ctx: ConvertContext) -> ConvertOutput:
    """
    Executor: onnx → tflite
    Deterministic, validated, observable.
    """
    try:
        from mlbuild.backends.tflite.converter import (
            TFLiteConverter,
            TFLiteConversionError,
        )
    except ImportError as e:
        raise ConvertError(
            f"TFLite backend unavailable: {e}. "
            "Install: pip install tensorflow onnx2tf",
            stage="convert_onnx_to_tflite",
            error_code=ErrorCode.DEPENDENCY_MISSING,
        )

    start = time.time()
    warnings: List[str] = []

    params = ctx.params
    quantize = params.quantize.value  # strict — no fallback

    _check_quantize(quantize, stage="convert_onnx_to_tflite")

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    expected_output = ctx.output_dir / f"{ctx.input_path.stem}.tflite"

    try:
        converter = TFLiteConverter()

        t0 = time.time()
        converter.convert(
            onnx_path=ctx.input_path,
            output_path=expected_output,
            quantization=quantize,
        )
        convert_duration = time.time() - t0

    except TFLiteConversionError as e:
        raise ConvertError(
            f"ONNX → TFLite conversion failed: {e}",
            stage="convert_onnx_to_tflite",
            error_code=ErrorCode.CONVERSION_FAILED,
        )

    # Resolve actual output deterministically
    output_path = expected_output if expected_output.exists() else _resolve_single_tflite(
        ctx.output_dir,
        stage="convert_onnx_to_tflite",
    )

    # Validate artifact integrity
    _validate_nonempty_file(output_path, stage="convert_onnx_to_tflite")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    sha256 = compute_source_hash(output_path)

    # Backend version
    try:
        import onnx2tf
        backend_version = onnx2tf.__version__
    except Exception:
        backend_version = None
        warnings.append("Failed to detect onnx2tf version")

    duration = time.time() - start

    return ConvertOutput(
        path=output_path,
        converter_version=CONVERTER_VERSION_OX,
        metadata={
            "backend": "onnx2tf",
            "backend_version": backend_version,
            "quantize": quantize,
            "quantization_impl": "onnx2tf",
            "size_mb": size_mb,
            "sha256": sha256,
            "convert_time_sec": convert_duration,
            "warnings": warnings,
        },
        warnings=warnings,
        duration_seconds=duration,
    )


@register_conversion("savedmodel", "tflite")
def convert_savedmodel_to_tflite(ctx: ConvertContext) -> ConvertOutput:
    """
    Executor: savedmodel → tflite
    Uses TensorFlow native converter with validation + determinism.
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ConvertError(
            "tensorflow is required for SavedModel conversion. "
            "Install: pip install tensorflow",
            stage="convert_savedmodel_to_tflite",
            error_code=ErrorCode.DEPENDENCY_MISSING,
        )

    start = time.time()
    warnings: List[str] = []

    params = ctx.params
    quantize = params.quantize.value  # strict

    _check_quantize(quantize, stage="convert_savedmodel_to_tflite")

    # Strong validation (not file existence)
    try:
        tf.saved_model.load(str(ctx.input_path))
    except Exception as e:
        raise ConvertError(
            f"Invalid SavedModel: {e}",
            stage="convert_savedmodel_to_tflite",
            error_code=ErrorCode.MODEL_LOAD_FAILED,
        )

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = ctx.output_dir / f"{ctx.input_path.name}.tflite"

    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(str(ctx.input_path))

        if quantize == "fp16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

        t0 = time.time()
        tflite_model = converter.convert()
        convert_duration = time.time() - t0

        output_path.write_bytes(tflite_model)

    except Exception as e:
        raise ConvertError(
            f"SavedModel → TFLite conversion failed: {e}",
            stage="convert_savedmodel_to_tflite",
            error_code=ErrorCode.CONVERSION_FAILED,
        )

    # Validate artifact
    _validate_nonempty_file(output_path, stage="convert_savedmodel_to_tflite")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    sha256 = compute_source_hash(output_path)

    duration = time.time() - start

    return ConvertOutput(
        path=output_path,
        converter_version=CONVERTER_VERSION_TF,
        metadata={
            "backend": "tensorflow",
            "backend_version": tf.__version__,
            "quantize": quantize,
            "quantization_impl": "tensorflow_native",
            "size_mb": size_mb,
            "sha256": sha256,
            "convert_time_sec": convert_duration,
            "warnings": warnings,
        },
        warnings=warnings,
        duration_seconds=duration,
    )