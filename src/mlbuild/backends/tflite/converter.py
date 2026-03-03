"""
Production-Grade ONNX → TFLite Conversion Pipeline

Conversion Flow:
1. ONNX → TFLite (via onnx2tf, direct conversion)
2. Apply quantization with optional calibration
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional, Callable, Iterator, List, Any

# Lazy imports for TensorFlow and ONNX
def _import_tf_and_onnx() -> tuple[Any, Any, Any]:
    try:
        import tensorflow as tf
        import onnx2tf
        import onnx
        return tf, onnx2tf, onnx
    except ImportError as e:
        raise ImportError(
            "TensorFlow, ONNX, and onnx2tf are required for TFLite conversion. "
            "Install with: pip install mlbuild[tflite]"
        ) from e


class TFLiteConversionError(Exception):
    """Raised when ONNX → TFLite conversion fails."""


class TFLiteConverter:
    """
    Convert ONNX models to TensorFlow Lite format efficiently and safely.

    Supports:
    - FP32 (full precision)
    - FP16 (half precision)
    - INT8 (integer quantization, requires representative dataset)
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("mlbuild.tflite")
        self.tf, self.onnx2tf, self.onnx = _import_tf_and_onnx()

    def convert(
        self,
        onnx_path: Path,
        output_path: Path,
        quantization: str = "fp32",
        representative_dataset: Optional[Callable[[], Iterator[List[Any]]]] = None,
        saved_model_path: Optional[Path] = None,
        input_shapes: Optional[list] = None,
    ) -> Path:
        """
        Convert an ONNX model to TFLite format.

        Args:
            onnx_path: Path to ONNX model.
            output_path: Path to save TFLite model.
            quantization: "fp32", "fp16", or "int8".
            representative_dataset: Callable generator for INT8 calibration.
            saved_model_path: Unused, kept for API compatibility.
            input_shapes: Optional list of (name, shape) tuples, unused directly
                          (onnx2tf handles shape inference internally).

        Returns:
            Path to generated TFLite file.

        Raises:
            TFLiteConversionError: If conversion fails at any step.
        """
        self.logger.info(f"Starting conversion: {onnx_path} → {quantization.upper()} TFLite")

        if not onnx_path.exists():
            raise TFLiteConversionError(f"ONNX model not found: {onnx_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if quantization not in ("fp32", "fp16", "int8"):
            raise TFLiteConversionError(
                f"Unknown quantization: {quantization}. Must be one of: fp32, fp16, int8"
            )

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir = Path(tmp_dir)

                # onnx2tf writes <stem>.tflite into the output folder
                self.logger.info(f"Converting ONNX → TFLite via onnx2tf (quant={quantization})")
                self.onnx2tf.convert(
                    input_onnx_file_path=str(onnx_path),
                    output_folder_path=str(tmp_dir),
                    output_dynamic_range_quantized_tflite=(quantization == "fp16"),
                    output_integer_quantized_tflite=(quantization == "int8"),
                    not_use_onnxsim=False,
                    non_verbose=True,
                    verbosity="error",
                )

                # Find the generated .tflite file
                tflite_files = list(tmp_dir.glob("*.tflite"))
                if not tflite_files:
                    raise TFLiteConversionError(
                        f"onnx2tf did not produce a .tflite file in {tmp_dir}"
                    )

                # Move to final output path
                import shutil
                shutil.move(str(tflite_files[0]), str(output_path))

        except TFLiteConversionError:
            raise
        except Exception as e:
            raise TFLiteConversionError(f"ONNX → TFLite conversion failed: {e}") from e

        size_mb = output_path.stat().st_size / (1024 * 1024)
        self.logger.info(f"Saved TFLite model: {output_path} ({size_mb:.2f} MB)")

        return output_path

    def _apply_quantization(
        self,
        converter: "self.tf.lite.TFLiteConverter",
        quantization: str,
        representative_dataset: Optional[Callable[[], Iterator[List[Any]]]],
    ):
        """Apply quantization settings to TFLite converter."""

        if quantization not in ("fp32", "fp16", "int8"):
            raise ValueError(f"Unknown quantization: {quantization}. Must be one of: fp32, fp16, int8")

        if quantization == "fp32":
            self.logger.info("Using FP32 (full precision)")
            return

        converter.optimizations = [self.tf.lite.Optimize.DEFAULT]

        if quantization == "fp16":
            self.logger.info("Applying FP16 quantization")
            converter.target_spec.supported_types = [self.tf.float16]

        elif quantization == "int8":
            if representative_dataset is None or not callable(representative_dataset):
                self.logger.warning(
                    "INT8 quantization requested without a valid representative dataset. "
                    "Falling back to dynamic range quantization (weights only)."
                )
                converter.target_spec.supported_ops = [self.tf.lite.OpsSet.TFLITE_BUILTINS]
            else:
                self.logger.info("Applying INT8 quantization with calibration dataset")
                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [self.tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = self.tf.int8
                converter.inference_output_type = self.tf.int8


class TFLiteValidator:
    """
    Validate TFLite models safely, supporting multiple inputs and dynamic shapes.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("mlbuild.tflite")
        self.tf, _, _ = _import_tf_and_onnx()

    def validate(self, tflite_path: Path) -> dict:
        """
        Validate a TFLite model.

        Args:
            tflite_path: Path to TFLite file.

        Returns:
            Dictionary with validation results.

        Raises:
            FileNotFoundError: If TFLite file does not exist.
            RuntimeError: If model fails to load in TFLite interpreter.
        """
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite model not found: {tflite_path}")

        try:
            interpreter = self.tf.lite.Interpreter(model_path=str(tflite_path))
            interpreter.allocate_tensors()
        except Exception as e:
            raise RuntimeError(f"Failed to load TFLite model: {e}") from e

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        self.logger.info(f"Validated TFLite model: {tflite_path}, inputs: {len(input_details)}, outputs: {len(output_details)}")

        return {
            "valid": True,
            "inputs": [
                {"shape": inp["shape"].tolist(), "dtype": str(inp["dtype"])}
                for inp in input_details
            ],
            "outputs": [
                {"shape": out["shape"].tolist(), "dtype": str(out["dtype"])}
                for out in output_details
            ],
            "size_bytes": tflite_path.stat().st_size,
            "size_mb": tflite_path.stat().st_size / (1024 * 1024),
        }


def example_representative_dataset(input_shape, num_samples: int = 100) -> Callable[[], Iterator[List[Any]]]:
    """
    Example representative dataset generator for INT8 calibration.
    WARNING: Only for testing. Use real data in production.

    Args:
        input_shape: Tuple matching model input shape (e.g., (1, 224, 224, 3))
        num_samples: Number of calibration samples

    Returns:
        Generator function for calibration
    """
    import numpy as np

    def generator():
        for _ in range(num_samples):
            data = np.random.randn(*input_shape).astype(np.float32)
            yield [data]

    return generator