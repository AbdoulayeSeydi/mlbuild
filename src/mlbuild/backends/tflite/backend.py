"""
TensorFlow Lite Backend for MLBuild
Enterprise-grade TFLite backend for mobile and edge deployment.
"""

from __future__ import annotations

import hashlib
import logging
import platform
from pathlib import Path
from typing import Optional, Callable, Iterator, List, Tuple, Dict, Any

from ..base import Backend, BackendCapabilities, EnvironmentValidation
from ...core.types import Build
from ...core.hash import compute_source_hash

# Import converter & validator lazily
from .converter import TFLiteConverter, TFLiteValidator, TFLiteConversionError
from .benchmark_runner import TFLiteBenchmarkRunner

logger = logging.getLogger("mlbuild.tflite")


class TFLiteBackend(Backend):
    """
    TensorFlow Lite backend for mobile/edge deployment.

    Features:
    - Supports Android (ARM64, ARM32, x86), iOS, and edge devices (Raspberry Pi, Coral TPU)
    - Lazy TensorFlow & ONNX-TF validation
    - Multi-input ONNX support
    - Configurable INT8 representative dataset
    - Memory-efficient conversion for large models
    """

    def __init__(
        self,
        artifact_dir: Path,
        converter: Optional[TFLiteConverter] = None,
        benchmark_runner: Optional[TFLiteBenchmarkRunner] = None,
        logger_override: Optional[logging.Logger] = None,
    ):
        """
        Initialize TFLite backend.

        Args:
            artifact_dir: Directory to store TFLite artifacts.
            converter: Optional custom TFLiteConverter instance.
            benchmark_runner: Optional custom benchmark runner.
            logger_override: Optional logger instance.

        Raises:
            RuntimeError: If artifact_dir cannot be created.
        """
        self.logger = logger_override or logger

        try:
            self.artifact_dir = Path(artifact_dir)
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create artifact directory {artifact_dir}: {e}")

        # Lazy instantiation
        self._converter = converter
        self._benchmark_runner = benchmark_runner

    @property
    def name(self) -> str:
        return "tflite"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_fp32=True,
            supports_fp16=True,
            supports_int8=True,
            supports_dynamic_shapes=False,
            compute_units=["CPU_ONLY"],
        )

    @property
    def converter(self) -> TFLiteConverter:
        if self._converter is None:
            try:
                self._converter = TFLiteConverter()
            except ImportError as e:
                raise RuntimeError(
                    "TensorFlow and onnx-tf must be installed for TFLite conversion."
                ) from e
        return self._converter

    @property
    def benchmark_runner(self) -> TFLiteBenchmarkRunner:
        if self._benchmark_runner is None:
            self._benchmark_runner = TFLiteBenchmarkRunner()
        return self._benchmark_runner

    def validate_environment(self) -> EnvironmentValidation:
        """
        Validate that the backend environment is ready.

        Returns:
            EnvironmentValidation containing errors, warnings, and platform info.
        """
        errors: List[str] = []
        warnings: List[str] = []
        info: Dict[str, Any] = {}

        # TensorFlow
        try:
            import tensorflow as tf
            info["tensorflow"] = tf.__version__
            _ = tf.lite.TFLiteConverter
            info["tflite"] = "available"
        except ImportError:
            errors.append("TensorFlow not installed")
        except AttributeError:
            errors.append("TFLite not available in installed TensorFlow")

        # ONNX-TF
        try:
            import onnx2tf
            info["onnx2tf"] = onnx2tf.__version__
        except ImportError:
            errors.append("onnx2tf not installed")

        # Platform
        info["platform"] = platform.system()
        info["architecture"] = platform.machine()

        return EnvironmentValidation(
            is_valid=len(errors) == 0,
            backend_name=self.name,
            errors=errors,
            warnings=warnings,
            info=info,
        )

    def build(
        self,
        model_path: Path,
        target: str,
        quantize: str,
        name: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Build:
        """
        Full build pipeline is handled by the CLI build command.
        Use 'mlbuild build --backend tflite' directly.
        """
        raise NotImplementedError(
            "TFLiteBackend.build() is handled by the build command pipeline. "
            "Use 'mlbuild build --backend tflite' directly."
        )

    def _export(
        self,
        model_path: Path,
        quantize: str,
        name: Optional[str] = None,
        representative_dataset: Optional[Callable[[], Iterator[List[Any]]]] = None,
    ) -> Path:
        """
        Internal: Convert ONNX model to TFLite file.

        Args:
            model_path: Path to ONNX model (.onnx)
            quantize: Quantization type ('fp32', 'fp16', 'int8')
            name: Optional output file stem
            representative_dataset: Optional generator for INT8 calibration.
                Required when quantize='int8'.

        Returns:
            Path to generated TFLite model

        Raises:
            FileNotFoundError: If ONNX model not found
            ValueError: Invalid file type or missing calibration for INT8
            TFLiteConversionError: Conversion or quantization failed
        """
        model_path = Path(model_path)
        self.logger.info(f"Exporting {model_path} to TFLite...")
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        if model_path.suffix.lower() != ".onnx":
            raise ValueError(f"Expected .onnx file, got: {model_path.suffix}")

        output_name = name or model_path.stem
        output_path = self.artifact_dir / f"{output_name}.tflite"

        # Load ONNX model safely
        try:
            import onnx
            onnx_model = onnx.load(str(model_path))
        except Exception as e:
            raise TFLiteConversionError(f"Failed to load ONNX model: {model_path}") from e

        input_shapes = self._get_input_shapes(onnx_model)
        self.logger.debug(f"Detected ONNX input shapes: {input_shapes}")

        # INT8 quantization requires representative dataset
        if quantize == "int8":
            if representative_dataset is None:
                raise ValueError(
                    "INT8 quantization requires a representative_dataset generator "
                    "for proper calibration. Provide one via the `representative_dataset` argument."
                )
        else:
            representative_dataset = None  # ignore any provided dataset

        # Conversion wrapped with error context
        try:
            tflite_path = self.converter.convert(
                onnx_path=model_path,
                output_path=output_path,
                quantization=quantize,
                representative_dataset=representative_dataset,
                input_shapes=input_shapes,
            )
        except Exception as e:
            raise TFLiteConversionError(
                f"TFLite conversion failed for {model_path} with quantization={quantize}"
            ) from e

        # Validate model (multi-input/output aware)
        try:
            validation = TFLiteValidator().validate(tflite_path)
        except Exception as e:
            raise TFLiteConversionError(f"TFLite validation failed: {tflite_path}") from e

        self.logger.info(
            f"✓ Model exported and validated: {tflite_path} "
            f"({validation.get('size_mb', 0):.2f} MB)"
        )
        return tflite_path

    @staticmethod
    def _tf_version() -> str:
        """Safely retrieve installed TensorFlow version."""
        try:
            import tensorflow as tf
            return tf.__version__
        except ImportError:
            return "unknown"

    @staticmethod
    def _get_input_shapes(onnx_model) -> List[Tuple[str, Tuple[int, ...]]]:
        """
        Extract input shapes from ONNX model.

        Returns:
            List of tuples (input_name, shape), dynamic dimensions replaced with 1

        Logs warning if dynamic dimensions are present.
        """
        input_shapes = []
        for input_info in onnx_model.graph.input:
            shape = []
            has_dynamic = False
            for dim in input_info.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    shape.append(1)
                    has_dynamic = True
            if has_dynamic:
                logger.warning(
                    f"Dynamic dimensions detected in input '{input_info.name}', "
                    f"defaulting to 1 for INT8 calibration."
                )
            input_shapes.append((input_info.name, tuple(shape)))
        return input_shapes