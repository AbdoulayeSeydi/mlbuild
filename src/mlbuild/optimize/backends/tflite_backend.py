"""
TFLite backend for MLBuild.

Provides compilation and quantization capabilities for TensorFlow Lite models.

Design Principles
-----------------
• Backend exposes capabilities, not optimization policy.
• Quantization is only supported when the original ONNX graph is available.
• Post-hoc quantization of compiled .tflite artifacts is intentionally disallowed
  because TFLiteConverter cannot reliably operate on flatbuffer models.

Supported Workflow
------------------

compile_from_graph()
    ONNX → SavedModel → TFLite

quantize_from_graph()
    ONNX → SavedModel → quantized TFLite

Optimizer Layer Flow
--------------------

compile_from_graph()
↓
(optional)
quantize_from_graph()

"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Dict
from uuid import uuid4

logger = logging.getLogger(__name__)


class TFLiteBackendError(RuntimeError):
    """Raised when a TFLite backend operation fails."""


class TFLiteBackend:
    """
    TensorFlow Lite backend implementation.
    """

    name = "tflite"

    capabilities = {
        "graph_compile": True,
        "quantize_from_graph": True,
        "posthoc_quantization": False,
        "benchmark": False,
    }

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def compile_from_graph(
        self,
        graph_path: str | Path,
        target_device: str | None = None,
    ) -> Path:
        """
        Compile ONNX graph into a TFLite artifact.

        Parameters
        ----------
        graph_path
            Path to ONNX model.
        target_device
            Reserved for future delegate configuration.

        Returns
        -------
        Path
            Compiled .tflite artifact.
        """

        graph_path = Path(graph_path)

        if not graph_path.exists():
            raise FileNotFoundError(f"ONNX graph not found: {graph_path}")

        logger.info(
            "tflite_compile_start",
            extra={
                "backend": "tflite",
                "graph": str(graph_path),
                "target_device": target_device,
            },
        )

        try:
            import tensorflow as tf
            import onnx2tf
        except ImportError as e:
            raise TFLiteBackendError(
                "TensorFlow and onnx2tf are required for TFLite compilation. "
                "Install with: pip install tensorflow onnx2tf"
            ) from e

        workspace = Path(tempfile.mkdtemp(prefix="mlbuild_tflite_workspace_"))

        onnx_dir = workspace / "onnx"
        saved_model_dir = workspace / "saved_model"
        output_dir = workspace / "output"

        onnx_dir.mkdir()
        saved_model_dir.mkdir()
        output_dir.mkdir()

        shutil.copy(graph_path, onnx_dir / graph_path.name)

        try:

            # ----------------------------------------
            # ONNX → SavedModel
            # ----------------------------------------

            onnx2tf.convert(
                input_onnx_file_path=str(onnx_dir / graph_path.name),
                output_folder_path=str(saved_model_dir),
                non_verbose=True,
            )

            # ----------------------------------------
            # SavedModel → TFLite
            # ----------------------------------------

            converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

            tflite_model = converter.convert()

        except Exception as e:

            logger.exception(
                "tflite_compile_failed",
                extra={
                    "backend": "tflite",
                    "graph": str(graph_path),
                },
            )

            raise TFLiteBackendError("TFLite compilation failed") from e

        finally:
            shutil.rmtree(workspace, ignore_errors=True)

        artifact = self._write_artifact(tflite_model, "compiled")

        logger.info(
            "tflite_compile_complete",
            extra={
                "backend": "tflite",
                "artifact": str(artifact),
            },
        )

        return artifact

    # --------------------------------------------------

    def quantize_from_graph(
        self,
        graph_path: str | Path,
        weight_precision: str,
    ) -> Dict:
        """
        Quantize model weights during compilation.

        Parameters
        ----------
        graph_path
            Path to ONNX graph.
        weight_precision
            Supported:
            • float16
            • int8 (dynamic range quantization)

        Returns
        -------
        dict
            {
                "artifact_path": Path,
                "weight_precision": str,
                "activation_precision": "fp32"
            }
        """

        graph_path = Path(graph_path)

        if not graph_path.exists():
            raise FileNotFoundError(f"ONNX graph not found: {graph_path}")

        if weight_precision not in {"float16", "int8"}:
            raise ValueError(
                f"Unsupported weight_precision '{weight_precision}'. "
                "Supported: float16, int8"
            )

        logger.info(
            "tflite_quantization_start",
            extra={
                "backend": "tflite",
                "graph": str(graph_path),
                "precision": weight_precision,
            },
        )

        try:
            import tensorflow as tf
            import onnx2tf
        except ImportError as e:
            raise TFLiteBackendError(
                "TensorFlow and onnx2tf are required for quantization."
            ) from e

        workspace = Path(tempfile.mkdtemp(prefix="mlbuild_tflite_quant_workspace_"))

        onnx_dir = workspace / "onnx"
        saved_model_dir = workspace / "saved_model"

        onnx_dir.mkdir()
        saved_model_dir.mkdir()

        shutil.copy(graph_path, onnx_dir / graph_path.name)

        try:

            # ----------------------------------------
            # ONNX → SavedModel
            # ----------------------------------------

            onnx2tf.convert(
                input_onnx_file_path=str(onnx_dir / graph_path.name),
                output_folder_path=str(saved_model_dir),
                non_verbose=True,
            )

            # ----------------------------------------
            # SavedModel → Quantized TFLite
            # ----------------------------------------

            converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

            if weight_precision == "float16":

                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]

            elif weight_precision == "int8":

                # Dynamic range quantization
                converter.optimizations = [tf.lite.Optimize.DEFAULT]

            tflite_model = converter.convert()

        except Exception as e:

            logger.exception(
                "tflite_quantization_failed",
                extra={
                    "backend": "tflite",
                    "graph": str(graph_path),
                    "precision": weight_precision,
                },
            )

            raise TFLiteBackendError("TFLite quantization failed") from e

        finally:
            shutil.rmtree(workspace, ignore_errors=True)

        artifact = self._write_artifact(tflite_model, weight_precision)

        logger.info(
            "tflite_quantization_complete",
            extra={
                "backend": "tflite",
                "artifact": str(artifact),
                "precision": weight_precision,
            },
        )

        return {
            "artifact_path": artifact,
            "weight_precision": weight_precision,
            "activation_precision": "fp32",
        }

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    def _write_artifact(self, tflite_model: bytes, label: str) -> Path:
        """
        Persist compiled artifact.
        """

        out_dir = Path(tempfile.mkdtemp(prefix="mlbuild_tflite_out_"))
        out_path = out_dir / f"model_{label}_{uuid4().hex}.tflite"

        out_path.write_bytes(tflite_model)

        return out_path


# --------------------------------------------------
# Backend registration
# --------------------------------------------------

try:
    from mlbuild.backends.registry import BackendRegistry

    BackendRegistry.register(TFLiteBackend())

except Exception:
    # Allow module import without registry during testing
    pass