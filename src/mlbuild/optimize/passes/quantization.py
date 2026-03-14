"""
Quantization optimization pass.

This pass translates CLI quantization methods into backend capability
calls without encoding backend-specific behavior.

Design Principles
-----------------
• Backend-agnostic: no format branching.
• Uses backend capability discovery.
• Single quantization interface across all backends.
• Structured logging for CI / telemetry systems.
• Consistent OptimizationResult return contract.

Expected Backend Capabilities
-----------------------------

Backends must declare capabilities such as:

backend.capabilities = {
    "compile_from_graph": bool,
    "quantize_from_graph": bool,
    "quantize_weights": bool,
}

The pass will automatically route operations based on these capabilities.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ...core.types import Build

logger = logging.getLogger(__name__)


METHOD_TO_PRECISION: Dict[str, str] = {
    "fp16": "float16",
    "int8": "int8",
    "int8_static": "int8",
}


class QuantizationPassError(RuntimeError):
    """Raised when a quantization pass cannot proceed."""


class QuantizationPass:
    """
    Backend-agnostic quantization pass.

    The optimizer injects a backend instance which declares
    its supported capabilities.
    """

    pass_name = "quantization"

    def __init__(self, backend):
        self.backend = backend

    # --------------------------------------------------

    def apply(
        self,
        source: "Build",
        method: str,
        graphs_root: Path,
        calibration_data: "Path | None" = None,
    ) -> Dict:
        """
        Apply quantization to a build.

        Parameters
        ----------
        source
            Build to optimize.
        method
            CLI quantization method ('fp16', 'int8').
        graphs_root
            Root directory containing stored ONNX graphs.
        calibration_data
            Optional path to calibration data (images, .npy dir, or .npz).
            When provided with int8, uses static quantization instead of
            dynamic range.

        Returns
        -------
        OptimizationResult
            {
                "artifact_path": Path,
                "weight_precision": str,
                "activation_precision": str
            }
        """

        if method not in METHOD_TO_PRECISION:
            raise QuantizationPassError(
                f"Unknown quantization method '{method}'. "
                f"Supported methods: {list(METHOD_TO_PRECISION)}"
            )

        precision = METHOD_TO_PRECISION[method]

        logger.info(
            "quantization_pass_start",
            extra={
                "pass": self.pass_name,
                "build": source.build_id,
                "method": method,
                "precision": precision,
                "backend": getattr(self.backend, "name", "unknown"),
            },
        )

        graph_path = self._resolve_graph_path(source, graphs_root)

        result = self._execute_quantization(
            source, precision, graph_path, calibration_data=calibration_data
        )

        logger.info(
            "quantization_pass_complete",
            extra={
                "pass": self.pass_name,
                "build": source.build_id,
                "artifact": str(result["artifact_path"]),
                "weight_precision": result["weight_precision"],
                "activation_precision": result["activation_precision"],
            },
        )

        return result

    # --------------------------------------------------
    # Quantization routing
    # --------------------------------------------------

    def _execute_quantization(
        self,
        source: "Build",
        precision: str,
        graph_path: Path | None,
        calibration_data: "Path | None" = None,
    ) -> Dict:

        caps = getattr(self.backend, "capabilities", {})

        # --------------------------------------------------
        # Static INT8: calibration data provided → use PostTrainingQuantizer
        # --------------------------------------------------

        if (
            precision == "int8"
            and calibration_data is not None
            and caps.get("quantize_weights_static", False)
        ):
            from ...core.accuracy.calibration import CalibrationLoader
            import coremltools as ct

            # Read input spec from the artifact
            model = ct.models.MLModel(str(source.artifact_path))
            spec = model.get_spec()
            inputs = list(spec.description.input)
            if not inputs:
                raise QuantizationPassError("Model has no inputs — cannot load calibration data")

            inp = inputs[0]
            input_name = inp.name
            input_shape = tuple(inp.type.multiArrayType.shape)

            # Detect layout from shape
            layout = None
            if len(input_shape) == 4:
                layout = "nchw" if input_shape[1] in (1, 3) else "nhwc"

            loader = CalibrationLoader(
                source=calibration_data,
                input_name=input_name,
                input_shape=input_shape,
                layout=layout,
                max_samples=200,
            )
            samples = loader.as_list()
            logger.info(
                "static_int8 calibration loaded samples=%d input=%s shape=%s",
                len(samples),
                input_name,
                input_shape,
            )
            return self.backend.quantize_weights_static(
                source.artifact_path, precision, samples
            )

        # --------------------------------------------------
        # Dynamic INT8: no calibration data — weights only
        # --------------------------------------------------

        if precision == "int8" and caps.get("quantize_weights", False):
            return self.backend.quantize_weights(source.artifact_path, precision)

        # --------------------------------------------------
        # FP16: prefer compile_from_graph when graph is available
        # (CT9 cannot do post-hoc FP16 on compiled artifacts)
        # --------------------------------------------------

        if precision == "float16" and caps.get("compile_from_graph", False):
            if graph_path is None:
                raise QuantizationPassError(
                    f"FP16 optimization requires the original ONNX graph. "
                    f"Build {source.build_id[:16]} was registered without one. "
                    f"Re-register using 'mlbuild build'."
                )
            result = self.backend.compile_from_graph(
                graph_path,
                source.target_device,
                compiler_options={"quantization": precision},
            )
            if isinstance(result, Path):
                return {
                    "artifact_path": result,
                    "weight_precision": precision,
                    "activation_precision": "fp32",
                }
            return result

        # --------------------------------------------------
        # TFLite: quantize_from_graph handles both fp16 + int8
        # --------------------------------------------------

        if caps.get("quantize_from_graph", False):
            if graph_path is None:
                raise QuantizationPassError(
                    f"Backend '{getattr(self.backend, 'name', 'backend')}' requires "
                    f"the original ONNX graph. Build {source.build_id[:16]} has none."
                )
            return self.backend.quantize_from_graph(graph_path, precision)

        raise QuantizationPassError(
            f"No supported quantization path for precision={precision} "
            f"on backend '{getattr(self.backend, 'name', 'unknown')}'."
        )

    # --------------------------------------------------
    # Graph resolution
    # --------------------------------------------------

    def _resolve_graph_path(
        self,
        source: "Build",
        graphs_root: Path,
    ) -> Path | None:

        if not source.has_graph:
            return None

        # source.graph_path is stored as "graphs/{build_id}.onnx"
        # graphs_root is already ".mlbuild/graphs/" — use filename only
        filename = Path(source.graph_path).name
        return graphs_root / filename