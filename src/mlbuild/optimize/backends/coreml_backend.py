"""
CoreML backend for MLBuild optimization passes.

This backend exposes compilation and weight-quantization capabilities
for CoreML models. It intentionally does NOT understand optimization
methods (fp16/int8/etc). Optimization passes translate those into
backend operations.

Responsibilities:
- Compile CoreML models from ONNX graphs
- Perform weight quantization on compiled artifacts
- Validate backend capabilities
- Ensure deterministic logging and reproducibility
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4
from typing import Dict, Optional

logger = logging.getLogger(__name__)


SUPPORTED_DEVICES = {"cpu", "gpu", "ane", "all"}


class CoreMLBackendError(RuntimeError):
    """Raised when a CoreML backend operation fails."""


class CoreMLBackend:
    """
    Backend providing CoreML compilation and quantization capabilities.

    Important:
    This backend does NOT understand optimization methods like 'fp16' or 'int8'.
    It exposes primitive capabilities that optimization passes call.
    """

    backend_name = "coreml"

    capabilities = {
        "graph_compile": True,
        "quantize_from_graph": False,
        "quantize_weights": True,
        "quantize_weights_static": True,
        "compile_from_graph": True,
        "prune_weights": True,
        "benchmark": False,
    }

    # -----------------------------
    # Public API
    # -----------------------------

    def compile_from_graph(
        self,
        graph_path: str | Path,
        target_device: str,
        compiler_options: Optional[Dict] = None,
    ) -> Path:
        """
        Compile an ONNX graph into a CoreML .mlpackage.

        Parameters
        ----------
        graph_path : Path
            Path to ONNX graph.
        target_device : str
            Target device (cpu/gpu/ane/all).
        compiler_options : dict
            Optional compile flags passed through to exporter.

        Returns
        -------
        Path
            Path to compiled .mlpackage artifact.
        """

        graph_path = Path(graph_path)

        if not graph_path.exists():
            raise FileNotFoundError(f"ONNX graph not found: {graph_path}")

        self._validate_device(target_device)

        logger.info(
            "coreml_compile_start",
            extra={
                "backend": "coreml",
                "graph": str(graph_path),
                "target_device": target_device,
            },
        )

        # Lazy imports to avoid hard dependency for non-CoreML users
        import coremltools as ct
        from ...loaders import load_model
        from ...backends.coreml import CoreMLExporter

        logger.info(
            "coreml_environment",
            extra={
                "coremltools_version": ct.__version__,
                "target_device": target_device,
            },
        )

        try:
            ir = load_model(str(graph_path))

            tmp_dir = Path(tempfile.mkdtemp(prefix="mlbuild_coreml_compile_"))

            exporter = CoreMLExporter(target=target_device)

            # Extract quantization from compiler_options
            # Pass uses internal precision names (float16/int8)
            # Exporter expects short names (fp16/fp32/int8)
            _PRECISION_TO_QUANT = {"float16": "fp16", "int8": "int8", "fp16": "fp16", "fp32": "fp32"}
            raw = (compiler_options or {}).get("quantization", "fp32")
            quantization = _PRECISION_TO_QUANT.get(raw, raw)

            mlpackage_path, _ = exporter.export(
                ir,
                output_dir=tmp_dir,
                quantization=quantization,
            )

        except Exception as e:
            logger.exception(
                "coreml_compile_failed",
                extra={
                    "graph": str(graph_path),
                    "target_device": target_device,
                },
            )
            raise CoreMLBackendError("CoreML compilation failed") from e

        final_path = self._finalize_artifact(Path(mlpackage_path), "compiled")

        logger.info(
            "coreml_compile_complete",
            extra={"artifact": str(final_path)},
        )

        return final_path

    # ----------------------------------------

    def quantize_weights(
            self,
            artifact_path: str | Path,
            weight_precision: str,
        ) -> Dict:

            artifact_path = Path(artifact_path)

            if not artifact_path.exists():
                raise FileNotFoundError(f"Artifact not found: {artifact_path}")

            if weight_precision not in {"float16", "int8"}:
                raise ValueError(f"Unsupported weight precision: {weight_precision}")

            logger.info(f"coreml_quantization_start artifact={artifact_path} precision={weight_precision}")

            import coremltools as ct

            model = ct.models.MLModel(str(artifact_path))

            try:
                if weight_precision == "float16":
                    # CT9 does not support post-hoc FP16 on compiled artifacts.
                    # FP16 requires recompilation from the ONNX graph.
                    raise CoreMLBackendError(
                        "FP16 quantization requires the original ONNX graph and cannot be "
                        "applied to a compiled .mlpackage directly. This build was registered "
                        "via 'mlbuild import' without a graph. To use FP16, re-register with "
                        "'mlbuild build' (from ONNX) or 'mlbuild import --graph model.onnx'."
                    )

                else:  # int8
                    from coremltools.optimize.coreml import (
                        OpLinearQuantizerConfig,
                        OptimizationConfig,
                        linear_quantize_weights,
                    )
                    op_config = OpLinearQuantizerConfig(dtype="int8")
                    config = OptimizationConfig(global_config=op_config)
                    optimized = linear_quantize_weights(model, config=config)

            except CoreMLBackendError:
                raise  # don't swallow our own errors
            except Exception as e:
                logger.exception("coreml_quantization_failed")
                raise CoreMLBackendError("CoreML quantization failed") from e

            tmp_dir = Path(tempfile.mkdtemp(prefix="mlbuild_coreml_quant_"))
            artifact_name = f"quantized_{weight_precision}_{uuid4().hex}.mlpackage"
            out_path = tmp_dir / artifact_name
            optimized.save(str(out_path))

            final_path = self._finalize_artifact(out_path, "quantized")

            logger.info(f"coreml_quantization_complete artifact={final_path} precision={weight_precision}")

            return {
                "artifact_path": final_path,
                "weight_precision": weight_precision,
                "activation_precision": "fp32",
            }


    def quantize_weights_static(
        self,
        artifact_path: str | Path,
        weight_precision: str,
        calibration_samples: list[dict],
    ) -> Dict:
        """
        Static INT8 quantization using representative calibration data.

        Uses CT9 PostTrainingQuantizer to quantize both weights and
        activations using the provided calibration samples.

        Parameters
        ----------
        artifact_path : Path to compiled .mlpackage
        weight_precision : must be "int8"
        calibration_samples : list of {input_name: np.ndarray} dicts
        """

        artifact_path = Path(artifact_path)

        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")

        if weight_precision != "int8":
            raise ValueError(
                f"quantize_weights_static only supports int8, got {weight_precision}"
            )

        if not calibration_samples:
            raise ValueError("calibration_samples must not be empty")

        logger.info(
            "coreml_static_quant_start artifact=%s samples=%d",
            artifact_path,
            len(calibration_samples),
        )

        import coremltools as ct
        from coremltools.optimize.coreml import (
            OpLinearQuantizerConfig,
            OptimizationConfig,
            linear_quantize_weights,
        )

        # Check for CT9 static quantization API
        try:
            from coremltools.optimize.coreml import (
                calibrate_activations,
                linear_quantize_activations,
                OpActivationLinearQuantizerConfig,
            )
            _has_static = True
        except ImportError:
            _has_static = False
            logger.warning(
                "calibrate_activations not available in coremltools %s — "
                "falling back to dynamic range INT8 (weights only)",
                ct.__version__,
            )

        model = ct.models.MLModel(str(artifact_path))

        try:
            if _has_static:
                # Step 1: quantize weights
                w_config = OpLinearQuantizerConfig(
                    mode="linear_symmetric",
                    dtype="int8",
                    weight_threshold=512,
                )
                opt_config = OptimizationConfig(global_config=w_config)
                model = linear_quantize_weights(model, config=opt_config)

                # Step 2: calibrate activations
                logger.info("coreml_static_quant calibrating activations samples=%d", len(calibration_samples))
                model = calibrate_activations(
                    model,
                    data_loader=calibration_samples,
                )

                # Step 3: quantize activations
                a_config = OpActivationLinearQuantizerConfig(dtype="int8")
                act_opt_config = OptimizationConfig(global_config=a_config)
                optimized = linear_quantize_activations(model, config=act_opt_config)
                activation_precision = "int8"
                logger.info("coreml_static_quant complete weights=int8 activations=int8")
            else:
                # Fallback: dynamic range weights only
                w_config = OpLinearQuantizerConfig(dtype="int8")
                opt_config = OptimizationConfig(global_config=w_config)
                optimized = linear_quantize_weights(model, config=opt_config)
                activation_precision = "fp32"

        except Exception as e:
            logger.exception("coreml_static_quant_failed")
            raise CoreMLBackendError("CoreML static quantization failed") from e

        tmp_dir = Path(tempfile.mkdtemp(prefix="mlbuild_coreml_static_quant_"))
        artifact_name = f"static_int8_{uuid4().hex}.mlpackage"
        out_path = tmp_dir / artifact_name
        optimized.save(str(out_path))

        final_path = self._finalize_artifact(out_path, "static_quantized")

        logger.info(
            "coreml_static_quant_complete artifact=%s activation_precision=%s",
            final_path,
            activation_precision,
        )

        return {
            "artifact_path": final_path,
            "weight_precision": "int8",
            "activation_precision": activation_precision,
        }


    def prune_weights(
        self,
        artifact_path: "str | Path",
        sparsity: float,
    ) -> Dict:
        """
        Post-hoc magnitude pruning on a compiled .mlpackage.

        Uses CT9 OpMagnitudePrunerConfig in percentile_based mode.
        Only used for imported builds with no ONNX graph.

        Parameters
        ----------
        artifact_path : Path to compiled .mlpackage
        sparsity      : Fraction of weights to zero (0.0-1.0)
        """
        artifact_path = Path(artifact_path)

        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")

        if not 0.0 < sparsity < 1.0:
            raise ValueError(f"sparsity must be between 0 and 1 exclusive, got {sparsity}")

        logger.info(
            "coreml_prune_start artifact=%s sparsity=%.3f",
            artifact_path,
            sparsity,
        )

        import coremltools as ct
        from coremltools.optimize.coreml import (
            OpMagnitudePrunerConfig,
            OptimizationConfig,
            prune_weights,
        )

        model = ct.models.MLModel(str(artifact_path))

        try:
            op_config = OpMagnitudePrunerConfig(
                mode="percentile_based",
                percentile=sparsity,
                weight_threshold=512,
            )
            config = OptimizationConfig(global_config=op_config)
            pruned = prune_weights(model, config=config)

        except Exception as e:
            logger.exception("coreml_prune_failed")
            raise CoreMLBackendError("CoreML pruning failed") from e

        tmp_dir = Path(tempfile.mkdtemp(prefix="mlbuild_coreml_prune_"))
        artifact_name = f"pruned_{sparsity:.2f}_{uuid4().hex}.mlpackage"
        out_path = tmp_dir / artifact_name
        pruned.save(str(out_path))

        final_path = self._finalize_artifact(out_path, "pruned")

        logger.info(
            "coreml_prune_complete artifact=%s sparsity=%.3f",
            final_path,
            sparsity,
        )

        return {
            "artifact_path": final_path,
            "weight_precision": "pruned",
            "activation_precision": "fp32",
        }

    # -----------------------------
    # Capability Checks
    # -----------------------------

    def _supports_quantization(self, model) -> bool:
        """
        Placeholder for future operator capability checks.
        """
        # In practice you'd inspect the model spec here
        return True

    def _is_already_quantized(self, model, precision: str) -> bool:
        """
        Detect if model weights are already quantized.
        """
        try:
            spec = model.get_spec()

            for layer in spec.neuralNetwork.layers:
                if layer.WhichOneof("layer") == "quantize":
                    return True

        except Exception:
            pass

        return False

    # -----------------------------
    # Internal Helpers
    # -----------------------------

    def _validate_device(self, device: str):
        # MLBuild uses apple_m1/apple_a17/android_arm64 style names
        # CoreML is valid for any Apple target or generic compute units
        valid_prefixes = ("apple_", "cpu", "gpu", "ane", "all")
        if not any(device.startswith(p) for p in valid_prefixes):
            raise ValueError(
                f"Unsupported target_device '{device}'. "
                f"Expected apple_m1/apple_a17/etc or cpu/gpu/ane/all."
            )

    def _finalize_artifact(self, src_path: Path, label: str) -> Path:
        """
        Move artifact from temp workspace into isolated temp location
        used by MLBuild registry ingestion.
        """

        final_dir = Path(tempfile.mkdtemp(prefix=f"mlbuild_{label}_"))
        final_path = final_dir / src_path.name

        shutil.move(str(src_path), str(final_path))

        return final_path