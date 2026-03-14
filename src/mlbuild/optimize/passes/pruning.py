"""
Magnitude-based unstructured weight pruning pass.

Routing
-------
build.has_graph=True
    → ONNX pruning via OnnxBackend
    → re-convert using existing build pipeline
    → works for coreml AND tflite

build.has_graph=False + format=coreml
    → CoreML post-hoc pruning via CT9 OpMagnitudePrunerConfig

build.has_graph=False + format=tflite
    → PruningPassError with actionable message
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from ...core.types import Build

logger = logging.getLogger(__name__)


class PruningPassError(RuntimeError):
    pass


class PruningPass:
    """
    Backend-agnostic pruning pass.

    Parameters
    ----------
    coreml_backend : CoreMLBackend instance (for post-hoc CoreML pruning)
    """

    pass_name = "pruning"

    def __init__(self, coreml_backend):
        self.coreml_backend = coreml_backend

    def apply(
        self,
        source: "Build",
        sparsity: float,
        graphs_root: Path,
        registry=None,
    ) -> Dict:
        """
        Apply pruning to a build.

        Parameters
        ----------
        source      : Build to prune
        sparsity    : Fraction of weights to zero (0.0-1.0)
        graphs_root : Root directory containing stored ONNX graphs
        registry    : LocalRegistry (needed for ONNX path re-conversion)

        Returns
        -------
        dict with keys: artifact_path, weight_precision, activation_precision
        """

        if not 0.0 < sparsity < 1.0:
            raise PruningPassError(
                f"sparsity must be between 0.0 and 1.0 exclusive, got {sparsity}"
            )

        logger.info(
            "pruning_pass_start build=%s sparsity=%.2f has_graph=%s format=%s",
            source.build_id[:16],
            sparsity,
            source.has_graph,
            source.format,
        )

        if source.has_graph:
            return self._prune_onnx(source, sparsity, graphs_root, registry)
        elif source.format == "coreml":
            return self._prune_coreml(source, sparsity)
        elif source.format == "tflite":
            raise PruningPassError(
                f"TFLite pruning requires the original ONNX graph. "
                f"Build {source.build_id[:16]} was imported without one. "
                f"Re-register using 'mlbuild build' or 'mlbuild import --graph model.onnx'."
            )
        else:
            raise PruningPassError(
                f"No pruning path for format='{source.format}' has_graph={source.has_graph}."
            )

    # --------------------------------------------------
    # ONNX pruning path
    # --------------------------------------------------

    def _prune_onnx(
        self,
        source: "Build",
        sparsity: float,
        graphs_root: Path,
        registry,
    ) -> Dict:
        """
        Prune ONNX graph initializers then re-convert via existing build pipeline.
        """
        from ...cli.commands.build import run_build

        # Resolve graph path
        filename = Path(source.graph_path).name
        graph_path = graphs_root / filename

        if not graph_path.exists():
            raise PruningPassError(
                f"ONNX graph not found at {graph_path}. "
                f"Was it deleted? Re-register with 'mlbuild build'."
            )

        # Prune the ONNX graph
        pruned_onnx_path = self._magnitude_prune_onnx(graph_path, sparsity)

        logger.info(
            "onnx_pruning_complete sparsity=%.2f pruned_graph=%s",
            sparsity,
            pruned_onnx_path,
        )

        # Re-convert using existing build pipeline
        pruned_build = run_build(
            model_path=pruned_onnx_path,
            target=source.target_device,
            name=source.name,
            quantization=source.weight_precision or "fp32",
            format=source.format,
            registry=registry,
        )

        return {
            "artifact_path": Path(pruned_build.artifact_path),
            "weight_precision": "pruned",
            "activation_precision": "fp32",
            "via_build_id": pruned_build.build_id,
        }

    def _magnitude_prune_onnx(self, graph_path: Path, sparsity: float) -> Path:
        """
        Zero out the smallest weights by absolute value in all large initializers.

        Skips tensors with fewer than 512 parameters (biases, BN params, etc.)
        to avoid degrading normalisation layers.
        """
        import onnx
        import numpy as np
        from onnx import numpy_helper

        model = onnx.load(str(graph_path))

        pruned = 0
        skipped = 0
        total_zeroed = 0
        total_params = 0

        for initializer in model.graph.initializer:
            arr = numpy_helper.to_array(initializer).copy()

            if arr.size < 512:
                skipped += 1
                continue

            threshold = np.percentile(np.abs(arr), sparsity * 100)
            mask = np.abs(arr) < threshold
            total_zeroed += mask.sum()
            total_params += arr.size
            arr[mask] = 0.0

            new_tensor = numpy_helper.from_array(arr, initializer.name)
            initializer.CopyFrom(new_tensor)
            pruned += 1

        actual_sparsity = total_zeroed / total_params if total_params > 0 else 0.0

        logger.info(
            "onnx_magnitude_pruning tensors_pruned=%d tensors_skipped=%d "
            "target_sparsity=%.2f actual_sparsity=%.4f",
            pruned,
            skipped,
            sparsity,
            actual_sparsity,
        )

        # Save to temp file
        tmp = Path(tempfile.mkdtemp(prefix="mlbuild_pruned_onnx_"))
        out_path = tmp / f"pruned_{sparsity:.2f}_{graph_path.stem}.onnx"
        onnx.save(model, str(out_path))

        return out_path

    # --------------------------------------------------
    # CoreML post-hoc pruning path
    # --------------------------------------------------

    def _prune_coreml(self, source: "Build", sparsity: float) -> Dict:
        """
        Post-hoc CoreML pruning via CT9 OpMagnitudePrunerConfig.
        Only available for imported .mlpackage builds without an ONNX graph.
        """
        return self.coreml_backend.prune_weights(
            artifact_path=source.artifact_path,
            sparsity=sparsity,
        )
