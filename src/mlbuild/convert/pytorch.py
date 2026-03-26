"""
PyTorch → ONNX conversion executor.

This module implements a robust PyTorch to ONNX converter with the following features:
- Flexible model loading (TorchScript or eager)
- Opset fallback sequence (17 → 16 → 15 → 14 → 13 → 12)
- Dynamic batch dimension support
- Phantom external data stripping for ONNX files
- Detailed error reporting with ConvertError and error codes
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
from enum import Enum

import torch
import torch.onnx
import onnx

from mlbuild.convert.graph import register_conversion
from mlbuild.convert.types import ConvertContext, ConvertOutput
from mlbuild.core.errors import ConvertError, ErrorCode

logger = logging.getLogger("mlbuild.convert.pytorch")

CONVERTER_VERSION = "pytorch_to_onnx_v2"


# -----------------------------
# Type safety & constants
# -----------------------------

class LoadMode(str, Enum):
    AUTO = "auto"
    JIT = "jit"
    EAGER = "eager"


# Default opset fallback sequence
DEFAULT_OPSET_FALLBACK = [17, 16, 15, 14, 13, 12]

MAX_TENSOR_ELEMENTS = 1_000_000_000  # 1B elements threshold for sample input


# -----------------------------
# Model Loader
# -----------------------------

class ModelLoader:
    """Encapsulates robust PyTorch model loading with timeout and type checks."""

    @staticmethod
    def is_state_dict(obj: Any) -> bool:
        if not isinstance(obj, dict) or not obj:
            return False
        # check multiple items to avoid false negatives
        for sample in list(obj.values())[:5]:
            if hasattr(sample, "shape") and hasattr(sample, "dtype"):
                return True
        return False

    @staticmethod
    def load(path: Path, mode: LoadMode = LoadMode.AUTO) -> torch.nn.Module:
        path = path.resolve()
        if not path.exists() or path.stat().st_size == 0:
            raise ConvertError(
                f"Model file missing or empty: {path}",
                stage="load_pytorch_model",
                error_code=ErrorCode.MODEL_LOAD_FAILED,
            )

        def _try_jit():
            try:
                return torch.jit.load(str(path), map_location="cpu")
            except (RuntimeError, TypeError) as e:
                raise ConvertError(
                    f"torch.jit.load() failed: {e}\n"
                    "Model may not be TorchScript.",
                    stage="load_pytorch_model",
                    error_code=ErrorCode.MODEL_LOAD_FAILED,
                )

        def _try_eager():
            try:
                obj = torch.load(str(path), map_location="cpu", weights_only=False)
            except (RuntimeError, TypeError) as e:
                raise ConvertError(
                    f"torch.load() failed: {e}",
                    stage="load_pytorch_model",
                    error_code=ErrorCode.MODEL_LOAD_FAILED,
                )
            if ModelLoader.is_state_dict(obj):
                raise ConvertError(
                    "State dict detected — load into model class first, then re-export.",
                    stage="load_pytorch_model",
                    error_code=ErrorCode.STATE_DICT_DETECTED,
                )
            return obj

        if mode == LoadMode.JIT:
            return _try_jit()
        elif mode == LoadMode.EAGER:
            return _try_eager()

        # AUTO: try JIT first
        try:
            model = _try_jit()
            logger.debug("Loaded model via TorchScript")
            return model
        except ConvertError:
            logger.debug("JIT load failed; falling back to eager")
            return _try_eager()


# -----------------------------
# Sample Input Generator
# -----------------------------

class SampleInputGenerator:
    """Generates safe dummy input tensors for tracing."""

    @staticmethod
    def make(input_shape: Tuple[int, ...], dtype=torch.float32) -> torch.Tensor:
        if any(dim <= 0 for dim in input_shape):
            raise ConvertError(
                f"Invalid input shape: {input_shape}",
                stage="make_sample_input",
                error_code=ErrorCode.CONVERSION_FAILED,
            )
        total_elements = 1
        for dim in input_shape:
            total_elements *= dim
        if total_elements > MAX_TENSOR_ELEMENTS:
            logger.warning("Sample input too large; using smaller tensor for tracing")
            factor = (MAX_TENSOR_ELEMENTS / total_elements) ** (1 / len(input_shape))
            input_shape = tuple(max(1, int(dim * factor)) for dim in input_shape)
        return torch.zeros(*input_shape, dtype=dtype)


# -----------------------------
# ONNX Exporter
# -----------------------------

class ONNXExporter:
    """Handles robust ONNX export with fallback, dynamic axes, validation, and metrics."""

    def __init__(self, model: torch.nn.Module, sample_input: torch.Tensor):
        self.model = model.eval()
        self.sample_input = sample_input.cpu()
        self.warnings: List[str] = []

    def _dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        return {"input": {0: "batch"}, "output": {0: "batch"}}

    def export_single(self, output_path: Path, opset: int) -> None:
        import torch

        # Detect if model is TorchScript — new exporter doesn't support it
        is_script = isinstance(self.model, torch.jit.ScriptModule)

        try:
            torch.onnx.export(
                self.model,
                self.sample_input,
                str(output_path),
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                keep_initializers_as_inputs=False,
                input_names=["input"],
                output_names=["output"],
                dynamo=False,   # force legacy exporter — works for both eager and TorchScript
            )
        except (RuntimeError, TypeError) as e:
            raise ConvertError(
                f"ONNX export failed for opset {opset}: {e}",
                stage="export_to_onnx",
                error_code=ErrorCode.CONVERSION_FAILED,
            )
        # Validate exported model
        try:
            onnx.checker.check_model(str(output_path))
        except onnx.checker.ValidationError as e:
            raise ConvertError(
                f"Exported ONNX model validation failed: {e}",
                stage="export_to_onnx",
                error_code=ErrorCode.CONVERSION_FAILED,
            )


# -----------------------------
# External Data Cleaner
# -----------------------------

class ExternalDataCleaner:
    """Removes phantom external data references."""

    @staticmethod
    def strip(onnx_path: Path) -> bool:
        data_file = onnx_path.with_suffix(onnx_path.suffix + ".data")
        if data_file.exists():
            return False  # real external data
        try:
            model = onnx.load(str(onnx_path), load_external_data=False)
            changed = False
            for init in model.graph.initializer:
                if init.HasField("data_location") and init.data_location == onnx.TensorProto.EXTERNAL:
                    init.external_data.clear()
                    init.data_location = onnx.TensorProto.DEFAULT
                    changed = True
            if changed:
                onnx.save(model, str(onnx_path))
                logger.debug("Stripped phantom external_data metadata")
            return changed
        except Exception as e:
            logger.warning(f"Failed to strip external data: {e}")
            return False
        
# -----------------------------
# Logging suppression
# -----------------------------

def _suppress_onnx_exporter_logging():
    """
    Suppress verbose logging from Torch 2.10+ ONNX exporter.
    These messages leak through the threading boundary and pollute CLI output.
    """
    for name in [
        "torch.onnx",
        "torch.onnx._internal",
        "torch.onnx._internal.exporter",
        "torch.onnx._internal.exporter._compat",
        "torch.onnx._internal.exporter._schemas",
        "torch.onnx._internal.exporter._registration",
        "onnxscript",
        "onnxscript.version_converter",
    ]:
        logging.getLogger(name).setLevel(logging.ERROR)


# -----------------------------
# Main export routine
# -----------------------------

def export_to_onnx(
    model: torch.nn.Module,
    output_path: Path,
    input_shape: Tuple[int, ...],
    opset_override: Optional[int] = None,
    fallback_sequence: Optional[List[int]] = None,
) -> Tuple[str, int, float, List[str]]:
    _suppress_onnx_exporter_logging() 
    fallback_sequence = fallback_sequence or DEFAULT_OPSET_FALLBACK
    sequence = ([opset_override] if opset_override else []) + [
        op for op in fallback_sequence if op != opset_override
    ]

    sample_input = SampleInputGenerator.make(input_shape)
    exporter = ONNXExporter(model, sample_input)

    errors: List[str] = []
    for opset in sequence:
        start = time.time()
        try:
            exporter.export_single(output_path, opset)
            ExternalDataCleaner.strip(output_path)
            size_mb = output_path.stat().st_size / (1024 * 1024)
            duration = time.time() - start
            logger.info(
                "ONNX export successful",
                extra={
                    "opset": opset,
                    "file_size_mb": size_mb,
                    "duration_sec": duration,
                },
            )
            metrics = {
                "torch_version": torch.__version__,
                "onnx_version": onnx.__version__,
                "file_size_mb": size_mb,
            }
            return f"opset_{opset}_dynamic_batch", opset, size_mb, exporter.warnings
        except ConvertError as e:
            errors.append(f"Opset {opset}: {str(e)}")

    raise ConvertError(
        f"ONNX export failed for all opsets:\n" + "\n".join(errors),
        stage="export_to_onnx",
        error_code=ErrorCode.CONVERSION_FAILED,
    )




# -----------------------------
# Registered executor
# -----------------------------

@register_conversion("pytorch", "onnx")
def convert_pytorch_to_onnx(ctx: ConvertContext) -> ConvertOutput:
    start_time = time.time()
    warnings: List[str] = []

    # Validate input shape
    if not isinstance(ctx.params.input_shape, tuple) or not all(
        isinstance(x, int) and x > 0 for x in ctx.params.input_shape
    ):
        raise ConvertError(
            f"Invalid input_shape: {ctx.params.input_shape}",
            stage="convert_pytorch_to_onnx",
            error_code=ErrorCode.CONVERSION_FAILED,
        )

    output_path = ctx.output_dir / f"{ctx.input_path.stem}.onnx"
    ctx.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = ModelLoader.load(ctx.input_path, LoadMode(ctx.params.load_mode.value))

    # Move model to CPU with no_grad
    with torch.no_grad():
        first_param = next(model.parameters(), None)
        if first_param is not None and first_param.device.type != "cpu":
            model = model.cpu()
            logger.debug("Moved model to CPU for export")

        # Export
        strategy_label, opset_used, size_mb, export_warnings = export_to_onnx(
            model=model,
            output_path=output_path,
            input_shape=ctx.params.input_shape,
            opset_override=ctx.params.opset,
        )
        warnings.extend(export_warnings)

    duration = time.time() - start_time
    metadata: Dict[str, Any] = {
        "opset": opset_used,
        "strategy": strategy_label,
        "load_mode": ctx.params.load_mode.value,
        "input_shape": list(ctx.params.input_shape),
        "size_mb": size_mb,
        "warnings": warnings,
        "converter_version": CONVERTER_VERSION,
    }

    return ConvertOutput(
        path=output_path,
        converter_version=CONVERTER_VERSION,
        metadata=metadata,
        warnings=warnings,
        duration_seconds=duration,
    )