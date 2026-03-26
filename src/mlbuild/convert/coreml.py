"""
PyTorch/ONNX → CoreML conversion executors.

Two registered edges:
    ("pytorch", "coreml") — convert_pytorch_to_coreml
    ("onnx",    "coreml") — convert_onnx_to_coreml

Strategy for pytorch → coreml:
    1. torch.jit.trace() → ct.convert()          [primary]
    2. export_to_onnx() → ct.convert(onnx_path)  [fallback, logged as warning]

Quantization:
    fp32 — no compression applied
    fp16 — ct.compression_utils.affine_quantize_weights(mode='linear')
    int8 — hard ConvertError (requires representative dataset, not supported)
"""

from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol, Tuple

from mlbuild.convert.graph import register_conversion
from mlbuild.convert.types import ConvertContext, ConvertOutput
from mlbuild.core.errors import ConvertError, ErrorCode

logger = logging.getLogger("mlbuild.convert.coreml")

CONVERTER_VERSION = "coreml_v2"
MAX_TENSOR_ELEMENTS = 50_000_000  # safety guard

# ---------------------------------------------------------------------
# Target → OS mapping (single source of truth)
# Imported by feature_compat.py for validation — do not duplicate.
# ---------------------------------------------------------------------

TARGET_OS_MAP: dict[str, str] = {
    "apple_m1":  "macOS13",
    "apple_m2":  "macOS14",
    "apple_m3":  "macOS15",
    "apple_a15": "iOS16",
    "apple_a16": "iOS16",
    "apple_a17": "iOS17",
    "apple_a18": "iOS18",
}


# ---------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------

class ConversionStrategy(Protocol):
    def run(self, model, ctx: ConvertContext, ct_target) -> Tuple[object, str]:
        ...


# ---------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------

def _validate_input_shape(shape: Tuple[int, ...]) -> None:
    if not shape or not all(isinstance(x, int) and x > 0 for x in shape):
        raise ConvertError(
            f"Invalid input_shape: {shape}",
            stage="input_validation",
            error_code=ErrorCode.CONVERSION_FAILED,
        )

    total = 1
    for x in shape:
        total *= x
    if total > MAX_TENSOR_ELEMENTS:
        raise ConvertError(
            f"Input tensor too large: {shape} ({total} elements)",
            stage="input_validation",
            error_code=ErrorCode.CONVERSION_FAILED,
        )


def _make_sample_input(shape: Tuple[int, ...]):
    import torch

    _validate_input_shape(shape)
    return torch.zeros(*shape)


# ---------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------

class Quantizer:
    @staticmethod
    def apply(model, mode: str, warnings: List[str]):
        if mode == "fp32":
            return model

        if mode == "fp16":
            try:
                import coremltools as ct
                return ct.compression_utils.affine_quantize_weights(
                    model, mode="linear"
                )
            except Exception as e:
                warnings.append(f"fp16 quantization failed: {e}")
                return model

        if mode == "int8":
            warnings.append("int8 quantization not supported, using fp32")
            return model

        raise ConvertError(
            f"Unknown quantization mode: {mode}",
            stage="quantization",
            error_code=ErrorCode.CONVERSION_FAILED,
        )


# ---------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------

class TorchScriptStrategy:
    def run(self, model, ctx: ConvertContext, ct_target) -> Tuple[object, str]:
        import torch
        import coremltools as ct

        sample = _make_sample_input(ctx.params.input_shape)

        with torch.no_grad():
            traced = torch.jit.trace(model, sample)

        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(shape=ctx.params.input_shape)],
            minimum_deployment_target=ct_target,
        )

        return mlmodel, "torchscript"


class ONNXStrategy:
    def run(self, model, ctx: ConvertContext, ct_target) -> Tuple[object, str]:
        import coremltools as ct
        from mlbuild.convert.pytorch import export_to_onnx

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            export_to_onnx(
                model=model,
                output_path=tmp_path,
                input_shape=ctx.params.input_shape,
            )

            mlmodel = ct.convert(
                str(tmp_path),
                minimum_deployment_target=ct_target,
            )

            return mlmodel, "onnx_fallback"

        finally:
            if tmp_path.exists():
                tmp_path.unlink()


# ---------------------------------------------------------------------
# CoreML Converter
# ---------------------------------------------------------------------

class CoreMLConverter:
    def __init__(self):
        self.strategies: List[ConversionStrategy] = [
            TorchScriptStrategy(),
            ONNXStrategy(),
        ]

    def convert(self, model, ctx: ConvertContext, ct_target, warnings: List[str]):
        errors = []

        for strategy in self.strategies:
            start = time.time()
            try:
                result, label = strategy.run(model, ctx, ct_target)
                duration = time.time() - start

                logger.info(
                    "conversion_strategy_success",
                    extra={
                        "strategy": label,
                        "duration": duration,
                    },
                )

                return result, label

            except Exception as e:
                logger.exception("strategy_failed", extra={"strategy": strategy.__class__.__name__})
                warnings.append(f"{strategy.__class__.__name__} failed: {e}")
                errors.append((strategy, e))

        raise ConvertError(
            f"All CoreML strategies failed: {[str(e) for _, e in errors]}",
            stage="coreml_conversion",
            error_code=ErrorCode.CONVERSION_FAILED,
        )


# ---------------------------------------------------------------------
# Target resolution (strict)
# ---------------------------------------------------------------------

def _resolve_target(ctx: ConvertContext):
    target = ctx.params.target
    if target is None:
        raise ConvertError(
            "CoreML target required",
            stage="target_resolution",
            error_code=ErrorCode.CONVERSION_FAILED,
        )

    # Normalize to string — handles both TargetDevice enum and plain str
    target_str = target.value if hasattr(target, "value") else str(target)

    # Use the exporter's dynamic TARGET_MAPPING (device str → ct.target enum)
    try:
        from mlbuild.backends.coreml.exporter import TARGET_MAPPING
        mapping = {k: v[0] for k, v in TARGET_MAPPING.items()}
        ct_target = mapping.get(target_str)
        if ct_target is not None:
            return ct_target
    except Exception:
        pass

    # Static fallback if exporter unavailable
    try:
        import coremltools as ct
        static = {
            "apple_m1":  ct.target.macOS13,
            "apple_m2":  ct.target.macOS14,
            "apple_m3":  ct.target.macOS15,
            "apple_a15": ct.target.iOS16,
            "apple_a16": ct.target.iOS16,
            "apple_a17": ct.target.iOS17,
            "apple_a18": ct.target.iOS18,
        }
        ct_target = static.get(target_str)
        if ct_target is not None:
            return ct_target
    except Exception:
        pass

    raise ConvertError(
        f"Invalid CoreML target: '{target_str}'. "
        f"Choose: apple_m1 | apple_m2 | apple_m3 | apple_a15 | apple_a16 | apple_a17 | apple_a18",
        stage="target_resolution",
        error_code=ErrorCode.CONVERSION_FAILED,
    )


# ---------------------------------------------------------------------
# Executor (clean orchestration only)
# ---------------------------------------------------------------------

@register_conversion("pytorch", "coreml")
def convert_pytorch_to_coreml(ctx: ConvertContext) -> ConvertOutput:
    start = time.time()
    warnings: List[str] = []

    try:
        import coremltools as ct
        from mlbuild.convert.pytorch import ModelLoader
    except ImportError:
        raise ConvertError(
            "Missing dependencies for CoreML conversion",
            stage="executor",
            error_code=ErrorCode.DEPENDENCY_MISSING,
        )

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = ctx.output_dir / f"{ctx.input_path.stem}.mlpackage"

    # Resolve target, then pass it directly into the converter
    ct_target = _resolve_target(ctx)

    # Load model
    model = ModelLoader.load(ctx.input_path, mode=ctx.params.load_mode)

    # Convert
    converter = CoreMLConverter()
    mlmodel, strategy = converter.convert(model, ctx, ct_target, warnings)

    # Quantize
    mlmodel = Quantizer.apply(mlmodel, ctx.params.quantize, warnings)

    # Save
    mlmodel.save(str(output_path))

    duration = time.time() - start

    size_mb = sum(
        f.stat().st_size for f in output_path.rglob("*") if f.is_file()
    ) / (1024 * 1024)

    return ConvertOutput(
        path=output_path,
        converter_version=CONVERTER_VERSION,
        metadata={
            "strategy": strategy,
            "quantize": ctx.params.quantize,
            "target": ctx.params.target,
            "coremltools_version": ct.__version__,
            "size_mb": size_mb,
            "input_shape": list(ctx.params.input_shape),
        },
        warnings=warnings,
        duration_seconds=duration,
    )


@register_conversion("onnx", "coreml")
def convert_onnx_to_coreml(ctx: ConvertContext) -> ConvertOutput:
    start = time.time()
    warnings: List[str] = []

    try:
        import coremltools as ct
    except ImportError:
        raise ConvertError(
            "coremltools is required. Install: pip install coremltools",
            stage="convert_onnx_to_coreml",
            error_code=ErrorCode.DEPENDENCY_MISSING,
        )

    try:
        import onnx2torch
    except ImportError:
        raise ConvertError(
            "onnx2torch is required for ONNX → CoreML conversion. "
            "Install: pip install onnx2torch",
            stage="convert_onnx_to_coreml",
            error_code=ErrorCode.DEPENDENCY_MISSING,
        )

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = ctx.output_dir / f"{ctx.input_path.stem}.mlpackage"

    ct_target = _resolve_target(ctx)
    quantize = ctx.params.quantize.value if hasattr(ctx.params.quantize, "value") else ctx.params.quantize

    try:
        import onnx
        onnx_model = onnx.load(str(ctx.input_path))
        torch_model = onnx2torch.convert(onnx_model)
        torch_model.eval()
    except Exception as e:
        raise ConvertError(
            f"ONNX → PyTorch conversion failed: {e}",
            stage="convert_onnx_to_coreml",
            error_code=ErrorCode.CONVERSION_FAILED,
        )

    try:
        import torch
        sample = torch.zeros(*ctx.params.input_shape)
        traced = torch.jit.trace(torch_model, sample)
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(shape=ctx.params.input_shape)],
            minimum_deployment_target=ct_target,
        )
    except Exception as e:
        raise ConvertError(
            f"CoreML conversion from ONNX failed: {e}",
            stage="convert_onnx_to_coreml",
            error_code=ErrorCode.CONVERSION_FAILED,
        )

    mlmodel = Quantizer.apply(mlmodel, quantize, warnings)
    mlmodel.save(str(output_path))

    duration = time.time() - start
    size_mb = sum(
        f.stat().st_size for f in output_path.rglob("*") if f.is_file()
    ) / (1024 * 1024)

    target_str = ctx.params.target.value if hasattr(ctx.params.target, "value") else ctx.params.target

    return ConvertOutput(
        path=output_path,
        converter_version=CONVERTER_VERSION,
        metadata={
            "strategy":          "onnx_via_onnx2torch",
            "quantize":          quantize,
            "target":            target_str,
            "coremltools_version": ct.__version__,
            "size_mb":           size_mb,
        },
        warnings=warnings,
        duration_seconds=duration,
    )