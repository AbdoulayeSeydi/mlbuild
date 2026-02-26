"""
Modern CoreML exporter - Phase 1 implementation.

Layer 1: Ingestion (ONNX/PyTorch → PyTorch normalization)
Layer 2: Optimization (PyTorch → MLProgram with tracking)
"""

from __future__ import annotations

from importlib.metadata import metadata
import logging
import platform
import subprocess
from pathlib import Path
from typing import Any

import coremltools as ct
import torch

from ...core.ir import ModelIR
from ...core.errors import ConversionError, PlatformError

logger = logging.getLogger(__name__)


# ============================================================
# INT8 Capability Detection
# ============================================================

def _supports_native_int8(target: str) -> bool:
    """
    Check if target supports native INT8 (W8A8) acceleration.
    
    Native INT8 = A17 Pro, M4 and newer
    M1/M2/M3 = Weight-only (W8A16) recommended
    """
    # A17 Pro and newer support W8A8
    if target in ["apple_a17"]:
        return True
    
    # M4 would go here (not in our target list yet)
    # if target in ["apple_m4"]:
    #     return True
    
    # M1/M2/M3 do NOT have native INT8 compute
    if target in ["apple_m1", "apple_m2", "apple_m3"]:
        return False
    
    # Unknown targets - assume weight-only to be safe
    return False


# ============================================================
# Dynamic Target Discovery
# ============================================================

def discover_available_targets() -> dict[str, tuple]:
    """
    Dynamically discover available CoreMLTools targets.
    
    Returns:
        Dict mapping target names to (ct.target enum, display name)
    """
    available = {}
    
    # Try to get all possible targets
    target_candidates = {
    "apple_a17": ("iOS17", "iOS17"),  # Use iOS17
    "apple_a16": ("iOS17", "iOS17"),
    "apple_a15": ("iOS16", "iOS16"),
    "apple_m3": ("macOS14", "macOS14"),  # Use macOS14
    "apple_m2": ("macOS14", "macOS14"),
    "apple_m1": ("macOS13", "macOS13"),
}
    
    for device, (target_attr, display) in target_candidates.items():
        try:
            target_enum = getattr(ct.target, target_attr)
            available[device] = (target_enum, display)
        except AttributeError:
            logger.debug(f"Target {target_attr} not available in coremltools {ct.__version__}")
    
    return available


# Build target mapping dynamically
TARGET_MAPPING = discover_available_targets()


# ============================================================
# Environment Capture
# ============================================================

def capture_build_environment() -> dict[str, Any]:
    """
    Capture complete build environment for reproducibility.
    
    Returns:
        Environment metadata dict
    """
    env = {
        "coremltools": ct.__version__,
        "python": platform.python_version(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "torch": torch.__version__,
    }
    
    # Get macOS-specific info
    if platform.system() == "Darwin":
        env["macos_version"] = platform.mac_ver()[0]
        
        # Get chip info
        try:
            chip = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()
            env["chip"] = chip
        except (subprocess.CalledProcessError, FileNotFoundError):
            env["chip"] = "unknown"
    
    # Check for onnx2torch
    try:
        import onnx2torch
        env["onnx2torch"] = onnx2torch.__version__
    except ImportError:
        env["onnx2torch"] = None
    
    return env


# ============================================================
# Layer 1: Model Ingestion
# ============================================================

class ModelIngestion:
    """Normalize any format to PyTorch"""
    
    @staticmethod
    def extract_input_specs(ir: ModelIR) -> tuple[list, list, list]:
        """
        Extract input specifications from ModelIR.
        
        Preserves dynamic dimensions and logs them.
        
        Args:
            ir: ModelIR from loader
            
        Returns:
            (input_specs list, dynamic_dims list, shape_tuples list)
        """
        input_specs = []
        dynamic_dims_found = []
        shape_tuples = []  # NEW: Store actual shape tuples for torch.randn
        
        for tensor_id in ir.graph.inputs:
            tensor = ir.graph.tensors[tensor_id]
            
            # Build shape, handling dynamic dims
            shape = []
            for i, dim in enumerate(tensor.shape.dims):
                if dim.is_static():
                    shape.append(dim.value)
                else:
                    # Dynamic dimension - freeze to 1 and log
                    logger.warning(
                        f"Dynamic dimension in input '{tensor.name}' at position {i}. "
                        f"Freezing to 1."
                    )
                    shape.append(1)
                    dynamic_dims_found.append({
                        "input": tensor.name,
                        "position": i,
                        "symbol": dim.symbol,
                        "frozen_to": 1
                    })
            
            shape_tuple = tuple(shape)
            shape_tuples.append(shape_tuple)  # Store for later use
            
            input_specs.append(ct.TensorType(
                name=tensor.name,
                shape=shape_tuple
            ))
        
        # Log dynamic dims
        if dynamic_dims_found:
            logger.info(f"Found {len(dynamic_dims_found)} dynamic dimensions across inputs")
        
        return input_specs, dynamic_dims_found, shape_tuples
    
    @staticmethod
    def ingest_onnx(onnx_model) -> tuple[torch.nn.Module, dict]:
        """
        Convert ONNX to PyTorch via onnx2torch.
        
        Args:
            onnx_model: ONNX ModelProto
            
        Returns:
            (PyTorch model, metadata dict)
        """
        from onnx2torch import convert as onnx2torch_convert
        
        torch_model = onnx2torch_convert(onnx_model)
        torch_model.eval()
        
        # Extract metadata
        metadata = {
            "bridge": "onnx2torch",
            "onnx_opset": max([op.version for op in onnx_model.opset_import]),
            "ir_version": onnx_model.ir_version,
            "producer": onnx_model.producer_name,
        }
        
        return torch_model, metadata
    
    @staticmethod
    def torch_to_jit(model: torch.nn.Module, example_input: torch.Tensor) -> tuple[torch.jit.ScriptModule, str]:
        """
        Convert PyTorch model to TorchScript.
        
        Tries script first (preserves control flow), falls back to trace.
        
        Args:
            model: PyTorch model
            example_input: Example input tensor
            
        Returns:
            (TorchScript model, conversion_mode: "script" or "trace")
        """
        try:
            # Try scripting first (preserves control flow)
            logger.info("Attempting torch.jit.script...")
            jit_model = torch.jit.script(model)
            logger.info("✓ Successfully scripted model")
            return jit_model, "script"
        except Exception as e:
            logger.warning(f"Scripting failed: {e}. Falling back to trace.")
            logger.warning("⚠ Using trace mode - control flow may be frozen")
            # Fallback to trace
            with torch.no_grad():
                jit_model = torch.jit.trace(model, example_input)
                return jit_model, "trace"


# ============================================================
# Layer 2: CoreML Optimization
# ============================================================

class CoreMLExporter:
    """
    Modern CoreML exporter with full reproducibility tracking.
    
    Converts normalized PyTorch models to MLProgram format.
    """
    
    def __init__(
        self,
        target: str,
        compute_unit: str = "cpu_only",
        pass_pipeline: str | None = None
    ):
        """
        Initialize exporter.
        
        Args:
            target: Target device (apple_m3, apple_a17, etc)
            compute_unit: Compute unit ('cpu_only', 'all', 'cpu_gpu', 'cpu_ne')
            pass_pipeline: Pass pipeline name (None=default)
        """
        if target not in TARGET_MAPPING:
            available = ", ".join(TARGET_MAPPING.keys())
            raise ValueError(
                f"Unsupported target: {target}. "
                f"Available: {available}"
            )
        
        self.target = target
        self.deployment_target, self.target_name = TARGET_MAPPING[target]
        
        # Map compute unit string to enum
        compute_unit_map = {
            "cpu_only": ct.ComputeUnit.CPU_ONLY,
            "all": ct.ComputeUnit.ALL,
            "cpu_gpu": ct.ComputeUnit.CPU_AND_GPU,
        }
        
        if compute_unit not in compute_unit_map:
            compute_unit = "all"  # Default fallback
        
        self.compute_unit = compute_unit_map[compute_unit]
        self.compute_unit_name = compute_unit
        
        # Get pass pipeline
        self.pass_pipeline = self._get_pass_pipeline(pass_pipeline)
        self.pass_pipeline_name = pass_pipeline or "default"
    
    def _get_pass_pipeline(self, name: str | None):
        """Get CoreML pass pipeline by name"""
        if name is None:
            return None  # Use default
        
        # Pass pipeline support will be added in Phase 2
        return None
    
    def export(
        self,
        ir: ModelIR,
        output_dir: Path,
        quantization: str,
        calibration_data: list = None  # NEW
    ) -> tuple[Path, dict]:
        """
        Export ModelIR to CoreML.
        
        Args:
            ir: ModelIR from loader
            output_dir: Directory to write .mlpackage
            quantization: 'fp32' or 'fp16'
            
        Returns:
            (Path to .mlpackage, conversion metadata dict)
        """
        # Validate framework
        if ir.framework not in ["onnx", "pytorch"]:
            raise ConversionError(
                f"Unsupported framework: {ir.framework}",
                details={"framework": ir.framework, "supported": ["onnx", "pytorch"]},
            )
        
        # Route to appropriate converter
        if ir.framework == "onnx":
            return self._export_onnx_via_torch(ir, output_dir, quantization, calibration_data)  # Pass it through
        elif ir.framework == "pytorch":
            raise ConversionError("Direct PyTorch conversion not yet implemented")
    
    def _export_onnx_via_torch(
    self,
    ir: ModelIR,
    output_dir: Path,
    quantization: str,
    calibration_data: list = None
) -> tuple[Path, dict]:
        """ONNX → onnx2torch → PyTorch → MLProgram"""
        import onnx
        
        onnx_model = ir.original_model
        
        if not isinstance(onnx_model, onnx.ModelProto):
            raise ConversionError(
                "ONNX model must be onnx.ModelProto",
                details={"actual_type": str(type(onnx_model))},
            )
        
        metadata = {}
        
        try:
            # Step 1: ONNX → PyTorch
            logger.info("Converting ONNX → PyTorch via onnx2torch...")
            torch_model, onnx_metadata = ModelIngestion.ingest_onnx(onnx_model)
            metadata.update(onnx_metadata)
            
            # Count nodes before conversion
            graph_nodes_before = len(onnx_model.graph.node)
            metadata["graph_nodes_before"] = graph_nodes_before
            
            # Step 2: Extract input specs
            input_specs, dynamic_dims, shape_tuples = ModelIngestion.extract_input_specs(ir)
            metadata["dynamic_dimensions"] = dynamic_dims
            metadata["shape_tuples"] = shape_tuples

            # Step 3: Create example input
            example_input = torch.randn(*shape_tuples[0]) 
            
            # Step 4: Convert to TorchScript
            logger.info("Converting to TorchScript...")
            traced_model, conversion_mode = ModelIngestion.torch_to_jit(torch_model, example_input)
            metadata["torch_conversion_mode"] = conversion_mode 

            # Log conversion mode to user
            if conversion_mode == "trace":
                logger.warning("Using trace mode (control flow frozen)")

            # Step 5: Convert to CoreML
            logger.info(f"Converting to CoreML (target={self.target_name})...")
            mlmodel, conversion_metadata = self._convert_to_coreml(
                traced_model,
                input_specs,
                quantization,
                calibration_data=calibration_data
            )
            metadata.update(conversion_metadata)
            
            # Step 6: Save
            output_path = output_dir / f"{ir.graph.name}.mlpackage"
            mlmodel.save(str(output_path))
            
            logger.info(f"Saved .mlpackage to: {output_path}")
            
            return output_path, metadata
            
        except Exception as exc:
            raise ConversionError(
                f"ONNX → PyTorch → CoreML conversion failed: {str(exc)}",
                details={
                    "framework": "onnx",
                    "bridge": "onnx2torch",
                    "target": self.target,
                    "coremltools_version": ct.__version__,
                },
            ) from exc
    
    def _convert_to_coreml(
        self,
        traced_model: torch.jit.ScriptModule,
        input_specs: list,
        quantization: str,
        calibration_data: list = None  # NEW: Optional calibration data for INT8
    ) -> tuple[ct.models.MLModel, dict]:
        """Convert TorchScript to CoreML with tracking."""
        metadata = {}
        
        # Determine compute precision and pass pipeline
        if quantization == "fp16":
            # Use FP16ComputePrecision transform for actual weight quantization
            compute_precision = ct.transform.FP16ComputePrecision(
                op_selector=lambda op: True  # Convert all ops
            )
            pass_pipeline = None  # Must use default pipeline with FP16ComputePrecision
            metadata["compute_precision"] = "FLOAT16"
            metadata["pass_pipeline"] = "default"
            
        elif quantization == "int8":
            # NEW: INT8 quantization
            if calibration_data is None:
                raise ValueError(
                    "INT8 quantization requires calibration_data. "
                    "Use CalibrationDataset to generate or load calibration samples."
                )
            
            # Convert calibration data to the format coremltools expects
            # coremltools expects: List[Dict[str, np.ndarray]]
            calibration_samples = []
            input_name = input_specs[0].name  # Assuming single input for now
            
            for sample in calibration_data:
                calibration_samples.append({input_name: sample})
            
            # Use quantization-aware conversion
            compute_precision = ct.precision.FLOAT32  # Base precision
            pass_pipeline = ct.PassPipeline.DEFAULT
            metadata["compute_precision"] = "INT8"
            metadata["pass_pipeline"] = "DEFAULT"
            metadata["calibration_samples"] = len(calibration_samples)
            
        elif quantization == "fp32":
            compute_precision = ct.precision.FLOAT32
            pass_pipeline = ct.PassPipeline.CLEANUP  # Conservative for FP32
            metadata["compute_precision"] = "FLOAT32"
            metadata["pass_pipeline"] = "CLEANUP"
            
        else:
            compute_precision = ct.precision.FLOAT32
            pass_pipeline = ct.PassPipeline.CLEANUP
            metadata["compute_precision"] = "FLOAT32"
            metadata["pass_pipeline"] = "CLEANUP"
        
        # Convert to CoreML
        mlmodel = ct.convert(
            traced_model,
            inputs=input_specs,
            source='pytorch',
            minimum_deployment_target=self.deployment_target,
            compute_precision=compute_precision,
            compute_units=ct.ComputeUnit.CPU_ONLY,
            pass_pipeline=pass_pipeline,
        )
        
        # Apply INT8 quantization post-conversion if needed
        if quantization == "int8":
            import coremltools.optimize.coreml as cto
            
            supports_native = _supports_native_int8(self.target)
            
            if supports_native:
                # A17 Pro / M4: Use W8A8 (Weight + Activation quantization)
                logger.info("Target supports native INT8 - using W8A8 quantization...")
                logger.info(f"Applying INT8 quantization with {len(calibration_samples)} calibration samples...")
                
                try:
                    # Step 1: Quantize activations (requires calibration)
                    activation_config = cto.OptimizationConfig(
                        global_config=cto.OpActivationLinearQuantizerConfig(
                            mode="linear_symmetric"
                        )
                    )
                    
                    mlmodel = cto.linear_quantize_activations(
                        mlmodel,
                        activation_config,
                        calibration_samples[:min(100, len(calibration_samples))],
                    )
                    
                    # Step 2: Quantize weights
                    weight_config = cto.OptimizationConfig(
                        global_config=cto.OpLinearQuantizerConfig(
                            mode="linear_symmetric",
                            weight_threshold=512,
                        )
                    )
                    
                    mlmodel = cto.linear_quantize_weights(
                        mlmodel,
                        weight_config,
                    )
                    
                    metadata["quantization_method"] = "w8a8_linear_symmetric"
                    metadata["weight_threshold"] = 512
                    
                except Exception as e:
                    logger.warning(f"W8A8 quantization failed: {e}")
                    logger.warning("Falling back to weight-only quantization")
                    supports_native = False  # Fall through to weight-only
            
            if not supports_native:
                # M1/M2/M3: Use W8A16 (Weight-only quantization)
                logger.info("Target uses weight-only INT8 (W8A16) - optimal for M1/M2/M3...")
                
                try:
                    # Weight-only quantization (no activation quantization)
                    weight_config = cto.OptimizationConfig(
                        global_config=cto.OpLinearQuantizerConfig(
                            mode="linear_symmetric",
                            weight_threshold=512,
                        )
                    )
                    
                    mlmodel = cto.linear_quantize_weights(
                        mlmodel,
                        weight_config,
                    )
                    
                    metadata["quantization_method"] = "w8a16_linear_symmetric"
                    metadata["weight_threshold"] = 512
                    metadata["optimization_note"] = "Weight-only for M1/M2/M3 Neural Engine"
                    
                except Exception as e:
                    logger.warning(f"INT8 quantization failed: {e}")
                    logger.warning("Falling back to FP32")
                    metadata["quantization_method"] = "failed_fallback_fp32"
                    
                    mlmodel = cto.linear_quantize_weights(
                        mlmodel,
                        weight_config,
                    )
                    
                    metadata["quantization_method"] = "w8a16_linear_symmetric"
                    metadata["weight_threshold"] = 512
                    metadata["optimization_note"] = "Weight-only for M1/M2/M3 Neural Engine"
                    
                except Exception as e:
                    logger.warning(f"INT8 quantization failed: {e}")
                    logger.warning("Falling back to FP32")
                    metadata["quantization_method"] = "failed_fallback_fp32"
        
        # Count nodes after conversion
        spec = mlmodel.get_spec()
        graph_nodes_after = 0
        
        try:
            # MLProgram models store ops in functions
            if hasattr(spec, 'mlProgram'):
                for func_name in spec.mlProgram.functions:
                    func = spec.mlProgram.functions[func_name]
                    # Get the main block
                    for block_name in func.block_specializations:
                        block = func.block_specializations[block_name]
                        graph_nodes_after += len(block.operations)
            # Legacy neuralnetwork models
            elif hasattr(spec, 'neuralNetwork'):
                graph_nodes_after = len(spec.neuralNetwork.layers)
        except Exception as e:
            logger.warning(f"Could not count graph nodes: {e}")
            graph_nodes_after = -1  # Use -1 to indicate failure, not 0
        
        metadata["graph_nodes_after"] = graph_nodes_after
        
        # Strip non-deterministic metadata
        self._strip_metadata(mlmodel)
        
        return mlmodel, metadata
    
    def _strip_metadata(self, mlmodel: ct.models.MLModel):
        """Strip non-deterministic metadata for reproducibility"""
        spec = mlmodel.get_spec()
        spec.description.metadata.author = ""
        spec.description.metadata.license = ""
        spec.description.metadata.shortDescription = ""
        if hasattr(spec.description.metadata, 'versionString'):
            spec.description.metadata.versionString = ""
        return mlmodel
    
    def get_optimizer_config(self) -> dict:
        """Get optimizer configuration for hashing"""
        return {
            "target": self.target,
            "deployment_target": self.target_name,
            "compute_unit": self.compute_unit_name,
            "pass_pipeline": self.pass_pipeline_name,
            "coremltools_version": ct.__version__,
        }