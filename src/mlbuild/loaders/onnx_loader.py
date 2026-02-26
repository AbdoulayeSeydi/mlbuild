"""
Multi-framework model loader for MLBuild.

Supports:
- ONNX (.onnx)
- PyTorch (.pt, .pth) - coming soon
"""

from __future__ import annotations
from pathlib import Path
import logging

import onnx
from onnx import shape_inference
from onnx.checker import ValidationError
from onnx.shape_inference import InferenceError

from ..core.ir import ModelIR, Graph, Tensor, Op, Shape, Dim, DType
from ..core.errors import ModelLoadError, ModelValidationError

logger = logging.getLogger(__name__)


class ONNXLoader:
    """Enterprise-ready ONNX loader with shape inference and validation"""
    
    format_name = "onnx"
    supported_opset_min = 11
    supported_opset_max = 18

    @staticmethod
    def load(path: Path, *, strict: bool = True) -> ModelIR:
        """Load ONNX model with full validation"""
        path = Path(path).expanduser().resolve()
        
        if not path.exists():
            raise ModelLoadError(f"Model not found: {path}")
        
        if path.suffix.lower() != ".onnx":
            raise ModelLoadError(f"Invalid file extension: {path.suffix}")

        try:
            model = onnx.load(str(path))
        except Exception as e:
            raise ModelLoadError(f"Failed to parse ONNX file: {path}") from e

        # Validate structure
        try:
            onnx.checker.check_model(model)
        except ValidationError as exc:
            raise ModelValidationError(f"ONNX validation failed: {path}") from exc

        # Validate opsets
        ONNXLoader._validate_opsets(model)

        # Shape inference
        try:
            model = shape_inference.infer_shapes(model)
        except InferenceError as exc:
            if strict:
                raise ModelLoadError(f"Shape inference failed: {path}") from exc
            logger.warning("Shape inference failed for %s; continuing without full shapes", path)

        # Convert to ModelIR
        return ONNXLoader._convert_to_ir(model, path.name, strict=strict)

    @staticmethod
    def _validate_opsets(model: onnx.ModelProto):
        """Validate ONNX opset versions"""
        for opset in model.opset_import:
            version = opset.version
            if not (ONNXLoader.supported_opset_min <= version <= ONNXLoader.supported_opset_max):
                logger.warning(
                    "ONNX opset %s outside tested range (%s-%s)",
                    version,
                    ONNXLoader.supported_opset_min,
                    ONNXLoader.supported_opset_max
                )

    @staticmethod
    def _convert_to_ir(model: onnx.ModelProto, source_name: str, *, strict: bool) -> ModelIR:
        """Convert ONNX model to ModelIR"""
        from onnx import TensorProto
        
        graph = model.graph
        ir_graph = Graph(name=graph.name or source_name)
        tensor_table = {}

        # Parse inputs
        for value in graph.input:
            dims = []
            if value.type.HasField('tensor_type'):
                tensor_type = value.type.tensor_type
                for dim in tensor_type.shape.dim:
                    if dim.HasField('dim_value'):
                        dims.append(Dim(value=dim.dim_value))
                    elif dim.HasField('dim_param'):
                        dims.append(Dim(symbol=dim.dim_param))
                    else:
                        dims.append(Dim())
                
                dtype_map = {
                    TensorProto.FLOAT: DType.FLOAT32,
                    TensorProto.FLOAT16: DType.FLOAT16,
                    TensorProto.INT32: DType.INT32,
                    TensorProto.INT64: DType.INT64,
                }
                dtype = dtype_map.get(tensor_type.elem_type, DType.FLOAT32)
            else:
                dtype = DType.UNKNOWN
            
            tensor = Tensor(
                id=value.name,
                name=value.name,
                shape=Shape(dims=tuple(dims)),
                dtype=dtype,
                is_input=True,
            )
            tensor_table[tensor.name] = tensor
            ir_graph.add_input(tensor)

        # Parse initializers
        for init in graph.initializer:
            dims = [Dim(value=d) for d in init.dims]
            
            dtype_map = {
                TensorProto.FLOAT: DType.FLOAT32,
                TensorProto.FLOAT16: DType.FLOAT16,
                TensorProto.INT32: DType.INT32,
                TensorProto.INT64: DType.INT64,
            }
            dtype = dtype_map.get(init.data_type, DType.FLOAT32)
            
            tensor = Tensor(
                id=init.name,
                name=init.name,
                shape=Shape(dims=tuple(dims)),
                dtype=dtype,
                is_initializer=True,
            )
            tensor_table[tensor.name] = tensor
            ir_graph.add_constant(tensor)

        # Parse nodes
        for idx, node in enumerate(graph.node):
            # Create placeholders for unknown tensors
            for name in node.input:
                if name and name not in tensor_table:
                    tensor_table[name] = Tensor.placeholder(name)
            
            for name in node.output:
                if name not in tensor_table:
                    tensor_table[name] = Tensor.placeholder(name)
            
            # Parse attributes
            attrs = {}
            for attr in node.attribute:
                attrs[attr.name] = Op.parse_onnx_attribute(attr)
            
            ir_node = Op(
                id=node.name or f"{node.op_type}_{idx}",
                op_type=node.op_type,
                domain=node.domain or "onnx",
                inputs=[n for n in node.input if n],
                outputs=list(node.output),
                attributes=attrs,
            )
            ir_graph.add_node(ir_node)

        # Parse outputs
        for value in graph.output:
            name = value.name
            if name in tensor_table:
                existing = tensor_table[name]
                # Create new tensor with is_output=True
                tensor = Tensor(
                    id=existing.id,
                    name=existing.name,
                    shape=existing.shape,
                    dtype=existing.dtype,
                    is_output=True,
                    is_input=existing.is_input,
                    is_initializer=existing.is_initializer,
                )
                tensor_table[name] = tensor
                ir_graph.add_output(tensor)

        # Create ModelIR with original ONNX model stored
        return ModelIR(
            graph=ir_graph,
            framework="onnx",  # NEW
            original_model=model,  # NEW: Store raw ONNX model
            metadata={
                "framework_version": onnx.__version__,
                "strict_mode": strict,
                "source_model": source_name,
                "producer": model.producer_name or "",
                "opsets": {op.domain or "onnx": op.version for op in model.opset_import},
            },
        )


# Public API
def load_model(path: str, strict: bool = True) -> ModelIR:
    """
    Load model from file.
    
    Supported formats:
    - .onnx (ONNX models)
    
    Args:
        path: Path to model file
        strict: Enable strict validation
        
    Returns:
        ModelIR with framework detection
    """
    path = Path(path)
    
    if path.suffix.lower() == ".onnx":
        return ONNXLoader.load(path, strict=strict)
    
    raise ModelLoadError(f"Unsupported model format: {path.suffix}")