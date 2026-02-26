"""
Refactor AI – Canonical Intermediate Representation (IR)

This IR is:
- Format-agnostic (ONNX is just one frontend)
- Explicitly graph-based
- Typed and shape-safe
- Hardware- and quantization-aware
- Designed for transformation, partitioning, and scheduling

ONNX, CoreML, TFLite, etc. are IMPORTERS / EXPORTERS — not the IR.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple, TypeAlias
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# ONNX Type Mapping (Centralized)
# ---------------------------------------------------------------------

def get_onnx_dtype(dtype_value):
    """
    Convert any dtype representation to ONNX TensorProto enum.
    
    Handles:
    - JAX dtype objects (jnp.float32)
    - NumPy dtype objects (np.float32)
    - String representations ("float32")
    - Already-converted ONNX type codes (1)
    
    Returns: ONNX TensorProto enum integer
    """
    from onnx import TensorProto
    
    # If it's already an integer, assume it's an ONNX type code
    if isinstance(dtype_value, int):
        return dtype_value
    
    # Convert to string for mapping
    dtype_str = str(dtype_value).lower()
    
    # Clean up common prefixes
    dtype_str = dtype_str.replace('jax.numpy.', '')
    dtype_str = dtype_str.replace('numpy.', '')
    dtype_str = dtype_str.replace('jnp.', '')
    dtype_str = dtype_str.replace('np.', '')
    dtype_str = dtype_str.replace('dtype(', '').replace(')', '')
    dtype_str = dtype_str.replace("'", "").replace('"', '')
    
    # Map to ONNX TensorProto enum
    ONNX_DTYPE_MAP = {
        'float16': TensorProto.FLOAT16,
        'float32': TensorProto.FLOAT,
        'float64': TensorProto.DOUBLE,
        'double': TensorProto.DOUBLE,
        'bfloat16': TensorProto.BFLOAT16,
        'int8': TensorProto.INT8,
        'int16': TensorProto.INT16,
        'int32': TensorProto.INT32,
        'int64': TensorProto.INT64,
        'uint8': TensorProto.UINT8,
        'uint16': TensorProto.UINT16,
        'uint32': TensorProto.UINT32,
        'uint64': TensorProto.UINT64,
        'bool': TensorProto.BOOL,
        'bool_': TensorProto.BOOL,
    }
    
    if dtype_str not in ONNX_DTYPE_MAP:
        logger.warning(f"Unknown dtype '{dtype_value}' (parsed as '{dtype_str}'), defaulting to FLOAT32")
        return TensorProto.FLOAT
    
    return ONNX_DTYPE_MAP[dtype_str]

# ---------------------------------------------------------------------
# Canonical Types (NO STRINGS)
# ---------------------------------------------------------------------

class DType(Enum):
    FLOAT32 = auto()
    FLOAT16 = auto()
    BFLOAT16 = auto()
    FLOAT8 = auto()
    INT8 = auto()
    UINT8 = auto()
    INT16 = auto()
    INT32 = auto()
    INT64 = auto()
    BOOL = auto()
    STRING = auto()
    COMPLEX64 = auto()
    COMPLEX128 = auto()
    UNKNOWN = auto()


class Layout(Enum):
    NCHW = auto()
    NHWC = auto()
    CHW = auto()
    HWC = auto()
    UNKNOWN = auto()


class Device(Enum):
    CPU = auto()
    GPU = auto()
    NPU = auto()
    DSP = auto()
    TPU = auto()
    UNKNOWN = auto()

# ---------------------------------------------------------------------
# Shape System (Preserves Symbolic + Dynamic Info)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Dim:
    value: Optional[int] = None      # Static dimension
    symbol: Optional[str] = None     # Symbolic name (e.g. "batch")

    def is_static(self) -> bool:
        return self.value is not None

    def is_dynamic(self) -> bool:
        return self.value is None

    def __repr__(self) -> str:
        if self.value is not None:
            return str(self.value)
        return self.symbol or "?"


@dataclass(frozen=True)
class Shape:
    dims: Tuple[Dim, ...]

    def rank(self) -> int:
        return len(self.dims)

    def is_fully_static(self) -> bool:
        return all(d.is_static() for d in self.dims)

    def numel(self) -> Optional[int]:
        if not self.is_fully_static():
            return None
        n = 1
        for d in self.dims:
            n *= d.value
        return n

    def __repr__(self) -> str:
        return "(" + ", ".join(map(str, self.dims)) + ")"

# ---------------------------------------------------------------------
# Tensor Representation
# ---------------------------------------------------------------------

@dataclass
class QuantParams:
    scale: float
    zero_point: int
    axis: Optional[int] = None


@dataclass
class Tensor:
    """
    Canonical tensor in the IR graph.
    """
    id: str
    name: str
    shape: Shape
    dtype: DType
    layout: Layout = Layout.UNKNOWN

    is_input: bool = False
    is_output: bool = False
    is_initializer: bool = False

    producer: Optional[str] = None
    consumers: Set[str] = field(default_factory=set)

    quant_params: Optional[QuantParams] = None

    preferred_device: Device = Device.UNKNOWN
    memory_bytes: Optional[int] = None
    
    # Actual tensor data (numpy array for initializers/constants)
    data: Optional[Any] = None

    # -------------------------
    # Factory methods
    # -------------------------

    @classmethod
    def from_onnx_value_info(cls, value_info):
        """Create Tensor from ONNX ValueInfoProto"""
        import onnx
        from onnx import TensorProto
        
        name = value_info.name
        
        # Parse shape
        dims = []
        if value_info.type.HasField('tensor_type'):
            tensor_type = value_info.type.tensor_type
            for dim in tensor_type.shape.dim:
                if dim.HasField('dim_value'):
                    dims.append(Dim(value=dim.dim_value))
                elif dim.HasField('dim_param'):
                    dims.append(Dim(symbol=dim.dim_param))
                else:
                    dims.append(Dim())  # Dynamic/unknown
            
            # Parse dtype
            onnx_dtype = tensor_type.elem_type
            dtype_map = {
                TensorProto.FLOAT: DType.FLOAT32,
                TensorProto.FLOAT16: DType.FLOAT16,
                TensorProto.DOUBLE: DType.FLOAT32,
                TensorProto.INT8: DType.INT8,
                TensorProto.INT16: DType.INT16,
                TensorProto.INT32: DType.INT32,
                TensorProto.INT64: DType.INT64,
                TensorProto.UINT8: DType.UINT8,
                TensorProto.BOOL: DType.BOOL,
                TensorProto.STRING: DType.STRING,
            }
            dtype = dtype_map.get(onnx_dtype, DType.FLOAT32)
        else:
            dims = []
            dtype = DType.UNKNOWN
        
        shape = Shape(dims=tuple(dims))
        
        return cls(
            id=name,
            name=name,
            shape=shape,
            dtype=dtype,
            is_input=False,
            is_output=False,
        )

    @classmethod
    def from_onnx_initializer(cls, initializer):
        """Create Tensor from ONNX TensorProto (initializer/constant)"""
        import onnx
        from onnx import TensorProto
        
        name = initializer.name
        
        # Parse shape
        dims = [Dim(value=d) for d in initializer.dims]
        shape = Shape(dims=tuple(dims))
        
        # Parse dtype
        dtype_map = {
            TensorProto.FLOAT: DType.FLOAT32,
            TensorProto.FLOAT16: DType.FLOAT16,
            TensorProto.INT8: DType.INT8,
            TensorProto.INT32: DType.INT32,
            TensorProto.INT64: DType.INT64,
            TensorProto.UINT8: DType.UINT8,
            TensorProto.BOOL: DType.BOOL,
        }
        dtype = dtype_map.get(initializer.data_type, DType.FLOAT32)
        
        # Calculate memory
        numel = shape.numel()
        bytes_per_elem = {
            DType.FLOAT32: 4,
            DType.FLOAT16: 2,
            DType.INT8: 1,
            DType.UINT8: 1,
            DType.INT32: 4,
            DType.INT64: 8,
            DType.BOOL: 1,
        }.get(dtype, 4)
        
        memory_bytes = numel * bytes_per_elem if numel else 0
        
        return cls(
            id=name,
            name=name,
            shape=shape,
            dtype=dtype,
            is_initializer=True,
            memory_bytes=memory_bytes,
        )

    @classmethod
    def placeholder(cls, name: str):
        """Create a placeholder tensor (for intermediate values)"""
        return cls(
            id=name,
            name=name,
            shape=Shape(dims=()),  # Empty shape - unknown
            dtype=DType.UNKNOWN,
        )

# ---------------------------------------------------------------------
# Operator Representation
# ---------------------------------------------------------------------

@dataclass
class Op:
    """
    Canonical operator node.
    """
    id: str
    op_type: str
    domain: str = ""

    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)

    attributes: Dict[str, Any] = field(default_factory=dict)

    supported_devices: List[Device] = field(default_factory=list)
    estimated_latency_ms: Optional[float] = None

    # -------------------------
    # Helpers
    # -------------------------

    @staticmethod
    def parse_onnx_attribute(attr):
        """Parse ONNX AttributeProto to Python value"""
        import onnx
        from onnx import AttributeProto
        
        attr_type = attr.type
        
        if attr_type == AttributeProto.FLOAT:
            return attr.f
        elif attr_type == AttributeProto.INT:
            return attr.i
        elif attr_type == AttributeProto.STRING:
            return attr.s.decode('utf-8')
        elif attr_type == AttributeProto.TENSOR:
            return attr.t  # Return raw tensor proto
        elif attr_type == AttributeProto.FLOATS:
            return list(attr.floats)
        elif attr_type == AttributeProto.INTS:
            return list(attr.ints)
        elif attr_type == AttributeProto.STRINGS:
            return [s.decode('utf-8') for s in attr.strings]
        else:
            return None

# ---------------------------------------------------------------------
# Graph Representation
# ---------------------------------------------------------------------

@dataclass
class Graph:
    """
    Explicit directed acyclic graph.
    """
    name: str

    tensors: Dict[str, Tensor] = field(default_factory=dict)
    ops: Dict[str, Op] = field(default_factory=dict)

    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)

    # -------------------------
    # Graph building
    # -------------------------

    def add_input(self, tensor: Tensor):
        """Add input tensor to graph"""
        tensor.is_input = True
        self.tensors[tensor.id] = tensor
        if tensor.id not in self.inputs:
            self.inputs.append(tensor.id)

    def add_output(self, tensor: Tensor):
        """Add output tensor to graph"""
        tensor.is_output = True
        self.tensors[tensor.id] = tensor
        if tensor.id not in self.outputs:
            self.outputs.append(tensor.id)

    def add_constant(self, tensor: Tensor):
        """Add constant/initializer tensor to graph"""
        tensor.is_initializer = True
        self.tensors[tensor.id] = tensor

    def add_node(self, node: Op):
        """Add operator node to graph"""
        self.ops[node.id] = node
        
        # Update tensor producer/consumer relationships
        for output_id in node.outputs:
            if output_id in self.tensors:
                self.tensors[output_id].producer = node.id
        
        for input_id in node.inputs:
            if input_id in self.tensors:
                self.tensors[input_id].consumers.add(node.id)

    # -------------------------
    # Validation
    # -------------------------

    def validate(self) -> None:
        # Every op input/output must reference a tensor
        for op in self.ops.values():
            for tid in op.inputs + op.outputs:
                if tid not in self.tensors:
                    raise ValueError(f"Op {op.id} references missing tensor {tid}")

        # Tensor producer/consumer consistency
        for tid, tensor in self.tensors.items():
            if tensor.producer and tensor.producer not in self.ops:
                raise ValueError(f"Tensor {tid} has invalid producer {tensor.producer}")
            for c in tensor.consumers:
                if c not in self.ops:
                    raise ValueError(f"Tensor {tid} has invalid consumer {c}")

    # -------------------------
    # Analysis helpers
    # -------------------------

    def num_parameters(self) -> int:
        total = 0
        for t in self.tensors.values():
            if t.is_initializer:
                n = t.shape.numel()
                if n is not None:
                    total += n
        return total

    def operator_histogram(self) -> Dict[str, int]:
        hist: Dict[str, int] = {}
        for op in self.ops.values():
            hist[op.op_type] = hist.get(op.op_type, 0) + 1
        return hist

# ---------------------------------------------------------------------
# Model IR (Top-Level Object)
# ---------------------------------------------------------------------

@dataclass
class ModelIR:
    """
    Canonical, backend-agnostic model representation.
    """
    graph: Graph
    framework: str = "onnx"  # NEW: "pytorch", "tensorflow", "onnx", etc.
    original_model: Any = None  # NEW: Store raw model (torch.nn.Module, onnx.ModelProto, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.graph.validate()

    # -------------------------
    # Introspection
    # -------------------------

    @property
    def num_nodes(self) -> int:
        return len(self.graph.ops)

    @property
    def num_tensors(self) -> int:
        return len(self.graph.tensors)

    @property
    def num_parameters(self) -> int:
        return self.graph.num_parameters()

    @property
    def input_tensors(self):
        """Get list of input tensors"""
        return [self.graph.tensors[tid] for tid in self.graph.inputs]

    @property
    def output_tensors(self):
        """Get list of output tensors"""
        return [self.graph.tensors[tid] for tid in self.graph.outputs]

    def get_operators(self):
        """Get operator histogram {op_type: count}"""
        return self.graph.operator_histogram()

    # -------------------------
    # Factory methods
    # -------------------------

    @classmethod
    def from_onnx_model(cls, onnx_model, source_format: str = 'onnx'):
        """Create ModelIR from ONNX ModelProto"""
        # This will be implemented by ONNXLoader._convert_to_ir
        # Just create a minimal IR for now
        graph = Graph(name=onnx_model.graph.name or "model")
        
        metadata = {
            'source_format': source_format,
            'producer': onnx_model.producer_name,
        }
        
        return cls(graph=graph, metadata=metadata)

    # -------------------------
    # Export methods
    # -------------------------

    def to_onnx(self, output_path: str, validate: bool = True):
        """
        Export IR to ONNX file
        
        Args:
            output_path: Path to save ONNX model
            validate: Whether to validate the model after saving (default: True)
        """
        import onnx
        from pathlib import Path
        
        onnx_model = self.to_onnx_model()
        
        # Save model
        logger.info(f"Saving ONNX model to: {output_path}")
        onnx.save(onnx_model, output_path)
        
        # Get file size
        file_size_mb = Path(output_path).stat().st_size / 1024 / 1024
        logger.info(f"ONNX model saved: {file_size_mb:.2f} MB")
        
        # Validate
        if validate:
            logger.info("Validating ONNX model...")
            if file_size_mb > 2000:  # >2GB needs file-based validation
                logger.info("Model >2GB, using file-based validation")
                onnx.checker.check_model(output_path)
            else:
                logger.info("Model <2GB, using in-memory validation")
                onnx.checker.check_model(onnx_model)
            logger.info("✓ ONNX model is valid")
        
        return onnx_model

    def to_onnx_model(self):
        """Convert IR to ONNX ModelProto with proper op lowering"""
        import onnx
        from onnx import helper, TensorProto
        import numpy as np
        
        logger.info("="*60)
        logger.info("STARTING IR → ONNX CONVERSION")
        logger.info(f"Total ops to convert: {len(self.graph.ops)}")
        logger.info(f"Op types in IR: {list(self.graph.operator_histogram().keys())}")
        logger.info("="*60)
        
        # ONNX DType mapping
        dtype_map = {
            DType.FLOAT32: TensorProto.FLOAT,
            DType.FLOAT16: TensorProto.FLOAT16,
            DType.BFLOAT16: TensorProto.BFLOAT16,
            DType.INT8: TensorProto.INT8,
            DType.INT32: TensorProto.INT32,
            DType.INT64: TensorProto.INT64,
            DType.UINT8: TensorProto.UINT8,
            DType.BOOL: TensorProto.BOOL,
            DType.UNKNOWN: TensorProto.FLOAT,
        }
        
        # Step 1: Create inputs
        logger.info(f"\n🔵 STEP 1: Creating {len(self.graph.inputs)} inputs")
        onnx_inputs = []
        for tensor_id in self.graph.inputs:
            tensor = self.graph.tensors[tensor_id]
            shape = [d.value if d.is_static() else None for d in tensor.shape.dims]
            onnx_type = dtype_map.get(tensor.dtype, TensorProto.FLOAT)
            
            value_info = helper.make_tensor_value_info(tensor.id, onnx_type, shape)
            onnx_inputs.append(value_info)
            logger.info(f"  Input: {tensor.id} shape={shape}")
        
        # Step 2: Create initializers FIRST
        logger.info(f"\n🔵 STEP 2: Creating initializers")
        onnx_initializers = []
        initializer_count = 0
        for tensor in self.graph.tensors.values():
            if tensor.is_initializer and tensor.data is not None:
                onnx_type = dtype_map.get(tensor.dtype, TensorProto.FLOAT)
                
                if not isinstance(tensor.data, np.ndarray):
                    tensor_data = np.array(tensor.data, dtype=np.float32)
                else:
                    tensor_data = tensor.data
                
                onnx_tensor = helper.make_tensor(
                    name=tensor.id,
                    data_type=onnx_type,
                    dims=tensor_data.shape,
                    vals=tensor_data.flatten().tolist()
                )
                onnx_initializers.append(onnx_tensor)
                initializer_count += 1
        logger.info(f"  Created {initializer_count} initializers")
        
        # Step 3: Topologically sort ops
        logger.info(f"\n🔵 STEP 3: Topologically sorting ops")
        sorted_ops = self._topological_sort_ops()
        logger.info(f"  Sorted {len(sorted_ops)} ops")
        
        # Step 4: Lower IR ops to ONNX ops (CRITICAL STEP)
        logger.info(f"\n🔵 STEP 4: Lowering {len(sorted_ops)} ops to ONNX")
        onnx_nodes = []
        for i, op in enumerate(sorted_ops):
            logger.info(f"\n--- Lowering op {i+1}/{len(sorted_ops)} ---")
            onnx_node = self._lower_op_to_onnx(op)
            if onnx_node:
                logger.info(f"✓ Lowered to ONNX node: {onnx_node.op_type}")
                onnx_nodes.append(onnx_node)
        
        logger.info(f"\n🔵 STEP 4 COMPLETE: Created {len(onnx_nodes)} ONNX nodes")
        logger.info(f"ONNX node types: {[n.op_type for n in onnx_nodes[:5]]}...")
        
        # Step 5: Create outputs
        logger.info(f"\n🔵 STEP 5: Creating {len(self.graph.outputs)} outputs")
        onnx_outputs = []
        for tensor_id in self.graph.outputs:
            tensor = self.graph.tensors[tensor_id]
            shape = [d.value if d.is_static() else None for d in tensor.shape.dims]
            onnx_type = dtype_map.get(tensor.dtype, TensorProto.FLOAT)
            
            value_info = helper.make_tensor_value_info(tensor.id, onnx_type, shape)
            onnx_outputs.append(value_info)
            logger.info(f"  Output: {tensor.id} shape={shape}")
        
        # Step 6: Create graph
        logger.info(f"\n🔵 STEP 6: Creating ONNX graph")
        onnx_graph = helper.make_graph(
            nodes=onnx_nodes,
            name=self.graph.name,
            inputs=onnx_inputs,
            outputs=onnx_outputs,
            initializer=onnx_initializers
        )
        logger.info(f"  Graph created: {len(onnx_graph.node)} nodes")
        

        import onnx
        # Step 7: Create model
        logger.info(f"\n🔵 STEP 7: Creating ONNX model")
        onnx_model = helper.make_model(
            onnx_graph,
            producer_name=self.metadata.get('producer', 'Refactor'),
            opset_imports=[helper.make_opsetid("", 17)],
        )

        onnx_model.ir_version = onnx.IR_VERSION
        
        logger.info("="*60)
        logger.info("IR → ONNX CONVERSION COMPLETE")
        logger.info("="*60)
        
        return onnx_model

    def _lower_op_to_onnx(self, op: Op):
        """
        Lower IR op to ONNX op.
        
        CRITICAL: This is where IR semantic ops are translated to ONNX ops.
        Never pass through op.op_type directly to ONNX.
        """
        from onnx import helper
        
        # DEBUG: Print what we're lowering
        logger.info(f"🔴 LOWERING OP: {op.id} | type={op.op_type} | domain={op.domain}")
        
        # Get tensor IDs
        input_ids = [tid for tid in op.inputs if tid in self.graph.tensors]
        output_ids = [tid for tid in op.outputs if tid in self.graph.tensors]
        
        logger.info(f"🔴   inputs={input_ids[:2]}... ({len(input_ids)} total)")
        logger.info(f"🔴   outputs={output_ids}")
        
        # Forbidden attributes
        FORBIDDEN = {"name", "dtype", "trainable", "batch_input_shape", "input_shape"}
        clean_attrs = {k: v for k, v in op.attributes.items() if k not in FORBIDDEN}
        
        logger.info(f"🔴   attributes={list(clean_attrs.keys())}")
        
        # ============================================================
        # Keras Op Lowering
        # ============================================================
        
        if op.domain == "keras":
            if op.op_type == "Conv2D":
                logger.info(f"🟢 MATCHED Keras Conv2D → Conv")
                attrs = {}
                if "kernel_size" in clean_attrs:
                    attrs["kernel_shape"] = clean_attrs["kernel_size"]
                if "strides" in clean_attrs:
                    attrs["strides"] = clean_attrs["strides"]
                if "padding" in clean_attrs:
                    if clean_attrs["padding"] == "same":
                        attrs["auto_pad"] = "SAME_UPPER"
                    else:
                        attrs["auto_pad"] = "VALID"
                
                logger.info(f"🟢 Creating ONNX Conv node with attrs={attrs}")
                node = helper.make_node("Conv", input_ids, output_ids, name=op.id, **attrs)
                logger.info(f"🟢 Created node: op_type={node.op_type}")
                return node
            
            elif op.op_type == "Dense":
                logger.info(f"🟢 MATCHED Keras Dense → Gemm")
                node = helper.make_node("Gemm", input_ids, output_ids, name=op.id)
                logger.info(f"🟢 Created node: op_type={node.op_type}")
                return node
            
            elif op.op_type == "MaxPooling2D":
                logger.info(f"🟢 MATCHED Keras MaxPooling2D → MaxPool")
                attrs = {}
                if "pool_size" in clean_attrs:
                    attrs["kernel_shape"] = clean_attrs["pool_size"]
                if "strides" in clean_attrs:
                    attrs["strides"] = clean_attrs["strides"]
                node = helper.make_node("MaxPool", input_ids, output_ids, name=op.id, **attrs)
                logger.info(f"🟢 Created node: op_type={node.op_type}")
                return node
            
            elif op.op_type == "Flatten":
                logger.info(f"🟢 MATCHED Keras Flatten → Flatten")
                node = helper.make_node("Flatten", input_ids, output_ids, name=op.id)
                logger.info(f"🟢 Created node: op_type={node.op_type}")
                return node
            
            elif op.op_type == "Activation":
                # Handle Keras Activation layers
                activation = clean_attrs.get("activation", "linear")
                if activation == "relu":
                    logger.info(f"🟢 MATCHED Keras Activation(relu) → Relu")
                    node = helper.make_node("Relu", input_ids, output_ids, name=op.id)
                elif activation == "sigmoid":
                    logger.info(f"🟢 MATCHED Keras Activation(sigmoid) → Sigmoid")
                    node = helper.make_node("Sigmoid", input_ids, output_ids, name=op.id)
                elif activation == "tanh":
                    logger.info(f"🟢 MATCHED Keras Activation(tanh) → Tanh")
                    node = helper.make_node("Tanh", input_ids, output_ids, name=op.id)
                elif activation == "softmax":
                    logger.info(f"🟢 MATCHED Keras Activation(softmax) → Softmax")
                    node = helper.make_node("Softmax", input_ids, output_ids, name=op.id, axis=-1)
                else:
                    logger.warning(f"Unknown activation: {activation}, using Identity")
                    node = helper.make_node("Identity", input_ids, output_ids, name=op.id)
                logger.info(f"🟢 Created node: op_type={node.op_type}")
                return node
            
            elif op.op_type == "ReLU":
                logger.info(f"🟢 MATCHED Keras ReLU → Relu")
                node = helper.make_node("Relu", input_ids, output_ids, name=op.id)
                logger.info(f"🟢 Created node: op_type={node.op_type}")
                return node
            
            elif op.op_type == "Softmax":
                logger.info(f"🟢 MATCHED Keras Softmax → Softmax")
                node = helper.make_node("Softmax", input_ids, output_ids, name=op.id, axis=-1)
                logger.info(f"🟢 Created node: op_type={node.op_type}")
                return node
            
            elif op.op_type == "BatchNormalization":
                logger.info(f"🟢 MATCHED Keras BatchNormalization → BatchNormalization")
                attrs = {}
                if "epsilon" in clean_attrs:
                    attrs["epsilon"] = clean_attrs["epsilon"]
                if "momentum" in clean_attrs:
                    attrs["momentum"] = clean_attrs["momentum"]
                node = helper.make_node("BatchNormalization", input_ids, output_ids, name=op.id, **attrs)
                logger.info(f"🟢 Created node: op_type={node.op_type}")
                return node
            
            elif op.op_type == "Dropout":
                # Dropout is a no-op during inference
                logger.info(f"🟢 MATCHED Keras Dropout → Identity (inference mode)")
                node = helper.make_node("Identity", input_ids, output_ids, name=op.id)
                logger.info(f"🟢 Created node: op_type={node.op_type}")
                return node
            
            elif op.op_type == "AveragePooling2D":
                logger.info(f"🟢 MATCHED Keras AveragePooling2D → AveragePool")
                attrs = {}
                if "pool_size" in clean_attrs:
                    attrs["kernel_shape"] = clean_attrs["pool_size"]
                if "strides" in clean_attrs:
                    attrs["strides"] = clean_attrs["strides"]
                node = helper.make_node("AveragePool", input_ids, output_ids, name=op.id, **attrs)
                logger.info(f"🟢 Created node: op_type={node.op_type}")
                return node
            
            elif op.op_type == "GlobalAveragePooling2D":
                logger.info(f"🟢 MATCHED Keras GlobalAveragePooling2D → GlobalAveragePool")
                node = helper.make_node("GlobalAveragePool", input_ids, output_ids, name=op.id)
                logger.info(f"🟢 Created node: op_type={node.op_type}")
                return node
            
            elif op.op_type == "Add":
                logger.info(f"🟢 MATCHED Keras Add → Add")
                node = helper.make_node("Add", input_ids, output_ids, name=op.id)
                logger.info(f"🟢 Created node: op_type={node.op_type}")
                return node
            
            elif op.op_type == "Concatenate":
                logger.info(f"🟢 MATCHED Keras Concatenate → Concat")
                attrs = {}
                if "axis" in clean_attrs:
                    attrs["axis"] = clean_attrs["axis"]
                node = helper.make_node("Concat", input_ids, output_ids, name=op.id, **attrs)
                logger.info(f"🟢 Created node: op_type={node.op_type}")
                return node
            
            elif op.op_type == "Reshape":
                logger.info(f"🟢 MATCHED Keras Reshape → Identity (passthrough)")
                node = helper.make_node("Identity", input_ids, output_ids, name=op.id)
                logger.info(f"🟢 Created node: op_type={node.op_type}")
                return node
            
            else:
                logger.error(f"🔴 NO MATCH for Keras layer type={op.op_type}")
                raise NotImplementedError(
                    f"Keras layer '{op.op_type}' has no ONNX lowering rule. "
                    f"Add explicit lowering in _lower_op_to_onnx()."
                )
        
        # ============================================================
        # JAX Primitive Lowering
        # ============================================================
        
        elif op.domain == "jax":
            if op.op_type == "conv_general_dilated":
                # JAX conv → ONNX Conv
                logger.info(f"🟢 MATCHED JAX conv_general_dilated → Conv")
                params = op.attributes
                attrs = {}
                
                # Extract convolution attributes
                if 'window_strides' in params:
                    attrs['strides'] = list(params['window_strides'])
                if 'padding' in params:
                    # JAX padding format: [(pad_before, pad_after), ...]
                    padding = params['padding']
                    if padding == 'SAME':
                        attrs['auto_pad'] = 'SAME_UPPER'
                    elif padding == 'VALID':
                        attrs['auto_pad'] = 'VALID'
                    else:
                        # Explicit padding
                        attrs['pads'] = [p for pair in padding for p in pair]
                
                logger.info(f"🟢 Creating ONNX Conv node with attrs={attrs}")
                node = helper.make_node("Conv", input_ids, output_ids, name=op.id, **attrs)
                logger.info(f"🟢 Created node: op_type={node.op_type}")
                return node
            
            elif op.op_type == "add":
                logger.info(f"🟢 MATCHED JAX add → Add")
                node = helper.make_node("Add", input_ids, output_ids, name=op.id)
                logger.info(f"🟢 Created node: op_type={node.op_type}")
                return node
            
            elif op.op_type == "reshape":
                # Skip reshape for now - ONNX reshape is complex
                # Just use Identity (passthrough)
                logger.info(f"🟢 MATCHED JAX reshape → Identity (passthrough)")
                node = helper.make_node("Identity", input_ids, output_ids, name=op.id)
                logger.info(f"🟢 Created node: op_type={node.op_type}")
                return node
            
            elif op.op_type == "custom_jvp_call":
                # This is JAX's ReLU activation
                logger.info(f"🟢 MATCHED JAX custom_jvp_call → Relu")
                call_jaxpr = op.attributes.get('call_jaxpr')
                if call_jaxpr:
                    # Check if it's relu
                    node = helper.make_node("Relu", input_ids, output_ids, name=op.id)
                else:
                    # Generic activation
                    node = helper.make_node("Relu", input_ids, output_ids, name=op.id)
                logger.info(f"🟢 Created node: op_type={node.op_type}")
                return node
            
            elif op.op_type == "reduce_window_sum":
                # JAX reduce_window_sum is used for pooling operations
                # When followed by div, it's average pooling
                # Use AveragePool directly (the div will be optimized out)
                logger.info(f"🟢 MATCHED JAX reduce_window_sum → AveragePool")
                params = op.attributes
                attrs = {}
                
                if 'window_dimensions' in params:
                    # Skip batch and channel dims, take spatial dims
                    kernel = list(params['window_dimensions'])[2:]  # [2, 2]
                    attrs['kernel_shape'] = kernel
                
                if 'window_strides' in params:
                    strides = list(params['window_strides'])[2:]
                    attrs['strides'] = strides
                
                # Use AveragePool instead of ReduceSum
                logger.info(f"🟢 Creating ONNX AveragePool node with attrs={attrs}")
                node = helper.make_node("AveragePool", input_ids, output_ids, name=op.id, **attrs)
                logger.info(f"🟢 Created node: op_type={node.op_type}")
                return node
            
            elif op.op_type == "convert_element_type":
                # Cast operation
                logger.info(f"🟢 MATCHED JAX convert_element_type → Cast")
                params = op.attributes
                
                # Get dtype from attributes (could be under 'new_dtype' or 'dtype')
                raw_dtype = params.get('new_dtype', params.get('dtype'))
                
                if raw_dtype is None:
                    logger.error(f"convert_element_type missing dtype attribute. Params: {params}")
                    raise ValueError(f"convert_element_type op {op.id} missing dtype attribute")
                
                # Convert to ONNX TensorProto enum using centralized function
                to_dtype = get_onnx_dtype(raw_dtype)
                
                logger.info(f"🟢 Converting dtype {raw_dtype} → ONNX type code {to_dtype}")
                node = helper.make_node("Cast", input_ids, output_ids, name=op.id, to=to_dtype)
                logger.info(f"🟢 Created node: op_type={node.op_type}")
                return node
            
            elif op.op_type == "div":
                logger.info(f"🟢 MATCHED JAX div → Div")
                node = helper.make_node("Div", input_ids, output_ids, name=op.id)
                logger.info(f"🟢 Created node: op_type={node.op_type}")
                return node
            
            elif op.op_type == "dot_general":
                # Matrix multiplication → Gemm
                logger.info(f"🟢 MATCHED JAX dot_general → Gemm")
                node = helper.make_node("Gemm", input_ids, output_ids, name=op.id)
                logger.info(f"🟢 Created node: op_type={node.op_type}")
                return node
            
            else:
                logger.error(f"🔴 NO MATCH for JAX op_type={op.op_type}")
                raise NotImplementedError(
                    f"JAX primitive '{op.op_type}' has no ONNX lowering rule. "
                    f"Add explicit lowering in _lower_op_to_onnx()."
                )
        
        # ============================================================
        # Generic ONNX Ops (domain-agnostic, used by GGUF transformer graphs)
        # ============================================================
        
        elif op.op_type in ["MatMul", "Add", "Mul", "Gather", "LayerNormalization", "Silu", "Identity"]:
            # These ops map directly to ONNX without transformation
            logger.info(f"🟢 MATCHED Generic {op.op_type} → {op.op_type}")
            
            # Extract clean attributes
            attrs = {k: v for k, v in clean_attrs.items()}
            
            node = helper.make_node(op.op_type, input_ids, output_ids, name=op.id, **attrs)
            logger.info(f"🟢 Created node: op_type={node.op_type}")
            return node
        
        else:
            # Unsupported op
            logger.error(f"🔴 NO MATCH for op_type={op.op_type} domain={op.domain}")
            raise NotImplementedError(
                f"IR op '{op.op_type}' (domain='{op.domain}') has no ONNX lowering rule. "
                f"Add explicit lowering in _lower_op_to_onnx()."
            )

    def _topological_sort_ops(self) -> List[Op]:
        """Topologically sort operations in the graph"""
        in_degree = {op_id: 0 for op_id in self.graph.ops}
        adjacency = {op_id: [] for op_id in self.graph.ops}
        
        for op_id, op in self.graph.ops.items():
            for input_id in op.inputs:
                if input_id in self.graph.tensors:
                    tensor = self.graph.tensors[input_id]
                    if tensor.producer and tensor.producer in self.graph.ops:
                        adjacency[tensor.producer].append(op_id)
                        in_degree[op_id] += 1
        
        queue = [op_id for op_id, degree in in_degree.items() if degree == 0]
        sorted_ops = []
        
        while queue:
            op_id = queue.pop(0)
            sorted_ops.append(self.graph.ops[op_id])
            
            for neighbor in adjacency[op_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(sorted_ops) != len(self.graph.ops):
            logger.warning("Graph has cycles or disconnected components, using original order")
            return list(self.graph.ops.values())
        
        return sorted_ops

    def __repr__(self) -> str:
        return (
            f"ModelIR("
            f"nodes={self.num_nodes}, "
            f"tensors={self.num_tensors}, "
            f"params={self.num_parameters:,}"
            f")"
        )

# ---------------------------------------------------------------------
# Backwards Compatibility Type Aliases
# ---------------------------------------------------------------------

IRGraph: TypeAlias = Graph
IROp: TypeAlias = Op
IRTensor: TypeAlias = Tensor
IRNode: TypeAlias = Op
QuantizationInfo: TypeAlias = QuantParams

# Export all
__all__ = [
    'ModelIR', 'Graph', 'Tensor', 'Op', 'Shape', 'Dim',
    'DType', 'Layout', 'Device', 'QuantParams',
    'IRGraph', 'IROp', 'IRTensor', 'IRNode', 'QuantizationInfo',
]