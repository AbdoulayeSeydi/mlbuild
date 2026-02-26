"""ONNX Runtime backend for cross-platform CPU/GPU inference."""

from .backend import ONNXRuntimeBackend

# Auto-register
from ..registry import BackendRegistry
BackendRegistry.register("onnxruntime", ONNXRuntimeBackend)

__all__ = ["ONNXRuntimeBackend"]