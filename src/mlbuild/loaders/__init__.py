"""
Model loaders for MLBuild.

Supports:
- ONNX models (.onnx)
- PyTorch models (.pt, .pth) - coming soon
"""

from .onnx_loader import load_model

__all__ = ["load_model"]