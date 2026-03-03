"""
MLBuild Backends

Available backends:
- CoreML: Apple devices (iOS, macOS)
- TFLite: Android, edge devices
"""

from .base import Backend, BackendCapabilities, EnvironmentValidation
from .registry import BackendRegistry
from .coreml.backend import CoreMLBackend
from .tflite.backend import TFLiteBackend

__all__ = [
    "Backend",
    "BackendCapabilities",
    "EnvironmentValidation",
    "BackendRegistry",
    "CoreMLBackend",
    "TFLiteBackend",
]