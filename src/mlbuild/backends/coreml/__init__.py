"""CoreML backend for model conversion and benchmarking."""

from .exporter import CoreMLExporter
from .backend import CoreMLBackend

# Auto-register with backend registry
from ..registry import BackendRegistry
BackendRegistry.register("coreml", CoreMLBackend)

__all__ = ["CoreMLExporter", "CoreMLBackend"]