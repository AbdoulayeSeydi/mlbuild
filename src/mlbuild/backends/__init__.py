"""
MLBuild backends.

This file ensures all backends are auto-registered on import.
"""

from .registry import BackendRegistry
from .base import Backend, BackendCapabilities, EnvironmentValidation

# Force import of all backends to trigger auto-registration
from . import coreml
from . import onnxruntime

__all__ = [
    'BackendRegistry',
    'Backend',
    'BackendCapabilities',
    'EnvironmentValidation',
]