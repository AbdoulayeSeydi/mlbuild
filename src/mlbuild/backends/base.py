"""
Abstract backend interface.
All backends (CoreML, ONNX Runtime, future TensorRT) implement this.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ..core.types import Build


@dataclass
class BackendCapabilities:
    """What this backend can do."""
    supports_fp16: bool = True
    supports_fp32: bool = True
    supports_int8: bool = False
    supports_dynamic_shapes: bool = False
    compute_units: List[str] = None  # ["CPU_ONLY", "CPU_AND_GPU", "ALL"]


@dataclass
class EnvironmentValidation:
    """Environment check result."""
    is_valid: bool
    backend_name: str
    errors: List[str]
    warnings: List[str]
    info: Dict[str, Any]


class Backend(ABC):
    """
    Abstract backend interface.
    All model conversion backends implement this.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name (e.g., 'coreml', 'onnxruntime')."""
        pass
    
    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """What this backend supports."""
        pass
    
    @abstractmethod
    def validate_environment(self) -> EnvironmentValidation:
        """
        Check if this backend can run on current system.
        Returns errors/warnings/info.
        """
        pass
    
    @abstractmethod
    def build(
        self,
        model_path: Path,
        target: str,
        quantize: str,
        name: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Build:
        """
        Convert source model to backend-specific format.
        Returns Build object with all hashes and metadata.
        """
        pass