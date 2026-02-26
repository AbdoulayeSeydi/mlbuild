"""
CoreML backend implementation.
Wraps existing CoreMLExporter into Backend interface.
"""

import platform
from pathlib import Path
from typing import Optional

from ..base import Backend, BackendCapabilities, EnvironmentValidation
from ...core.types import Build


class CoreMLBackend(Backend):
    """CoreML backend for Apple Silicon."""
    
    @property
    def name(self) -> str:
        return "coreml"
    
    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_fp16=True,
            supports_fp32=True,
            supports_int8=False,  # TODO: Add INT8
            supports_dynamic_shapes=False,
            compute_units=["CPU_ONLY", "CPU_AND_GPU", "ALL"]
        )
    
    def validate_environment(self) -> EnvironmentValidation:
        """Check if CoreML can run."""
        errors = []
        warnings = []
        info = {}
        
        # Check platform
        if platform.system() != "Darwin":
            errors.append("CoreML requires macOS")
        else:
            mac_version = platform.mac_ver()[0]
            info["platform"] = f"macOS {mac_version}"
        
        # Check Apple Silicon
        machine = platform.machine()
        if machine != "arm64":
            warnings.append(
                f"Running on {machine}. CoreML works best on Apple Silicon (M1/M2/M3)"
            )
        else:
            info["chip"] = "Apple Silicon"
        
        # Check coremltools
        try:
            import coremltools as ct
            info["coremltools"] = ct.__version__
        except ImportError:
            errors.append("coremltools not installed. Run: pip install coremltools")
        
        # Check onnx
        try:
            import onnx
            info["onnx"] = onnx.__version__
        except ImportError:
            errors.append("onnx not installed. Run: pip install onnx")
        
        return EnvironmentValidation(
            is_valid=len(errors) == 0,
            backend_name=self.name,
            errors=errors,
            warnings=warnings,
            info=info
        )
    
    def build(
        self,
        model_path: Path,
        target: str,
        quantize: str,
        name: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Build:
        """
        Build CoreML model.
        
        This is a thin wrapper around your existing build command logic.
        For now, it just calls the existing build flow.
        """
        # Import your existing build function
        from ...cli.commands.build import build as existing_build_logic
        
        # For now, raise NotImplementedError
        # We'll integrate properly in next step
        raise NotImplementedError(
            "CoreMLBackend.build() needs integration with existing build command.\n"
            "Use 'mlbuild build' command directly for now."
        )