"""
Backend discovery and registration.
"""

from typing import Dict, Type, List
from pathlib import Path
from .base import Backend, EnvironmentValidation
import inspect


class BackendRegistry:
    """
    Central registry for all backends.
    Auto-discovers and validates backends.
    """
    
    _backends: Dict[str, Type[Backend]] = {}
    
    @classmethod
    def register(cls, name: str, backend_cls: Type[Backend]):
        """Register a backend class."""
        cls._backends[name] = backend_cls
    
    @classmethod
    def get_backend(cls, name: str, artifact_dir: Path = None) -> Backend:
        """Get backend instance by name."""
        if name not in cls._backends:
            available = ", ".join(cls._backends.keys())
            raise ValueError(
                f"Unknown backend: '{name}'. Available: {available}"
            )
        
        backend_cls = cls._backends[name]
        
        # Default artifact dir
        if artifact_dir is None:
            artifact_dir = Path.cwd() / ".mlbuild" / "artifacts"
        
        # Check if backend accepts artifact_dir parameter
        sig = inspect.signature(backend_cls.__init__)
        params = list(sig.parameters.keys())
        
        if 'artifact_dir' in params:
            backend = backend_cls(artifact_dir=artifact_dir)
        else:
            # Legacy backend (like CoreML) - no artifact_dir
            backend = backend_cls()
        
        # Validate environment
        validation = backend.validate_environment()
        if not validation.is_valid:
            errors = "\n  ".join(validation.errors)
            raise RuntimeError(
                f"Backend '{name}' not available:\n  {errors}\n\n"
                f"Run 'mlbuild doctor' for details"
            )
        
        return backend
    
    @classmethod
    def list_backends(cls) -> Dict[str, EnvironmentValidation]:
        """List all backends with their validation status."""
        results = {}
        artifact_dir = Path.cwd() / ".mlbuild" / "artifacts"
        
        for name, backend_cls in cls._backends.items():
            try:
                # Check if backend accepts artifact_dir
                sig = inspect.signature(backend_cls.__init__)
                params = list(sig.parameters.keys())
                
                if 'artifact_dir' in params:
                    backend = backend_cls(artifact_dir=artifact_dir)
                else:
                    backend = backend_cls()
                
                results[name] = backend.validate_environment()
            except Exception as e:
                results[name] = EnvironmentValidation(
                    is_valid=False,
                    backend_name=name,
                    errors=[str(e)],
                    warnings=[],
                    info={}
                )
        return results
    
    @classmethod
    def available_backends(cls) -> List[str]:
        """List names of available (valid) backends."""
        return [
            name for name, validation 
            in cls.list_backends().items()
            if validation.is_valid
        ]


# Auto-register backends
from .coreml.backend import CoreMLBackend
from .tflite.backend import TFLiteBackend

BackendRegistry.register("coreml", CoreMLBackend)
BackendRegistry.register("tflite", TFLiteBackend)