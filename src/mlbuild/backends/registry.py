"""
Backend discovery and registration.
"""

from typing import Dict, Type, List
from .base import Backend, EnvironmentValidation


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
    def get_backend(cls, name: str) -> Backend:
        """Get backend instance by name."""
        if name not in cls._backends:
            available = ", ".join(cls._backends.keys())
            raise ValueError(
                f"Unknown backend: '{name}'. Available: {available}"
            )
        
        backend_cls = cls._backends[name]
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
        for name, backend_cls in cls._backends.items():
            try:
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