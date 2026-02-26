"""
MLBuild - Deterministic build system for CoreML models with real device benchmarking.
"""

__version__ = "0.1.0"

# Lazy imports - don't load heavy dependencies at module level
def __getattr__(name):
    if name == "LocalRegistry":
        from .registry.local import LocalRegistry
        return LocalRegistry
    elif name == "load_model":
        from .loaders.loader import load_model
        return load_model
    elif name == "ModelIR":
        from .core.ir import ModelIR
        return ModelIR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["__version__"]