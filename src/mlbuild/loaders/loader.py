"""
Model loading entrypoint for MLBuild.

This module resolves the appropriate loader implementation
based on file extension and delegates model parsing to it.

Design principles:
- Extension-based resolution (explicit, not magic-byte detection)
- Registry-driven (no branching sprawl)
- Path-native (no implicit string coercion)
- Clear error semantics
- Future-proof for directory-based formats (e.g. .mlpackage)
"""

from pathlib import Path
from typing import Dict, Protocol, Type

from ..core.ir import ModelIR
from ..core.errors import ModelLoadError
from .onnx_loader import ONNXLoader


class BaseLoader(Protocol):
    """
    Loader interface contract.

    All loaders must implement:
        load(path: Path) -> ModelIR
    """
    @staticmethod
    def load(path: Path) -> ModelIR: ...


# Central loader registry.
# Adding support for new formats requires updating this map only.
_LOADERS: Dict[str, Type[BaseLoader]] = {
    ".onnx": ONNXLoader,
}


def supported_formats() -> tuple[str, ...]:
    """
    Returns a tuple of supported file extensions.
    """
    return tuple(sorted(_LOADERS.keys()))


def load_model(model_path: str | Path) -> ModelIR:
    """
    Load a model from disk based on file extension.

    Format detection is strictly extension-based.
    No content-based auto-detection is performed.

    Args:
        model_path: Path to model file or directory.

    Returns:
        ModelIR instance.

    Raises:
        ModelLoadError:
            - If path does not exist.
            - If path is neither file nor directory.
            - If extension is unsupported.
            - If loader fails internally.
    """
    path = Path(model_path).expanduser().resolve()

    # Existence check
    if not path.exists():
        raise ModelLoadError(f"Model path does not exist: {path}")

    # Sanity check: must be file or directory
    if not (path.is_file() or path.is_dir()):
        raise ModelLoadError(
            f"Invalid model path type: {path}. "
            "Expected file or directory."
        )

    suffix = path.suffix.lower()

    loader_cls = _LOADERS.get(suffix)

    if loader_cls is None:
        raise ModelLoadError(
            f"Unsupported model format: '{suffix}'. "
            f"Supported formats: {', '.join(supported_formats())}"
        )

    try:
        return loader_cls.load(path)
    except ModelLoadError:
        # Preserve domain errors exactly
        raise
    except Exception as exc:
        # Prevent leaking arbitrary backend exceptions upward
        raise ModelLoadError(
            f"Failed to load model '{path}' using "
            f"{loader_cls.__name__}: {exc}"
        ) from exc