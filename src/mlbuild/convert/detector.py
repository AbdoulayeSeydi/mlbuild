"""
Model format detector.

Detects the source format of a model from its path.
Used by service.py before path resolution.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from mlbuild.core.errors import ConvertError, ErrorCode

logger = logging.getLogger("mlbuild.convert.detector")


# ---------------------------------------------------------------------
# Detector registry
# ---------------------------------------------------------------------

DetectorFn = Callable[[Path], Optional[str]]
_DETECTORS: List[Tuple[int, DetectorFn]] = []  # (priority, fn)


def register_detector(priority: int):
    """
    Register a format detector.

    Higher priority runs first.
    Detector returns:
        - format string if match
        - None if not applicable
    """
    def wrapper(fn: DetectorFn):
        _DETECTORS.append((priority, fn))
        _DETECTORS.sort(key=lambda x: -x[0])
        return fn
    return wrapper


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except Exception:
        return False


def _read_magic(path: Path, n: int = 16) -> bytes:
    try:
        with path.open("rb") as f:
            return f.read(n)
    except Exception:
        return b""


# ---------------------------------------------------------------------
# File-based detectors
# ---------------------------------------------------------------------

@register_detector(priority=100)
def _detect_onnx(path: Path) -> Optional[str]:
    if not _is_nonempty_file(path):
        return None

    # Extension hint
    if path.suffix.lower() == ".onnx":
        return "onnx"

    # Magic fallback (protobuf-based → weak heuristic)
    magic = _read_magic(path)
    if b"ir_version" in magic or b"onnx" in magic:
        return "onnx"

    return None


@register_detector(priority=100)
def _detect_tflite(path: Path) -> Optional[str]:
    if not _is_nonempty_file(path):
        return None

    if path.suffix.lower() == ".tflite":
        return "tflite"

    magic = _read_magic(path)
    # Flatbuffer files often contain "TFL3"
    if b"TFL3" in magic:
        return "tflite"

    return None


@register_detector(priority=100)
def _detect_pytorch(path: Path) -> Optional[str]:
    if not _is_nonempty_file(path):
        return None

    if path.suffix.lower() in (".pt", ".pth"):
        return "pytorch"

    # Weak heuristic: PyTorch pickles often start with 0x80
    magic = _read_magic(path)
    if magic.startswith(b"\x80"):
        return "pytorch"

    return None


@register_detector(priority=100)
def _detect_mlmodel(path: Path) -> Optional[str]:
    if not _is_nonempty_file(path):
        return None

    if path.suffix.lower() == ".mlmodel":
        return "mlmodel"

    return None


# ---------------------------------------------------------------------
# Directory-based detectors
# ---------------------------------------------------------------------

@register_detector(priority=200)
def _detect_mlpackage(path: Path) -> Optional[str]:
    if not path.is_dir():
        return None

    # Structure-based detection (NOT suffix-based)
    manifest = path / "Manifest.json"
    if manifest.exists():
        return "mlpackage"

    return None


@register_detector(priority=200)
def _detect_savedmodel(path: Path) -> Optional[str]:
    if not path.is_dir():
        return None

    pb = path / "saved_model.pb"
    pbtxt = path / "saved_model.pbtxt"

    if not (pb.exists() or pbtxt.exists()):
        return None

    # Optional lightweight validation
    try:
        import tensorflow as tf
        tf.saved_model.load(str(path))
        return "savedmodel"
    except Exception as e:
        logger.warning(f"SavedModel structure detected but failed to load: {e}")
        return None


# ---------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------

def detect_format(path: Path) -> str:
    """
    Detect model format using:
    1. Structural checks (directories)
    2. Magic bytes
    3. Extension hints (last)

    Returns:
        format string

    Raises:
        ConvertError with detailed diagnostics
    """
    path = Path(path)

    if not path.exists():
        raise ConvertError(
            f"Model path does not exist: {path}",
            stage="detect_format",
            error_code=ErrorCode.MODEL_LOAD_FAILED,
        )

    # Reject empty files early
    if path.is_file() and path.stat().st_size == 0:
        raise ConvertError(
            f"Model file is empty (0 bytes): {path}",
            stage="detect_format",
            error_code=ErrorCode.MODEL_LOAD_FAILED,
        )

    matches: List[str] = []

    for _, detector in _DETECTORS:
        try:
            result = detector(path)
            if result:
                matches.append(result)
        except Exception as e:
            logger.debug(f"Detector {detector.__name__} failed: {e}")

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise ConvertError(
            f"Ambiguous model format for {path}.\n"
            f"Matched formats: {matches}\n"
            f"Resolve by renaming file or specifying format explicitly.",
            stage="detect_format",
            error_code=ErrorCode.UNSUPPORTED_MODEL,
        )

    # No matches → detailed error
    suffix = path.suffix.lower()
    kind = "directory" if path.is_dir() else "file"
    size = path.stat().st_size if path.is_file() else "N/A"

    raise ConvertError(
        f"Could not detect model format.\n"
        f"Path: {path}\n"
        f"Type: {kind}\n"
        f"Extension: '{suffix or 'none'}'\n"
        f"Size: {size}\n\n"
        f"Supported formats:\n"
        f"  - PyTorch (.pt, .pth)\n"
        f"  - ONNX (.onnx)\n"
        f"  - TFLite (.tflite)\n"
        f"  - CoreML (.mlmodel, .mlpackage)\n"
        f"  - TensorFlow SavedModel (directory with saved_model.pb)\n",
        stage="detect_format",
        error_code=ErrorCode.UNSUPPORTED_MODEL,
    )