"""
Backend-agnostic artifact hashing utilities.
"""

import hashlib
from pathlib import Path
from typing import Literal


def compute_file_hash(file_path: Path) -> str:
    """
    Simple SHA256 hash of a single file.
    Used for ONNX, TensorRT, and other single-file formats.
    """
    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def compute_directory_hash(dir_path: Path) -> str:
    """
    Deterministic hash of all files in a directory.
    Used for non-CoreML multi-file formats.
    """
    hasher = hashlib.sha256()
    
    # Sort files for deterministic order
    files = sorted(dir_path.rglob('*'))
    
    for file_path in files:
        if file_path.is_file():
            # Hash relative path
            rel_path = file_path.relative_to(dir_path)
            hasher.update(str(rel_path).encode('utf-8'))
            
            # Hash file contents
            hasher.update(file_path.read_bytes())
    
    return hasher.hexdigest()


def compute_backend_artifact_hash(
    artifact_path: Path,
    backend: Literal["coreml", "onnxruntime", "tensorrt"]
) -> str:
    """
    Compute artifact hash based on backend type.
    
    Args:
        artifact_path: Path to artifact (directory or file)
        backend: Backend name
        
    Returns:
        SHA256 hash string
    """
    if backend == "coreml":
        # Use existing CoreML-specific hashing
        from .hash import compute_artifact_hash as compute_coreml_hash
        return compute_coreml_hash(artifact_path)
    
    elif backend == "onnxruntime":
        # ONNX Runtime stores model as single .onnx file in directory
        model_file = artifact_path / "model.onnx"
        if not model_file.exists():
            raise ValueError(f"ONNX model not found: {model_file}")
        return compute_file_hash(model_file)
    
    elif backend == "tensorrt":
        # TensorRT stores engine as single .engine file
        engine_file = artifact_path / "model.engine"
        if not engine_file.exists():
            raise ValueError(f"TensorRT engine not found: {engine_file}")
        return compute_file_hash(engine_file)
    
    else:
        raise ValueError(f"Unsupported backend for hashing: {backend}")