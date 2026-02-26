"""
Deterministic artifact and configuration diffing utilities.

Provides:
- Binary artifact comparison for .mlpackage directories
- Configuration deep-diff with type-aware comparison
- Deterministic, canonical output
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Any
import json


# ============================================================
# Artifact Comparison (Binary-Level)
# ============================================================

def compare_artifacts(path_a: str, path_b: str) -> Dict[str, Any]:
    """
    Deep comparison of two .mlpackage artifacts.
    
    Compares:
    - File structure (presence/absence of files)
    - File sizes
    - File hashes (SHA256)
    
    Args:
        path_a: Path to first .mlpackage directory
        path_b: Path to second .mlpackage directory
        
    Returns:
        Dictionary with diff results (empty if identical)
    """
    path_a = Path(path_a)
    path_b = Path(path_b)
    
    if not path_a.exists() or not path_a.is_dir():
        raise ValueError(f"Invalid artifact path: {path_a}")
    
    if not path_b.exists() or not path_b.is_dir():
        raise ValueError(f"Invalid artifact path: {path_b}")
    
    # Get all files in both packages
    files_a = {f.relative_to(path_a): f for f in path_a.rglob("*") if f.is_file()}
    files_b = {f.relative_to(path_b): f for f in path_b.rglob("*") if f.is_file()}
    
    all_files = sorted(set(files_a.keys()) | set(files_b.keys()))
    
    diff = {}
    
    for rel_path in all_files:
        file_a = files_a.get(rel_path)
        file_b = files_b.get(rel_path)
        
        # File only in A
        if file_a and not file_b:
            diff[str(rel_path)] = {
                "status": "only_in_a",
                "size_a": file_a.stat().st_size,
            }
            continue
        
        # File only in B
        if file_b and not file_a:
            diff[str(rel_path)] = {
                "status": "only_in_b",
                "size_b": file_b.stat().st_size,
            }
            continue
        
        # File in both - compare
        size_a = file_a.stat().st_size
        size_b = file_b.stat().st_size
        
        # Quick check: different sizes = different files
        if size_a != size_b:
            diff[str(rel_path)] = {
                "status": "different_size",
                "size_a": size_a,
                "size_b": size_b,
                "delta": size_b - size_a,
            }
            continue
        
        # Same size - hash to confirm
        hash_a = _hash_file(file_a)
        hash_b = _hash_file(file_b)
        
        if hash_a != hash_b:
            diff[str(rel_path)] = {
                "status": "different_content",
                "size": size_a,  # Same size, different content
                "hash_a": hash_a,
                "hash_b": hash_b,
            }
    
    return diff


def _hash_file(path: Path) -> str:
    """Compute SHA256 hash of file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


# ============================================================
# Configuration Comparison (Deep-Diff)
# ============================================================

def compare_configs(config_a: Dict[str, Any], config_b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep comparison of two configuration dictionaries.
    
    Type-aware comparison with canonical output.
    
    Args:
        config_a: First configuration
        config_b: Second configuration
        
    Returns:
        Dictionary with diff results (empty if identical)
    """
    diff = {}
    
    # Get all keys
    all_keys = sorted(set(config_a.keys()) | set(config_b.keys()))
    
    for key in all_keys:
        val_a = config_a.get(key)
        val_b = config_b.get(key)
        
        # Key only in A
        if key in config_a and key not in config_b:
            diff[key] = {
                "status": "only_in_a",
                "value_a": _serialize_value(val_a),
            }
            continue
        
        # Key only in B
        if key in config_b and key not in config_a:
            diff[key] = {
                "status": "only_in_b",
                "value_b": _serialize_value(val_b),
            }
            continue
        
        # Key in both - compare values
        if _values_equal(val_a, val_b):
            # Equal - don't include in diff
            continue
        
        # Different values
        diff[key] = {
            "status": "different",
            "value_a": _serialize_value(val_a),
            "value_b": _serialize_value(val_b),
            "type_a": type(val_a).__name__,
            "type_b": type(val_b).__name__,
        }
    
    return diff


def _values_equal(a: Any, b: Any) -> bool:
    """
    Type-aware equality comparison.
    
    Handles nested dicts, lists, and primitive types.
    """
    # Type mismatch (except int/float)
    if type(a) != type(b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return float(a) == float(b)
        return False
    
    # Dict comparison
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_values_equal(a[k], b[k]) for k in a.keys())
    
    # List comparison
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(_values_equal(a[i], b[i]) for i in range(len(a)))
    
    # Primitive comparison
    return a == b


def _serialize_value(val: Any) -> Any:
    """
    Serialize value for JSON output.
    
    Handles nested structures and non-JSON types.
    """
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    
    if isinstance(val, dict):
        return {k: _serialize_value(v) for k, v in val.items()}
    
    if isinstance(val, (list, tuple)):
        return [_serialize_value(v) for v in val]
    
    # Fallback: convert to string
    return str(val)