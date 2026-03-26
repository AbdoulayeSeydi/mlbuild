"""
Conversion cache — deterministic cache key generation and lookup.

Guarantees:
- Fully deterministic SHA256 cache keys
- Canonical payload (stable across environments)
- Strict typing (no runtime type branching)
- Environment-aware invalidation
- Strategy-aware (no false reuse across execution paths)
- Safe registry lookup (no silent corruption masking)

Contract:
- compute_source_hash(path) MUST:
    - hash file contents only (NOT mtime)
    - resolve symlinks
    - recursively hash directories in sorted order
"""

from __future__ import annotations

import hashlib
import importlib
import json
from importlib.metadata import version as pkg_version, PackageNotFoundError
from pathlib import Path
from typing import Dict, Optional, Any

from mlbuild.convert.types import ConvertParams
from mlbuild.core.hash import compute_source_hash


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

CACHE_SCHEMA_VERSION = "v1"

TRACKED_LIBS = [
    "torch",
    "onnx",
    "coremltools",
    "tensorflow",
    "onnx2tf",
]


# ---------------------------------------------------------------------
# Environment Version Collection
# ---------------------------------------------------------------------

def _get_version(lib: str) -> Optional[str]:
    """
    Robust version resolution:
    - Primary: importlib.metadata (correct for installed packages)
    - Fallback: module.__version__
    - None if not installed
    """
    try:
        return pkg_version(lib)
    except PackageNotFoundError:
        try:
            mod = importlib.import_module(lib)
            return getattr(mod, "__version__", None)
        except ImportError:
            return None


def collect_env_versions() -> Dict[str, Optional[str]]:
    """
    Collect versions for all tracked libraries.

    Returns a normalized dict:
        {lib_name: version_or_None}

    Ordering is NOT relied on here — normalization happens later.
    """
    return {lib: _get_version(lib) for lib in TRACKED_LIBS}


# ---------------------------------------------------------------------
# Canonicalization Helpers
# ---------------------------------------------------------------------

def _normalize_env(env: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """
    Ensure:
    - All tracked libs present
    - Stable ordering
    - No missing keys
    """
    return {lib: env.get(lib) for lib in sorted(TRACKED_LIBS)}


def _canonical_json(payload: Dict[str, Any]) -> str:
    """
    Strict canonical JSON encoding:
    - Sorted keys
    - No whitespace
    - Stable representation
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


# ---------------------------------------------------------------------
# Cache Key
# ---------------------------------------------------------------------

def build_cache_key(
    input_path: Path,
    src_format: str,
    dst_format: str,
    params: ConvertParams,
    converter_version: str,
    env_versions: Dict[str, Optional[str]],
    *,
    executor_name: str,
    strategy_hint: Optional[str] = None,
) -> str:
    """
    Build deterministic cache key for a single conversion step.

    Includes:
    - Input file hash (per-step input, NOT original model)
    - Source + destination formats
    - All semantic conversion parameters
    - Converter version + executor identity
    - Optional strategy hint (if execution path varies)
    - Normalized environment versions
    - Cache schema version

    Excludes:
    - name, notes (cosmetic)
    - run_id (non-semantic)
    """

    # --- normalize path (prevents aliasing bugs) ---
    input_path = input_path.resolve()

    # --- hash input deterministically ---
    input_hash = compute_source_hash(input_path)

    # --- normalize env ---
    env_normalized = _normalize_env(env_versions)

    # --- build canonical payload ---
    payload: Dict[str, Any] = {
        "schema": CACHE_SCHEMA_VERSION,
        "input_hash": input_hash,
        "src": src_format,
        "dst": dst_format,
        "params": {
            "input_shape": list(params.input_shape),
            "quantize": params.quantize.value,
            "load_mode": params.load_mode.value,
            "opset": params.opset,
            "target": params.target.value if params.target else None,
        },
        "execution": {
            "converter_version": converter_version,
            "executor": executor_name,
            "strategy_hint": strategy_hint,
        },
        "env": env_normalized,
    }

    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------
# Registry Lookup
# ---------------------------------------------------------------------

def find_cached_build(cache_key: str, registry) -> Optional[Any]:
    """
    Lookup cached build by cache key.

    Behavior:
    - Returns Build if found
    - Returns None if not found
    - Raises on unexpected errors (no silent corruption masking)

    Registry contract:
        find_by_cache_key(key: str) -> Optional[Build]
    """
    finder = getattr(registry, "find_by_cache_key", None)

    if finder is None:
        return None  # registry does not support caching

    try:
        return finder(cache_key)
    except (KeyError, ValueError):
        # expected "not found" / lookup errors
        return None
    except Exception as e:
        # DO NOT silently swallow corruption / infra bugs
        raise RuntimeError(f"[cache] registry lookup failed: {e}") from e