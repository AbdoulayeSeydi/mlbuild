"""
Enterprise-grade environment fingerprinting for ML reproducibility.

Design goals:
- Explicit entropy capture (if it affects numerics, we fingerprint it)
- Structured sub-collectors (maintainable)
- No bare excepts
- Canonical JSON hashing
- Hardware-aware
- Toolchain-aware
- Dependency-lock aware
"""

from __future__ import annotations

import hashlib
import json
import locale
import os
import platform
import random
import subprocess
import sys
import time
from typing import Any, Dict, List


# ============================================================
# Utilities
# ============================================================

def _safe_run(cmd: List[str], timeout: int = 5) -> str | None:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None


# ============================================================
# Python / Runtime
# ============================================================

def collect_python() -> Dict[str, Any]:
    return {
        "version": sys.version,
        "implementation": platform.python_implementation(),
        "compiler": platform.python_compiler(),
        "build": platform.python_build(),
        "executable": sys.executable,
    }


# ============================================================
# Dependency State
# ============================================================

def collect_pip_freeze() -> List[str]:
    output = _safe_run([sys.executable, "-m", "pip", "freeze"])
    if not output:
        return []
    lines = sorted(line.strip() for line in output.splitlines() if line.strip())
    return lines


def collect_poetry_lock_hash() -> str | None:
    if os.path.exists("poetry.lock"):
        with open("poetry.lock", "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    return None


def collect_conda_env() -> str | None:
    output = _safe_run(["conda", "env", "export"])
    if output:
        return hashlib.sha256(output.encode()).hexdigest()
    return None


def collect_dependencies() -> Dict[str, Any]:
    return {
        "pip_freeze": collect_pip_freeze(),
        "poetry_lock_hash": collect_poetry_lock_hash(),
        "conda_env_hash": collect_conda_env(),
    }


# ============================================================
# Numerical Libraries (LAZY IMPORTS INSIDE FUNCTIONS)
# ============================================================

def collect_numpy() -> Dict[str, Any]:
    try:
        import numpy as np
    except ImportError:
        return {"installed": False}

    try:
        from io import StringIO
        buf = StringIO()
        np.__config__.show(buf)
        blas_info = buf.getvalue()
    except Exception:
        blas_info = "unavailable"

    return {
        "installed": True,
        "version": np.__version__,
        "blas_info": blas_info,
    }


def collect_torch() -> Dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"installed": False}

    cuda_available = torch.cuda.is_available()

    return {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "initial_seed": torch.initial_seed(),
    }


def collect_tensorflow() -> Dict[str, Any]:
    try:
        import tensorflow as tf
    except ImportError:
        return {"installed": False}

    build_info = tf.sysconfig.get_build_info()

    return {
        "installed": True,
        "version": tf.__version__,
        "cuda_version": build_info.get("cuda_version"),
        "cudnn_version": build_info.get("cudnn_version"),
    }


# ============================================================
# ONNX Runtime (sorted providers)
# ============================================================

def collect_onnxruntime() -> Dict[str, Any]:
    try:
        import onnxruntime as ort
    except ImportError:
        return {"installed": False}

    providers = sorted(ort.get_available_providers())

    return {
        "installed": True,
        "version": ort.__version__,
        "providers": providers,
    }


# ============================================================
# Hardware
# ============================================================

def collect_cpu_info() -> Dict[str, Any]:
    uname = platform.uname()
    cpu_model = _safe_run(["sysctl", "-n", "machdep.cpu.brand_string"]) if uname.system == "Darwin" else None

    return {
        "system": uname.system,
        "node": uname.node,
        "release": uname.release,
        "version": uname.version,
        "machine": uname.machine,
        "processor": uname.processor,
        "cpu_model": cpu_model,
    }


def collect_gpu_driver() -> Dict[str, Any]:
    nvidia_smi = _safe_run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    return {
        "nvidia_driver_version": nvidia_smi,
    }


# ============================================================
# Toolchain
# ============================================================

def collect_compilers() -> Dict[str, Any]:
    return {
        "gcc_version": _safe_run(["gcc", "--version"]),
        "clang_version": _safe_run(["clang", "--version"]),
        "cxx_version": _safe_run(["c++", "--version"]),
    }


def collect_macos_sdk() -> Dict[str, Any]:
    if platform.system() != "Darwin":
        return {}

    return {
        "xcode_version": _safe_run(["xcodebuild", "-version"]),
        "sdk_path": _safe_run(["xcrun", "--show-sdk-path"]),
        "clt_version": _safe_run(["pkgutil", "--pkg-info=com.apple.pkg.CLTools_Executables"]),
    }


# ============================================================
# Random State
# ============================================================

def collect_random_state() -> Dict[str, Any]:
    state = random.getstate()
    return {
        "python_random_state_hash": hashlib.sha256(str(state).encode()).hexdigest(),
    }


# ============================================================
# Locale / TZ
# ============================================================

def collect_locale_tz() -> Dict[str, Any]:
    return {
        "locale": locale.getlocale(),
        "preferred_encoding": locale.getpreferredencoding(),
        "tzname": time.tzname,
        "LANG": os.environ.get("LANG"),
        "LC_ALL": os.environ.get("LC_ALL"),
    }


# ============================================================
# Master Collector
# ============================================================

def collect_environment() -> Dict[str, Any]:
    return {
        "python": collect_python(),
        "dependencies": collect_dependencies(),
        "numpy": collect_numpy(),
        "torch": collect_torch(),
        "tensorflow": collect_tensorflow(),
        "onnxruntime": collect_onnxruntime(),
        "hardware": {
            "cpu": collect_cpu_info(),
            "gpu_driver": collect_gpu_driver(),
        },
        "toolchain": {
            "compilers": collect_compilers(),
            "macos_sdk": collect_macos_sdk(),
        },
        "random_state": collect_random_state(),
        "locale_tz": collect_locale_tz(),
    }


# ============================================================
# Deterministic Hash
# ============================================================

def hash_environment(env_data: Dict[str, Any]) -> str:
    canonical = json.dumps(
        env_data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ============================================================
# Reproducibility Validation - Returns (bool, list[str])
# ============================================================

def validate_reproducibility() -> tuple[bool, list[str]]:
    """
    Enterprise-grade reproducibility validation.

    Returns:
        is_reproducible (bool): True if environment is fully deterministic.
        warnings (List[str]): List of non-fatal warnings affecting reproducibility.
    """
    warnings: list[str] = []

    # Collect environment snapshot
    env = collect_environment()

    # --------------------------------------
    # Python runtime checks
    # --------------------------------------
    py_version = env["python"]["version"]
    py_impl = env["python"]["implementation"]

    if py_impl != "CPython":
        warnings.append(f"Non-CPython runtime detected: {py_impl}")

    if py_version.startswith("3.13"):
        warnings.append("Python 3.13 may produce subtle FP16 nondeterminism")

    # --------------------------------------
    # Random state
    # --------------------------------------
    if "python_random_state_hash" not in env["random_state"]:
        warnings.append("Python random state not captured")

    # --------------------------------------
    # Numpy checks
    # --------------------------------------
    np_info = env.get("numpy", {})
    if not np_info.get("installed", False):
        warnings.append("NumPy not installed, numerical reproducibility may be affected")
    elif "blas_info" in np_info and "unavailable" in np_info["blas_info"]:
        warnings.append("NumPy BLAS info unavailable, numerical reproducibility may be affected")

    # --------------------------------------
    # Torch / CUDA checks
    # --------------------------------------
    torch_info = env.get("torch", {})
    if torch_info.get("installed", False):
        if torch_info.get("cuda_available"):
            warnings.append(
                f"Torch GPU detected: {torch_info.get('gpu_name')} "
                f"(CUDA {torch_info.get('cuda_version')}, cuDNN {torch_info.get('cudnn_version')}) "
                "—GPU computations may be non-deterministic"
            )

    # --------------------------------------
    # TensorFlow checks
    # --------------------------------------
    tf_info = env.get("tensorflow", {})
    if tf_info.get("installed", False):
        if tf_info.get("cuda_version"):
            warnings.append(
                f"TensorFlow GPU detected (CUDA {tf_info.get('cuda_version')}, "
                f"cuDNN {tf_info.get('cudnn_version')}) — GPU computations may be non-deterministic"
            )

    # --------------------------------------
    # ONNXRuntime
    # --------------------------------------
    ort_info = env.get("onnxruntime", {})
    if ort_info.get("installed", False):
        providers = ort_info.get("providers", [])
        if "CPUExecutionProvider" not in providers:
            warnings.append(f"ONNXRuntime default CPUExecutionProvider not available: {providers}")

    # --------------------------------------
    # Dependency state
    # --------------------------------------
    deps = env.get("dependencies", {})
    if not deps.get("pip_freeze"):
        warnings.append("pip freeze output is empty, cannot verify package versions")
    if not deps.get("poetry_lock_hash") and os.path.exists("poetry.lock"):
        warnings.append("poetry.lock exists but hash could not be calculated")
    if not deps.get("conda_env_hash"):
        warnings.append("conda environment hash not available, cannot verify conda dependencies")

    # --------------------------------------
    # Hardware / compilers
    # --------------------------------------
    cpu_info = env.get("hardware", {}).get("cpu", {})
    if cpu_info.get("cpu_model") is None:
        warnings.append("CPU model info unavailable")

    compilers = env.get("toolchain", {}).get("compilers", {})
    if not any(compilers.values()):
        warnings.append("No compiler info available (gcc/clang/c++), reproducibility may be affected")

    # --------------------------------------
    # macOS SDK
    # --------------------------------------
    macos_sdk = env.get("toolchain", {}).get("macos_sdk", {})
    if macos_sdk and not macos_sdk.get("xcode_version"):
        warnings.append("Xcode version unavailable, toolchain reproducibility may be affected")

    # --------------------------------------
    # Determine reproducibility
    # --------------------------------------
    is_reproducible = len(warnings) == 0

    return is_reproducible, warnings