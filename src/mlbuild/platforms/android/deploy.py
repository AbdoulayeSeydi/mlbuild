"""
mlbuild.platforms.android.deploy

Model and binary deployment to Android devices.

Responsibilities:
- Create run-scoped remote directories
- Select the correct benchmark binary for the device ABI
- Push model + binary to device
- Set correct permissions
- Clean up run-scoped directories after use

Design rules:
- Every run gets an isolated directory: /data/local/tmp/mlbuild/<run_id>/
- Cleanup only wipes <run_id>/ — never the parent mlbuild/ dir
- ABI selection fails immediately and loudly — no silent fallbacks
- All failures are typed DeployError — no raw subprocess errors leak
- Stateless: no shared mutable state between runs
"""

from __future__ import annotations

import sys
import json
import os
import re
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict
from uuid import uuid4

from mlbuild.platforms.android import adb
from mlbuild.core.errors import DeployError, UnsupportedABIError


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DEBUG = os.getenv("MLBUILD_DEBUG") == "1"

REMOTE_BASE_DIR = "/data/local/tmp/mlbuild"
BINARY_NAME = "benchmark_model"
BINARIES_DIR = Path(__file__).parent / "binaries"
MANIFEST_PATH = BINARIES_DIR / "manifest.json"

DEFAULT_TIMEOUTS = {
    "mkdir": 10,
    "push": 60,
    "chmod": 10,
    "verify": 5,
}

_MANIFEST_CACHE: Optional[dict] = None


def _log(msg: str) -> None:
    if DEBUG:
        print(f"[mlbuild.deploy] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------

@dataclass
class DeployedRun:
    run_id: str
    serial: str
    remote_dir: str
    remote_model_path: str
    remote_binary_path: str
    binary_version: str


# ---------------------------------------------------------------------
# Manifest + Binary Integrity
# ---------------------------------------------------------------------

def _load_manifest() -> dict:
    global _MANIFEST_CACHE

    if _MANIFEST_CACHE is not None:
        return _MANIFEST_CACHE

    if not MANIFEST_PATH.exists():
        raise DeployError(f"Missing manifest: {MANIFEST_PATH}")

    try:
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
    except Exception as exc:
        raise DeployError(f"Invalid manifest JSON: {exc}")

    # Validate structure
    for abi, entry in manifest.items():
        if not isinstance(entry, dict):
            raise DeployError(f"Malformed manifest entry for {abi}")

    _MANIFEST_CACHE = manifest
    return manifest


def _verify_sha256(path: Path, expected: Optional[str]) -> None:
    if not expected:
        _log(f"No SHA256 for {path.name} — skipping integrity check")
        return

    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise DeployError(f"Binary integrity check failed for {path.name}")


def _select_binary(abi: str) -> tuple[Path, str]:
    manifest = _load_manifest()

    if abi not in manifest:
        raise UnsupportedABIError(abi=abi)

    binary_path = BINARIES_DIR / abi / BINARY_NAME
    if not binary_path.exists():
        raise UnsupportedABIError(abi=abi)

    entry = manifest[abi]
    version = entry.get("version", "unknown")
    sha256 = entry.get("sha256")

    _verify_sha256(binary_path, sha256)

    return binary_path, version


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name)


def _run_id() -> str:
    return uuid4().hex[:12]  # 48-bit


def _adb_error(detail: str) -> DeployError:
    d = detail.lower()

    if "no space left" in d:
        return DeployError("Device storage full")

    if "permission denied" in d:
        return DeployError("Permission denied on device")

    if "read-only file system" in d:
        return DeployError("Device filesystem is read-only")

    return DeployError(detail)


def _timed(label: str, fn, *args, **kwargs):
    start = time.time()
    result = fn(*args, **kwargs)
    _log(f"{label} took {time.time() - start:.2f}s")
    return result


# ---------------------------------------------------------------------
# Core Steps
# ---------------------------------------------------------------------

def _create_remote_dir(remote_dir: str, serial: Optional[str], timeouts):
    _timed(
        "mkdir",
        adb.mkdir,
        remote_dir,
        serial=serial,
    )

    # Verify existence
    result = adb.shell(
        f"test -d {remote_dir} && echo OK",
        serial=serial,
        timeout=timeouts["verify"],
    )

    if "OK" not in result.stdout:
        raise DeployError(f"Failed to verify remote dir: {remote_dir}")


def _push_binary(local: Path, remote: str, serial: Optional[str], timeouts):
    result = adb.run(
        ["push", str(local), remote],
        serial=serial,
        timeout=timeouts["push"],
        retries=1,
    )

    if not result.ok:
        raise _adb_error(result.stderr or result.stdout)

    adb.chmod(remote, "+x", serial=serial)


def _push_model(local: Path, remote: str, serial: Optional[str], timeouts):
    if not local.exists():
        raise DeployError(f"Model not found: {local}")

    result = adb.run(
        ["push", str(local), remote],
        serial=serial,
        timeout=timeouts["push"],
        retries=1,
    )

    if not result.ok:
        raise _adb_error(result.stderr or result.stdout)


def _safe_cleanup(remote_dir: str, serial: Optional[str]):
    try:
        adb.rm(remote_dir, serial=serial, recursive=True)
        _log(f"Rollback cleanup succeeded: {remote_dir}")
    except Exception as exc:
        _log(f"Rollback cleanup FAILED: {exc}")


# ---------------------------------------------------------------------
# Public: Deploy (Transactional)
# ---------------------------------------------------------------------

def deploy(
    model_path: Path,
    abi: str,
    serial: Optional[str] = None,
    timeouts: Optional[Dict[str, int]] = None,
) -> DeployedRun:

    timeouts = {**DEFAULT_TIMEOUTS, **(timeouts or {})}
    model_path = Path(model_path)

    binary, version = _select_binary(abi)

    run_id = _run_id()
    remote_dir = f"{REMOTE_BASE_DIR}/{run_id}"

    safe_model_name = _safe_name(model_path.name)
    remote_model = f"{remote_dir}/{safe_model_name}"
    remote_binary = f"{remote_dir}/{BINARY_NAME}"

    _log(f"[{serial}] Deploy start run_id={run_id}")

    try:
        _create_remote_dir(remote_dir, serial, timeouts)
        _push_binary(binary, remote_binary, serial, timeouts)
        _push_model(model_path, remote_model, serial, timeouts)

    except Exception:
        _safe_cleanup(remote_dir, serial)
        raise

    return DeployedRun(
        run_id=run_id,
        serial=serial or "",
        remote_dir=remote_dir,
        remote_model_path=remote_model,
        remote_binary_path=remote_binary,
        binary_version=version,
    )


# ---------------------------------------------------------------------
# Public: Cleanup
# ---------------------------------------------------------------------

def cleanup(deployed: DeployedRun) -> None:
    try:
        adb.rm(deployed.remote_dir, serial=deployed.serial or None, recursive=True)

        if DEBUG:
            result = adb.shell(
                f"test -d {deployed.remote_dir} && echo EXISTS",
                serial=deployed.serial or None,
            )
            if "EXISTS" in result.stdout:
                _log(f"Cleanup verification failed: {deployed.remote_dir}")
            else:
                _log(f"Cleanup verified: {deployed.remote_dir}")

    except Exception as exc:
        _log(f"Cleanup failed: {exc}")