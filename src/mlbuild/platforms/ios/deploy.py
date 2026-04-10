"""
mlbuild.platforms.ios.deploy

Model deployment to iOS simulator and real devices.

Responsibilities:
- Select correct app bundle (arm64-sim or user-provided signed arm64)
- Install benchmark runner app if not already present
- Push model into app sandbox (run-scoped path)
- Clean up run-scoped sandbox directory after use

Design rules:
- Every run gets an isolated sandbox path: Documents/mlbuild/<run_id>/
- Cleanup only wipes <run_id>/ — never the parent mlbuild/ dir
- Real device requires --signed-app — no signing attempted, hard stop if missing
- App install is skipped if bundle already present — idempotent
- All failures are typed IDBDeployError — no raw subprocess errors leak
- Stateless: no shared mutable state between runs

Key difference from Android:
- No free filesystem access. All file ops route through the app sandbox
  via idb file push/delete with --bundle-id.
- No binary push per run — the runner app is installed once and reused.
- No chmod — app bundle handles its own permissions.
"""

from __future__ import annotations

import subprocess
import json
import os
import re
import sys
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict
from uuid import uuid4

from mlbuild.platforms.ios import idb
from mlbuild.core.errors import IDBDeployError, UnsignedBinaryError


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DEBUG = os.getenv("MLBUILD_DEBUG") == "1"

BUNDLE_ID = "com.mlbuild.MLBuildRunner"
REMOTE_BASE_DIR = "Documents/mlbuild"
BINARIES_DIR = Path(__file__).parent / "binaries"
MANIFEST_PATH = BINARIES_DIR / "manifest.json"

SIM_APP_DIR = BINARIES_DIR / "arm64-sim"
SIM_APP_NAME = "MLBuildRunner.app"

DEFAULT_TIMEOUTS = {
    "install":     60,
    "file_push":   60,
    "file_delete": 15,
    "list_apps":   10,
}

_MANIFEST_CACHE: Optional[dict] = None


def _log(msg: str) -> None:
    if DEBUG:
        print(f"[mlbuild.ios.deploy] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------

@dataclass
class DeployedRun:
    run_id:          str
    udid:            str
    bundle_id:       str
    remote_dir:      str           # e.g. Documents/mlbuild/<run_id>
    remote_model_path: str         # full sandbox path to pushed model
    binary_version:  str
    is_simulator:    bool


# ---------------------------------------------------------------------
# Manifest + Integrity
# ---------------------------------------------------------------------

def _load_manifest() -> dict:
    global _MANIFEST_CACHE

    if _MANIFEST_CACHE is not None:
        return _MANIFEST_CACHE

    if not MANIFEST_PATH.exists():
        raise IDBDeployError(detail=f"Missing manifest: {MANIFEST_PATH}")

    try:
        manifest = json.loads(MANIFEST_PATH.read_text())
    except Exception as exc:
        raise IDBDeployError(detail=f"Invalid manifest JSON: {exc}")

    for key, entry in manifest.items():
        if not isinstance(entry, dict):
            raise IDBDeployError(detail=f"Malformed manifest entry for {key}")

    _MANIFEST_CACHE = manifest
    return manifest


def _verify_sha256(path: Path, expected: Optional[str]) -> None:
    if not expected:
        _log(f"No SHA256 for {path.name} — skipping integrity check")
        return

    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise IDBDeployError(detail=f"Integrity check failed for {path.name}")


# ---------------------------------------------------------------------
# App Bundle Selection
# ---------------------------------------------------------------------

def _select_app(
    is_simulator: bool,
    signed_app: Optional[Path],
) -> tuple[Path, str]:
    """
    Returns (app_path, binary_version).

    Simulator: use bundled arm64-sim app — no account needed.
    Real device: require caller-provided signed app via --signed-app.
                 Never attempt signing ourselves.
    """
    manifest = _load_manifest()

    if is_simulator:
        app_path = SIM_APP_DIR / SIM_APP_NAME
        if not app_path.exists():
            raise IDBDeployError(
                detail=f"Bundled simulator app not found: {app_path}"
            )
        entry = manifest.get("arm64-sim", {})
        _verify_sha256(app_path, entry.get("sha256"))
        return app_path, entry.get("version", "unknown")

    # Real device path
    if signed_app is None:
        raise UnsignedBinaryError()

    signed_app = Path(signed_app)
    if not signed_app.exists():
        raise IDBDeployError(detail=f"Signed app not found: {signed_app}")

    # Version from manifest if available, else unknown
    entry = manifest.get("arm64", {})
    return signed_app, entry.get("version", "unknown")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name)


def _run_id() -> str:
    return uuid4().hex[:12]


def _timed(label: str, fn, *args, **kwargs):
    start = time.time()
    result = fn(*args, **kwargs)
    _log(f"{label} took {time.time() - start:.2f}s")
    return result


def _idb_error(detail: str) -> IDBDeployError:
    d = detail.lower()

    if "no space" in d:
        return IDBDeployError(detail="Device storage full")

    if "permission denied" in d:
        return IDBDeployError(detail="Permission denied on device")

    if "not installed" in d:
        return IDBDeployError(detail=f"App bundle not installed: {BUNDLE_ID}")

    return IDBDeployError(detail=detail)


# ---------------------------------------------------------------------
# Core Steps
# ---------------------------------------------------------------------

def _ensure_app_installed(
    app_path: Path,
    udid: Optional[str],
    timeouts: dict,
) -> None:
    """
    Install the benchmark runner app if not already present.
    Idempotent — skips install if bundle ID already listed.
    """
    try:
        installed = idb.list_apps(udid=udid, bundle_id=BUNDLE_ID)
        if BUNDLE_ID in installed:
            _log(f"Runner app already installed: {BUNDLE_ID}")
            return
    except Exception as exc:
        _log(f"list_apps failed — proceeding with install: {exc}")

    _log(f"Installing runner app: {app_path}")
    _timed("install", idb.install, app_path, udid=udid)

def _compile_model(model_path: Path) -> Path:
    """
    Compile .mlmodel to .mlmodelc using xcrun coremlcompiler.
    Returns path to compiled .mlmodelc (in a temp dir).
    Already-compiled .mlmodelc passes through unchanged.
    """
    if model_path.suffix == ".mlmodelc":
        return model_path

    import tempfile
    out_dir = Path(tempfile.mkdtemp(prefix="mlbuild_compile_"))
    result = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(model_path), str(out_dir)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise IDBDeployError(
            detail=f"coremlcompiler failed: {result.stderr.strip()}"
        )
    compiled = out_dir / (model_path.stem + ".mlmodelc")
    if not compiled.exists():
        raise IDBDeployError(detail=f"Compiled model not found: {compiled}")
    return compiled

def _push_model(
    model_path: Path,
    remote_path: str,
    udid: Optional[str],
    timeouts: dict,
) -> None:
    if not model_path.exists():
        raise IDBDeployError(detail=f"Model not found: {model_path}")

    try:
        _timed(
            "file_push",
            idb.file_push,
            model_path,
            remote_path,
            bundle_id=BUNDLE_ID,
            udid=udid,
        )
    except IDBDeployError:
        raise
    except Exception as exc:
        raise _idb_error(str(exc))


def _safe_cleanup_remote(remote_dir: str, udid: Optional[str]) -> None:
    try:
        idb.file_delete(remote_dir, bundle_id=BUNDLE_ID, udid=udid)
        _log(f"Rollback cleanup succeeded: {remote_dir}")
    except Exception as exc:
        _log(f"Rollback cleanup FAILED: {exc}")


# ---------------------------------------------------------------------
# Public: Deploy (Transactional)
# ---------------------------------------------------------------------

def deploy(
    model_path: Path,
    *,
    is_simulator: bool,
    udid: Optional[str] = None,
    signed_app: Optional[Path] = None,
    timeouts: Optional[Dict[str, int]] = None,
) -> DeployedRun:
    """
    Full deploy sequence:
    1. Select app bundle (simulator bundled vs user-provided signed)
    2. Install app if not already present
    3. Push model into run-scoped sandbox directory

    On any failure: rollback remote dir, re-raise typed error.
    """
    timeouts = {**DEFAULT_TIMEOUTS, **(timeouts or {})}
    model_path = Path(model_path)

    app_path, binary_version = _select_app(is_simulator, signed_app)

    run_id = _run_id()
    remote_dir = f"{REMOTE_BASE_DIR}/{run_id}"
    safe_model_name = _safe_name(model_path.name)
    remote_model_path = f"{remote_dir}/{safe_model_name}"

    _log(f"[{udid}] Deploy start run_id={run_id} simulator={is_simulator}")

    compiled_path = _compile_model(model_path)
    compiled_name = _safe_name(compiled_path.name)
    remote_model_path = f"{remote_dir}/{compiled_name}"

    try:
        _ensure_app_installed(app_path, udid, timeouts)
        _push_model(compiled_path, remote_model_path, udid, timeouts)

    except Exception:
        _safe_cleanup_remote(remote_dir, udid)
        raise

    _log(f"Deploy complete: {remote_model_path}")

    return DeployedRun(
        run_id=run_id,
        udid=udid or "",
        bundle_id=BUNDLE_ID,
        remote_dir=remote_dir,
        remote_model_path=remote_model_path,
        binary_version=binary_version,
        is_simulator=is_simulator,
    )


# ---------------------------------------------------------------------
# Public: Cleanup
# ---------------------------------------------------------------------

def cleanup(deployed: DeployedRun) -> None:
    """
    Wipe run_id/ from sandbox only. Never touches parent mlbuild/ dir.
    Failure is logged, not raised — cleanup is best-effort.
    """
    try:
        idb.file_delete(
            deployed.remote_dir,
            bundle_id=deployed.bundle_id,
            udid=deployed.udid or None,
        )

        _log(f"Cleanup succeeded: {deployed.remote_dir}")

    except Exception as exc:
        _log(f"Cleanup failed (non-fatal): {exc}")