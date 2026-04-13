"""
mlbuild.platforms.ios.deploy

Model deployment to iOS simulator and real devices.

Responsibilities:
- Select correct app bundle (arm64-sim or user-provided signed arm64)
- Install benchmark runner app if not already present
- Push model into app sandbox (run-scoped path)
- Clean up run-scoped sandbox directory after use

Design rules:
- Every run gets an isolated sandbox path: Documents/mlb/<run_id>/
- Simulator: idb file push via simctl filesystem copy
- Real device: pymobiledevice3 HouseArrest into Documents/mlb/<run_id>/
- Real device requires --signed-app — no signing attempted, hard stop if missing
- App install is skipped if bundle already present — idempotent
- All failures are typed IDBDeployError — no raw subprocess errors leak
- Stateless: no shared mutable state between runs
"""

from __future__ import annotations

import asyncio
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

BUNDLE_ID        = "com.mlbuild.MLBuildRunner"
REAL_DEVICE_BASE = "Documents/mlb"   # HouseArrest-writable, Documents-relative
SIM_BASE         = "Documents/mlbuild"

BINARIES_DIR  = Path(__file__).parent / "binaries"
MANIFEST_PATH = BINARIES_DIR / "manifest.json"
SIM_APP_DIR   = BINARIES_DIR / "arm64-sim"
SIM_APP_NAME  = "MLBuildRunner.app"

DEFAULT_TIMEOUTS = {
    "install":     60,
    "file_push":   120,
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
    run_id:            str
    udid:              str
    bundle_id:         str
    remote_dir:        str   # Documents/mlb/<run_id> or Documents/mlbuild/<run_id>
    remote_model_path: str   # Documents-relative path passed to app
    binary_version:    str
    is_simulator:      bool


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
    manifest = _load_manifest()

    if is_simulator:
        app_path = SIM_APP_DIR / SIM_APP_NAME
        if not app_path.exists():
            raise IDBDeployError(detail=f"Bundled simulator app not found: {app_path}")
        entry = manifest.get("arm64-sim", {})
        _verify_sha256(app_path, entry.get("sha256"))
        return app_path, entry.get("version", "unknown")

    if signed_app is None:
        raise UnsignedBinaryError()
    signed_app = Path(signed_app)
    if not signed_app.exists():
        raise IDBDeployError(detail=f"Signed app not found: {signed_app}")
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


# ---------------------------------------------------------------------
# Core Steps
# ---------------------------------------------------------------------

def _ensure_app_installed(
    app_path: Path,
    udid: Optional[str],
    timeouts: dict,
    is_simulator: bool = False,
) -> None:
    try:
        if not is_simulator:
            r = subprocess.run(
                ["xcrun", "devicectl", "device", "info", "apps",
                 "--device", udid, "--bundle-id", BUNDLE_ID],
                capture_output=True, text=True, timeout=15,
            )
            installed = [BUNDLE_ID] if BUNDLE_ID in r.stdout else []
        else:
            installed = idb.list_apps(udid=udid, bundle_id=BUNDLE_ID)
        if BUNDLE_ID in installed:
            _log(f"Runner app already installed: {BUNDLE_ID} — reinstalling fresh")
            # Don't skip — always install fresh since we uninstall after each run
    except Exception as exc:
        _log(f"list_apps failed — proceeding with install: {exc}")

    _log(f"Installing runner app: {app_path}")
    _timed("install", idb.install, app_path, udid=udid, is_simulator=is_simulator)


def _compile_model(model_path: Path) -> Path:
    if model_path.suffix == ".mlmodelc":
        return model_path
    import tempfile
    out_dir = Path(tempfile.mkdtemp(prefix="mlbuild_compile_"))
    result = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(model_path), str(out_dir)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise IDBDeployError(detail=f"coremlcompiler failed: {result.stderr.strip()}")
    compiled = out_dir / (model_path.stem + ".mlmodelc")
    if not compiled.exists():
        raise IDBDeployError(detail=f"Compiled model not found: {compiled}")
    return compiled


def _push_model_real_device(
    model_path: Path,
    remote_dir: str,
    compiled_name: str,
) -> None:
    """
    Push .mlmodelc to real device via pymobiledevice3 HouseArrest.
    remote_dir is Documents-relative e.g. Documents/mlb/<run_id>
    """
    async def _push():
        from pymobiledevice3.lockdown import create_using_usbmux
        from pymobiledevice3.services.house_arrest import HouseArrestService

        lockdown = await create_using_usbmux()
        afc = await HouseArrestService.create(lockdown, bundle_id=BUNDLE_ID)

        model_root = f"{remote_dir}/{compiled_name}"

        # Create model root dir
        try:
            await afc.makedirs(model_root)
        except Exception:
            pass

        # Push all files recursively
        for f in sorted(model_path.rglob("*")):
            if f.is_file():
                rel = f.relative_to(model_path)
                remote = f"{model_root}/{rel}"
                parent = remote.rsplit("/", 1)[0]
                try:
                    await afc.makedirs(parent)
                except Exception:
                    pass
                await afc.set_file_contents(remote, f.read_bytes())
                _log(f"pushed: {remote}")

        await afc.close()

    try:
        asyncio.run(_push())
    except Exception as exc:
        raise IDBDeployError(detail=f"HouseArrest push failed: {exc}")


def _push_model_simulator(
    model_path: Path,
    remote_path: str,
    udid: str,
) -> None:
    """Push .mlmodelc to simulator via idb file push."""
    idb.file_push(
        model_path,
        remote_path,
        bundle_id=BUNDLE_ID,
        udid=udid,
        is_simulator=True,
    )


def _safe_cleanup_remote(
    remote_dir: str,
    udid: Optional[str],
    is_simulator: bool = False,
) -> None:
    if not is_simulator:
        _log(f"Cleanup skipped for real device (no delete API): {remote_dir}")
        return
    try:
        idb.file_delete(remote_dir, bundle_id=BUNDLE_ID, udid=udid, is_simulator=True)
        _log(f"Cleanup succeeded: {remote_dir}")
    except Exception as exc:
        _log(f"Cleanup failed (non-fatal): {exc}")


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
    timeouts   = {**DEFAULT_TIMEOUTS, **(timeouts or {})}
    model_path = Path(model_path)

    app_path, binary_version = _select_app(is_simulator, signed_app)

    run_id        = _run_id()
    compiled_path = _compile_model(model_path)
    compiled_name = _safe_name(compiled_path.name)

    if is_simulator:
        remote_dir        = f"{SIM_BASE}/{run_id}"
        remote_model_path = f"{remote_dir}/{compiled_name}"
        app_model_path    = remote_model_path  # passed as-is to simctl spawn
    else:
        remote_dir        = f"{REAL_DEVICE_BASE}/{run_id}"
        # Documents-relative path passed to app via --model=
        app_model_path    = f"mlb/{run_id}/{compiled_name}"

    _log(f"[{udid}] Deploy start run_id={run_id} simulator={is_simulator}")

    try:
        _ensure_app_installed(app_path, udid, timeouts, is_simulator=is_simulator)

        if is_simulator:
            _push_model_simulator(compiled_path, remote_model_path, udid)
        else:
            _push_model_real_device(compiled_path, remote_dir, compiled_name)

    except Exception:
        _safe_cleanup_remote(remote_dir, udid, is_simulator=is_simulator)
        raise

    _log(f"Deploy complete: {app_model_path}")

    return DeployedRun(
        run_id            = run_id,
        udid              = udid or "",
        bundle_id         = BUNDLE_ID,
        remote_dir        = remote_dir,
        remote_model_path = app_model_path,
        binary_version    = binary_version,
        is_simulator      = is_simulator,
    )


# ---------------------------------------------------------------------
# Public: Cleanup
# ---------------------------------------------------------------------

def cleanup(deployed: DeployedRun) -> None:
    _safe_cleanup_remote(
        deployed.remote_dir,
        deployed.udid or None,
        is_simulator=deployed.is_simulator,
    )
