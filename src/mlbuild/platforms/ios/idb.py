"""
mlbuild.platforms.ios.idb

idb transport layer.

Principles:
- Minimal but not naive: handles transport-level failures explicitly.
- Retry-aware: only retries transient failures.
- Observable: debug logging + timing.
- Safe: controlled companion restart usage.
- Stateless per call (no shared mutable runtime state).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from mlbuild.core.errors import (
    IDBCompanionNotRunningError,
    IDBDeployError,
    IDBNotFoundError,
    IDBOfflineError,
    IDBTimeoutError,
)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

TIMEOUTS = {
    "list_targets": 5,
    "describe": 10,
    "install": 60,
    "list_apps": 10,
    "launch": 30,
    "file_push": 60,
    "file_pull": 60,
    "file_delete": 15,
    "benchmark": 600,
}

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 1.0

DEBUG = os.getenv("MLBUILD_DEBUG") == "1"


# ---------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------

@dataclass
class IDBResult:
    stdout: str
    stderr: str
    exit_code: int

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


# ---------------------------------------------------------------------
# IDB Binary (cached)
# ---------------------------------------------------------------------

_IDB_PATH: Optional[str] = None


def _idb_binary() -> str:
    global _IDB_PATH
    if _IDB_PATH:
        return _IDB_PATH

    path = shutil.which("idb")
    if not path:
        raise IDBNotFoundError()

    _IDB_PATH = path
    return path


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _build_cmd(args: list[str], udid: Optional[str]) -> list[str]:
    cmd = [_idb_binary()]
    cmd += args
    if udid:
        cmd += ["--udid", udid]
    return cmd


def _log(msg: str) -> None:
    if DEBUG:
        print(f"[mlbuild.idb] {msg}", file=sys.stderr)


def _is_transient_error(stderr: str) -> bool:
    s = stderr.lower()
    return any(
        phrase in s
        for phrase in [
            "companion not running",
            "connection refused",
            "connection reset",
            "broken pipe",
            "timeout",
            "target not found",
        ]
    )


def _handle_transport_errors(stderr: str) -> None:
    s = stderr.lower()

    if "companion not running" in s or "connection refused" in s:
        raise IDBCompanionNotRunningError()

    if "target not found" in s or "no target" in s:
        raise IDBOfflineError()


def _run_subprocess(
    cmd: list[str],
    timeout: int,
    retries: int,
    *,
    allow_restart_companion: bool = False,
) -> IDBResult:
    last_exc: Optional[Exception] = None

    for attempt in range(retries + 1):
        start = time.time()
        proc = None

        try:
            _log(f"RUN: {' '.join(cmd)} (attempt {attempt})")

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            stdout, stderr = proc.communicate(timeout=timeout)
            duration = time.time() - start

            _log(f"DONE ({duration:.2f}s) exit={proc.returncode}")

            if proc.returncode != 0:
                _handle_transport_errors(stderr)

                if _is_transient_error(stderr) and attempt < retries:
                    _log("Transient error detected, retrying...")
                    time.sleep(RETRY_DELAY)
                    continue

            return IDBResult(stdout.strip(), stderr.strip(), proc.returncode)

        except subprocess.TimeoutExpired as exc:
            last_exc = exc

            if proc:
                proc.kill()
                proc.communicate()

            _log("TIMEOUT")

            if attempt < retries:
                time.sleep(RETRY_DELAY)
                continue

            if allow_restart_companion:
                _log("Restarting idb_companion (last resort)")
                _restart_companion()

            raise IDBTimeoutError(command=" ".join(cmd)) from exc

        except FileNotFoundError:
            raise IDBNotFoundError()

    raise IDBTimeoutError(command=" ".join(cmd)) from last_exc


def _restart_companion() -> None:
    """
    Last-resort companion restart. Mirrors adb kill-server semantics.
    Kills idb_companion process and restarts it in daemon mode.
    """
    try:
        subprocess.run(
            ["pkill", "-f", "idb_companion"],
            timeout=5,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1.0)
        subprocess.Popen(
            ["idb_companion", "--daemon"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1.5)  # give companion time to come up before next attempt
    except Exception:
        pass


# ---------------------------------------------------------------------
# Companion Check
# ---------------------------------------------------------------------

def ensure_companion(udid: Optional[str] = None) -> None:
    """
    Start companion only if not already running.
    Never kill a healthy companion.
    """
    # Check if companion is already up
    try:
        targets = list_targets()
        if targets:
            _log(f"Companion already running, {len(targets)} target(s) visible")
            return
    except Exception:
        pass

    _log(f"Companion not running — starting for udid={udid}")

    subprocess.Popen(
        [IDB_COMPANION_BIN,
        "--udid", udid or "only",
        "--grpc-domain-sock", f"/tmp/idb/{udid}_companion.sock"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2.5)

    targets = list_targets()
    if not targets:
        raise IDBCompanionNotRunningError()

    _log(f"Companion started, {len(targets)} target(s) visible")

    _log("Companion not running — attempting daemon launch")
    _restart_companion()

    # One retry after restart
    try:
        result = _run_subprocess(
            [_idb_binary(), "list-targets"],
            timeout=TIMEOUTS["list_targets"],
            retries=0,
        )
        if result.ok:
            return
    except Exception:
        pass

    raise IDBCompanionNotRunningError()


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def run(
    args: list[str],
    *,
    udid: Optional[str] = None,
    timeout: Optional[int] = None,
    retries: int = 0,
    allow_restart_companion: bool = False,
) -> IDBResult:
    cmd = _build_cmd(args, udid)
    return _run_subprocess(
        cmd,
        timeout=timeout or DEFAULT_TIMEOUT,
        retries=retries,
        allow_restart_companion=allow_restart_companion,
    )


# ---------------------------------------------------------------------
# Device Discovery
# ---------------------------------------------------------------------

IDB_COMPANION_BIN = "/Users/abdoulayeseydi/idb/build/Build/Products/Release/idb_companion"


def list_targets() -> list[tuple[str, str, str]]:
    """
    Calls idb_companion --list 1 directly — no gRPC, no port conflict.
    Returns list of (udid, name, state).
    """
    result = subprocess.run(
        [IDB_COMPANION_BIN, "--list", "1"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    targets = []
    for line in result.stdout.splitlines():
        try:
            import json as _json
            entry = _json.loads(line)
            udid  = entry.get("udid", "")
            name  = entry.get("name", "unknown")
            state = entry.get("state", "unknown")
            if udid:
                targets.append((udid, name, state))
        except Exception:
            # fallback: pipe-delimited
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                targets.append((parts[1], parts[0], parts[2]))

    return targets


def describe(*, udid: Optional[str] = None) -> dict:
    """
    Get device info.
    Real device: xcrun devicectl device info details
    Simulator: idb_companion --list 1
    """
    import json as _json
    import tempfile

    # Try devicectl for real device first
    if udid:
        try:
            tmp = tempfile.mktemp(suffix=".json")
            result = subprocess.run(
                ["xcrun", "devicectl", "device", "info", "details",
                 "--device", udid, "--json-output", tmp],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                data = _json.loads(open(tmp).read())
                # devicectl info details structure
                dev = data.get("result", {}).get("deviceProperties", {})
                hw  = data.get("result", {}).get("hardwareProperties", {})
                if dev or hw:
                    return {
                        "udid":         udid,
                        "name":         hw.get("marketingName", dev.get("name", "unknown")),
                        "os_version":   dev.get("osVersionNumber", "unknown"),
                        "model":        hw.get("productType", hw.get("deviceType", "unknown")),
                        "state":        "connected",
                        "is_simulator": False,
                    }
        except Exception as e:
            _log(f"devicectl describe failed: {e}")

    # Fallback: idb_companion --list 1 for simulator
    result = subprocess.run(
        [IDB_COMPANION_BIN, "--list", "1"],
        capture_output=True, text=True, timeout=10,
    )
    for line in result.stdout.splitlines():
        try:
            entry = _json.loads(line)
            if udid and entry.get("udid") != udid:
                continue
            return {
                "udid":         entry.get("udid", ""),
                "name":         entry.get("name", "unknown"),
                "os_version":   entry.get("os_version", "unknown"),
                "model":        entry.get("model", "unknown"),
                "state":        entry.get("state", "unknown"),
                "is_simulator": entry.get("type", "").lower() == "simulator",
            }
        except Exception:
            continue
    return {}


# ---------------------------------------------------------------------
# App Management
# ---------------------------------------------------------------------

def install(app_path: Path, *, udid: Optional[str] = None, is_simulator: bool = False) -> None:
    if not app_path.exists():
        raise IDBDeployError(detail=f"App bundle not found: {app_path}")

    if udid and is_simulator:
        # Simulator: simctl install — bypasses gRPC entirely
        result = subprocess.run(
            ["xcrun", "simctl", "install", udid, str(app_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise IDBDeployError(detail=result.stderr or result.stdout)
        return

    # Real device: use devicectl (Xcode 15+) — reliable, no gRPC needed
    if udid:
        result = subprocess.run(
            ["xcrun", "devicectl", "device", "install", "app",
             "--device", udid, str(app_path)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise IDBDeployError(detail=result.stderr or result.stdout)
        return

    result = run(
        ["install", str(app_path)],
        timeout=TIMEOUTS["install"],
        retries=1,
    )
    if not result.ok:
        raise IDBDeployError(detail=result.stderr or result.stdout)


def list_apps(*, udid: Optional[str] = None, bundle_id: Optional[str] = None) -> list[str]:
    if udid and bundle_id:
        result = subprocess.run(
            ["xcrun", "simctl", "get_app_container", udid, bundle_id, "data"],
            capture_output=True, text=True, timeout=10,
        )
        return [bundle_id] if result.returncode == 0 else []

    # Fallback to idb
    result = run(["list-apps"], udid=udid, timeout=TIMEOUTS["list_apps"])
    bundle_ids = []
    for line in result.stdout.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if parts:
            bundle_ids.append(parts[0])
    return bundle_ids


def launch(
    bundle_id: str,
    args: Optional[list[str]] = None,
    *,
    udid: Optional[str] = None,
) -> IDBResult:
    if udid:
        # Simulator: use simctl launch --console to capture stdout
        cmd = ["xcrun", "simctl", "launch", "--console", udid, bundle_id]
        if args:
            cmd += args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUTS["launch"],
        )
        return IDBResult(result.stdout.strip(), result.stderr.strip(), result.returncode)

    # Real device: idb gRPC path
    cmd = ["launch", bundle_id]
    if args:
        cmd += args
    return run(cmd, udid=udid, timeout=TIMEOUTS["launch"], retries=0)


# ---------------------------------------------------------------------
# File Transfer (Sandbox-Aware)
#
# Unlike ADB, iOS sandboxing means all file ops are routed through
# the app's Documents directory via --bundle-id. No free filesystem access.
# ---------------------------------------------------------------------

def _sim_container(udid: str, bundle_id: str) -> Path:
    """Get the data container path for an installed simulator app."""
    result = subprocess.run(
        ["xcrun", "simctl", "get_app_container", udid, bundle_id, "data"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise IDBDeployError(
            detail=f"App not installed on simulator {udid}: {bundle_id}. "
                   f"Run failed: {result.stderr.strip()}"
        )
    return Path(result.stdout.strip())

def _device_container(udid: str, bundle_id: str) -> str:
    """Get the data container path for a real device app via devicectl."""
    import tempfile, json as _json
    tmp = tempfile.mktemp(suffix=".json")
    result = subprocess.run(
        ["xcrun", "devicectl", "device", "copy", "to",
         "--device", udid,
         "--source", "/dev/null",
         "--destination", "mlbuild_probe",
         "--domain-type", "appDataContainer",
         "--domain-identifier", bundle_id,
         "--json-output", tmp],
        capture_output=True, text=True, timeout=30,
    )
    try:
        data = _json.loads(open(tmp).read())
        path = data["result"]["files"][0]["path"]
        # path is like /private/var/mobile/Containers/Data/Application/<UUID>/mlbuild_probe
        return str(Path(path).parent)
    except Exception:
        # Fallback: parse from stderr
        for line in result.stdout.splitlines():
            if "/Containers/Data/Application/" in line:
                import re
                m = re.search(r"(/private/var/mobile/Containers/Data/Application/[A-F0-9\-]+)", line)
                if m:
                    return m.group(1)
        raise IDBDeployError(detail="Could not determine device container path")


def file_push(
    src: Path,
    remote_path: str,
    *,
    bundle_id: str,
    udid: Optional[str] = None,
    is_simulator: bool = False,
) -> None:
    if not src.exists():
        raise IDBDeployError(detail=f"Missing file: {src}")

    if udid and is_simulator:
        container = _sim_container(udid, bundle_id)
        dest = container / remote_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil as _shutil
        if src.is_dir():
            if dest.exists():
                _shutil.rmtree(str(dest))
            _shutil.copytree(str(src), str(dest))
        else:
            _shutil.copy2(str(src), str(dest))
        return

    # Real device: use devicectl appDataContainer
    if udid:
        # remote_path is like "Documents/mlbuild/<run_id>/model.mlmodelc"
        # devicectl destination is relative to app data container
        dest_dir = str(Path(remote_path).parent)
        result = subprocess.run(
            ["xcrun", "devicectl", "device", "copy", "to",
             "--device", udid,
             "--source", str(src),
             "--destination", dest_dir,
             "--domain-type", "appDataContainer",
             "--domain-identifier", bundle_id],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise IDBDeployError(detail=result.stderr or result.stdout)
        return

    result = run(
        ["file", "push", "--bundle-id", bundle_id, str(src), remote_path],
        timeout=TIMEOUTS["file_push"],
        retries=2,
    )
    if not result.ok:
        raise IDBDeployError(detail=result.stderr or result.stdout)


def file_pull(
    remote_path: str,
    dest: Path,
    *,
    bundle_id: str,
    udid: Optional[str] = None,
    is_simulator: bool = False,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if udid and is_simulator:
        container = _sim_container(udid, bundle_id)
        src = container / remote_path
        if not src.exists():
            raise IDBDeployError(detail=f"Remote file not found: {remote_path}")
        import shutil as _shutil
        _shutil.copy2(str(src), str(dest))
        return

    result = run(
        ["file", "pull", "--bundle-id", bundle_id, remote_path, str(dest)],
        udid=udid,
        timeout=TIMEOUTS["file_pull"],
        retries=2,
    )
    if not result.ok:
        raise IDBDeployError(detail=result.stderr or result.stdout)


def file_delete(
    remote_path: str,
    *,
    bundle_id: str,
    udid: Optional[str] = None,
    is_simulator: bool = False,
) -> None:
    if udid and is_simulator:
        container = _sim_container(udid, bundle_id)
        target = container / remote_path
        if target.exists():
            import shutil as _shutil
            _shutil.rmtree(str(target)) if target.is_dir() else target.unlink()
        return

    # Real device: devicectl has no delete — skip, files are scoped per run
    if udid:
        _log(f"file_delete skipped on real device (no devicectl delete): {remote_path}")
        return

    result = run(
        ["file", "delete", "--bundle-id", bundle_id, remote_path],
        timeout=TIMEOUTS["file_delete"],
    )
    if not result.ok:
        _log(f"WARNING: file_delete failed for {remote_path}: {result.stderr}")