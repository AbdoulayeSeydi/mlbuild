"""
mlbuild.platforms.android.adb

ADB transport layer.

Principles:
- Minimal but not naive: handles transport-level failures explicitly.
- Retry-aware: only retries transient failures.
- Observable: debug logging + timing.
- Safe: shell quoting, controlled kill-server usage.
- Stateless per call (no shared mutable runtime state).
"""

from __future__ import annotations

import sys
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

from mlbuild.core.errors import (
    ADBNotFoundError,
    ADBOfflineError,
    ADBTimeoutError,
    DeployError,
)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

TIMEOUTS = {
    "devices": 5,
    "getprop": 10,
    "shell": 30,
    "push": 60,
    "pull": 60,
    "chmod": 10,
    "mkdir": 10,
    "rm": 15,
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
class ADBResult:
    stdout: str
    stderr: str
    exit_code: int

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


# ---------------------------------------------------------------------
# ADB Binary (cached)
# ---------------------------------------------------------------------

_ADB_PATH: Optional[str] = None


def _adb_binary() -> str:
    global _ADB_PATH
    if _ADB_PATH:
        return _ADB_PATH

    path = shutil.which("adb")
    if not path:
        raise ADBNotFoundError()

    _ADB_PATH = path
    return path


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _build_cmd(args: list[str], serial: Optional[str]) -> list[str]:
    cmd = [_adb_binary()]
    if serial:
        cmd += ["-s", serial]
    cmd += args
    return cmd


def _log(msg: str) -> None:
    if DEBUG:
        print(f"[mlbuild.adb] {msg}", file=sys.stderr)


def _is_transient_error(stderr: str) -> bool:
    s = stderr.lower()
    return any(
        phrase in s
        for phrase in [
            "device offline",
            "device not found",
            "closed",
            "protocol fault",
            "connection reset",
        ]
    )


def _handle_transport_errors(stderr: str) -> None:
    s = stderr.lower()

    if "device offline" in s or "device not found" in s:
        raise ADBOfflineError()


def _run_subprocess(
    cmd: list[str],
    timeout: int,
    retries: int,
    *,
    allow_kill_server: bool = False,
) -> ADBResult:
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

            # Transport-level failure detection
            if proc.returncode != 0:
                _handle_transport_errors(stderr)

                if _is_transient_error(stderr) and attempt < retries:
                    _log("Transient error detected, retrying...")
                    time.sleep(RETRY_DELAY)
                    continue

            return ADBResult(stdout.strip(), stderr.strip(), proc.returncode)

        except subprocess.TimeoutExpired as exc:
            last_exc = exc

            if proc:
                proc.kill()
                proc.communicate()

            _log("TIMEOUT")

            if attempt < retries:
                time.sleep(RETRY_DELAY)
                continue

            if allow_kill_server:
                _log("Killing adb server (last resort)")
                _kill_server()

            raise ADBTimeoutError(command=" ".join(cmd)) from exc

        except FileNotFoundError:
            raise ADBNotFoundError()

    raise ADBTimeoutError(command=" ".join(cmd)) from last_exc


def _kill_server() -> None:
    try:
        subprocess.run(
            [_adb_binary(), "kill-server"],
            timeout=5,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def run(
    args: list[str],
    *,
    serial: Optional[str] = None,
    timeout: Optional[int] = None,
    retries: int = 0,
    allow_kill_server: bool = False,
) -> ADBResult:
    cmd = _build_cmd(args, serial)
    return _run_subprocess(
        cmd,
        timeout=timeout or DEFAULT_TIMEOUT,
        retries=retries,
        allow_kill_server=allow_kill_server,
    )


def shell(
    cmd: str,
    *,
    serial: Optional[str] = None,
    timeout: Optional[int] = None,
    retries: int = 0,
) -> ADBResult:
    return run(
        ["shell", cmd],
        serial=serial,
        timeout=timeout or TIMEOUTS["shell"],
        retries=retries,
    )


# ---------------------------------------------------------------------
# Safe Shell Helpers
# ---------------------------------------------------------------------

def _q(s: str) -> str:
    return shlex.quote(s)


def mkdir(path: str, *, serial: Optional[str] = None) -> None:
    result = shell(f"mkdir -p {_q(path)}", serial=serial)
    if not result.ok:
        raise DeployError(detail=result.stderr)


def chmod(path: str, mode: str = "+x", *, serial: Optional[str] = None) -> None:
    result = shell(f"chmod {mode} {_q(path)}", serial=serial)
    if not result.ok:
        raise DeployError(detail=result.stderr)


def rm(
    path: str,
    *,
    serial: Optional[str] = None,
    recursive: bool = True,
) -> None:
    flag = "-rf" if recursive else "-f"
    result = shell(f"rm {flag} {_q(path)}", serial=serial)

    if not result.ok:
        _log(f"WARNING: rm failed for {path}: {result.stderr}")


# ---------------------------------------------------------------------
# File Transfer
# ---------------------------------------------------------------------

def push(
    src: Path,
    dest: str,
    *,
    serial: Optional[str] = None,
) -> None:
    if not src.exists():
        raise DeployError(detail=f"Missing file: {src}")

    result = run(
        ["push", str(src), dest],
        serial=serial,
        timeout=TIMEOUTS["push"],
        retries=2,
        allow_kill_server=True,
    )

    if not result.ok:
        raise DeployError(detail=result.stderr or result.stdout)


def pull(
    src: str,
    dest: Path,
    *,
    serial: Optional[str] = None,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    result = run(
        ["pull", src, str(dest)],
        serial=serial,
        timeout=TIMEOUTS["pull"],
        retries=2,
    )

    if not result.ok:
        raise DeployError(detail=result.stderr or result.stdout)


# ---------------------------------------------------------------------
# Devices (normalized, not raw garbage)
# ---------------------------------------------------------------------

def devices() -> list[tuple[str, str]]:
    """
    Returns:
        List of (serial, state)
    """
    result = run(["devices"], timeout=TIMEOUTS["devices"])

    lines = result.stdout.splitlines()
    if lines and "devices attached" in lines[0]:
        lines = lines[1:]

    out = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            out.append((parts[0], parts[1]))

    return out


# ---------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------

def getprop(
    key: str,
    *,
    serial: Optional[str] = None,
) -> Optional[str]:
    result = shell(
        f"getprop {_q(key)}",
        serial=serial,
        timeout=TIMEOUTS["getprop"],
        retries=2,
    )

    if not result.ok:
        raise ADBOfflineError()

    val = result.stdout.strip()
    return val if val else None