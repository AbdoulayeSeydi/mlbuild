# src/mlbuild/core/machine.py

from __future__ import annotations

import os
import platform
import tempfile
import uuid
from pathlib import Path
from typing import Any

# Future-proof directory layout
MLBUILD_DIR = Path.home() / ".mlbuild"
STATE_DIR = MLBUILD_DIR / "state"

MACHINE_ID_PATH = STATE_DIR / "machine_id"

# In-process cache
_MACHINE_ID: str | None = None


# ---------------------------------------------------------
# Hardware Machine ID Detection (best effort)
# ---------------------------------------------------------

def _get_hardware_machine_id() -> str | None:
    """
    Attempts to retrieve a stable OS-level machine identifier.
    Used only on first install if available.
    """
    try:
        system = platform.system()

        if system == "Linux":
            path = Path("/etc/machine-id")
            if path.exists():
                return path.read_text().strip()

        if system == "Darwin":
            import subprocess

            result = subprocess.run(
                ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                capture_output=True,
                text=True,
                check=False,
            )
            for line in result.stdout.splitlines():
                if "IOPlatformUUID" in line:
                    return line.split('"')[-2]

        if system == "Windows":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Cryptography",
            )
            value, _ = winreg.QueryValueEx(key, "MachineGuid")
            return value

    except Exception:
        pass

    return None


# ---------------------------------------------------------
# Atomic File Write
# ---------------------------------------------------------

def _atomic_write(path: Path, data: str) -> None:
    """
    Atomically writes data to a file to prevent corruption.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        dir=str(path.parent),
    ) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        temp_path = Path(tmp.name)

    temp_path.replace(path)


# ---------------------------------------------------------
# Machine ID
# ---------------------------------------------------------

def get_machine_id() -> str:
    """
    Returns a stable identifier for this MLBuild installation.

    Priority:
    1. Existing stored ID
    2. OS hardware machine ID
    3. Generated UUID

    Once generated it never changes.
    """

    global _MACHINE_ID

    if _MACHINE_ID:
        return _MACHINE_ID

    # Attempt to read existing ID
    try:
        if MACHINE_ID_PATH.exists():
            raw = MACHINE_ID_PATH.read_text().strip()
            _MACHINE_ID = str(uuid.UUID(raw))
            return _MACHINE_ID
    except Exception:
        pass

    # Try hardware ID
    hardware_id = _get_hardware_machine_id()

    if hardware_id:
        machine_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, hardware_id))
    else:
        machine_id = str(uuid.uuid4())

    # Persist if possible
    try:
        _atomic_write(MACHINE_ID_PATH, machine_id)
    except OSError:
        # Non-fatal (e.g. CI environment)
        pass

    _MACHINE_ID = machine_id
    return machine_id


# ---------------------------------------------------------
# Machine Metadata
# ---------------------------------------------------------

def get_machine_name() -> str:
    """
    Returns hostname.
    Treated as mutable display metadata.
    """
    return platform.node()


def get_machine_info() -> dict[str, Any]:
    """
    Returns machine identity and metadata.

    Called once per CLI invocation by MLBuild instrumentation.
    """

    return {
        "machine_id": get_machine_id(),
        "machine_name": get_machine_name(),
        "platform": platform.system().lower(),
        "arch": platform.machine().lower(),
        "python_version": platform.python_version(),
    }