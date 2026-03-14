# src/mlbuild/core/device.py

from __future__ import annotations

import logging
import platform
import subprocess
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)


DEVICE_LABELS = {
    "apple_m1": "Apple M1",
    "apple_m2": "Apple M2",
    "apple_m3": "Apple M3",
    "apple_arm": "Apple Silicon",
    "apple_intel": "Apple Intel",
    "cpu": "CPU",
}


@dataclass(frozen=True)
class DeviceInfo:
    """
    Structured device profile used across MLBuild.

    Attributes
    ----------
    target : str
        MLBuild target identifier (e.g. apple_m2).
    arch : str
        CPU architecture (arm64, x86_64, etc).
    brand : str
        Raw CPU brand string when available.
    """

    target: str
    arch: str
    brand: str

    def label(self) -> str:
        """Human-readable label for CLI output."""
        label = DEVICE_LABELS.get(self.target, self.target)
        return f"{label} (detected)"


def _run_sysctl(keys: list[str]) -> dict[str, str]:
    """
    Run sysctl once and return parsed key/value output.

    This avoids spawning multiple subprocesses.
    """

    try:
        proc = subprocess.run(
            ["sysctl", "-n"] + keys,
            capture_output=True,
            text=True,
            timeout=1,
            check=True,
        )

        lines = proc.stdout.strip().splitlines()

        return dict(zip(keys, lines))

    except Exception as e:
        logger.debug("sysctl probe failed: %s", e)
        return {}


def _detect_apple_target(brand: str) -> str:
    """
    Determine Apple chip generation from brand string.
    """

    brand = brand.lower()

    if brand.startswith("apple m3"):
        return "apple_m3"

    if brand.startswith("apple m2"):
        return "apple_m2"

    if brand.startswith("apple m1"):
        return "apple_m1"

    # Future Apple chips (M4, M5, etc.)
    if brand.startswith("apple"):
        return "apple_arm"

    return "apple_arm"


@lru_cache(maxsize=1)
def detect_device() -> DeviceInfo:
    """
    Detect the current machine.

    Returns
    -------
    DeviceInfo
        Structured device profile.
    """

    system = platform.system()
    arch = platform.machine()

    # Non-macOS systems
    if system != "Darwin":
        return DeviceInfo(
            target="cpu",
            arch=arch,
            brand="",
        )

    # Single sysctl probe
    data = _run_sysctl([
        "hw.optional.arm64",
        "machdep.cpu.brand_string",
    ])

    arm64 = data.get("hw.optional.arm64", "0")
    brand = data.get("machdep.cpu.brand_string", "")

    # Intel mac
    if arm64 != "1":
        return DeviceInfo(
            target="apple_intel",
            arch=arch,
            brand=brand,
        )

    # Apple Silicon
    target = _detect_apple_target(brand)

    return DeviceInfo(
        target=target,
        arch=arch,
        brand=brand,
    )


def device_label(target: str) -> str:
    """
    Convert a target string into a CLI label.

    This does NOT perform device detection.
    """

    label = DEVICE_LABELS.get(target, target)
    return f"{label} (detected)"