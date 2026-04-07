"""
mlbuild.platforms.android.introspect

Device discovery + introspection.

Upgrades:
- Observability (debug logging + timing)
- Retry-aware device resolution (handles transient offline)
- Explicit failure handling (no silent swallowing)
- Auditable delegate detection
- Defensive normalization with warnings
"""

from __future__ import annotations

import sys
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

from mlbuild.platforms.android import adb
from mlbuild.core.errors import (
    ADBNoDeviceError,
    ADBUnauthorizedError,
    ADBOfflineError,
    ADBMultipleDevicesError,
)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

GETPROP_RETRIES = 3
GETPROP_RETRY_DELAY = 1.0

DEVICE_RETRIES = 3
DEVICE_RETRY_DELAY = 1.0

DEBUG = os.getenv("MLBUILD_DEBUG") == "1"


def _log(msg: str) -> None:
    if DEBUG:
        print(f"[mlbuild.introspect] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------
# DeviceProfile
# ---------------------------------------------------------------------

@dataclass
class DeviceProfile:
    serial: str
    model: str
    manufacturer: str
    api_level: int
    primary_abi: str
    abi_list: list[str]
    chipset: str
    heap_mb: int
    build_fingerprint: str
    delegate_candidates: list[str] = field(default_factory=list)

    @property
    def fingerprint(self) -> str:
        return f"{self.model}:{self.api_level}:{self.primary_abi}:{self.build_fingerprint}"

    @property
    def is_emulator(self) -> bool:
        """
        True if this device is an Android emulator rather than real hardware.
        Emulators produce non-representative benchmark results.
        """
        return (
            "emulator"   in self.serial.lower()  or
            "ranchu"     in self.chipset.lower() or
            "sdk_gphone" in self.model.lower()   or
            "generic"    in self.model.lower()   or
            "emulator"   in self.model.lower()
        )
    
    @property
    def is_android(self) -> bool:
        """True if this device is a physical or virtual Android device."""
        return not self.is_ios
    
    @property
    def is_ios(self) -> bool:
        """
        True if this device is an Apple iOS device.
        Currently always False — iOS support not yet implemented.
        Placeholder for when iOS ADB/libimobiledevice bridge is added.
        """
        return False


# ---------------------------------------------------------------------
# Device Resolution (retry-aware)
# ---------------------------------------------------------------------

def resolve_device(
    serial: Optional[str] = None,
    *,
    retry_on_offline: bool = True,
) -> Tuple[str, str]:
    """
    Returns (serial, state)

    Adds:
    - retry on transient offline
    - logging of all devices
    - trimmed serial matching
    """

    serial = serial.strip() if serial else None

    last_seen = None

    for attempt in range(DEVICE_RETRIES):
        devices = adb.devices()
        _log(f"Devices detected: {devices}")

        if not devices:
            raise ADBNoDeviceError()

        if serial:
            devices = [(s.strip(), st) for s, st in devices if s.strip() == serial]
            if not devices:
                raise ADBNoDeviceError()

        ready = [(s, st) for s, st in devices if st == "device"]
        unauthorized = [(s, st) for s, st in devices if st == "unauthorized"]
        offline = [(s, st) for s, st in devices if st == "offline"]

        last_seen = devices

        if unauthorized and not ready:
            raise ADBUnauthorizedError(serial=unauthorized[0][0])

        if ready:
            if len(ready) > 1 and not serial:
                raise ADBMultipleDevicesError(serials=[s for s, _ in ready])
            return ready[0]

        # No ready devices — retry if offline
        if offline and retry_on_offline and attempt < DEVICE_RETRIES - 1:
            _log("Device offline, retrying...")
            time.sleep(DEVICE_RETRY_DELAY)
            continue

        if offline:
            raise ADBOfflineError(serial=offline[0][0])

        raise ADBNoDeviceError()

    raise ADBOfflineError(serial=last_seen[0][0] if last_seen else None)


# ---------------------------------------------------------------------
# Property Fetching (observable + failure-aware)
# ---------------------------------------------------------------------

def _fetch_props(serial: str) -> Dict[str, Optional[str]]:
    keys = [
        "ro.product.model",
        "ro.product.manufacturer",
        "ro.build.version.sdk",
        "ro.product.cpu.abi",
        "ro.product.cpu.abilist",
        "ro.hardware",
        "ro.board.platform",
        "ro.build.fingerprint",
        "dalvik.vm.heapsize",
    ]

    last_exc: Optional[Exception] = None

    for attempt in range(GETPROP_RETRIES):
        start = time.time()
        props = {}

        try:
            for key in keys:
                props[key] = adb.getprop(key, serial=serial)

            duration = time.time() - start
            _log(f"getprop batch completed in {duration:.2f}s")

            if props.get("ro.product.model"):
                return props

            _log("Device not ready yet (missing model), retrying...")

        except ADBOfflineError as exc:
            _log("Device went offline during getprop")
            raise

        except Exception as exc:
            last_exc = exc
            _log(f"getprop error: {exc}")

        time.sleep(GETPROP_RETRY_DELAY)

    raise ADBOfflineError(serial=serial) from last_exc


# ---------------------------------------------------------------------
# Normalization (defensive + logged)
# ---------------------------------------------------------------------

def _parse_api_level(raw: Optional[str]) -> int:
    try:
        return int(raw or "0")
    except ValueError:
        _log(f"Invalid API level: {raw}")
        return 0


def _parse_abi_list(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    return [abi.strip() for abi in raw.split(",") if abi.strip()]


def _parse_heap_mb(raw: Optional[str]) -> int:
    if not raw:
        return 0

    raw = raw.strip().lower()

    try:
        if raw.endswith("k"):
            return int(float(raw[:-1]) / 1024)
        if raw.endswith("m"):
            return int(float(raw[:-1]))
        if raw.endswith("g"):
            return int(float(raw[:-1]) * 1024)

        return int(raw)
    except Exception:
        _log(f"Invalid heap size: {raw}")
        return 0


def _resolve_chipset(props: Dict[str, Optional[str]]) -> str:
    raw = props.get("ro.hardware") or props.get("ro.board.platform") or "unknown"
    return raw.lower()


# ---------------------------------------------------------------------
# Delegate Detection (auditable)
# ---------------------------------------------------------------------

def detect_delegate_candidates(api_level: int, chipset: str) -> list[str]:
    candidates = ["GPU"]

    if api_level >= 27:
        candidates.append("NNAPI")

    is_qualcomm = bool(
        re.search(r"(qualcomm|qcom|snapdragon|sm\d+)", chipset)
    )

    if is_qualcomm and api_level >= 27:
        candidates.append("HEXAGON")

    if is_qualcomm and api_level >= 31:
        candidates.append("HEXAGON_HTP")

    _log(f"Delegate candidates: {candidates} (chipset={chipset}, api={api_level})")

    return candidates


# ---------------------------------------------------------------------
# Public Entry Point (timed + resilient)
# ---------------------------------------------------------------------

def build_profile(serial: Optional[str] = None) -> DeviceProfile:
    start = time.time()

    confirmed_serial, state = resolve_device(serial)

    _log(f"Resolved device: {confirmed_serial} ({state})")

    props = _fetch_props(confirmed_serial)

    api_level = _parse_api_level(props.get("ro.build.version.sdk"))
    chipset = _resolve_chipset(props)

    primary_abi = props.get("ro.product.cpu.abi") or "unknown"
    abi_list = _parse_abi_list(props.get("ro.product.cpu.abilist"))

    if not abi_list and primary_abi != "unknown":
        abi_list = [primary_abi]

    profile = DeviceProfile(
        serial=confirmed_serial,
        model=props.get("ro.product.model") or "unknown",
        manufacturer=props.get("ro.product.manufacturer") or "unknown",
        api_level=api_level,
        primary_abi=primary_abi,
        abi_list=abi_list,
        chipset=chipset,
        heap_mb=_parse_heap_mb(props.get("dalvik.vm.heapsize")),
        build_fingerprint=props.get("ro.build.fingerprint") or "unknown",
        delegate_candidates=detect_delegate_candidates(api_level, chipset),
    )

    duration = time.time() - start
    _log(f"DeviceProfile built in {duration:.2f}s: {profile}")

    return profile