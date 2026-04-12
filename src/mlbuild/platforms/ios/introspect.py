"""
mlbuild.platforms.ios.introspect

Device discovery + introspection.

- Pipe-delimited idb list-targets parsing (switch to --json if it breaks)
- Chip resolved from chip_map.json — no inline hardcoding
- ANE availability is structural: always False on simulator, always True on real device
- Retry-aware target resolution
- Explicit failure handling, no silent swallowing
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from mlbuild.platforms.ios import idb
from mlbuild.core.errors import (
    IDBNoDeviceError,
    IDBUnauthorizedError,
    IDBOfflineError,
    IDBMultipleDevicesError,
    SimulatorBootError,
)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

TARGET_RETRIES = 3
TARGET_RETRY_DELAY = 1.0

DESCRIBE_RETRIES = 3
DESCRIBE_RETRY_DELAY = 1.0

DEBUG = os.getenv("MLBUILD_DEBUG") == "1"


def _log(msg: str) -> None:
    if DEBUG:
        print(f"[mlbuild.ios.introspect] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------
# Chip Map (loaded once at module init)
# ---------------------------------------------------------------------

_CHIP_MAP_PATH = Path(__file__).parent / "chip_map.json"

try:
    CHIP_MAP: dict[str, str] = json.loads(_CHIP_MAP_PATH.read_text())
except FileNotFoundError:
    _log(f"WARNING: chip_map.json not found at {_CHIP_MAP_PATH} — chip names will be 'unknown'")
    CHIP_MAP = {}
except json.JSONDecodeError as e:
    _log(f"WARNING: chip_map.json is malformed ({e}) — chip names will be 'unknown'")
    CHIP_MAP = {}


def resolve_chip(model_identifier: str) -> str:
    chip = CHIP_MAP.get(model_identifier)
    if not chip:
        _log(f"Unknown model identifier: {model_identifier} — chip will be 'unknown'")
    return chip or "unknown"


# ---------------------------------------------------------------------
# DeviceProfile
# ---------------------------------------------------------------------

@dataclass
class DeviceProfile:
    udid: str
    name: str                       # e.g. "iPhone 15 Pro" or "iPhone 15 Pro Simulator"
    model: str                      # e.g. "iPhone16,1"
    ios_version: str                # e.g. "17.4"
    chip: str                       # e.g. "A17 Pro" — resolved from chip_map.json
    is_simulator: bool
    has_ane: bool                   # always False on simulator, always True on real device
    delegate_candidates: list[str] = field(default_factory=list)

    @property
    def fingerprint(self) -> str:
        return f"{self.model}:{self.ios_version}:{self.chip}:{self.udid[:8]}"


# ---------------------------------------------------------------------
# Target Parsing
# ---------------------------------------------------------------------

def _parse_targets(stdout: str) -> list[dict]:
    """
    Parse idb list-targets pipe-delimited output.

    Expected format per line:
        <name> | <udid> | <state> | <type> | <os_version> | <model>

    Fields beyond the first three are optional — don't crash if absent.
    """
    targets = []

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split("|")]

        if len(parts) < 3:
            _log(f"Skipping malformed target line: {repr(line)}")
            continue

        target = {
            "name":        parts[0] if len(parts) > 0 else "unknown",
            "udid":        parts[1] if len(parts) > 1 else "",
            "state":       parts[2] if len(parts) > 2 else "unknown",
            "type":        parts[3] if len(parts) > 3 else "unknown",
            "os_version":  parts[4] if len(parts) > 4 else "unknown",
            "model":       parts[5] if len(parts) > 5 else "unknown",
        }

        if not target["udid"]:
            _log(f"Skipping target with no udid: {repr(line)}")
            continue

        targets.append(target)

    return targets


def _is_simulator(target: dict) -> bool:
    return (
        target.get("is_simulator", False)
        or target.get("type", "").lower() == "simulator"
        or "simulator" in target.get("name", "").lower()
    )


# ---------------------------------------------------------------------
# Target Resolution (retry-aware)
# ---------------------------------------------------------------------

def resolve_target(
    udid: Optional[str] = None,
    *,
    retry_on_offline: bool = True,
) -> dict:
    """
    Returns a single resolved target dict.

    State handling:
        Booted (simulator)    → proceed
        Shutdown (simulator)  → attempt boot, retry
        connected (device)    → proceed
        disconnected (device) → retry up to TARGET_RETRIES → IDBOfflineError
        none found            → IDBNoDeviceError
        multiple, no udid     → IDBMultipleDevicesError
    """
    udid = udid.strip() if udid else None
    last_seen: Optional[list] = None

    for attempt in range(TARGET_RETRIES):
        result = idb.list_targets()
        raw_stdout = "\n".join(f"{n} | {u} | {s}" for u, n, s in result)
        targets = _parse_targets(raw_stdout)

        _log(f"Targets detected: {[(t['udid'], t['state']) for t in targets]}")

        if not targets:
            raise IDBNoDeviceError()

        if udid:
            targets = [t for t in targets if t["udid"] == udid]
            if not targets:
                raise IDBNoDeviceError()

        last_seen = targets

        booted    = [t for t in targets if t["state"].lower() == "booted"]
        connected = [t for t in targets if t["state"].lower() == "connected"]
        shutdown  = [t for t in targets if t["state"].lower() == "shutdown"]
        disconnected = [t for t in targets if t["state"].lower() == "disconnected"]
        unauthorized = [t for t in targets if t["state"].lower() == "unauthorized"]

        ready = booted + connected

        if unauthorized and not ready:
            raise IDBUnauthorizedError(udid=unauthorized[0]["udid"])

        if ready:
            if len(ready) > 1 and not udid:
                raise IDBMultipleDevicesError(udids=[t["udid"] for t in ready])
            return ready[0]

        # Simulator is shutdown — attempt boot and retry
        if shutdown and _is_simulator(shutdown[0]) and attempt < TARGET_RETRIES - 1:
            target_udid = shutdown[0]["udid"]
            _log(f"Simulator is shutdown — attempting boot: {target_udid}")
            _boot_simulator(target_udid)
            time.sleep(TARGET_RETRY_DELAY)
            continue

        # Real device disconnected — retry if allowed
        if disconnected and retry_on_offline and attempt < TARGET_RETRIES - 1:
            _log("Device disconnected, retrying...")
            time.sleep(TARGET_RETRY_DELAY)
            continue

        if disconnected:
            raise IDBOfflineError(udid=disconnected[0]["udid"])

        raise IDBNoDeviceError()

    raise IDBOfflineError(
        udid=last_seen[0]["udid"] if last_seen else None
    )


def _boot_simulator(udid: str) -> None:
    """
    Boot a shutdown simulator via xcrun simctl.
    idb can't boot simulators directly — simctl is the right tool.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["xcrun", "simctl", "boot", udid],
            timeout=30,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise SimulatorBootError(udid=udid)
    except subprocess.TimeoutExpired:
        raise SimulatorBootError(udid=udid)


# ---------------------------------------------------------------------
# Device Description (retry-aware)
# ---------------------------------------------------------------------

def _fetch_describe(udid: str) -> dict:
    """
    Call idb describe and return parsed key-value dict.
    Retries on empty or missing model field.
    """
    last_exc: Optional[Exception] = None

    for attempt in range(DESCRIBE_RETRIES):
        start = time.time()
        try:
            info = idb.describe(udid=udid)
            duration = time.time() - start
            _log(f"idb describe completed in {duration:.2f}s: {info}")

            if info.get("name") or info.get("model"):
                return info

            _log("describe returned empty — retrying...")

        except Exception as exc:
            last_exc = exc
            _log(f"describe error: {exc}")

        time.sleep(DESCRIBE_RETRY_DELAY)

    raise IDBOfflineError(udid=udid) from last_exc


# ---------------------------------------------------------------------
# Delegate Detection
# ---------------------------------------------------------------------

def detect_delegate_candidates(has_ane: bool, is_simulator: bool) -> list[str]:
    """
    CPU and GPU always available.
    ANE candidates only on real device — skipped on simulator entirely.
    SKIPPED status is set in delegate.py, not here — this list only contains
    delegates that will actually be probed.
    """
    candidates = ["CPU", "GPU"]

    if has_ane and not is_simulator:
        candidates.append("ANE")
        candidates.append("ANE_EXPLICIT")

    _log(f"Delegate candidates: {candidates} (has_ane={has_ane}, is_simulator={is_simulator})")

    return candidates


# ---------------------------------------------------------------------
# Public Entry Point
# ---------------------------------------------------------------------

def build_profile(udid: Optional[str] = None) -> DeviceProfile:
    start = time.time()

    target = resolve_target(udid)
    confirmed_udid = target["udid"]

    _log(f"Resolved target: {confirmed_udid} ({target['state']})")

    info = _fetch_describe(confirmed_udid)

    # Use is_simulator from describe() — it has the type field from JSON
    # _is_simulator(target) misses it because list_targets() tuples drop type
    is_sim = info.get("is_simulator", False) or _is_simulator(target)

    model_identifier = info.get("model") or target.get("model") or "unknown"
    ios_version = info.get("os_version") or target.get("os_version") or "unknown"
    chip = resolve_chip(model_identifier)
    has_ane = not is_sim

    profile = DeviceProfile(
        udid=confirmed_udid,
        name=info.get("name") or target.get("name") or "unknown",
        model=model_identifier,
        ios_version=ios_version,
        chip=chip,
        is_simulator=is_sim,
        has_ane=has_ane,
        delegate_candidates=detect_delegate_candidates(has_ane, is_sim),
    )

    duration = time.time() - start
    _log(f"DeviceProfile built in {duration:.2f}s: {profile}")

    return profile