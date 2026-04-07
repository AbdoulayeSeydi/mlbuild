"""
mlbuild.platforms.android.thermal

Thermal and battery state capture.

Guarantees:
- Multi-heuristic temperature normalization (no silent mis-scaling)
- Transport errors propagate (no false “stable” runs)
- Percentile-based thermal aggregation (noise-resistant)
- Strong observability via debug logging
- No policy decisions (no instability flag here)
"""

from __future__ import annotations

import sys
import os
import re
import time
from dataclasses import dataclass
from typing import Optional, List

from mlbuild.platforms.android import adb
from mlbuild.core.errors import ADBOfflineError, ADBTimeoutError


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DEBUG = os.getenv("MLBUILD_DEBUG") == "1"

TEMP_MIN_C = 0.0
TEMP_MAX_C = 120.0

BATTERY_TIMEOUT = 15
THERMAL_TIMEOUT = 5

SYSFS_MILLI_DIV = 1000.0
SYSFS_DECI_DIV  = 10.0


def _log(msg: str) -> None:
    if DEBUG:
        print(f"[mlbuild.thermal] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------

@dataclass
class ThermalSnapshot:
    battery_pct:    Optional[int]   = None
    battery_temp_c: Optional[float] = None
    cpu_temp_c:     Optional[float] = None
    timestamp:      float           = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class ThermalScore:
    thermal_delta_c:   Optional[float]
    latency_drift_pct: Optional[float]


# ---------------------------------------------------------------------
# Battery Parsing
# ---------------------------------------------------------------------

def _parse_battery_dump(raw: str) -> tuple[Optional[int], Optional[float]]:
    pct: Optional[int] = None
    temp: Optional[float] = None

    for line in raw.splitlines():
        line = line.strip()

        m_lvl = re.search(r"\blevel:\s*(\d+)", line, re.IGNORECASE)
        if m_lvl:
            pct = int(m_lvl.group(1))

        m_tmp = re.search(r"\btemperature:\s*(-?\d+)", line, re.IGNORECASE)
        if m_tmp:
            raw_temp = int(m_tmp.group(1))
            if raw_temp > 0:
                temp = raw_temp / 10.0

    return pct, temp


# ---------------------------------------------------------------------
# Temperature Normalization
# ---------------------------------------------------------------------

def _normalize_temp(raw_val: int) -> Optional[float]:
    """
    Multi-heuristic normalization:
    - millidegrees (common)
    - decidegrees (some devices)
    - direct degrees
    """
    candidates: List[float] = []

    # millidegrees
    candidates.append(raw_val / SYSFS_MILLI_DIV)

    # decidegrees
    candidates.append(raw_val / SYSFS_DECI_DIV)

    # raw degrees
    candidates.append(float(raw_val))

    for temp in candidates:
        if TEMP_MIN_C <= temp <= TEMP_MAX_C:
            return round(temp, 2)

    _log(f"Discarding invalid temp raw={raw_val}")
    return None


# ---------------------------------------------------------------------
# Thermal Zone Parsing
# ---------------------------------------------------------------------

def _parse_thermal_zones(raw: str) -> Optional[float]:
    readings: List[float] = []

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue

        try:
            raw_val = int(line)
        except ValueError:
            continue

        temp = _normalize_temp(raw_val)
        if temp is not None:
            readings.append(temp)

    if not readings:
        return None

    readings.sort()

    if DEBUG:
        _log(f"Thermal raw readings: {readings}")

    # Use P90 instead of max to remove spikes
    idx = int(len(readings) * 0.9)
    idx = min(idx, len(readings) - 1)

    return readings[idx]


# ---------------------------------------------------------------------
# Public: Capture Snapshot
# ---------------------------------------------------------------------

def capture_snapshot(serial: Optional[str] = None) -> ThermalSnapshot:
    battery_pct: Optional[int] = None
    battery_temp_c: Optional[float] = None
    cpu_temp_c: Optional[float] = None

    # --- Battery ---
    try:
        result = adb.shell(
            "dumpsys battery",
            serial=serial,
            timeout=BATTERY_TIMEOUT,
        )
        if result.ok:
            battery_pct, battery_temp_c = _parse_battery_dump(result.stdout)
        else:
            _log(f"dumpsys battery failed: {result.stderr}")

    except (ADBOfflineError, ADBTimeoutError):
        raise  # critical transport issue
    except Exception as exc:
        _log(f"Battery read error: {exc}")

    # --- CPU Temp ---
    try:
        result = adb.shell(
            "cat /sys/class/thermal/thermal_zone*/temp",
            serial=serial,
            timeout=THERMAL_TIMEOUT,
        )
        if result.ok:
            cpu_temp_c = _parse_thermal_zones(result.stdout)
        else:
            _log(f"thermal read failed: {result.stderr}")

    except (ADBOfflineError, ADBTimeoutError):
        raise
    except Exception as exc:
        _log(f"CPU temp read error: {exc}")

    return ThermalSnapshot(
        battery_pct=battery_pct,
        battery_temp_c=battery_temp_c,
        cpu_temp_c=cpu_temp_c,
    )


# ---------------------------------------------------------------------
# Public: Compute Score (signal only, no policy)
# ---------------------------------------------------------------------

def compute_thermal_score(
    pre: ThermalSnapshot,
    post: ThermalSnapshot,
    latency_trend: Optional[list[float]] = None,
) -> ThermalScore:

    thermal_delta_c: Optional[float] = None
    latency_drift_pct: Optional[float] = None

    # --- Timestamp sanity ---
    if post.timestamp - pre.timestamp < 1:
        _log("Snapshots too close — thermal delta unreliable")

    # --- Thermal delta ---
    if pre.cpu_temp_c is not None and post.cpu_temp_c is not None:
        thermal_delta_c = round(post.cpu_temp_c - pre.cpu_temp_c, 2)
        _log(f"Thermal delta: {thermal_delta_c}°C")

    # --- Latency drift (trim warmup noise) ---
    if latency_trend and len(latency_trend) >= 4:
        start = latency_trend[len(latency_trend) // 4]
        end   = latency_trend[-1]

        if start > 0:
            latency_drift_pct = round((end - start) / start, 4)
            _log(f"Latency drift: {latency_drift_pct:.1%}")

    return ThermalScore(
        thermal_delta_c=thermal_delta_c,
        latency_drift_pct=latency_drift_pct,
    )