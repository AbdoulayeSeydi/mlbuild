"""
mlbuild.platforms.ios.thermal

Thermal state capture and scoring.

Key differences from Android:
- iOS exposes discrete thermal states (nominal/fair/serious/critical),
  not raw temperatures. No sysfs, no millidegree normalization.
- Simulator: no thermal data. Returns immediately with state=None.
  Nothing is faked.
- Thermal state is reported by the benchmark runner app in its stdout.
  capture_snapshot() parses a state string — it does not query the OS directly.
- ThermalScore uses state transitions + latency drift. No delta_c.
- No policy decisions here (no unstable flag). That lives in stability.py.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from mlbuild.core.errors import InternalError


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DEBUG = os.getenv("MLBUILD_DEBUG") == "1"

LATENCY_DRIFT_MIN_SAMPLES = 4


def _log(msg: str) -> None:
    if DEBUG:
        print(f"[mlbuild.ios.thermal] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------
# Thermal State
# ---------------------------------------------------------------------

class IOSThermalState(str, Enum):
    NOMINAL  = "nominal"   # no throttling
    FAIR     = "fair"      # minor throttling possible
    SERIOUS  = "serious"   # significant throttling
    CRITICAL = "critical"  # aggressive throttling — results unreliable

    @classmethod
    def from_string(cls, raw: str) -> Optional["IOSThermalState"]:
        """
        Case-insensitive parse. Returns None on unrecognized input
        rather than crashing — runner output may vary.
        """
        normalized = raw.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        _log(f"Unrecognized thermal state: {repr(raw)}")
        return None

    def severity(self) -> int:
        """
        Numeric severity for comparison.
        Higher = worse.
        """
        return {
            IOSThermalState.NOMINAL:  0,
            IOSThermalState.FAIR:     1,
            IOSThermalState.SERIOUS:  2,
            IOSThermalState.CRITICAL: 3,
        }[self]


# ---------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------

@dataclass
class ThermalSnapshot:
    state:        Optional[IOSThermalState]   # None if is_simulated=True
    is_simulated: bool
    timestamp:    float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class ThermalScore:
    pre_state:         Optional[IOSThermalState]
    post_state:        Optional[IOSThermalState]
    state_degraded:    bool                      # post severity > pre severity
    latency_drift_pct: Optional[float]


# ---------------------------------------------------------------------
# Simulator Fast Path
# ---------------------------------------------------------------------

def simulator_snapshot() -> ThermalSnapshot:
    """
    Simulator processes don't throttle.
    Return immediately with no state — nothing faked.
    """
    _log("Simulator target — skipping thermal capture")
    return ThermalSnapshot(state=None, is_simulated=True)


# ---------------------------------------------------------------------
# State Parsing (from runner stdout)
# ---------------------------------------------------------------------

def parse_thermal_state(runner_stdout: str) -> Optional[IOSThermalState]:
    """
    Extract thermal state from benchmark runner output.

    Expected line format from runner:
        thermal_state: nominal

    Returns None if the line is absent or unparseable.
    Caller decides whether to treat None as a warning or hard error.
    """
    import json as _json

    # Primary: parse from JSON result event
    for line in runner_stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                obj = _json.loads(line)
                if obj.get("event") == "result" and "thermal_state" in obj:
                    state = IOSThermalState.from_string(obj["thermal_state"])
                    if state is not None:
                        _log(f"Parsed thermal state from JSON: {state.value}")
                    return state
            except Exception:
                pass

    # Fallback: flat text format
    for line in runner_stdout.splitlines():
        line = line.strip().lower()
        if line.startswith("thermal_state:"):
            _, _, raw = line.partition(":")
            state = IOSThermalState.from_string(raw.strip())
            if state is not None:
                _log(f"Parsed thermal state from flat text: {state.value}")
            return state

    _log("No thermal_state found in runner output")
    return None


# ---------------------------------------------------------------------
# Public: Capture Snapshot
# ---------------------------------------------------------------------

def capture_snapshot(
    runner_stdout: str,
    *,
    is_simulated: bool,
) -> ThermalSnapshot:
    """
    Build a ThermalSnapshot from benchmark runner output.

    On simulator: returns immediately with state=None.
    On real device: parses thermal_state from runner stdout.
    Missing state on real device logs a warning but does not raise —
    thermal data is best-effort.
    """
    if is_simulated:
        return simulator_snapshot()

    state = parse_thermal_state(runner_stdout)

    if state is None:
        _log("WARNING: thermal state missing from runner output — snapshot will have state=None")

    return ThermalSnapshot(state=state, is_simulated=False)


# ---------------------------------------------------------------------
# Public: Compute Score (signal only, no policy)
# ---------------------------------------------------------------------

def compute_thermal_score(
    pre: ThermalSnapshot,
    post: ThermalSnapshot,
    latency_trend: Optional[list[float]] = None,
) -> Optional[ThermalScore]:
    """
    Returns None for simulator runs — no score, no faked data.

    For real device:
    - state_degraded: True if post severity > pre severity
    - latency_drift_pct: end vs start latency drift, trimming warmup noise
      (requires >= 4 samples, mirrors Android implementation)
    """
    if pre.is_simulated or post.is_simulated:
        _log("Simulator run — skipping thermal score")
        return None

    # --- Timestamp sanity ---
    if post.timestamp - pre.timestamp < 1:
        _log("Snapshots too close — thermal comparison unreliable")

    # --- State degradation ---
    state_degraded = False
    if pre.state is not None and post.state is not None:
        state_degraded = post.state.severity() > pre.state.severity()
        _log(
            f"Thermal state: {pre.state.value} -> {post.state.value} "
            f"(degraded={state_degraded})"
        )

    # --- Latency drift (trim warmup noise, mirrors Android) ---
    latency_drift_pct: Optional[float] = None

    if latency_trend and len(latency_trend) >= LATENCY_DRIFT_MIN_SAMPLES:
        start = latency_trend[len(latency_trend) // 4]
        end   = latency_trend[-1]

        if start > 0:
            latency_drift_pct = round((end - start) / start, 4)
            _log(f"Latency drift: {latency_drift_pct:.1%}")

    return ThermalScore(
        pre_state=pre.state,
        post_state=post.state,
        state_degraded=state_degraded,
        latency_drift_pct=latency_drift_pct,
    )