from __future__ import annotations
from datetime import datetime, timezone


def relative_time(dt: datetime) -> str:
    """Shared relative time formatter. Used by inspect, prune, and any future commands."""
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = now - dt
    s = int(delta.total_seconds())
    if s < 60:
        return "just now"
    if s < 3600:
        return f"{s // 60}m ago"
    if s < 86400:
        return f"{s // 3600}h ago"
    if s < 86400 * 7:
        return f"{s // 86400} days ago"
    return dt.strftime("%Y-%m-%d")

def parse_duration(value: str) -> float:
    """Parse duration string to days. Accepts: 30d, 7d, 24h, 1h, 30m."""
    value = value.strip()
    try:
        if value.endswith("d"):
            return float(value[:-1])
        if value.endswith("h"):
            return float(value[:-1]) / 24
        if value.endswith("m"):
            return float(value[:-1]) / (24 * 60)
    except ValueError:
        pass
    raise ValueError(f"Invalid duration: {value!r}. Expected format: 30d, 7d, 24h, 1h")