"""
budget.toml schema (STRICT):

- Owned entirely by MLBuild
- Only contains [constraints]
- Any additional data must live in separate files

This file is intentionally minimal and stable.
"""


from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Optional

# ================================================================
# Constants / Schema
# ================================================================

VERSION = 1

KEY_MAP: dict[str, str] = {
    "max-latency": "max_latency_ms",
    "max-p95":     "max_p95_ms",
    "max-memory":  "max_memory_mb",
    "max-size":    "max_size_mb",
}

REVERSE_KEY_MAP: dict[str, str] = {v: k for k, v in KEY_MAP.items()}

KEYS: list[str] = [
    "max_latency_ms",
    "max_p95_ms",
    "max_memory_mb",
    "max_size_mb",
]

DISPLAY_MAP: dict[str, tuple[str, str]] = {
    "max_latency_ms": ("Max latency (p50)", "ms"),
    "max_p95_ms":     ("Max latency (p95)", "ms"),
    "max_memory_mb":  ("Max memory",        "MB"),
    "max_size_mb":    ("Max size",          "MB"),
}


# ================================================================
# Paths
# ================================================================

def get_budget_path() -> Path:
    return Path.cwd() / ".mlbuild" / "budget.toml"


def _ensure_dir() -> None:
    get_budget_path().parent.mkdir(parents=True, exist_ok=True)


# ================================================================
# Internal helpers
# ================================================================

def _empty_budget() -> dict[str, Optional[float]]:
    return {k: None for k in KEYS}


def _validate_keys(d: dict) -> None:
    for k in d:
        if k not in KEYS:
            raise KeyError(f"Invalid constraint key: {k}")


def _validate_constraints(constraints: dict) -> None:
    for key, val in constraints.items():
        if val is None:
            continue
        if not isinstance(val, (int, float)):
            raise ValueError(f"{key} must be numeric, got {type(val).__name__}")
        if val <= 0:
            label, unit = DISPLAY_MAP.get(key, (key, ""))
            raise ValueError(f"{label} must be > 0, got {val} {unit}")


def _normalize(data: dict) -> dict[str, Optional[float]]:
    """
    Strict parsing of TOML → internal schema.
    Raises on invalid types or unknown version.
    """
    if not isinstance(data, dict):
        raise ValueError("Invalid TOML structure")
    
    # Enforce strict top-level schema
    allowed_top_keys = {"version", "constraints"}
    for key in data:
        if key not in allowed_top_keys:
            raise ValueError(f"Unknown top-level key: {key}")

    version = data.get("version", 1)
    if version != VERSION:
        raise ValueError(f"Unsupported budget version: {version}")

    constraints = data.get("constraints", {})
    if not isinstance(constraints, dict):
        raise ValueError("constraints must be a table")

    result = _empty_budget()

    # allow both CLI-style and internal keys
    reverse_map = {**{k: k for k in KEYS}, **KEY_MAP}

    for raw_key, value in constraints.items():
        internal = reverse_map.get(raw_key)
        if internal is None:
            continue  # ignore unknown keys safely

        if value is None:
            continue

        if not isinstance(value, (int, float)):
            raise ValueError(
                f"{internal} must be a number, got {type(value).__name__}"
            )

        result[internal] = float(value)

    return result


def _serialize(constraints: dict) -> str:
    """
    Serialize to TOML string.
    Explicitly controls format and ordering.
    """
    active = {k: v for k, v in constraints.items() if v is not None}

    lines = [f"version = {VERSION}\n\n", "[constraints]\n"]

    for key in KEYS:
        if key in active:
            lines.append(f"{key} = {active[key]}\n")

    return "".join(lines)


def _atomic_write(path: Path, content: str) -> None:
    """
    Atomic write to prevent file corruption.
    """
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content)
    tmp.replace(path)


# ================================================================
# Public API
# ================================================================

def load_budget() -> dict[str, Optional[float]]:
    path = get_budget_path()

    if not path.exists():
        return _empty_budget()

    with open(path, "rb") as f:
        data = tomllib.load(f)

    return _normalize(data)


def save_budget(constraints: dict) -> None:
    _validate_keys(constraints)
    _validate_constraints(constraints)

    _ensure_dir()

    content = _serialize(constraints)
    _atomic_write(get_budget_path(), content)


def clear_budget() -> None:
    path = get_budget_path()
    if path.exists():
        path.unlink()


def clear_constraint(cli_key: str) -> None:
    if cli_key not in KEY_MAP:
        valid = ", ".join(KEY_MAP.keys())
        raise KeyError(f"Unknown constraint '{cli_key}'. Valid options: {valid}")

    path = get_budget_path()
    if not path.exists():
        raise FileNotFoundError("No budget file to modify")

    internal_key = KEY_MAP[cli_key]

    budget = load_budget()
    budget[internal_key] = None

    if budget_is_empty(budget):
        clear_budget()
    else:
        save_budget(budget)


def merge_constraints(explicit: dict, budget: dict) -> dict:
    _validate_keys(explicit)
    _validate_keys(budget)

    result = {}
    for key in KEYS:
        val = explicit.get(key)
        result[key] = val if val is not None else budget.get(key)

    return result


def budget_is_empty(budget: dict) -> bool:
    return all(v is None for v in budget.values())


def format_budget_display(budget: dict) -> str:
    def fmt(key: str, val: Optional[float]) -> str:
        if val is None:
            return "not set"
        _, unit = DISPLAY_MAP[key]
        return f"{val:.1f} {unit}"

    lines = []
    for key in KEYS:
        label, _ = DISPLAY_MAP[key]
        lines.append(f"  {label:<20} {fmt(key, budget.get(key))}")

    return "\n".join(lines)


def constraint_origin(key: str, explicit: dict, budget: dict) -> str:
    """
    Only call AFTER validation.
    """
    if explicit.get(key) is not None:
        return "explicit flag"
    if budget.get(key) is not None:
        return "budget"
    return "not set"