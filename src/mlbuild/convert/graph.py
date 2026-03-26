"""
Conversion graph — decorator registration + deterministic BFS path resolution.

Key properties:
- O(V + E) traversal via adjacency list
- Deterministic path resolution (sorted neighbors)
- Strict registration validation (format + function signature)
- PathStep includes executor (no desync risk)
- Supports multiple executors per edge (future-proof)
- Introspection APIs for debugging / CLI
"""

from __future__ import annotations

import inspect
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from mlbuild.convert.types import PathStep
from mlbuild.core.errors import ConvertError, ErrorCode


# ---------------------------------------------------------------------
# Graph Storage
# ---------------------------------------------------------------------

# (src, dst) → list of executors (supports multiple strategies per edge)
CONVERSION_GRAPH: Dict[Tuple[str, str], List[Callable]] = defaultdict(list)

# adjacency list: src → [dst1, dst2, ...]
ADJ: Dict[str, List[str]] = defaultdict(list)


# ---------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------

def register_conversion(src: str, dst: str) -> Callable:
    """
    Decorator to register a conversion executor.

    Enforces:
    - lowercase format names
    - correct function signature (1 arg: ConvertContext)
    - deterministic graph structure
    """

    if not src.islower() or not dst.islower():
        raise ValueError(
            f"Invalid formats: ({src} → {dst}). Must be lowercase strings."
        )

    def wrapper(fn: Callable) -> Callable:
        # --- signature validation ---
        sig = inspect.signature(fn)
        if len(sig.parameters) != 1:
            raise TypeError(
                f"{fn.__name__} must accept exactly one argument (ConvertContext)."
            )

        # --- register executor ---
        edge = (src, dst)
        CONVERSION_GRAPH[edge].append(fn)

        # --- maintain adjacency ---
        if dst not in ADJ[src]:
            ADJ[src].append(dst)

        return fn

    return wrapper


# ---------------------------------------------------------------------
# PathStep (override to include executor)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class GraphPathStep:
    src: str
    dst: str
    reason: str
    executor: Callable


# ---------------------------------------------------------------------
# Path Resolution (Deterministic BFS)
# ---------------------------------------------------------------------

def resolve_path(src: str, dst: str) -> List[GraphPathStep]:
    """
    Deterministic BFS over conversion graph.

    Returns:
        Ordered list of GraphPathStep (includes executor)

    Raises:
        ConvertError if no path exists
    """
    if src == dst:
        raise ConvertError(
            f"Source and target format are both '{src}'. Nothing to convert.",
            error_code=ErrorCode.NO_CONVERSION_PATH,
        )

    queue = deque([(src, [])])
    visited = set()

    while queue:
        current, path = queue.popleft()

        if current in visited:
            continue
        visited.add(current)

        for neighbor in sorted(ADJ.get(current, [])):  # deterministic
            new_path = path + [(current, neighbor)]

            if neighbor == dst:
                return _build_steps(new_path, dst)

            queue.append((neighbor, new_path))

    # --- better error ---
    outgoing = sorted(ADJ.get(src, []))
    raise ConvertError(
        _format_no_path_error(src, dst, outgoing),
        error_code=ErrorCode.NO_CONVERSION_PATH,
    )


# ---------------------------------------------------------------------
# Step Builder
# ---------------------------------------------------------------------

def _build_steps(
    raw_path: List[Tuple[str, str]],
    final_dst: str,
) -> List[GraphPathStep]:
    steps: List[GraphPathStep] = []
    total = len(raw_path)

    for i, (s, d) in enumerate(raw_path):
        # pick primary executor (future: ranking system)
        executors = CONVERSION_GRAPH[(s, d)]
        executor = executors[0]

        if i < total - 1:
            reason = f"{s} cannot convert directly to {final_dst}, routing via {d}"
        else:
            reason = "direct conversion"

        steps.append(
            GraphPathStep(
                src=s,
                dst=d,
                reason=reason,
                executor=executor,
            )
        )

    return steps


# ---------------------------------------------------------------------
# Error Formatting
# ---------------------------------------------------------------------

def _format_no_path_error(src: str, dst: str, outgoing: List[str]) -> str:
    if not outgoing:
        return f"No conversion path from {src} → {dst}. No outgoing edges from {src}."

    edges = "\n  - ".join(outgoing)
    return (
        f"No conversion path from {src} → {dst}.\n\n"
        f"Available conversions from {src}:\n"
        f"  - {edges}"
    )


# ---------------------------------------------------------------------
# Introspection APIs (for CLI, debugging, etc.)
# ---------------------------------------------------------------------

def list_formats() -> List[str]:
    """Return all known formats in the graph."""
    formats = set()
    for s, d in CONVERSION_GRAPH:
        formats.add(s)
        formats.add(d)
    return sorted(formats)


def list_edges() -> List[Tuple[str, str]]:
    """Return all registered edges."""
    return sorted(CONVERSION_GRAPH.keys())


def list_outgoing(src: str) -> List[str]:
    """Return all formats reachable in one step from src."""
    return sorted(ADJ.get(src, []))