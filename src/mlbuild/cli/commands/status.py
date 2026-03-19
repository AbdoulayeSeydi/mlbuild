# src/mlbuild/cli/commands/status.py

"""
mlbuild status — quick workspace health snapshot.

Reads from existing data only:
  - registry (via public API ONLY)
  - budget file
  - baseline
  - machine identity

No direct SQL access. No schema coupling.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import click
from rich.console import Console

console = Console()


# ================================================================
# Helpers
# ================================================================

def _time_ago(iso: str) -> str:
    """Human-readable time delta from ISO timestamp string."""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = now - dt
        seconds = int(delta.total_seconds())

        if seconds < 60:
            return f"{seconds}s ago"
        if seconds < 3600:
            return f"{seconds // 60}m ago"
        if seconds < 86400:
            return f"{seconds // 3600}h ago"
        return f"{seconds // 86400}d ago"

    except Exception:
        return "unknown"


def _safe_size_mb(size_bytes: Optional[Any]) -> str:
    if size_bytes is None:
        return "—"
    try:
        return f"{float(size_bytes) / (1024 * 1024):.2f} MB"
    except Exception:
        return "—"


def _safe_latency(val: Optional[Any]) -> str:
    if val is None:
        return "—"
    try:
        return f"{float(val):.2f} ms"
    except Exception:
        return "—"


def _load_budget_safe():
    from ...core.budget import load_budget

    try:
        return load_budget()
    except Exception as e:
        console.print(f"\n[red]Invalid .mlbuild/budget.toml:[/red] {e}")
        console.print("[dim]Run: mlbuild budget clear to reset[/dim]\n")
        return None


# ================================================================
# Data Collection
# ================================================================

def collect_status_data(registry) -> Dict[str, Any]:
    """Collect all status data via registry + safe loaders."""

    # Registry stats (NO SQL here)
    stats = registry.get_stats()
    last_build = registry.get_last_build()
    last_benchmark = registry.get_last_benchmark()
    last_validate = registry.get_last_validate()
    baseline = registry.get_baseline()

    # Budget
    from ...core.budget import budget_is_empty

    budget = _load_budget_safe()
    if budget is None or budget_is_empty(budget):
        budget_clean = None
    else:
        budget_clean = budget

    # Machine
    from ...core.machine import get_machine_info

    try:
        machine = get_machine_info()
        machine_name = machine.get("machine_name", "unknown")
    except Exception as e:
        machine_name = "unknown"
        console.print(f"[dim]Warning: machine info unavailable: {e}[/dim]")

    # Workspace health
    ws_path = Path.cwd() / ".mlbuild"
    registry_db = ws_path / "registry.db"

    workspace_ok = ws_path.exists() and registry_db.exists()

    return {
        "machine": machine_name,
        "workspace_ok": workspace_ok,
        "build_count": stats.get("builds", 0),
        "benchmark_count": stats.get("benchmarks", 0),
        "last_build": last_build,
        "last_benchmark": last_benchmark,
        "last_validate": last_validate,
        "baseline": baseline,
        "budget": budget_clean,
    }


# ================================================================
# Renderers
# ================================================================

def render_status_json(data: Dict[str, Any]):
    import json
    console.print(json.dumps(data, indent=2, default=str))


def render_status_text(data: Dict[str, Any]):
    console.print(f"\n[bold]MLBuild Status[/bold]  [dim]{data['machine']}[/dim]\n")

    # Workspace
    if data["workspace_ok"]:
        console.print("  [green]✓[/green] Workspace    .mlbuild/")
    else:
        console.print("  [red]✗[/red] Workspace    not initialized — run: mlbuild init")

    # Registry
    console.print(
        f"  [green]✓[/green] Registry     "
        f"{data['build_count']} build{'s' if data['build_count'] != 1 else ''}  |  "
        f"{data['benchmark_count']} benchmark{'s' if data['benchmark_count'] != 1 else ''}"
    )

    # Last build
    lb = data["last_build"]
    if lb:
        name = lb.get("name") or "(unnamed)"
        fmt = lb.get("format") or "—"
        size = _safe_size_mb(lb.get("size_bytes"))
        ago = _time_ago(lb.get("created_at", ""))
        console.print(f"  [dim]Last build:  {name} ({fmt}, {size}) — {ago}[/dim]")
    else:
        console.print("  [dim]Last build:  none[/dim]")

    # Last benchmark
    lbm = data["last_benchmark"]
    if lbm:
        p50 = _safe_latency(lbm.get("latency_p50_ms"))
        ago = _time_ago(lbm.get("measured_at", ""))
        console.print(f"  [dim]Last bench:  p50={p50} — {ago}[/dim]")
    else:
        console.print("  [dim]Last bench:  none[/dim]")

    console.print()

    # Baseline
    baseline = data["baseline"]
    if baseline and baseline.get("build"):
        b = baseline["build"]
        lat = _safe_latency(b.get("cached_latency_p50_ms"))
        size = _safe_size_mb(b.get("size_bytes"))
        console.print(
            f"  [green]✓[/green] Baseline     "
            f"{b.get('build_id', '')[:12]}  "
            f"{b.get('name') or '(unnamed)'}  "
            f"{lat}  {size}"
        )
    else:
        console.print(
            "  [yellow]—[/yellow] Baseline     not set  "
            "[dim](run: mlbuild baseline set <build_id>)[/dim]"
        )

    # Last validate
    lv = data["last_validate"]
    if lv:
        result = "[green]PASSED[/green]" if lv.get("exit_code") == 0 else "[red]FAILED[/red]"
        ago = _time_ago(lv.get("ran_at", ""))
        console.print(f"  [dim]Last validate: {result} — {ago}[/dim]")
    else:
        console.print("  [dim]Last validate: never run[/dim]")

    console.print()

    # Budget
    from ...core.budget import format_budget_display

    if data["budget"] is None:
        console.print(
            "  [yellow]—[/yellow] Budget       not set  "
            "[dim](run: mlbuild budget set --max-latency <ms>)[/dim]"
        )
    else:
        console.print("  [green]✓[/green] Budget       .mlbuild/budget.toml")
        for line in format_budget_display(data["budget"]).splitlines():
            console.print(f"  [dim]{line.strip()}[/dim]")

    console.print()


# ================================================================
# CLI Entry
# ================================================================

@click.command("status")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def status(as_json: bool):
    """Show workspace health and current state."""

    from ...registry import LocalRegistry

    try:
        registry = LocalRegistry()
    except Exception as e:
        console.print(f"\n[red]Registry error:[/red] {e}\n")
        sys.exit(1)

    data = collect_status_data(registry)

    if as_json:
        render_status_json(data)
    else:
        render_status_text(data)