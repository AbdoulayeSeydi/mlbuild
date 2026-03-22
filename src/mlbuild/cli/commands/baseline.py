# src/mlbuild/cli/commands/baseline.py

"""
Baseline management — clean UX wrapper around mlbuild tag.

Under the hood baseline set/update calls tag create with the
reserved tag name 'mlbuild-baseline'. mlbuild ci already resolves
baselines by tag name so this integrates with zero CI changes.
"""

from __future__ import annotations

import sys
import click
from rich.console import Console
from rich.table import Table

from ...registry import LocalRegistry

console = Console(width=None)

BASELINE_TAG = "mlbuild-baseline"


# ================================================================
# Helpers
# ================================================================

def _get_registry_or_exit() -> LocalRegistry:
    """Create registry or exit with clean error."""
    try:
        return LocalRegistry()
    except Exception as e:
        console.print(f"\n[red]Registry error:[/red] {e}\n")
        sys.exit(1)


def _get_baseline(registry: LocalRegistry) -> dict | None:
    """
    Return structured baseline info or None.

    Returns:
        {
            "build": build,
            "created_at": str | None
        }
    """
    build = registry.get_build_by_tag(BASELINE_TAG)
    if build is None:
        return None

    # NOTE: requires registry.get_tag_row() (no SQL in CLI)
    row = registry.get_tag_row(BASELINE_TAG)
    created_at = row["created_at"] if row else None

    return {
        "build": build,
        "created_at": created_at,
    }


def _format_size(build) -> str:
    """Safe size formatting."""
    if getattr(build, "size_mb", None) is None:
        return "—"
    try:
        return f"{float(build.size_mb):.2f} MB"
    except Exception:
        return "—"


def _print_no_baseline():
    console.print("\n[yellow]No baseline set.[/yellow]")
    console.print("[dim]Run: mlbuild baseline set <build_id>[/dim]\n")


def _print_baseline(info: dict):
    """Print baseline details (single source of truth)."""
    build = info["build"]
    created_at = info["created_at"]

    console.print(f"\n[bold]Current Baseline[/bold]\n")
    console.print(f"  Build:   {build.build_id[:12]}...")
    console.print(f"  Name:    {build.name or '(unnamed)'}")
    console.print(f"  Format:  {build.format}")
    console.print(f"  Target:  {build.target_device}")
    console.print(f"  Size:    {_format_size(build)}")

    if build.cached_latency_p50_ms is not None:
        console.print(f"  p50:     {build.cached_latency_p50_ms:.2f} ms")

    if created_at:
        console.print(f"  Tagged:  {created_at[:10]}")

    console.print(
        f"\n[dim]Use in CI: mlbuild ci --model model.onnx "
        f"--baseline {BASELINE_TAG}[/dim]\n"
    )


# ================================================================
# CLI
# ================================================================

@click.group(invoke_without_command=True)
@click.pass_context
def baseline(ctx):
    """Manage the performance baseline for CI comparison."""
    if ctx.invoked_subcommand is None:
        registry = _get_registry_or_exit()
        info = _get_baseline(registry)

        if info is None:
            _print_no_baseline()
            return

        _print_baseline(info)


@baseline.command("set")
@click.argument("build_ref")
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing baseline without prompting."
)
def baseline_set(build_ref: str, force: bool):
    """Set a build as the performance baseline."""
    registry = _get_registry_or_exit()

    build = registry.resolve_build(build_ref)
    if not build:
        console.print(f"\n[red]Build not found: {build_ref}[/red]\n")
        sys.exit(1)

    existing = _get_baseline(registry)

    if existing and not force:
        current = existing["build"]

        if current.build_id == build.build_id:
            console.print(
                f"\n[yellow]Baseline already set to {build.build_id[:12]}.[/yellow]\n"
            )
            return

        confirmed = click.confirm(
            f"Overwrite baseline {current.build_id[:12]}?",
            default=False,
        )
        if not confirmed:
            console.print("\n[dim]Cancelled.[/dim]\n")
            return

    # FIX: atomic baseline update
    registry.delete_tag(BASELINE_TAG)
    registry.add_tag(build.build_id, BASELINE_TAG)

    console.print(f"\n[green]✓ Baseline set[/green]")

    # reuse display logic (no duplication)
    _print_baseline({
        "build": build,
        "created_at": None
    })


@baseline.command("show")
def baseline_show():
    """Show the current baseline build."""
    registry = _get_registry_or_exit()
    info = _get_baseline(registry)

    if info is None:
        _print_no_baseline()
        return

    _print_baseline(info)


@baseline.command("history")
@click.option("--limit", default=20, type=click.IntRange(1, 1000))
def baseline_history(limit: int):
    """
    Show baseline history.

    NOTE:
    This currently relies on tag naming conventions.
    Future improvement: tag type = 'baseline'.
    """
    registry = _get_registry_or_exit()

    # NOTE: still uses registry abstraction — no SQL here
    rows = registry.get_baseline_history(limit=limit)

    if not rows:
        console.print("\n[dim]No baseline history found.[/dim]\n")
        return

    console.print(f"\n[bold]Baseline History[/bold]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Tag",      style="green", no_wrap=True)
    table.add_column("Build ID", style="cyan", no_wrap=True)
    table.add_column("Name", no_wrap=True)
    table.add_column("Format",   style="blue", no_wrap=True)
    table.add_column("Target",   style="yellow", no_wrap=True)
    table.add_column("p50",      justify="right", no_wrap=True)
    table.add_column("Size",     justify="right", no_wrap=True)
    table.add_column("Tagged",   style="dim", no_wrap=True)

    for row in rows:
        lat = (
            f"{row['cached_latency_p50_ms']:.2f} ms"
            if row['cached_latency_p50_ms'] is not None
            else "—"
        )

        size = (
            f"{row['size_bytes'] / (1024*1024):.2f} MB"
            if row['size_bytes'] is not None
            else "—"
        )

        table.add_row(
            row['tag'] or "—",
            (row['build_id'][:12] + "...") if row['build_id'] else "—",
            row['name'] or "(unnamed)",
            row['format'] or "—",
            row['target_device'] or "—",
            lat,
            size,
            (row['created_at'] or "—")[:10],
        )

    console.print(table)
    console.print()


@baseline.command("unset")
def baseline_unset():
    """Remove the current baseline."""
    registry = _get_registry_or_exit()
    info = _get_baseline(registry)

    if info is None:
        console.print("\n[yellow]No baseline to remove.[/yellow]\n")
        return

    build = info["build"]

    confirmed = click.confirm(
        f"Remove baseline {build.build_id[:12]}?",
        default=False,
    )
    if not confirmed:
        console.print("\n[dim]Cancelled.[/dim]\n")
        return

    registry.delete_tag(BASELINE_TAG)

    console.print("\n[green]✓ Baseline removed[/green]\n")