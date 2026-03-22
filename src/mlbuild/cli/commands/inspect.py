from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace, is_dataclass
from datetime import datetime
from typing import Any

import click
from rich.console import Console

console = Console(width=None)


# ── Registry Access (DI-ready) ─────────────────────────────────

def get_registry():
    # Future: swap for remote / hybrid / mock in tests
    from mlbuild.registry.local import LocalRegistry
    return LocalRegistry()


# ── Serialization ──────────────────────────────────────────────

def _serialize(obj: Any):
    if isinstance(obj, datetime):
        return obj.isoformat()

    # Strict mode (preferred for infra correctness)
    if is_dataclass(obj):
        return asdict(obj)

    raise TypeError(f"Unserializable type: {type(obj).__name__}")


# ── View Transformations (pure, no mutation) ───────────────────

def _shorten_view(view):
    """
    Return a derived BuildView with:
    - latest benchmark per compute unit
    - everything else unchanged
    """
    # NOTE: we do NOT sort here if renderer owns ordering
    # We only filter

    latest_by_cu: dict[str, Any] = {}

    for b in view.benchmarks:
        existing = latest_by_cu.get(b.compute_unit)
        if existing is None or b.ran_at > existing.ran_at:
            latest_by_cu[b.compute_unit] = b

    filtered = list(latest_by_cu.values())

    return replace(view, benchmarks=filtered)


# ── Command ───────────────────────────────────────────────────

@click.command(help="Show full details for a build.")
@click.argument("build_id")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
@click.option(
    "--short",
    is_flag=True,
    help="Build + artifact + latest benchmark per compute unit.",
)
def inspect(build_id: str, as_json: bool, short: bool):
    from mlbuild.cli.formatters.inspect import render_inspect

    # ── Load ───────────────────────────────────────────────────

    try:
        registry = get_registry()
        view = registry.get_build_view(build_id)
    except Exception as e:
        console.print(f"\n[red]Error loading build:[/red] {e}\n")
        sys.exit(1)

    if view is None:
        console.print(f"\n[red]No build found:[/red] {build_id}\n")
        sys.exit(1)

    # ── Transform (pure) ───────────────────────────────────────

    view_out = _shorten_view(view) if short else view

    # ── Output ─────────────────────────────────────────────────

    if as_json:
        try:
            payload = asdict(view_out)
            click.echo(
                json.dumps(
                    payload,
                    default=_serialize,
                    indent=2,
                    sort_keys=True,  # deterministic
                )
            )
        except TypeError as e:
            console.print(f"\n[red]Serialization error:[/red] {e}\n")
            sys.exit(1)
        return

    render_inspect(view_out, short=short)