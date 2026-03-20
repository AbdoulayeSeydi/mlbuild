from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import click
from rich.console import Console

console = Console()


# ── Internal helpers ──────────────────────────────────────────

def _fail(msg: str) -> None:
    console.print(f"\n[red]Error:[/red] {msg}\n")
    sys.exit(1)


def get_registry():
    from mlbuild.registry.local import LocalRegistry
    return LocalRegistry()


def _resolve_format(output: str | None, fmt: str | None) -> Tuple[str, bool]:
    """
    Returns (format, is_directory).
    format: "json" | "csv"
    """

    # ── No output → stdout ─────────────────────────────────────
    if output is None:
        return (fmt or "json", False)

    p = Path(output)

    # ── Directory mode (explicit) ──────────────────────────────
    if output.endswith(("/", "\\")):
        if fmt is not None and fmt != "csv":
            _fail("Directory output requires CSV format.")
        return ("csv", True)

    # ── Directory mode (existing path) ─────────────────────────
    if p.exists() and p.is_dir():
        if fmt is not None and fmt != "csv":
            _fail("Directory output requires CSV format.")
        return ("csv", True)

    # ── File mode ──────────────────────────────────────────────
    ext = p.suffix.lower()

    if not ext:
        _fail(
            f"Ambiguous output path: {output!r}\n"
            "Use a trailing / for directory mode or add .json/.csv extension."
        )

    if ext not in (".json", ".csv"):
        _fail(f"Unsupported extension: {ext}\nUse .json or .csv")

    inferred = "json" if ext == ".json" else "csv"

    # ── Conflict detection ─────────────────────────────────────
    if fmt is not None and fmt != inferred:
        _fail(
            f"--format {fmt} conflicts with --output {output}\n"
            "Remove --format or change the output extension."
        )

    return (inferred, False)


# ── CLI Command ───────────────────────────────────────────────

@click.command(help="Export a build and all its data as JSON or CSV.")
@click.argument("build_id")
@click.option(
    "--format", "fmt",
    type=click.Choice(["json", "csv"]),
    default=None,
    help="Output format. Inferred from --output extension if not set.",
)
@click.option(
    "--output", "-o",
    type=str,
    default=None,
    help="Output path. File (.json/.csv) or directory (trailing /).",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing directory output.",
)
def export(build_id: str, fmt: str | None, output: str | None, force: bool):
    from mlbuild.cli.formatters.export import (
        build_view_to_json,
        build_view_to_flat_csv,
        build_view_to_csv_dir,
    )

    # ── Resolve build ──────────────────────────────────────────
    try:
        registry = get_registry()
        view = registry.get_build_view(build_id)
    except Exception as e:
        _fail(f"Error loading build: {e}")

    if view is None:
        _fail(f"No build found: {build_id}")

    # ── Resolve format ─────────────────────────────────────────
    resolved_fmt, is_dir = _resolve_format(output, fmt)

    # ── Execute ────────────────────────────────────────────────
    try:
        if is_dir:
            out_path = Path(output)
            build_view_to_csv_dir(view, out_path, force=force)

            console.print(f"\n  [green]✓[/green] Exported to {out_path.resolve()}/\n")
            console.print("    build.csv, artifacts.csv, benchmarks.csv, accuracy.csv, tags.csv\n")

        elif output is not None:
            out_path = Path(output)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if resolved_fmt == "json":
                content = build_view_to_json(view)
            else:
                content = build_view_to_flat_csv(view)

            # Enforce single trailing newline contract
            content = content.rstrip("\n") + "\n"

            out_path.write_text(content, encoding="utf-8")

            console.print(f"\n  [green]✓[/green] Exported to {out_path.resolve()}\n")

        else:
            # stdout mode
            if resolved_fmt == "json":
                content = build_view_to_json(view)
            else:
                content = build_view_to_flat_csv(view)

            # Avoid formatter coupling
            click.echo(content.rstrip("\n"))

    except FileExistsError as e:
        _fail(str(e))
    except RuntimeError as e:
        _fail(str(e))
    except Exception as e:
        _fail(f"Unexpected error: {e}")