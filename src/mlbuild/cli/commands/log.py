"""
Enterprise-grade log command: Display build history.

Features:
- Soft-delete awareness
- Pagination with --limit and --offset
- ISO8601 timestamps
- Deterministic ordering
- Optional JSON/CSV output
- Optional full hashes for auditing
- Safe notes display
"""

from __future__ import annotations

import json
import csv
import sys
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import click
from rich.console import Console
from rich.table import Table
from rich.markup import escape

from ...registry import LocalRegistry
from ...core.errors import MLBuildError

if TYPE_CHECKING:
    from ...core.types import Build

console = Console()


# -------------------------
# Utility
# -------------------------

def format_iso8601(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_iso8601(value: Optional[str], field: str) -> Optional[str]:
    if value is None:
        return None
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value
    except Exception:
        raise click.UsageError(f"Invalid ISO8601 value for {field}: {value}")


def humanize_bytes(num: int) -> str:
    step = 1024.0
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < step:
            return f"{num:.2f} {unit}"
        num /= step
    return f"{num:.2f} PB"


def sanitize_notes(notes: Optional[str], max_len: int = 200) -> str:
    if not notes:
        return ""
    clean = " ".join(notes.split())
    clean = escape(clean)  # escape rich markup
    if len(clean) > max_len:
        return clean[:max_len] + "…"
    return clean


def safe_hash(value: Optional[str], full: bool) -> str:
    if not value:
        return "-"
    return value if full else value[:16] + "..."


# -------------------------
# CLI
# -------------------------

@click.command()
@click.argument('build_id', required=False)
@click.option('--limit', default=50, type=int)
@click.option('--offset', default=0, type=int)
@click.option('--json', 'as_json', is_flag=True)
@click.option('--csv', 'csv_path', type=str)
@click.option('--show-hashes', is_flag=True)
@click.option('--show-notes', is_flag=True)
@click.option('--full-id', is_flag=True)
@click.option('--full-hashes', is_flag=True)
@click.option('--target', default=None)
@click.option('--name', default=None)
@click.option('--tag', default=None)
@click.option('--date-from', default=None)
@click.option('--date-to', default=None)
def log(
    build_id: Optional[str], 
    limit: int,
    offset: int,
    as_json: bool,
    csv_path: Optional[str],
    show_hashes: bool,
    show_notes: bool,
    full_id: bool,
    full_hashes: bool,
    target: Optional[str],
    name: Optional[str],
    tag: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
):
    """
    Enterprise-grade build history inspection.
    
    Examples:
        mlbuild log                          # Show recent builds
        mlbuild log <build_id>               # Show specific build details
        mlbuild log --limit 20               # Show last 20 builds
        mlbuild log --target apple_m3        # Filter by target
        mlbuild log --json                   # JSON output
        mlbuild log --csv builds.csv         # CSV export
    """

    try:
        if as_json and csv_path:
            raise click.UsageError("--json and --csv are mutually exclusive")

        date_from = parse_iso8601(date_from, "--date-from")
        date_to = parse_iso8601(date_to, "--date-to")

        registry = LocalRegistry()

        # -------------------------
        # Show Specific Build Details
        # -------------------------
        if build_id:
            build = registry.resolve_build(build_id)
            
            if not build:
                console.print(f"\n[red]Build not found: {build_id}[/red]\n")
                sys.exit(1)
            
            if as_json:
                data = build.to_public_dict(include_hashes=True)
                data["created_at"] = format_iso8601(data["created_at"])
                console.print(json.dumps(data, indent=2, sort_keys=True))
                sys.exit(0)
            
            # Rich detailed view
            console.print(f"\n[bold cyan]{build.name or 'Unnamed Build'}[/bold cyan]")
            console.print(f"[dim]Created: {format_iso8601(build.created_at)}[/dim]\n")
            
            console.print("[bold]Identifiers:[/bold]")
            console.print(f"  Build ID:        {build.build_id}")
            console.print(f"  Artifact Hash:   {build.artifact_hash}")
            console.print(f"  Source Hash:     {build.source_hash}")
            console.print(f"  Config Hash:     {build.config_hash}")
            console.print(f"  Env Fingerprint: {build.env_fingerprint}")
            
            console.print(f"\n[bold]Configuration:[/bold]")
            console.print(f"  Target:       {build.target_device}")
            console.print(f"  Format:       {build.format}")
            console.print(f"  Quantization: {build.quantization_type}")
            console.print(f"  Size:         {build.size_mb:.2f} MB ({build.size_bytes:,} bytes)")
            
            console.print(f"\n[bold]Source:[/bold]")
            console.print(f"  Path: {build.source_path}")
            
            console.print(f"\n[bold]Artifact:[/bold]")
            console.print(f"  Path: {build.artifact_path}")
            
            console.print(f"\n[bold]Backend Versions:[/bold]")
            for k, v in sorted(build.backend_versions.items()):
                console.print(f"  {k:20s} {v}")
            
            console.print(f"\n[bold]Environment:[/bold]")
            console.print(f"  MLBuild Version: {build.mlbuild_version}")
            console.print(f"  Python:          {build.python_version}")
            console.print(f"  Platform:        {build.platform}")
            console.print(f"  OS Version:      {build.os_version}")
            
            if build.notes:
                console.print(f"\n[bold]Notes:[/bold]")
                console.print(f"  {build.notes}")
            
            console.print()
            sys.exit(0)

        # -------------------------
        # Show Build History (List)
        # -------------------------
        builds: List["Build"] = registry.list_builds(
            limit=limit,
            offset=offset,
            target=target,
            name_pattern=name,
            tag=tag,
            include_deleted=False,
            date_from=date_from,
            date_to=date_to,
            order_by=("created_at DESC", "build_id ASC"),
        )

        if not builds:
            console.print("\n[yellow]No builds found[/yellow]\n")
            sys.exit(0)

        # -------------------------
        # JSON (deterministic)
        # -------------------------

        if as_json:
            output = []
            for b in builds:
                data = b.to_public_dict(include_hashes=show_hashes)
                data["created_at"] = format_iso8601(data["created_at"])
                output.append(data)

            console.print(json.dumps(output, indent=2, sort_keys=True))
            sys.exit(0)

        # -------------------------
        # CSV (pipeline friendly)
        # -------------------------

        if csv_path is not None:
            if csv_path == "-":
                out_file = sys.stdout
            else:
                out_file = Path(csv_path).open("w", newline="", encoding="utf-8")

            writer = csv.writer(out_file)

            header = [
                "build_id",
                "name",
                "target_device",
                "format",
                "size_bytes",
                "created_at",
            ]

            if show_hashes:
                header += ["artifact_hash", "source_hash", "config_hash"]

            writer.writerow(header)

            for b in builds:
                data = b.to_public_dict(include_hashes=show_hashes)

                row = [
                    data["build_id"],
                    data["name"] or "",
                    data["target_device"],
                    data["format"],
                    data["size_bytes"],
                    format_iso8601(data["created_at"]),
                ]

                if show_hashes:
                    row += [
                        data.get("artifact_hash") or "",
                        data.get("source_hash") or "",
                        data.get("config_hash") or "",
                    ]

                writer.writerow(row)

            if csv_path != "-":
                out_file.close()

            sys.exit(0)

        # -------------------------
        # Rich Table View
        # -------------------------

        table = Table(
            title=f"Builds (showing {len(builds)} builds, offset {offset})"
        )

        table.add_column("Build ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Target", style="yellow")
        table.add_column("Size", justify="right")
        table.add_column("Created", style="dim")

        if show_hashes:
            table.add_column("Artifact Hash", style="magenta")
            table.add_column("Source Hash", style="magenta")
            table.add_column("Config Hash", style="magenta")

        for b in builds:
            data = b.to_public_dict(include_hashes=show_hashes)

            row = [
                data["build_id"] if full_id else data["build_id"][:16] + "...",
                data["name"] or "(unnamed)",
                data["target_device"],
                humanize_bytes(data["size_bytes"]),
                format_iso8601(data["created_at"]),
            ]

            if show_hashes:
                row += [
                    safe_hash(data.get("artifact_hash"), full_hashes),
                    safe_hash(data.get("source_hash"), full_hashes),
                    safe_hash(data.get("config_hash"), full_hashes),
                ]

            table.add_row(*row)

        console.print(table)

        if show_notes:
            console.print("\n[bold]Notes:[/bold]")
            for b in builds:
                note = sanitize_notes(b.notes)
                if note:
                    console.print(f"{b.build_id[:16]}: {note}")

        sys.exit(0)

    except MLBuildError as e:
        console.print("\n[bold red]Registry Error[/bold red]\n")
        console.print(e.format())
        sys.exit(1)

    except click.UsageError as e:
        console.print(f"\n[bold red]Usage Error:[/bold red] {e}\n")
        sys.exit(2)

    except Exception as e:
        console.print(f"\n[bold red]Unexpected Error:[/bold red] {e}\n")
        sys.exit(1)
