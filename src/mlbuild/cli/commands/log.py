"""
Log command: Display build history.

Features:
- Soft-delete awareness
- Pagination with --limit and --offset
- ISO8601 timestamps
- Deterministic ordering
- Optional JSON/CSV output
- Optional full hashes for auditing
- Safe notes display
- Format filter, roots-only, tree view, latency + method columns
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
from rich.tree import Tree

from ...registry import LocalRegistry
from ...core.errors import MLBuildError

if TYPE_CHECKING:
    from ...core.types import Build

console = Console(width=None)


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
    clean = escape(clean)
    if len(clean) > max_len:
        return clean[:max_len] + "…"
    return clean


def safe_hash(value: Optional[str], full: bool) -> str:
    if not value:
        return "-"
    return value if full else value[:16] + "..."


def is_imported(build: "Build") -> bool:
    return build.backend_versions.get("imported") == "true"

def get_machine_name(build: "Build") -> str:
    """Extract machine hostname from environment_data."""
    try:
        node = build.environment_data.get("hardware", {}).get("cpu", {}).get("node", "") or ""
        return node.replace(".local", "")
    except Exception:
        return ""

def display_task(build: "Build") -> str:
    return getattr(build, "task_type", None) or "unknown"


# -------------------------
# Tree view
# -------------------------

def _format_method_label(build) -> str:
    """Format optimization method as a readable step label."""
    method = getattr(build, "optimization_method", None)
    if not method:
        return "fp32"
    # prune_0.50 → prune(0.50)
    if method.startswith("prune_"):
        sparsity = method[len("prune_"):]
        return f"prune({sparsity})"
    # int8_static → int8(static)
    if method == "int8_static":
        return "int8(static)"
    return method


def _add_children(tree_node, build_id: str, children: dict) -> None:
    """Recursively add children to a Rich Tree node."""
    for child in sorted(children.get(build_id, []), key=lambda b: b.created_at):
        method = _format_method_label(child)
        latency = getattr(child, "cached_latency_p50_ms", None)
        lat_str = f"  [dim]{latency:.2f}ms[/dim]" if latency else ""
        child_label = (
            f"[cyan]{child.build_id[:16]}[/cyan]  "
            f"[blue]{child.format}[/blue]  "
            f"[yellow]{method}[/yellow]  "
            f"{float(child.size_mb):.2f} MB"
            f"{lat_str}"
        )
        subtree = tree_node.add(child_label)
        _add_children(subtree, child.build_id, children)


def _print_tree(builds: list, console: Console) -> None:
    """
    Render builds as a recursive parent-child DAG.

    Groups by actual parent_build_id lineage, not root_build_id,
    so optimization chains (prune → int8) display correctly.
    """
    # Build parent adjacency map
    children: dict = {}
    build_map: dict = {}
    for b in builds:
        build_map[b.build_id] = b
        if b.parent_build_id:
            children.setdefault(b.parent_build_id, []).append(b)

    # Roots: builds with no parent in the current set
    roots = [b for b in builds if b.parent_build_id is None]

    for root in roots:
        latency = getattr(root, "cached_latency_p50_ms", None)
        lat_str = f"  [dim]{latency:.2f}ms[/dim]" if latency else ""
        label = (
            f"[cyan]{root.build_id[:16]}[/cyan]  "
            f"[green]{root.name or '(unnamed)'}[/green]  "
            f"[blue]{root.format}[/blue]  fp32  "
            f"{float(root.size_mb):.2f} MB"
            f"{lat_str}"
        )
        tree = Tree(label)
        _add_children(tree, root.build_id, children)
        console.print(tree)
        console.print()


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
@click.option(
    '--task',
    default=None,
    type=click.Choice(['vision', 'nlp', 'audio', 'multimodal', 'unknown']),
    help='Filter builds by task type.',
)
@click.option(
    '--format', 'fmt',
    default=None,
    type=click.Choice(['coreml', 'tflite']),
    help='Filter by backend format.',
)
@click.option(
    '--roots-only',
    is_flag=True,
    default=False,
    help='Show only root builds, not variants.',
)
@click.option(
    '--source',
    default=None,
    help='Filter by source model path (substring match on filename).',
)
@click.option(
    '--tree',
    is_flag=True,
    default=False,
    help='Show baseline → variant hierarchy.',
)
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
    task: Optional[str],
    fmt: Optional[str],
    roots_only: bool,
    source: Optional[str],
    tree: bool,
):
    """
    Build history inspection.

    Examples:
        mlbuild log                          # Show recent builds
        mlbuild log <build_id>               # Show specific build details
        mlbuild log --name mobilenet         # All variants for an experiment
        mlbuild log --name mobilenet --tree  # Hierarchy view
        mlbuild log --format coreml          # Filter by backend
        mlbuild log --roots-only             # Baselines only, no variants
        mlbuild log --task vision            # Filter by task type
        mlbuild log --source mobilenet.onnx  # Filter by source model
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
                data["imported"] = is_imported(build)
                console.print(json.dumps(data, indent=2, sort_keys=True))
                sys.exit(0)

            imported_badge = " [bold yellow]\[imported][/bold yellow]" if is_imported(build) else ""
            console.print(f"\n[bold cyan]{build.name or 'Unnamed Build'}[/bold cyan]{imported_badge}")
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

            method = getattr(build, "optimization_method", None)
            if method:
                console.print(f"  Method:       {method}")
                console.print(f"  Weight Prec:  {getattr(build, 'weight_precision', '—')}")
                console.print(f"  Activation:   {getattr(build, 'activation_precision', '—')}")

            latency = getattr(build, "cached_latency_p50_ms", None)
            p95 = getattr(build, "cached_latency_p95_ms", None)
            if latency:
                console.print(f"\n[bold]Cached Benchmark:[/bold]")
                console.print(f"  p50 Latency:  {latency:.2f}ms")
                if p95:
                    console.print(f"  p95 Latency:  {p95:.2f}ms")

            parent = getattr(build, "parent_build_id", None)
            if parent:
                console.print(f"\n[bold]Lineage:[/bold]")
                console.print(f"  Parent:  {parent}")
                console.print(f"  Root:    {getattr(build, 'root_build_id', '—')}")

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

        # Client-side filters
        if task:
            builds = [b for b in builds if display_task(b) == task]
        if fmt:
            builds = [b for b in builds if b.format == fmt]
        if roots_only:
            builds = [b for b in builds if b.parent_build_id is None]
        if source:
            needle = source.lower()
            builds = [b for b in builds if needle in Path(b.source_path).name.lower()]

        if not builds:
            console.print("\n[yellow]No builds found[/yellow]\n")
            sys.exit(0)

        # -------------------------
        # JSON
        # -------------------------
        if as_json:
            output = []
            for b in builds:
                data = b.to_public_dict(include_hashes=show_hashes)
                data["created_at"] = format_iso8601(data["created_at"])
                data["imported"] = is_imported(b)
                data["task"] = display_task(b)
                data["optimization_method"] = getattr(b, "optimization_method", None)
                data["cached_latency_p50_ms"] = getattr(b, "cached_latency_p50_ms", None)
                output.append(data)
            console.print(json.dumps(output, indent=2, sort_keys=True))
            sys.exit(0)

        # -------------------------
        # CSV
        # -------------------------
        if csv_path is not None:
            if csv_path == "-":
                out_file = sys.stdout
            else:
                out_file = Path(csv_path).open("w", newline="", encoding="utf-8")

            writer = csv.writer(out_file)

            header = [
                "build_id", "name", "format", "optimization_method",
                "target_device", "task", "size_bytes",
                "cached_latency_p50_ms", "created_at", "imported",
            ]
            if show_hashes:
                header += ["artifact_hash", "source_hash", "config_hash"]

            writer.writerow(header)

            for b in builds:
                data = b.to_public_dict(include_hashes=show_hashes)
                row = [
                    data["build_id"],
                    data["name"] or "",
                    data["format"],
                    getattr(b, "optimization_method", None) or "",
                    data["target_device"],
                    display_task(b),
                    data["size_bytes"],
                    getattr(b, "cached_latency_p50_ms", None) or "",
                    format_iso8601(data["created_at"]),
                    is_imported(b),
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
        # Tree view
        # -------------------------
        if tree:
            _print_tree(builds, console)
            sys.exit(0)

        # -------------------------
        # Rich Table View
        # -------------------------

        # Detect multiple machines — only show column if builds came from more than one machine
        # uses environment_data node as display, but detection is smarter
        # Since old builds don't have machine_id in environment_data,
        # fall back to hostname but normalize it
        machine_names = [get_machine_name(b) for b in builds]
        try:
            distinct_machines = registry.get_distinct_machines()
            show_machine = len(distinct_machines) > 1
        except Exception:
            show_machine = False

        table = Table(
            title=f"Builds (showing {len(builds)}, offset {offset})"
        )

        table.add_column("Build ID", style="cyan", no_wrap=True)
        table.add_column("Name", style="green", no_wrap=True)
        table.add_column("Format", style="blue", no_wrap=True)
        table.add_column("Method", style="cyan", no_wrap=True)
        table.add_column("Target", style="yellow", no_wrap=True)
        table.add_column("Task", style="magenta", no_wrap=True)
        table.add_column("Size", justify="right", no_wrap=True)
        table.add_column("p50 Latency", justify="right", no_wrap=True)
        if show_machine:
            table.add_column("Machine", style="dim", no_wrap=True)
        table.add_column("Created", style="dim", no_wrap=True)

        if show_hashes:
            table.add_column("Artifact Hash", style="magenta", no_wrap=True)
            table.add_column("Source Hash", style="magenta", no_wrap=True)
            table.add_column("Config Hash", style="magenta", no_wrap=True)

        for b, machine_name in zip(builds, machine_names):
            data = b.to_public_dict(include_hashes=show_hashes)

            display_name = data["name"] or "(unnamed)"
            if is_imported(b):
                display_name = f"{display_name} [yellow]\[imported][/yellow]"

            method = getattr(b, "optimization_method", None) or "fp32"
            latency = getattr(b, "cached_latency_p50_ms", None)
            latency_str = f"{latency:.2f}ms" if latency else "—"

            row = [
                data["build_id"] if full_id else data["build_id"][:16] + "...",
                display_name,
                data["format"],
                method,
                data["target_device"],
                display_task(b),
                humanize_bytes(data["size_bytes"]),
                latency_str,
            ]

            if show_machine:
                row.append(machine_name or "—")

            row.append(format_iso8601(data["created_at"]))

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