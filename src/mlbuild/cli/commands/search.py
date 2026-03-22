from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List

import click
from rich.console import Console
from rich.table import Table
from rich import box

from mlbuild.registry.local import LocalRegistry

console = Console()


# ---------- Validation ----------

def _validate_limit(limit: int) -> int:
    if limit <= 0:
        raise click.ClickException("Limit must be greater than 0.")
    if limit > 1000:
        raise click.ClickException("Limit cannot exceed 1000.")
    return limit


def _validate_date(value: str | None, field: str) -> str | None:
    if value is None:
        return None
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return value
    except ValueError:
        raise click.ClickException(f"{field} must be in YYYY-MM-DD format.")


# ---------- Normalization (defensive layer) ----------

def _normalize_result(r: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "build_id": str(r.get("build_id") or ""),
        "name": r.get("name") or "(unnamed)",
        "format": r.get("format") or "—",
        "target": r.get("target_device") or "—",
        "task": r.get("task_type") or "—",
        "method": r.get("optimization_method") or "—",
        "latency_p50": r.get("cached_latency_p50_ms"),
        "latency_p95": r.get("cached_latency_p95_ms"),
        "memory": r.get("cached_memory_peak_mb"),
        "size": r.get("size_bytes"),
        "tags": r.get("tags"),
        "created_at": r.get("created_at"),
    }


# ---------- Formatting ----------

def _fmt_latency(val: Any) -> str:
    if val is None:
        return "—"
    try:
        return f"{float(val):.2f} ms"
    except (ValueError, TypeError):
        return "—"
    
def _fmt_memory(val: Any) -> str:
    if val is None:
        return "—"
    try:
        return f"{float(val):.1f} MB"
    except (ValueError, TypeError):
        return "—"


def _fmt_size(val: Any) -> str:
    if val is None:
        return "—"
    try:
        return f"{float(val) / (1024 * 1024):.2f} MB"
    except (ValueError, TypeError):
        return "—"


def _fmt_tags(raw: str | None) -> str:
    if not raw:
        return ""
    tags = [t.strip() for t in raw.split(",") if t.strip()]
    return " ".join(f"[dim][{t}][/dim]" for t in tags)


def _sort_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Stable sort: newest first, fallback to build_id
    return sorted(
        results,
        key=lambda r: (
            r.get("created_at") or "",
            r.get("build_id") or "",
        ),
        reverse=True,
    )


# ---------- CLI ----------

@click.command(help="Search builds by name, format, target, task, or tag.")
@click.argument("query", required=False)
@click.option("--target", default=None)
@click.option("--task", type=click.Choice(["vision", "nlp", "audio", "multimodal", "unknown"]))
@click.option("--format", "fmt", type=click.Choice(["coreml", "tflite", "onnx"]))
@click.option("--tag", default=None)
@click.option("--date-from", default=None)
@click.option("--date-to", default=None)
@click.option("--limit", default=20, type=int)
@click.option("--json", "as_json", is_flag=True)
def search(query, target, task, fmt, tag, date_from, date_to, limit, as_json):

    # ---------- Validate inputs ----------
    limit = _validate_limit(limit)
    date_from = _validate_date(date_from, "date-from")
    date_to = _validate_date(date_to, "date-to")

    registry = LocalRegistry()

    # ---------- Fetch ----------
    try:
        raw_results = registry.search_builds(
            query=query,
            target=target,
            task=task,
            fmt=fmt,
            tag=tag,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
        )
    except Exception as e:
        raise click.ClickException(f"Search failed: {e}")

    if not raw_results:
        label = f'"{query}"' if query else "your filters"
        console.print(f"[yellow]No builds found[/yellow] for {label}.")
        return

    # ---------- Normalize + sort ----------
    results = [_normalize_result(r) for r in raw_results]
    results = _sort_results(results)

    # ---------- JSON output ----------
    if as_json:
        # Strict JSON: fail if not serializable
        try:
            click.echo(json.dumps(results, indent=2))
        except TypeError as e:
            raise click.ClickException(f"Failed to serialize results: {e}")
        return

    # ---------- Table output ----------
    label = f'"{query}"' if query else "all builds"
    console.print(f"[bold]{len(results)} result{'s' if len(results) != 1 else ''}[/bold] for {label}")

    table = Table(box=box.SIMPLE, show_header=True, pad_edge=False)
    table.add_column("ID",      style="cyan",  min_width=10)
    table.add_column("Name",    min_width=20)
    table.add_column("Format",  min_width=8)
    table.add_column("Target",  min_width=14)
    table.add_column("Task",    min_width=8)
    table.add_column("Method",  min_width=10)
    table.add_column("p50",     justify="right")
    table.add_column("p95",     justify="right")
    table.add_column("Mem",     justify="right")
    table.add_column("Size",    justify="right")
    table.add_column("Tags")

    for r in results:
        table.add_row(
            r["build_id"][:8],
            r["name"],
            r["format"],
            r["target"],
            r["task"],
            r["method"],
            _fmt_latency(r["latency_p50"]),
            _fmt_latency(r["latency_p95"]),
            _fmt_memory(r["memory"]),
            _fmt_size(r["size"]),
            _fmt_tags(r["tags"]),
        )

    console.print(table)