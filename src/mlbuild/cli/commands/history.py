# src/mlbuild/cli/commands/history.py

from __future__ import annotations

import click
from datetime import datetime, timezone, timedelta

from rich.console import Console
from rich.table import Table
from rich import box

from ...registry.local import LocalRegistry

console = Console(width=None)


# ------------------------------------------------------------
# Registry helper
# ------------------------------------------------------------

def _get_registry(ctx: click.Context) -> LocalRegistry:
    if ctx.obj is None:
        ctx.obj = {}

    if "registry" not in ctx.obj:
        ctx.obj["registry"] = LocalRegistry()

    return ctx.obj["registry"]


# ------------------------------------------------------------
# Time parsing
# ------------------------------------------------------------

def _parse_since(value: str | None) -> datetime | None:
    if not value:
        return None

    value = value.strip().lower()
    now = datetime.now(timezone.utc)

    if value == "yesterday":
        return now - timedelta(days=1)

    if "days ago" in value:
        try:
            days = int(value.split()[0])
            return now - timedelta(days=days)
        except Exception:
            pass

    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        raise click.BadParameter(
            "Invalid --since format. Use 'yesterday', '7 days ago', or YYYY-MM-DD."
        )


# ------------------------------------------------------------
# Presentation helpers
# ------------------------------------------------------------

def _time_ago(ran_at: str) -> str:
    try:
        if ran_at.endswith("Z"):
            ran_at = ran_at.replace("Z", "+00:00")

        dt = datetime.fromisoformat(ran_at)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        diff = now - dt
        seconds = int(diff.total_seconds())

        if seconds < 60:
            return "just now"

        if seconds < 3600:
            return f"{seconds // 60}m ago"

        if seconds < 86400:
            return f"{seconds // 3600}h ago"

        if seconds < 172800:
            return "yesterday"

        days = seconds // 86400

        if days < 30:
            return f"{days}d ago"

        if days < 365:
            return f"{days // 7}w ago"

        return f"{days // 365}y ago"

    except Exception:
        return "unknown"


def _format_duration(duration_ms: int | None) -> str:
    if not duration_ms:
        return "0ms"

    if duration_ms < 1000:
        return f"{duration_ms}ms"

    if duration_ms < 60000:
        return f"{duration_ms / 1000:.1f}s"

    mins = duration_ms // 60000
    secs = (duration_ms % 60000) // 1000
    return f"{mins}m {secs}s"


def _format_result(row: dict) -> str:
    exit_code = row.get("exit_code", 0)
    command = row.get("command_name", "")

    if exit_code != 0:
        return "[red]FAILED[/red]"

    linked_build = row.get("linked_build_id")

    if command == "build" and linked_build:
        return f"[dim]{linked_build[:7]}[/dim]"

    if command == "validate":
        return "[green]PASSED[/green]"

    return "[green]✓[/green]"


def _render_history_table(rows: list[dict]) -> Table:

    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold",
        show_edge=False,
        pad_edge=False,
        expand=False,
    )

    table.add_column("ID", style="dim", min_width=8, no_wrap=True)
    table.add_column("Time", style="dim", min_width=9, no_wrap=True)
    table.add_column("Command", no_wrap=True, overflow="ellipsis", min_width=20)
    table.add_column("Result", min_width=8, no_wrap=True)
    table.add_column("Duration", min_width=8, no_wrap=True)

    for row in rows:

        table.add_row(
            row["id"][:8],
            _time_ago(row["ran_at"]),
            row["raw_command"],
            _format_result(row),
            _format_duration(row.get("duration_ms")),
        )

    return table


def _render_history_entry(row: dict) -> str:

    lines = [
        f"[bold]Command:[/bold] {row['raw_command']}",
        f"[bold]ID:[/bold] {row['id']}",
        f"[bold]Ran:[/bold] {_time_ago(row['ran_at'])}",
        f"[bold]Duration:[/bold] {_format_duration(row.get('duration_ms'))}",
        f"[bold]Exit Code:[/bold] {row.get('exit_code', 0)}",
    ]

    return "\n".join(lines)


# ------------------------------------------------------------
# history command
# ------------------------------------------------------------

@click.group(invoke_without_command=True, help="Show and manage command history.")
@click.option(
    "--filter",
    "filter_cmd",
    type=click.Choice([
    "accuracy", "baseline", "benchmark", "budget", "build",
    "ci", "ci-check", "compare", "compare-compute-units",
    "compare-quantization", "convert", "diff", "doctor", "experiment",
    "explore", "export", "import", "init", "inspect", "log", "optimize",
    "pin", "unpin", "profile", "prune", "pull", "push", "remote",
    "rename", "report", "run", "search", "status", "sync", "tag",
    "upgrade", "validate", "failed",
    ]),
)
@click.option("--since")
@click.option("--build-id")
@click.option("--limit", default=50, type=click.IntRange(1, 1000))
@click.pass_context
def history(ctx, filter_cmd, since, build_id, limit):

    if ctx.invoked_subcommand:
        return

    registry = _get_registry(ctx)

    since_dt = _parse_since(since)

    failed_only = filter_cmd == "failed"
    command_name = None if filter_cmd in (None, "failed") else filter_cmd

    rows = registry.get_history(
        command_name=command_name,
        since=since_dt,
        build_id=build_id,
        limit=limit,
        failed_only=failed_only,
    )

    if not rows:
        console.print("\n[dim]No history entries found.[/dim]\n")
        return

    console.print()
    console.print(_render_history_table(rows))
    console.print(
        f"  [dim]Showing {len(rows)} {'entry' if len(rows)==1 else 'entries'}[/dim]\n"
    )


# ------------------------------------------------------------
# history delete
# ------------------------------------------------------------

@history.command("delete")
@click.argument("entry_id")
@click.pass_context
def history_delete(ctx, entry_id):

    registry = _get_registry(ctx)

    matches = registry.get_history_by_prefix(entry_id)

    if not matches:
        console.print(f"\n[yellow]No entry found with ID:[/yellow] {entry_id}\n")
        return

    if len(matches) > 1:
        console.print("\n[red]Ambiguous ID. Matches:[/red]")
        for m in matches:
            console.print(f"  {m['id'][:8]}  {m['raw_command']}")
        console.print()
        return

    entry = matches[0]

    registry.delete_history(entry["id"])

    console.print(
        f"\n[green]✓ Deleted:[/green] {entry['raw_command']} "
        f"[dim]({entry['id'][:8]})[/dim]\n"
    )


# ------------------------------------------------------------
# history clear
# ------------------------------------------------------------

@history.command("clear")
@click.pass_context
def history_clear(ctx):

    registry = _get_registry(ctx)

    count = registry.count_history()

    if count == 0:
        console.print("\n[dim]No history entries to clear.[/dim]\n")
        return

    confirmed = click.confirm(
        f"\nThis will delete {count} history "
        f"{'entry' if count == 1 else 'entries'}. Continue?"
    )

    if not confirmed:
        console.print("\n[dim]Cancelled.[/dim]\n")
        return

    deleted = registry.clear_history()

    console.print(
        f"\n[green]✓ Cleared {deleted} "
        f"{'entry' if deleted == 1 else 'entries'}[/green]\n"
    )