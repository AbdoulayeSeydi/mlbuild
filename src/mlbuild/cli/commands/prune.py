from __future__ import annotations

import sys
import click
from rich.console import Console
from rich.table import Table
from rich import box


from mlbuild.cli.formatters.utils import parse_duration, relative_time

console = Console()


# ── Helpers ───────────────────────────────────────────────────

def _fmt_bytes(b: int) -> str:
    if b < 1024 * 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b / (1024 * 1024):.1f} MB"


def _validate_keep_last(value: int | None):
    if value is not None and value < 0:
        console.print("\n[red]Error:[/red] --keep-last must be >= 0\n")
        sys.exit(1)


def _error(msg: str):
    console.print(f"\n[red]Error:[/red] {msg}\n")
    sys.exit(1)


# ── Command ───────────────────────────────────────────────────

@click.command(help="Remove old builds from the local registry.")
@click.option("--keep-last", type=int, default=None, help="Keep N most recent builds globally.")
@click.option("--older-than", type=str, default=None, help="Prune builds older than: 30d, 7d, 24h.")
@click.option("--tag", type=str, default=None, help="Prune builds with this exact tag (case-sensitive).")
@click.option("--dry-run", is_flag=True, help="Show what would be pruned without touching anything.")
@click.option("--force", is_flag=True, help="Skip confirmation prompt.")
@click.option("--purge", is_flag=True, help="Hard delete rows and artifact files. Requires --force.")
def prune(keep_last, older_than, purge, dry_run, force, tag):
    # ── Validation ────────────────────────────────────────────

    _validate_keep_last(keep_last)

    if purge and not force:
        _error("--purge requires --force. This operation is irreversible.\nRe-run with: mlbuild prune --purge --force")

    # Normalize tag
    tag = tag.strip() if tag else None

    # Parse duration (shared util, not CLI logic)
    older_than_days: float | None = None
    if older_than:
        try:
            older_than_days = parse_duration(older_than)
        except ValueError as e:
            _error(str(e))

    # ── Load Plan ─────────────────────────────────────────────

    try:
        from mlbuild.registry.local import LocalRegistry
        registry = LocalRegistry()
        plan = registry.get_prune_plan(
            keep_last=keep_last,
            older_than_days=older_than_days,
            tag=tag,
        )
    except Exception as e:
        _error(f"Error loading prune plan: {e}")

    if not plan.candidates:
        console.print("\n  Nothing to prune.\n")
        return

    build_ids = [c.id for c in plan.candidates]

    n_candidates = len(plan.candidates)
    n_skipped = len(plan.skipped)

    skipped_note = f"  ([dim]{n_skipped} skipped — protected[/dim])" if n_skipped else ""

    # ── Dry Run ───────────────────────────────────────────────

    if dry_run:
        console.print()
        console.print(f"  [bold]{n_candidates} builds match[/bold]{skipped_note}")
        console.print()

        table = Table(box=box.SIMPLE, show_header=True, pad_edge=False)
        table.add_column("ID",     style="bold", min_width=10)
        table.add_column("Name",   min_width=18)
        table.add_column("Format", min_width=10)
        table.add_column("Size",   min_width=10, justify="right")
        table.add_column("Age",    min_width=12)

        for c in plan.candidates:
            table.add_row(
                c.id_short,
                c.name or "—",
                c.primary_format or "—",   # defensive fallback
                f"{c.size_mb:.2f} MB",
                relative_time(c.created_at),
            )

        console.print(table)

        total_mb = sum(c.size_mb for c in plan.candidates)

        console.print(f"  Estimated disk reclaim:  {total_mb:.1f} MB  (if run with --purge)")

        # Optional: show skipped builds for transparency
        if plan.skipped:
            console.print("\n  [dim]Skipped (protected):[/dim]")
            for s in plan.skipped[:5]:
                console.print(f"    {s.id_short}  {s.name or '—'}")
            if len(plan.skipped) > 5:
                console.print(f"    ... and {len(plan.skipped) - 5} more")

        console.print("  Run without --dry-run to apply.\n")
        return

    # ── Confirm ───────────────────────────────────────────────

    action = "hard-delete" if purge else "soft-delete"

    if not force:
        confirm = click.confirm(
            f"\n  This will {action} {n_candidates} builds. Continue?",
            default=False,
        )
        if not confirm:
            console.print("  Cancelled.\n")
            return

    # ── Execute ───────────────────────────────────────────────

    if purge:
        console.print("\n  [yellow]Hard deleting builds and artifact files. This is irreversible.[/yellow]\n")

        try:
            result = registry.hard_delete_builds(plan.candidates)
        except Exception as e:
            _error(f"Error during hard delete: {e}")

        file_note = (
            f"  ({result.file_errors} failed — file not found)"
            if result.file_errors else ""
        )

        console.print(f"  [green]✓[/green] Pruned {result.builds_deleted} builds  ([dim]{n_skipped} skipped — protected[/dim])")
        console.print(f"  [green]✓[/green] Deleted {result.files_deleted} artifact files{file_note}")
        console.print(f"  Disk reclaimed:  {_fmt_bytes(result.bytes_reclaimed)}\n")

    else:
        try:
            deleted = registry.soft_delete_builds(build_ids)
        except Exception as e:
            _error(f"Error during soft delete: {e}")

        console.print(f"\n  [green]✓[/green] Pruned {deleted} builds  ([dim]{n_skipped} skipped — protected[/dim])")
        console.print("  Builds hidden from all queries. Run with --purge to reclaim disk.\n")