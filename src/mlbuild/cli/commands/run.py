"""
Enterprise-grade run tracking CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from ...experiments.manager import (
    ExperimentManager,
    ValidationError,
    NotFoundError,
    ConflictError,
    InfrastructureError,
)
from ...experiments.active_run import ActiveRunStore

console = Console()


# ============================================================
# Manager Factory (Decoupled)
# ============================================================

def get_manager() -> ExperimentManager:
    """
    Factory for ExperimentManager.

    Future-safe:
    - Can inject config
    - Swap backend
    - Add dependency injection
    """
    from ...registry import LocalRegistry
    registry = LocalRegistry()
    return ExperimentManager(registry)


# ============================================================
# Active Run Persistence
# ============================================================

def save_active_run(run_id: str) -> None:
    store = ActiveRunStore()
    store.set(run_id)


def load_active_run() -> Optional[str]:
    store = ActiveRunStore()
    return store.get()


def clear_active_run() -> None:
    store = ActiveRunStore()
    store.clear()


# ============================================================
# Error Handling
# ============================================================

def handle_error(exc: Exception):
    if isinstance(exc, ValidationError):
        console.print(f"[red]Invalid input:[/red] {exc}")
        sys.exit(2)

    if isinstance(exc, NotFoundError):
        console.print(f"[red]Not found:[/red] {exc}")
        sys.exit(3)

    if isinstance(exc, ConflictError):
        console.print(f"[red]Conflict:[/red] {exc}")
        sys.exit(4)

    if isinstance(exc, InfrastructureError):
        console.print("[red]System failure. Check logs.[/red]")
        sys.exit(10)

    console.print(f"[red]Unexpected error:[/red] {exc}")
    sys.exit(1)


# ============================================================
# Run ID Resolution
# ============================================================

def resolve_run_id(
    manager: ExperimentManager,
    explicit_run_id: Optional[str],
) -> str:
    """
    Deterministic run resolution:
    1. Explicit --run-id
    2. Persistent active run file
    3. Fail if none
    """

    run_id = explicit_run_id or load_active_run()

    if not run_id:
        raise ValidationError(
            "No active run. Start a run or specify --run-id."
        )

    # Validate run exists and is active
    run = manager.get_run(run_id)

    if run.status.value != "running":
        raise ConflictError(
            f"Run {run_id} is not active (status: {run.status.value})."
        )

    return run_id


# ============================================================
# CLI Group
# ============================================================

@click.group()
@click.pass_context
def run(ctx):
    """
    Manage experimental runs.
    """
    ctx.obj = get_manager()


# ============================================================
# Start
# ============================================================

@run.command("start")
@click.option("--experiment", required=True)
@click.option("--name")
@click.pass_obj
def start_run(manager: ExperimentManager, experiment: str, name: str):
    try:
        run_obj = manager.start_run(experiment, name)

        save_active_run(str(run_obj.run_id))

        console.print("[green]✓ Started run[/green]")
        console.print(f"  ID: {run_obj.run_id}")
        console.print(f"  Experiment: {experiment}")
        if name:
            console.print(f"  Name: {name}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled.[/yellow]")
        sys.exit(130)

    except Exception as e:
        handle_error(e)


# ============================================================
# Log Param
# ============================================================

@run.command("log-param")
@click.argument("key")
@click.argument("value")
@click.option("--run-id")
@click.pass_obj
def log_param(manager: ExperimentManager, key: str, value: str, run_id: str):
    try:
        target_run_id = resolve_run_id(manager, run_id)
        manager.log_param(target_run_id, key, value)

        console.print(f"[green]✓[/green] Logged param: {key} = {value}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled.[/yellow]")
        sys.exit(130)

    except Exception as e:
        handle_error(e)


# ============================================================
# Log Metric
# ============================================================

@run.command("log-metric")
@click.argument("key")
@click.argument("value", type=float)
@click.option("--run-id")
@click.pass_obj
def log_metric(manager: ExperimentManager, key: str, value: float, run_id: str):
    try:
        target_run_id = resolve_run_id(manager, run_id)
        manager.log_metric(target_run_id, key, value)

        console.print(f"[green]✓[/green] Logged metric: {key} = {value}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled.[/yellow]")
        sys.exit(130)

    except Exception as e:
        handle_error(e)


# ============================================================
# Attach Build
# ============================================================

@run.command("attach-build")
@click.argument("build_id")
@click.option("--run-id")
@click.pass_obj
def attach_build(manager: ExperimentManager, build_id: str, run_id: str):
    try:
        target_run_id = resolve_run_id(manager, run_id)
        manager.attach_build(target_run_id, build_id)

        console.print(f"[green]✓[/green] Attached build: {build_id}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled.[/yellow]")
        sys.exit(130)

    except Exception as e:
        handle_error(e)


# ============================================================
# End
# ============================================================

@run.command("end")
@click.option("--run-id")
@click.option(
    "--status",
    type=click.Choice(["completed", "failed", "cancelled"]),
    default="completed",
)
@click.pass_obj
def end_run(manager: ExperimentManager, run_id: str, status: str):
    try:
        target_run_id = resolve_run_id(manager, run_id)

        manager.end_run(target_run_id, status)

        clear_active_run()

        console.print(f"[green]✓[/green] Ended run ({status})")

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled.[/yellow]")
        sys.exit(130)

    except Exception as e:
        handle_error(e)


# ============================================================
# List
# ============================================================

@run.command("list")
@click.option("--experiment")
@click.option("--limit", default=20, type=click.IntRange(1, 200))
@click.pass_obj
def list_runs(manager: ExperimentManager, experiment: str, limit: int):
    try:
        runs = manager.list_runs(experiment, limit)

        if not runs:
            console.print("[yellow]No runs found[/yellow]")
            return

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Run Name")
        table.add_column("ID", style="dim")
        table.add_column("Status")
        table.add_column("Build")
        table.add_column("Started")

        for run_obj in runs:
            table.add_row(
                run_obj.run_name or "(unnamed)",
                str(run_obj.run_id)[:16] + "...",
                run_obj.status.value,
                run_obj.build_id[:12] + "..." if run_obj.build_id else "-",
                run_obj.started_at.strftime("%Y-%m-%d %H:%M")
                if run_obj.started_at
                else "-",
            )

        console.print(table)

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled.[/yellow]")
        sys.exit(130)

    except Exception as e:
        handle_error(e)