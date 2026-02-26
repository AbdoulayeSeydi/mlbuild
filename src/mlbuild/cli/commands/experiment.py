"""
Enterprise-grade experiment management CLI.
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from dataclasses import asdict
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from ...experiments.manager import ExperimentManager

from pathlib import Path

console = Console()


# ------------------------------------------------------------
# Exit Codes (CI-safe)
# ------------------------------------------------------------

EXIT_SUCCESS = 0
EXIT_VALIDATION_ERROR = 2
EXIT_NOT_FOUND = 3
EXIT_CONFLICT = 4
EXIT_PERMISSION = 5
EXIT_INFRASTRUCTURE = 10
EXIT_UNKNOWN = 99


# ------------------------------------------------------------
# CLI Context (Dependency Injection + Global Options)
# ------------------------------------------------------------

class CLIContext:
    def __init__(self, config: Optional[str], verbose: bool, json_output: bool):
        self.verbose = verbose
        self.json_output = json_output

        # Use LocalRegistry with project-local DB
        from ...registry import LocalRegistry
        registry = LocalRegistry()  # Uses .mlbuild/registry.db in current dir
        self.manager = ExperimentManager(registry)

        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )


pass_context = click.make_pass_decorator(CLIContext)


@click.group()
@click.option("--config", type=click.Path(exists=True), help="Path to config file")
@click.option("--verbose", is_flag=True, help="Enable verbose debugging output")
@click.option("--json", "json_output", is_flag=True, help="Machine-readable JSON output")
@click.pass_context
def experiment(ctx, config, verbose, json_output):
    """
    Manage experiments (enterprise-grade).
    """
    ctx.obj = CLIContext(config, verbose, json_output)


# ------------------------------------------------------------
# Error Handling Wrapper
# ------------------------------------------------------------

def handle_error(e: Exception, verbose: bool):
    """Simple error handler without custom exception types."""
    console.print(f"[red]Error:[/red] {e}")
    
    if verbose:
        import traceback
        traceback.print_exc()
    
    # Exit with generic error code
    sys.exit(1)

# ------------------------------------------------------------
# Create Experiment
# ------------------------------------------------------------

@experiment.command("create")
@click.argument("name")
@click.option("--description", help="Experiment description")
@pass_context
def create_experiment(ctx: CLIContext, name: str, description: Optional[str] = None):
    """Create a new experiment."""
    try:
        existing = ctx.manager.get_experiment(name)
        if existing:
            console.print(f"[yellow]Experiment '{name}' already exists[/yellow]")
            return

        exp = ctx.manager.create_experiment(name=name, description=description)

        console.print(f"[green]✓ Created experiment[/green]")
        console.print(f"  Name: {exp['name']}")
        console.print(f"  ID:   {exp['experiment_id']}")

    except Exception as e:
        handle_error(e, ctx.verbose)

# ------------------------------------------------------------
# List Experiments (Pagination + Sorting + Filtering)
# ------------------------------------------------------------

@experiment.command("list")
@click.option("--limit", default=50, type=click.IntRange(1, 1000))
@click.option("--sort-by", type=click.Choice(["name", "created_at"]), default="created_at")
@click.option("--descending", is_flag=True)
@click.option("--include-deleted", is_flag=True, help="Include soft-deleted experiments")
@click.option("--name-prefix", help="Filter by name prefix")
@pass_context
def list_experiments(
    ctx: CLIContext,
    limit: int,
    sort_by: str,
    descending: bool,
    include_deleted: bool,
    name_prefix: Optional[str],
):
    try:
        experiments = ctx.manager.list_experiments(
            limit=limit,
            sort_by=sort_by,
            descending=descending,
            include_deleted=include_deleted,
            name_prefix=name_prefix,
        )

        if ctx.json_output:
            console.print(json.dumps([asdict(e) for e in experiments], default=str))
            sys.exit(EXIT_SUCCESS)

        if not experiments:
            console.print("[yellow]No experiments found[/yellow]")
            sys.exit(EXIT_SUCCESS)

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Name", style="green", overflow="fold")
        table.add_column("ID", style="dim")
        table.add_column("Created", style="dim")
        table.add_column("Deleted", style="red")

        for exp in experiments:
            table.add_row(
                exp["name"],
                str(exp["experiment_id"]),
                exp["created_at"].strftime("%Y-%m-%d %H:%M"),
                exp["deleted_at"].strftime("%Y-%m-%d %H:%M") if exp["deleted_at"] else "",
            )

        console.print(table)
        sys.exit(EXIT_SUCCESS)

    except Exception as e:
        handle_error(e, ctx.verbose)


# ------------------------------------------------------------
# Delete Experiment (ID-based + Confirmation + Soft Delete Explicit)
# ------------------------------------------------------------

@experiment.command("delete")
@click.argument("experiment_id")
@click.option("--yes", is_flag=True, help="Confirm deletion without prompt")
@click.option(
    "--hard",
    is_flag=True,
    help="Permanently delete (irreversible). Default is soft-delete.",
)
@pass_context
def delete_experiment(ctx: CLIContext, experiment_id: str, yes: bool, hard: bool):
    try:
        if not yes:
            click.confirm(
                f"Are you sure you want to {'permanently ' if hard else ''}delete experiment {experiment_id}?",
                abort=True,
            )

        # Look up by name first, fall back to treating input as UUID
        experiments = ctx.manager.list_experiments(limit=1000, include_deleted=False)
        match = next((e for e in experiments if e["name"] == experiment_id), None)

        if match:
            target_id = match["experiment_id"]
        else:
            import uuid
            try:
                target_id = uuid.UUID(experiment_id)
            except ValueError:
                console.print(f"[red]Error:[/red] No experiment found with name or ID '{experiment_id}'")
                sys.exit(EXIT_NOT_FOUND)

        if hard:
            ctx.manager.hard_delete_experiment(target_id)
        else:
            ctx.manager.soft_delete_experiment(target_id)

        console.print(
            f"[green]✓ {'Hard' if hard else 'Soft'} deleted experiment[/green] {experiment_id}"
        )

        sys.exit(EXIT_SUCCESS)

    except Exception as e:
        handle_error(e, ctx.verbose)