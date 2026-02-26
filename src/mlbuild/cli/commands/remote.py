"""
Remote storage management commands (enterprise-grade).
"""

from __future__ import annotations

import json
import re
import sys
import traceback
from pathlib import Path
from typing import Dict

import click
from rich.console import Console
from rich.table import Table

from ...storage import (
    RemoteRepository,
    RemoteConfig,
    Backend,
    ValidationError,
    NotFoundError,
    ConfigError,
)

console = Console()

# ---------------------------------------------------------------------
# Exit Codes (structured)
# ---------------------------------------------------------------------

EXIT_USER_ERROR = 1
EXIT_CONFIG_ERROR = 2
EXIT_SYSTEM_ERROR = 3

NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


# ---------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------

def handle_error(exc: Exception, debug: bool) -> None:
    if isinstance(exc, ValidationError):
        console.print(f"[red]Validation error:[/red] {exc}")
        sys.exit(EXIT_USER_ERROR)

    if isinstance(exc, NotFoundError):
        console.print(f"[red]Not found:[/red] {exc}")
        sys.exit(EXIT_USER_ERROR)

    if isinstance(exc, ConfigError):
        console.print(f"[red]Configuration error:[/red] {exc}")
        sys.exit(EXIT_CONFIG_ERROR)

    # Unexpected error
    if debug:
        console.print("[red]Unexpected error:[/red]")
        traceback.print_exc()
    else:
        console.print(f"[red]Unexpected system error:[/red] {exc}")
        console.print("Run with --debug for traceback.")

    sys.exit(EXIT_SYSTEM_ERROR)


# ---------------------------------------------------------------------
# CLI Root
# ---------------------------------------------------------------------

@click.group()
@click.option("--debug", is_flag=True, help="Show full tracebacks on errors.")
@click.pass_context
def remote(ctx: click.Context, debug: bool):
    """Manage remote storage backends."""
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def validate_name(name: str) -> None:
    if not NAME_PATTERN.match(name):
        raise ValidationError(
            "Remote name must match: [a-zA-Z0-9_-] and be <= 64 characters."
        )


def validate_backend_options(
    backend: str,
    bucket: str | None,
    path: str | None,
):
    if backend == "s3":
        if not bucket:
            raise ValidationError("--bucket is required for S3 backend.")
        if path:
            raise ValidationError("--path cannot be used with S3 backend.")

    if backend == "local":
        if not path:
            raise ValidationError("--path is required for local backend.")
        if bucket:
            raise ValidationError("--bucket cannot be used with local backend.")


def format_location(config: RemoteConfig) -> str:
    if config.backend == Backend.S3:
        prefix = config.prefix or ""
        return f"{config.bucket}/{prefix}"
    return str(Path(config.path).resolve())


def config_to_dict(name: str, config: RemoteConfig) -> Dict:
    return {
        "name": name,
        "backend": config.backend.value,
        "location": format_location(config),
        "default": config.default,
    }


# ---------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------

@remote.command("add")
@click.argument("name")
@click.option("--backend", type=click.Choice(["s3", "local"]), required=True)
@click.option("--bucket")
@click.option("--prefix", default="mlbuild/")
@click.option("--region")
@click.option("--endpoint")
@click.option("--path", type=click.Path())
@click.option("--default", "set_default_flag", is_flag=True)
@click.pass_context
def add_remote(
    ctx: click.Context,
    name,
    backend,
    bucket,
    prefix,
    region,
    endpoint,
    path,
    set_default_flag,
):
    """Add a new remote storage backend."""
    debug = ctx.obj["debug"]

    try:
        validate_name(name)
        validate_backend_options(backend, bucket, path)

        repo = RemoteRepository()

        config = RemoteConfig(
            name=name,
            backend=Backend(backend),
            bucket=bucket,
            prefix=prefix,
            region=region,
            endpoint_url=endpoint,
            path=Path(path) if path else None,
            default=False,  # Always false on creation
        )

        repo.add(config)

        if set_default_flag:
            repo.set_default(name)

        console.print(f"[green]✓ Added remote '{name}'[/green]")
        if set_default_flag:
            console.print("[yellow]Set as default[/yellow]")

    except Exception as e:
        handle_error(e, debug)


# ---------------------------------------------------------------------

@remote.command("list")
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
@click.pass_context
def list_remotes(ctx: click.Context, as_json: bool):
    """List all configured remotes."""
    debug = ctx.obj["debug"]

    try:
        repo = RemoteRepository()
        remotes = repo.load()

        if not remotes:
            if as_json:
                console.print(json.dumps([]))
            else:
                console.print("[yellow]No remotes configured[/yellow]")
            return

        # Deterministic ordering
        sorted_names = sorted(remotes.keys())

        if as_json:
            output = [
                config_to_dict(name, remotes[name])
                for name in sorted_names
            ]
            console.print(json.dumps(output, indent=2))
            return

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Name", style="green")
        table.add_column("Backend")
        table.add_column("Location")
        table.add_column("Default", justify="center")

        for name in sorted_names:
            config = remotes[name]
            table.add_row(
                name,
                config.backend.value,
                format_location(config),
                "✓" if config.default else "",
            )

        console.print(table)

    except Exception as e:
        handle_error(e, debug)


# ---------------------------------------------------------------------

@remote.command("remove")
@click.argument("name")
@click.option("--yes", is_flag=True)
@click.pass_context
def remove_remote(ctx: click.Context, name, yes):
    """Remove a remote storage backend."""
    debug = ctx.obj["debug"]

    try:
        validate_name(name)

        repo = RemoteRepository()
        config = repo.get(name)

        if config.default:
            console.print("[yellow]Warning: removing the default remote.[/yellow]")

        if not yes:
            click.confirm(f"Remove remote '{name}'?", abort=True)

        repo.remove(name)

        console.print(f"[green]✓ Removed remote '{name}'[/green]")

    except Exception as e:
        handle_error(e, debug)


# ---------------------------------------------------------------------

@remote.command("set-default")
@click.argument("name")
@click.pass_context
def set_default(ctx: click.Context, name):
    """Set a remote as the default."""
    debug = ctx.obj["debug"]

    try:
        validate_name(name)

        repo = RemoteRepository()
        repo.set_default(name)

        console.print(f"[green]✓ Set '{name}' as default remote[/green]")

    except Exception as e:
        handle_error(e, debug)


# ---------------------------------------------------------------------

@remote.command("show")
@click.argument("name")
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def show_remote(ctx: click.Context, name, as_json: bool):
    """Show detailed configuration for a remote."""
    debug = ctx.obj["debug"]

    try:
        validate_name(name)

        repo = RemoteRepository()
        config = repo.get(name)

        if as_json:
            console.print(json.dumps(config_to_dict(name, config), indent=2))
            return

        console.print(f"\n[bold cyan]Remote: {name}[/bold cyan]")
        console.print(f"Backend: {config.backend.value}")
        console.print(f"Location: {format_location(config)}")
        console.print(f"Default: {'Yes' if config.default else 'No'}")
        console.print()

    except Exception as e:
        handle_error(e, debug)