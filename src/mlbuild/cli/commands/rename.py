from __future__ import annotations

import click
from rich.console import Console

from mlbuild.registry.local import LocalRegistry

console = Console(width=None)


def _validate_name(name: str) -> str:
    name = name.strip()

    if not name:
        raise click.ClickException("Name cannot be empty.")

    if len(name) > 64:
        raise click.ClickException("Name cannot exceed 64 characters.")

    return name


@click.command(help="Rename a build.")
@click.argument("build_id")
@click.argument("new_name")
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="Skip confirmation prompt.",
)
def rename(build_id: str, new_name: str, yes: bool) -> None:
    registry = LocalRegistry()

    validated_name = _validate_name(new_name)

    # Resolve build
    try:
        build = registry.resolve_build(build_id)
    except Exception as e:
        # You don't have structured errors yet, so don't lie about it
        raise click.ClickException(f"Failed to resolve build: {e}")

    # If your API returns None, handle it explicitly
    if build is None:
        raise click.ClickException(f"Build not found: {build_id}")

    # No-op case
    if build.name == validated_name:
        console.print(
            f"[yellow]No change:[/yellow] already named '{validated_name}'"
        )
        return

    # Confirmation
    if not yes:
        click.confirm(
            f"Rename '{build.build_id[:8]}' "
            f"from '{build.name or '(unnamed)'}' → '{validated_name}'?",
            abort=True,
        )

    # Rename
    try:
        registry.rename_build(build.build_id, validated_name)
    except Exception as e:
        # Still unstructured, so be honest
        raise click.ClickException(f"Failed to rename build: {e}")

    console.print(
        f"[green]✓ Renamed[/green] {build.build_id[:8]} "
        f"[dim]{build.name or '(unnamed)'} → {validated_name}[/dim]"
    )