from __future__ import annotations

import click
from rich.console import Console

from mlbuild.registry.local import LocalRegistry

console = Console()

PIN_TAG = "mlbuild-pinned"


# ---------- Helpers (no duplication, no DB leakage) ----------

def _get_build(registry: LocalRegistry, build_id: str):
    try:
        return registry.resolve_build(build_id)
    except Exception as e:
        raise click.ClickException(f"Failed to resolve build: {e}")


def _ensure_build_exists(build, build_id: str):
    # If your API still returns None, we handle it.
    if build is None:
        raise click.ClickException(f"Build not found: {build_id}")
    return build


def _is_pinned(registry: LocalRegistry, build_id: str) -> bool:
    try:
        return registry.is_tagged(build_id, PIN_TAG)
    except AttributeError:
        # Hard fail — this is required for clean architecture
        raise click.ClickException(
            "Registry missing required method: is_tagged(build_id, tag)"
        )
    except Exception as e:
        raise click.ClickException(f"Failed to check pin status: {e}")


def _fmt_build(build) -> str:
    return f"{build.build_id[:8]} {build.name or '(unnamed)'}"


# ---------- CLI ----------

@click.group(invoke_without_command=True)
@click.argument("build_id")
@click.pass_context
def pin(ctx, build_id: str):
    """Pin a build to protect it from pruning."""
    if ctx.invoked_subcommand:
        return

    registry = LocalRegistry()

    build = _ensure_build_exists(
        _get_build(registry, build_id),
        build_id,
    )

    if _is_pinned(registry, build.build_id):
        console.print(f"[yellow]Already pinned:[/yellow] {_fmt_build(build)}")
        return

    try:
        registry.add_tag(build.build_id, PIN_TAG)
    except Exception as e:
        raise click.ClickException(f"Failed to pin build: {e}")

    console.print(f"[green]✓ Pinned[/green] {_fmt_build(build)}")
    console.print("[dim]Protected from pruning.[/dim]")


@click.command()
@click.argument("build_id")
def unpin(build_id: str):
    """Unpin a build."""
    registry = LocalRegistry()

    build = _ensure_build_exists(
        _get_build(registry, build_id),
        build_id,
    )

    if not _is_pinned(registry, build.build_id):
        console.print(f"[yellow]Not pinned:[/yellow] {_fmt_build(build)}")
        return

    try:
        registry.remove_tag(build.build_id, PIN_TAG)
    except TypeError:
        # catches your current broken signature
        raise click.ClickException(
            "remove_tag must accept (build_id, tag). Fix your registry API."
        )
    except Exception as e:
        raise click.ClickException(f"Failed to unpin build: {e}")

    console.print(f"[green]✓ Unpinned[/green] {_fmt_build(build)}")