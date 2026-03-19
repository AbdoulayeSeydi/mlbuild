from __future__ import annotations

import sys
import click
from rich.console import Console

from ...core.budget import (
    load_budget,
    save_budget,
    clear_budget,
    clear_constraint,
    budget_is_empty,
    format_budget_display,
    merge_constraints,
    constraint_origin,
    KEY_MAP,
    KEYS,
    DISPLAY_MAP,
)

console = Console()


# ================================================================
# Helpers
# ================================================================

def _load_or_exit() -> dict:
    try:
        return load_budget()
    except Exception as e:
        console.print(f"\n[red]Invalid .mlbuild/budget.toml:[/red] {e}")
        console.print("[dim]Run: mlbuild budget clear to reset[/dim]\n")
        sys.exit(1)


def _print_no_budget():
    console.print("\n[yellow]No budget set.[/yellow]")
    console.print("[dim]Run: mlbuild budget set --max-latency <ms>[/dim]\n")


# ================================================================
# CLI Group
# ================================================================

@click.group(invoke_without_command=True)
@click.pass_context
def budget(ctx):
    """Manage persistent performance constraints."""
    if ctx.invoked_subcommand is None:
        b = _load_or_exit()

        if budget_is_empty(b):
            _print_no_budget()
            return

        console.print("\n[bold]Budget[/bold] — .mlbuild/budget.toml\n")
        console.print(format_budget_display(b))
        console.print(
            "\n[dim]Applied automatically by mlbuild validate and mlbuild ci.[/dim]"
            "\n[dim]Explicit flags always override budget values.[/dim]\n"
        )


# ================================================================
# SET
# ================================================================

@budget.command("set")
@click.option("--max-latency", type=float)
@click.option("--max-p95",     type=float)
@click.option("--max-memory",  type=float)
@click.option("--max-size",    type=float)
def budget_set(max_latency, max_p95, max_memory, max_size):
    """Set or update performance budget constraints."""

    if all(v is None for v in [max_latency, max_p95, max_memory, max_size]):
        console.print("\n[red]At least one constraint required.[/red]")
        console.print("[dim]Example: mlbuild budget set --max-latency 10 --max-size 8[/dim]\n")
        sys.exit(1)

    existing = _load_or_exit()

    # ✅ Use KEY_MAP instead of hardcoding internal keys
    incoming = {
        KEY_MAP["max-latency"]: max_latency,
        KEY_MAP["max-p95"]:     max_p95,
        KEY_MAP["max-memory"]:  max_memory,
        KEY_MAP["max-size"]:    max_size,
    }

    # ✅ Use core merge logic (single source of truth)
    merged = merge_constraints(incoming, existing)

    try:
        save_budget(merged)
    except ValueError as e:
        console.print(f"\n[red]Invalid constraint:[/red] {e}\n")
        sys.exit(1)

    console.print("\n[green]✓ Budget updated[/green]\n")
    console.print(format_budget_display(merged))
    console.print(
        "\n[dim]Saved to .mlbuild/budget.toml[/dim]"
        "\n[dim]Tip: commit this file so your team enforces the same constraints.[/dim]\n"
    )


# ================================================================
# SHOW
# ================================================================

@budget.command("show")
def budget_show():
    """Show current budget constraints."""
    b = _load_or_exit()

    if budget_is_empty(b):
        _print_no_budget()
        return

    console.print("\n[bold]Budget[/bold] — .mlbuild/budget.toml\n")
    console.print(format_budget_display(b))
    console.print(
        "\n[dim]Applied automatically by:[/dim]"
        "\n[dim]  mlbuild validate <build_id>[/dim]"
        "\n[dim]  mlbuild ci --model model.onnx --baseline <tag>[/dim]"
        "\n[dim]Explicit flags always override budget values.[/dim]\n"
    )


# ================================================================
# CLEAR
# ================================================================

@budget.command("clear")
@click.option(
    "--constraint",
    type=click.Choice(list(KEY_MAP.keys())),
    help="Remove one specific constraint instead of all."
)
def budget_clear(constraint):
    """Clear all budget constraints or remove one."""

    if constraint:
        try:
            clear_constraint(constraint)
        except FileNotFoundError:
            console.print("\n[yellow]No budget file to modify.[/yellow]\n")
            return
        except KeyError as e:
            console.print(f"\n[red]{e}[/red]\n")
            sys.exit(1)

        b = _load_or_exit()  # ✅ fixed

        label, _ = DISPLAY_MAP[KEY_MAP[constraint]]
        console.print(f"\n[green]✓ Removed {label} from budget[/green]\n")

        if budget_is_empty(b):
            console.print("[dim]Budget is now empty.[/dim]\n")
        else:
            console.print(format_budget_display(b))
            console.print()

        return

    # Clear everything
    b = _load_or_exit()

    if budget_is_empty(b):
        console.print("\n[yellow]No budget to clear.[/yellow]\n")
        return

    confirmed = click.confirm(
        "Remove ALL budget constraints?",
        default=False
    )

    if not confirmed:
        console.print("\n[dim]Cancelled.[/dim]\n")
        return

    clear_budget()
    console.print("\n[green]✓ Budget cleared[/green]\n")


# ================================================================
# VALIDATE (Preview)
# ================================================================

@budget.command("validate")
@click.argument("build_id")
def budget_validate(build_id):
    """Preview which constraints would apply to a build."""

    from ...registry import LocalRegistry

    b = _load_or_exit()

    if budget_is_empty(b):
        _print_no_budget()
        return

    # ✅ Safe registry handling
    try:
        registry = LocalRegistry()
        build = registry.resolve_build(build_id)
    except Exception as e:
        console.print(f"\n[red]Registry error:[/red] {e}\n")
        sys.exit(1)

    if not build:
        console.print(f"\n[red]Build not found: {build_id}[/red]\n")
        sys.exit(1)

    if build.size_mb is None:
        console.print("\n[red]Build missing size information[/red]\n")
        sys.exit(1)

    size_mb = float(build.size_mb)

    console.print(
        f"\n[bold]Budget preview[/bold] — "
        f"{build.build_id[:8]} ({build.name or 'unnamed'}, {size_mb:.2f} MB)\n"
    )

    rows = []

    for key in KEYS:
        label, unit = DISPLAY_MAP[key]
        val = b.get(key)

        origin = constraint_origin(key, {}, b)

        if val is None:
            rows.append((label, "not set", "[dim]— skipped[/dim]"))
            continue

        if key == "max_size_mb":
            if size_mb <= val:
                status = f"[green]✓ passes[/green] ({size_mb:.2f} MB)"
            else:
                status = f"[red]✗ will fail[/red] ({size_mb:.2f} MB > {val:.1f} MB)"
        else:
            status = "[yellow]⚠ unknown (needs benchmark)[/yellow]"

        rows.append(
            (label, f"{val:.1f} {unit}", f"{status} [dim]({origin})[/dim]")
        )

    col1 = max(len(r[0]) for r in rows)
    col2 = max(len(r[1]) for r in rows)

    for label, limit, status in rows:
        console.print(f"  {label:<{col1}}  {limit:<{col2}}  {status}")

    console.print(
        f"\n[dim]Run 'mlbuild validate {build.build_id[:8]}' to enforce all constraints.[/dim]\n"
    )