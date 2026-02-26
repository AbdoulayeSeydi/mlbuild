"""
Tag management: Git-like tags for builds.
"""

import re
import click
from click import ClickException
from rich.console import Console
from rich.table import Table
from datetime import datetime, timezone

from ...registry import LocalRegistry

console = Console()

# ---------------------------
# Constants
# ---------------------------

TAG_REGEX = re.compile(r"^[a-zA-Z0-9._-]+$")
MAX_TAG_LENGTH = 64
BUILD_ID_DISPLAY_LENGTH = 12


def validate_tag_name(tag_name: str):
    if len(tag_name) > MAX_TAG_LENGTH:
        raise ClickException(
            f"Tag name exceeds {MAX_TAG_LENGTH} characters."
        )

    if not TAG_REGEX.match(tag_name):
        raise ClickException(
            "Invalid tag name. Allowed: letters, numbers, ., _, -"
        )


# ---------------------------
# CLI Group
# ---------------------------

@click.group()
def tag():
    """Manage build tags (similar to git tag)."""
    pass


# ============================================================
# CREATE
# ============================================================

@tag.command("create")
@click.argument("build_ref")
@click.argument("tag_name")
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite tag if it already exists.",
)
def create_tag(build_ref: str, tag_name: str, force: bool):
    """
    Create or update a tag pointing to a build.

    Examples:
        mlbuild tag create febc5f7 v1.0.0
        mlbuild tag create latest production --force
    """
    registry = LocalRegistry()

    validate_tag_name(tag_name)

    # Resolve build
    build = registry.resolve_build(build_ref)
    
    if not build:
        raise ClickException(f"Build not found: {build_ref}")
    
    # Check if tag already exists
    if not force:
        with registry._connect() as conn:
            existing = conn.execute(
                "SELECT build_id FROM tags WHERE tag = ?",
                (tag_name,)
            ).fetchone()
            
            if existing:
                raise ClickException(
                    f"Tag '{tag_name}' already exists. Use --force to overwrite."
                )
    
    # Create/update tag
    registry.add_tag(build.build_id, tag_name)

    console.print(
        f"[green]✓[/green] Tagged "
        f"{build.build_id[:BUILD_ID_DISPLAY_LENGTH]} "
        f"as '{tag_name}'"
    )


# ============================================================
# LIST
# ============================================================

@tag.command("list")
@click.option("--limit", default=50, type=int, show_default=True)
def list_tags(limit: int):
    """
    List all tags.

    Examples:
        mlbuild tag list
        mlbuild tag list --limit 20
    """
    registry = LocalRegistry()

    with registry._connect() as conn:
        rows = conn.execute(
            """
            SELECT tags.tag, tags.build_id, tags.created_at, builds.name as build_name
            FROM tags
            LEFT JOIN builds ON tags.build_id = builds.build_id
            ORDER BY tags.created_at DESC
            LIMIT ?
            """,
            (limit,)
        ).fetchall()

    if not rows:
        console.print("[yellow]No tags found[/yellow]")
        return

    console.print(f"\n[bold]Tags[/bold] ({len(rows)} shown)\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Tag", style="green")
    table.add_column("Build ID", style="cyan")
    table.add_column("Build Name")
    table.add_column("Created", style="dim")

    for row in rows:
        tag_name = row['tag']
        build_id = row['build_id']
        build_name = row['build_name'] or "(unnamed)"
        created = row['created_at']

        build_display = build_id[:BUILD_ID_DISPLAY_LENGTH] + "..."

        table.add_row(
            tag_name,
            build_display,
            build_name,
            created,
        )

    console.print(table)
    console.print()


# ============================================================
# DELETE
# ============================================================

@tag.command("delete")
@click.argument("tag_name")
def delete_tag(tag_name: str):
    """
    Delete a tag.

    Examples:
        mlbuild tag delete v1.0.0
    """
    registry = LocalRegistry()

    validate_tag_name(tag_name)

    # Check if exists
    with registry._connect() as conn:
        exists = conn.execute(
            "SELECT 1 FROM tags WHERE tag = ?",
            (tag_name,)
        ).fetchone()
        
        if not exists:
            raise ClickException(f"Tag not found: {tag_name}")
        
        # Delete
        conn.execute("DELETE FROM tags WHERE tag = ?", (tag_name,))
        conn.commit()

    console.print(f"[green]✓[/green] Deleted tag '{tag_name}'")


# ============================================================
# SHOW
# ============================================================

@tag.command("show")
@click.argument("tag_name")
def show_tag(tag_name: str):
    """
    Show what a tag points to.

    Examples:
        mlbuild tag show v1.0.0
    """
    registry = LocalRegistry()

    validate_tag_name(tag_name)

    # Get tag and build info
    with registry._connect() as conn:
        row = conn.execute(
            """
            SELECT tags.tag, tags.build_id, tags.created_at, 
                   builds.name, builds.target_device, builds.quantization,
                   builds.size_bytes, builds.created_at as build_created_at
            FROM tags
            LEFT JOIN builds ON tags.build_id = builds.build_id
            WHERE tags.tag = ?
            """,
            (tag_name,)
        ).fetchone()
    
    if not row:
        raise ClickException(f"Tag not found: {tag_name}")
    
    if not row['build_id']:
        raise ClickException(f"Tag '{tag_name}' is orphaned (build missing).")
    
    # Parse quantization JSON
    import json
    quantization = json.loads(row['quantization'])
    
    console.print(f"\n[bold]Tag:[/bold] {tag_name}")
    console.print(f"[bold]Build:[/bold] {row['build_id'][:BUILD_ID_DISPLAY_LENGTH]}...")

    if row['name']:
        console.print(f"[bold]Name:[/bold] {row['name']}")

    console.print(f"[bold]Target:[/bold] {row['target_device']}")
    console.print(f"[bold]Quantization:[/bold] {quantization.get('type', 'unknown')}")
    console.print(f"[bold]Size:[/bold] {row['size_bytes'] / (1024*1024):.2f} MB")
    console.print(f"[bold]Build Created:[/bold] {row['build_created_at']}")
    console.print(f"[bold]Tag Created:[/bold] {row['created_at']}")
    console.print()