"""
hylos projects -- manage Hylos Cloud projects.
"""
import os
import click
import httpx
from rich.console import Console
from rich.table import Table
from .login import get_valid_access_token, ConfigManager

console = Console()

SUPABASE_URL = os.getenv("HYLOS_SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("HYLOS_SUPABASE_ANON_KEY")


@click.group(name="projects")
def projects():
    """Manage Hylos Cloud projects."""
    pass


@projects.command(name="create")
@click.argument("name")
def projects_create(name):
    """Create a new Hylos project."""
    try:
        access_token = get_valid_access_token()
    except SystemExit:
        raise

    try:
        response = httpx.post(
            f"{SUPABASE_URL}/rest/v1/projects",
            headers={
                "apikey": SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "Prefer": "return=representation",
            },
            json={"name": name},
            timeout=10,
        )
        response.raise_for_status()
        project = response.json()[0]

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            console.print("[red]Auth expired. Run[/red] [bold]hylos login[/bold]")
        else:
            console.print(f"[red]Failed to create project: {e.response.status_code}[/red]")
        raise SystemExit(1)
    except httpx.RequestError:
        console.print("[red]Could not reach Hylos Cloud.[/red]")
        raise SystemExit(1)

    # Store project_id in config (non-sensitive)
    config = ConfigManager.load()
    config["project_id"] = project["id"]
    ConfigManager.save(config)

    console.print(f"\n[green]Project created ✓[/green]")
    console.print(f"[dim]  name:    {project['name']}[/dim]")
    console.print(f"  api_key: [bold]{project['api_key']}[/bold]")
    console.print(f"\n[dim]Project ID saved to ~/.hylos/config.json[/dim]")
    console.print(f"[dim]Use this API key when initializing the Hylos SDK.[/dim]\n")


@projects.command(name="list")
def projects_list():
    """List your Hylos projects."""
    try:
        access_token = get_valid_access_token()
    except SystemExit:
        raise

    try:
        response = httpx.get(
            f"{SUPABASE_URL}/rest/v1/projects",
            headers={
                "apikey": SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {access_token}",
            },
            params={"select": "id,name,api_key,created_at"},
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()

    except httpx.RequestError:
        console.print("[red]Could not reach Hylos Cloud.[/red]")
        raise SystemExit(1)

    if not result:
        console.print("[yellow]No projects yet. Run[/yellow] [bold]hylos projects create <name>[/bold]")
        return

    config = ConfigManager.load()
    current_project_id = config.get("project_id")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Name")
    table.add_column("API Key")
    table.add_column("Created")

    for p in result:
        name_display = p["name"]
        if p["id"] == current_project_id:
            name_display = f"[green]{p['name']} ← active[/green]"
        api_key_display = p["api_key"][:12] + "..."
        table.add_row(name_display, api_key_display, p["created_at"][:10])

    console.print(table)


@projects.command(name="use")
@click.argument("project_id")
def projects_use(project_id):
    """Set the active project by ID."""
    config = ConfigManager.load()
    config["project_id"] = project_id
    ConfigManager.save(config)
    console.print(f"[green]Active project set ✓[/green]")