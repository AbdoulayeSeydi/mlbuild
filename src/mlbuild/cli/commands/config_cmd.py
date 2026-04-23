"""
hylos config -- manage local Hylos configuration

- Non-sensitive config only in ~/.hylos/config.json
- Secrets (tokens, api_key) stored in OS keychain
- Prevents corruption + inconsistent auth state
"""

import json
from pathlib import Path
from typing import Optional

import click
import keyring
from rich.console import Console
from rich.table import Table

console = Console()

CONFIG_PATH = Path.home() / ".hylos" / "config.json"
SERVICE_NAME = "hylos-cli"

# Only NON-SENSITIVE keys allowed here
VALID_KEYS = {"project_id"}


# ------------------------
# Config Manager (cached + safe)
# ------------------------

class ConfigManager:
    _cache: Optional[dict] = None
    _corrupted: bool = False

    @classmethod
    def load(cls) -> dict:
        if cls._cache is not None:
            return cls._cache

        if not CONFIG_PATH.exists():
            cls._cache = {}
            return cls._cache

        try:
            data = json.loads(CONFIG_PATH.read_text())
            cls._cache = data
            return data
        except Exception:
            backup = CONFIG_PATH.with_suffix(".corrupt")
            CONFIG_PATH.rename(backup)
            console.print(f"[red]Config corrupted. Backed up to {backup}[/red]")
            cls._cache = {}
            cls._corrupted = True
            return cls._cache

    @classmethod
    def save(cls, data: dict):
        if cls._corrupted:
            console.print("[red]Refusing to overwrite corrupted config. Run `hylos login` first.[/red]")
            raise SystemExit(1)

        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

        tmp = CONFIG_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(CONFIG_PATH)

        CONFIG_PATH.chmod(0o600)
        cls._cache = data


# ------------------------
# Keychain Helpers
# ------------------------

def get_email() -> Optional[str]:
    return ConfigManager.load().get("email")


def get_token_state(email: str):
    access = keyring.get_password(SERVICE_NAME, f"{email}_access")
    expires = keyring.get_password(SERVICE_NAME, f"{email}_expires")

    if not access or not expires:
        return "missing"

    import time
    if time.time() >= int(expires):
        return "expired"

    return "valid"


def delete_credentials(email: str):
    for key in ["access", "refresh", "expires", "api_key"]:
        try:
            keyring.delete_password(SERVICE_NAME, f"{email}_{key}")
        except Exception:
            pass


# ------------------------
# CLI
# ------------------------

@click.group(name="config")
def config_cmd():
    """Manage Hylos configuration."""
    pass


# ------------------------
# SET
# ------------------------

@config_cmd.command(name="set")
@click.argument("key", type=click.Choice(list(VALID_KEYS)))
@click.argument("value")
def config_set(key, value):
    """Set config value (non-sensitive only)."""

    config = ConfigManager.load()

    # Validation
    if key == "project_id":
        if len(value) < 10:
            console.print("[red]Invalid project_id (too short)[/red]")
            raise SystemExit(1)

    config[key] = value
    ConfigManager.save(config)

    console.print(f"[green]Set {key} ✓[/green]")


# ------------------------
# GET
# ------------------------

@config_cmd.command(name="get")
@click.argument("key", type=click.Choice(["project_id", "email"]))
def config_get(key):
    """Get config value."""

    config = ConfigManager.load()
    value = config.get(key)

    if not value:
        console.print(f"[yellow]{key} not set[/yellow]")
        return

    console.print(value)


# ------------------------
# LIST
# ------------------------

@config_cmd.command(name="list")
def config_list():
    """Show config + session state."""

    config = ConfigManager.load()
    email = config.get("email")

    if not config:
        console.print("[yellow]No config found. Run hylos login[/yellow]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Key")
    table.add_column("Value")

    # Non-sensitive config
    for key in ["email", "project_id"]:
        value = config.get(key)
        display = value if value else "[dim]not set[/dim]"
        table.add_row(key, display)

    console.print(f"\n[dim]Config: {CONFIG_PATH}[/dim]")
    console.print(table)

    # Session state
    if email:
        state = get_token_state(email)

        if state == "valid":
            console.print("[green]Session: valid ✓[/green]")
        elif state == "expired":
            console.print("[yellow]Session: expired[/yellow]")
        else:
            console.print("[red]Session: missing credentials[/red]")
    else:
        console.print("[red]Not logged in[/red]")

    console.print()


# ------------------------
# CLEAR (FULL RESET)
# ------------------------

@config_cmd.command(name="clear")
@click.confirmation_option(prompt="Clear all Hylos config and credentials?")
def config_clear():
    """Fully reset local state (config + keychain)."""

    config = ConfigManager.load()
    email = config.get("email")

    if CONFIG_PATH.exists():
        CONFIG_PATH.unlink()

    if email:
        delete_credentials(email)

    console.print("[green]All config and credentials cleared ✓[/green]")