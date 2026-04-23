"""
hylos login -- authenticate with Hylos Cloud)

- Stores tokens securely in OS keychain (via keyring)
- Config stored in ~/.hylos/config.json (non-sensitive only)
- Handles token refresh automatically
- Uses env vars for Supabase config
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import click
import httpx
import keyring
from rich.console import Console

console = Console()

CONFIG_PATH = Path.home() / ".hylos" / "config.json"
SERVICE_NAME = "hylos-cli"

SUPABASE_URL = os.getenv("HYLOS_SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("HYLOS_SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    console.print("[red]Missing environment variables HYLOS_SUPABASE_URL / HYLOS_SUPABASE_ANON_KEY[/red]")
    raise SystemExit(1)


# ------------------------
# Config Management (non-sensitive only)
# ------------------------

class ConfigManager:
    _cache: Optional[dict] = None

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
            console.print(f"[yellow]Corrupted config detected. Backed up to {backup}[/yellow]")
            cls._cache = {}
            return cls._cache

    @classmethod
    def save(cls, data: dict):
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

        tmp = CONFIG_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(CONFIG_PATH)

        # enforce secure permissions
        CONFIG_PATH.chmod(0o600)

        cls._cache = data


# ------------------------
# Secure Token Storage (Keychain)
# ------------------------

def store_tokens(email: str, access_token: str, refresh_token: str, expires_at: int):
    keyring.set_password(SERVICE_NAME, f"{email}_access", access_token)
    keyring.set_password(SERVICE_NAME, f"{email}_refresh", refresh_token)
    keyring.set_password(SERVICE_NAME, f"{email}_expires", str(expires_at))


def load_tokens(email: str):
    access = keyring.get_password(SERVICE_NAME, f"{email}_access")
    refresh = keyring.get_password(SERVICE_NAME, f"{email}_refresh")
    expires = keyring.get_password(SERVICE_NAME, f"{email}_expires")

    if not access or not refresh or not expires:
        return None

    return {
        "access_token": access,
        "refresh_token": refresh,
        "expires_at": int(expires),
    }


# ------------------------
# Token Refresh Logic
# ------------------------

def refresh_access_token(email: str, refresh_token: str):
    try:
        r = httpx.post(
            f"{SUPABASE_URL}/auth/v1/token?grant_type=refresh_token",
            headers={"apikey": SUPABASE_ANON_KEY},
            json={"refresh_token": refresh_token},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPError:
        console.print("[red]Token refresh failed. Please login again.[/red]")
        raise SystemExit(1)

    access = data.get("access_token")
    refresh = data.get("refresh_token")
    expires_in = data.get("expires_in")

    if not access or not refresh or not expires_in:
        console.print("[red]Malformed refresh response[/red]")
        raise SystemExit(1)

    expires_at = int(time.time()) + int(expires_in)

    store_tokens(email, access, refresh, expires_at)
    return access


def get_valid_access_token() -> str:
    config = ConfigManager.load()
    email = config.get("email")

    if not email:
        console.print("[red]Not logged in. Run 'hylos login'[/red]")
        raise SystemExit(1)

    tokens = load_tokens(email)
    if not tokens:
        console.print("[red]Missing credentials. Please login again.[/red]")
        raise SystemExit(1)

    if time.time() >= tokens["expires_at"]:
        return refresh_access_token(email, tokens["refresh_token"])

    return tokens["access_token"]


# ------------------------
# CLI Command
# ------------------------

@click.command()
@click.option("--email", prompt=True)
@click.option("--password", prompt=True, hide_input=True)
def login(email, password):
    """Log in to Hylos Cloud."""

    existing = ConfigManager.load()

    if existing.get("email") == email:
        if not click.confirm("Already logged in as this user. Re-authenticate?"):
            console.print("[dim]Login cancelled.[/dim]")
            return

    console.print("[bold]Logging in...[/bold]")

    try:
        r = httpx.post(
            f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
            headers={
                "apikey": SUPABASE_ANON_KEY,
                "Content-Type": "application/json",
            },
            json={"email": email, "password": password},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()

    except httpx.TimeoutException:
        console.print("[red]Request timed out.[/red]")
        raise SystemExit(1)
    except httpx.ConnectError:
        console.print("[red]Network connection failed.[/red]")
        raise SystemExit(1)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 400:
            console.print("[red]Invalid credentials.[/red]")
        else:
            console.print(f"[red]HTTP error {e.response.status_code}[/red]")
        raise SystemExit(1)

    access = data.get("access_token")
    refresh = data.get("refresh_token")
    expires_in = data.get("expires_in")

    if not access or not refresh or not expires_in:
        console.print("[red]Malformed response from server[/red]")
        raise SystemExit(1)

    expires_at = int(time.time()) + int(expires_in)

    store_tokens(email, access, refresh, expires_at)

    ConfigManager.save({
        "email": email,
        "project_id": existing.get("project_id"),
        "api_key": existing.get("api_key"),
    })

    console.print(f"[green]Logged in as {email} ✓[/green]")
    console.print(f"[dim]Token expires in {expires_in // 60} minutes[/dim]")

    if not existing.get("project_id"):
        console.print("\n[dim]Next:[/dim] hylos projects create <name>\n")