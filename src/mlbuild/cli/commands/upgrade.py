from __future__ import annotations

import sys
import subprocess
import urllib.request
import json

import click
from rich.console import Console

from packaging.version import Version, InvalidVersion

console = Console(width=None)

PYPI_URL = "https://pypi.org/pypi/mlbuild/json"


# ── Version helpers ───────────────────────────────────────────

def _current_version() -> str:
    from importlib.metadata import version, PackageNotFoundError
    try:
        return version("mlbuild")
    except PackageNotFoundError:
        return "unknown"


def _latest_version(pre: bool) -> str:
    """
    Always compute from PyPI releases.
    Single source of truth. No divergence.
    """
    try:
        with urllib.request.urlopen(PYPI_URL, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception:
        raise ConnectionError(
            "could not reach PyPI. Check your connection.\n"
            "To upgrade manually: pip install --upgrade mlbuild"
        )

    releases = data.get("releases", {})
    if not releases:
        raise RuntimeError("Invalid PyPI response")

    parsed: list[Version] = []
    for v in releases.keys():
        try:
            pv = Version(v)
            if not pre and pv.is_prerelease:
                continue
            parsed.append(pv)
        except InvalidVersion:
            continue

    if not parsed:
        raise RuntimeError("No valid versions found on PyPI")

    return str(max(parsed))


def _is_up_to_date(current: str, latest: str) -> bool:
    if current == "unknown":
        return False
    try:
        return Version(current) >= Version(latest)
    except InvalidVersion:
        return False


# ── CLI ───────────────────────────────────────────────────────

@click.command(help="Upgrade MLBuild to the latest version.")
@click.option("--pre", is_flag=True, help="Include pre-release versions.")
def upgrade(pre: bool):
    current = _current_version()
    console.print(f"\n  Current:   {current}")
    console.print("  Checking PyPI...")

    try:
        latest = _latest_version(pre)
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}\n")
        sys.exit(1)

    console.print(f"  Latest:    {latest}")

    # Correct version comparison (handles >, ==, invalid)
    if _is_up_to_date(current, latest):
        console.print(f"\n  [green]✓[/green] Already on latest version ({current})\n")
        return

    console.print("  Upgrading...")

    pip_args = [sys.executable, "-m", "pip", "install", "--upgrade", "mlbuild"]
    if pre:
        pip_args.append("--pre")

    try:
        result = subprocess.run(
            pip_args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as e:
        console.print(f"\n[red]Error:[/red] Failed to invoke pip: {e}\n")
        sys.exit(1)

    if result.returncode != 0:
        console.print(f"\n[red]Error:[/red] pip upgrade failed:\n{result.stderr}\n")
        sys.exit(1)

    # Re-read version after upgrade
    new_version = _current_version()

    # Guard against false success
    if new_version == current:
        console.print(f"\n  [green]✓[/green] Already on latest version ({current})\n")
        return

    console.print(f"\n  [green]✓[/green] MLBuild upgraded to {new_version}\n")