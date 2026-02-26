"""
Production-grade diff command for Edge ML builds.
Deterministic, canonical, CI-safe.
"""

import json
import logging
import sys
from datetime import timezone
from pathlib import Path
from typing import Dict, Any

import click
from rich.console import Console
from rich.table import Table

from mlbuild.registry import LocalRegistry
from mlbuild.core.errors import MLBuildError
from mlbuild.core.diff_utils import compare_artifacts, compare_configs

console = Console()
logger = logging.getLogger("mlbuild.diff")


# ---------------------------------------------------
# Utility
# ---------------------------------------------------

def utc_iso(dt):
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def structured_diff(old, new):
    if old == new:
        return {"status": "equal", "old": old, "new": new}
    return {"status": "different", "old": old, "new": new}


def numeric_diff(old: int, new: int, threshold: int = None):
    if old == new:
        return {"status": "equal", "old": old, "new": new, "delta": 0}

    delta = new - old
    result = {
        "status": "different",
        "old": old,
        "new": new,
        "delta": delta,
    }

    if threshold is not None:
        result["threshold_exceeded"] = abs(delta) > threshold

    return result


def hash_state(a: str, b: str) -> Dict[str, Any]:
    if not a or not b:
        return {"status": "missing", "a": a, "b": b}
    if a == b:
        return {"status": "equal", "a": a, "b": b}
    return {"status": "different", "a": a, "b": b}


def validate_artifact_path(path: str):
    p = Path(path)
    if not p.exists():
        raise MLBuildError(f"Artifact path does not exist: {path}")
    if not p.is_dir():
        raise MLBuildError(f"Artifact path is not a directory: {path}")


# ---------------------------------------------------
# CLI
# ---------------------------------------------------

@click.command()
@click.argument("build_a")
@click.argument("build_b")
@click.option("--json", "as_json", is_flag=True, help="Output structured JSON")
@click.option("--ignore-size", is_flag=True, help="Ignore metadata size diff")
@click.option("--ignore-quant", is_flag=True, help="Ignore quantization diff")
@click.option("--deep", is_flag=True, help="Force deep artifact diff even if hashes match")
def diff(build_a: str, build_b: str, as_json: bool, ignore_size: bool, ignore_quant: bool, deep: bool):
    """
    Deterministic, structured diff between two builds.
    Exit codes:
      0 = no differences
      1 = differences detected
      2 = usage / resolution error
      3 = artifact diff failure
    """

    try:
        registry = LocalRegistry()

        build_obj_a = registry.resolve_build(build_a)
        build_obj_b = registry.resolve_build(build_b)

        if build_obj_a is None:
            console.print(f"\n[red]Build not found: {build_a}[/red]\n")
            sys.exit(2)
        if build_obj_b is None:
            console.print(f"\n[red]Build not found: {build_b}[/red]\n")
            sys.exit(2)

        result: Dict[str, Any] = {
            "build_a": build_obj_a.build_id,
            "build_b": build_obj_b.build_id,
            "metadata": {},
            "artifact": {},
            "config": {},
        }

        # ----------------------------
        # Metadata Diff (Canonical Units)
        # ----------------------------

        result["metadata"]["target_device"] = structured_diff(
            build_obj_a.target_device,
            build_obj_b.target_device,
        )

        if not ignore_quant:
            result["metadata"]["quantization_type"] = structured_diff(
                build_obj_a.quantization_type,
                build_obj_b.quantization_type,
            )

        if not ignore_size:
            threshold_bytes = 10 * 1024 * 1024
            result["metadata"]["size_bytes"] = numeric_diff(
                build_obj_a.size_bytes,
                build_obj_b.size_bytes,
                threshold=threshold_bytes,
            )

        result["metadata"]["created_at"] = structured_diff(
            utc_iso(build_obj_a.created_at),
            utc_iso(build_obj_b.created_at),
        )

        result["metadata"]["artifact_hash"] = hash_state(
            build_obj_a.artifact_hash,
            build_obj_b.artifact_hash,
        )

        result["metadata"]["config_hash"] = hash_state(
            build_obj_a.config_hash,
            build_obj_b.config_hash,
        )

        result["metadata"]["source_hash"] = hash_state(
            build_obj_a.source_hash,
            build_obj_b.source_hash,
        )

        # ----------------------------
        # Artifact Diff (Only if Needed)
        # ----------------------------

        artifact_status = result["metadata"]["artifact_hash"]["status"]

        if artifact_status == "different" or deep:
            validate_artifact_path(build_obj_a.artifact_path)
            validate_artifact_path(build_obj_b.artifact_path)

            try:
                artifact_diff = compare_artifacts(
                    build_obj_a.artifact_path,
                    build_obj_b.artifact_path,
                )
                result["artifact"] = {
                    "status": "compared",
                    "layers": dict(sorted(artifact_diff.items()))
                }
            except Exception as e:
                logger.exception("Artifact diff failed")
                result["artifact"] = {"status": "error", "error": str(e)}
                if as_json:
                    console.print(json.dumps(result, indent=2, sort_keys=True))
                sys.exit(3)
        else:
            result["artifact"] = {"status": "skipped"}

        # ----------------------------
        # Config Diff
        # ----------------------------

        config_diff = compare_configs(build_obj_a.config, build_obj_b.config)
        result["config"] = {
            "status": "different" if config_diff else "equal",
            "diff": dict(sorted(config_diff.items())) if isinstance(config_diff, dict) else config_diff
        }

        # ----------------------------
        # Determine Overall Status
        # ----------------------------

        differences = any(
            v.get("status") == "different"
            for v in result["metadata"].values()
            if isinstance(v, dict)
        ) or result["config"]["status"] == "different"

        if result["artifact"].get("status") == "compared":
            differences = differences or bool(result["artifact"]["layers"])

        # ----------------------------
        # Output
        # ----------------------------

        if as_json:
            console.print(json.dumps(result, indent=2, sort_keys=True))
        else:
            console.print(f"[bold]Diff:[/bold] {build_obj_a.build_id[:12]} → {build_obj_b.build_id[:12]}")
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Field")
            table.add_column("Status")

            for field in sorted(result["metadata"].keys()):
                table.add_row(field, result["metadata"][field]["status"])

            console.print(table)

            if not differences:
                console.print("\n[green]No differences found[/green]")
            else:
                console.print("\n[yellow]Differences detected[/yellow]")

        sys.exit(1 if differences else 0)

    except MLBuildError as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        sys.exit(2)

    except Exception:
        logger.exception("Unexpected error in diff")
        console.print("\n[bold red]Unexpected error. See logs.[/bold red]\n")
        sys.exit(2)
