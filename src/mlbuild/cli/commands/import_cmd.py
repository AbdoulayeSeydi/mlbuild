"""
Import command for MLBuild.

Features:
- Streamed, memory-safe hashing
- Deterministic cross-platform directory hashing
- Atomic artifact copy with temp staging
- Registry-driven target families (no SKU explosion)
- Deterministic build IDs independent of environment
- Proper duplicate handling via exceptions
- Sanitized human-friendly build names
- Unified content-addressed artifact naming
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from mlbuild import __version__ as MLBUILD_VERSION
from mlbuild.core.format_detection import (
    detect_and_validate_format,
    validate_format_target_compat,
    ModelFormatError,
    TargetCompatibilityError,
)
from mlbuild.registry import LocalRegistry
from mlbuild.core.hash import compute_config_hash, compute_source_hash
from mlbuild.core.environment import collect_environment, hash_environment
from mlbuild.core.types import Build
from mlbuild.core.errors import InternalError

console = Console()

# ================================================================
# Helpers
# ================================================================

CHUNK_SIZE = 1024 * 1024  # 1MB


def hash_file_stream(path: Path) -> str:
    """SHA256 hash of a file using streaming to avoid memory spikes."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_dir_stream(path: Path) -> str:
    """Deterministic SHA256 of a directory (cross-platform)."""
    h = hashlib.sha256()
    for child in sorted(path.rglob("*"), key=lambda p: p.as_posix()):
        if child.is_file():
            rel = child.relative_to(path).as_posix().encode("utf-8")
            h.update(rel)
            with child.open("rb") as f:
                for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
                    h.update(chunk)
    return h.hexdigest()


def artifact_hash(path: Path) -> str:
    """Hash either a file or a directory artifact deterministically."""
    return hash_dir_stream(path) if path.is_dir() else hash_file_stream(path)


def sanitize_name(name: str) -> str:
    """Remove problematic characters from human-friendly build names."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def build_id(source_hash: str, config_hash: str, artifact_hash: str) -> str:
    """Deterministic build ID for imported artifacts (no env fingerprint)."""
    return hashlib.sha256(
        b"\x00".join([
            bytes.fromhex(source_hash),
            bytes.fromhex(config_hash),
            bytes.fromhex(artifact_hash),
            MLBUILD_VERSION.encode("utf-8"),
        ])
    ).hexdigest()


def dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


# ================================================================
# CLI Command
# ================================================================

@click.command("import")
@click.option(
    "--model",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the pre-built model (.tflite, .mlmodel, or .mlpackage).",
)
@click.option(
    "--target",
    required=True,
    type=click.Choice([
        "apple_a17", "apple_a16", "apple_a15",
        "apple_m3",  "apple_m2",  "apple_m1",
        "android_arm64", "android_arm32", "android_x86",
        "raspberry_pi", "coral_tpu", "generic_linux",
    ]),
    help="Target device this model was built for.",
)
@click.option(
    "--quantize",
    type=click.Choice(["fp32", "fp16", "int8"]),
    default="fp32",
    show_default=True,
    help="Quantization level of the model (informational — not applied).",
)
@click.option("--name", default=None, help="Human-friendly name for this build.")
@click.option("--notes", default=None, help="Optional notes about the model's origin.")
@click.option("--json", "as_json", is_flag=True, help="Output result as JSON.")
def import_cmd(model: Path, target: str, quantize: str, name: Optional[str], notes: Optional[str], as_json: bool):
    try:
        _run_import(model, target, quantize, name, notes, as_json)
    except (ModelFormatError, TargetCompatibilityError) as exc:
        console.print(f"[bold red]Import failed:[/bold red] {exc}")
        sys.exit(1)
    except Exception as exc:
        console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        console.print("[dim]Use --debug for traceback[/dim]")
        sys.exit(1)


# ================================================================
# Core Import Logic
# ================================================================

def _run_import(model: Path, target: str, quantize: str, name: Optional[str], notes: Optional[str], as_json: bool):
    # Step 1: Detect format
    console.print(f"[bold]Importing:[/bold] {model.name}")
    fmt = detect_and_validate_format(model)
    console.print(f"Format:        {fmt}")
    console.print(f"Target:        {target}")
    console.print(f"Quantization:  {quantize}")

    # Step 2: Validate target compatibility
    validate_format_target_compat(fmt, target)

    # Step 3: Compute hashes
    console.print("[dim]Hashing source...[/dim]")
    source_hash = compute_source_hash(model)

    console.print("[dim]Hashing artifact...[/dim]")
    art_hash = artifact_hash(model)

    # Step 4: Build config hash
    config = {
        "target": target,
        "quantization": {"type": quantize},
        "optimizer": {},
    }
    config_hash = compute_config_hash(config)

    # Step 5: Collect environment — provides valid env_fingerprint and
    # system fields required by the Build dataclass and registry schema.
    # Zeroed sentinel values are avoided so the build remains queryable.
    console.print("[dim]Collecting environment...[/dim]")
    env_data = collect_environment()
    env_fingerprint = hash_environment(env_data)

    # Step 6: Prepare artifact store (atomic copy)
    artifacts_root = Path(".mlbuild/artifacts").resolve()
    artifacts_root.mkdir(parents=True, exist_ok=True)

    tmp_dir = artifacts_root / f".tmp_{art_hash[:16]}"
    final_path = artifacts_root / art_hash

    if final_path.exists():
        console.print(f"[yellow]Reusing existing artifact {art_hash[:12]}[/yellow]")
    else:
        # Atomic copy
        if model.is_dir():
            shutil.copytree(model, tmp_dir)
        else:
            tmp_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(model, tmp_dir / sanitize_name(model.name))

        tmp_dir.rename(final_path)  # atomic on same volume

    # Step 7: Build ID (deterministic, independent of environment)
    bid = build_id(source_hash, config_hash, art_hash)

    # Step 8: Compute size (safe after atomic move)
    size_bytes = dir_size_bytes(final_path) if final_path.is_dir() else final_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    # Step 9: Construct Build object
    build_obj = Build(
        build_id=bid,
        artifact_hash=art_hash,
        source_hash=source_hash,
        config_hash=config_hash,
        env_fingerprint=env_fingerprint,
        name=sanitize_name(name or model.stem),
        notes=notes,
        created_at=datetime.now(timezone.utc),
        source_path=str(model.resolve()),
        target_device=target,
        format=fmt,
        quantization=config["quantization"],
        optimizer_config=config["optimizer"],
        backend_versions={"imported": "true"},  # str to match Dict[str, str] type
        environment_data=env_data,
        mlbuild_version=MLBUILD_VERSION,
        python_version=env_data["python"]["version"],
        platform=env_data["hardware"]["cpu"]["system"],
        os_version=env_data["hardware"]["cpu"]["release"],
        artifact_path=str(final_path),
        size_mb=size_mb,
    )

    # Step 10: Register build (handle duplicates explicitly)
    registry = LocalRegistry()
    try:
        registry.save_build(build_obj)
    except InternalError as exc:
        if "already exists" in str(exc).lower() or "uniqueness" in str(exc).lower():
            console.print("[yellow]Build already registered — skipping duplicate import.[/yellow]")
        else:
            raise

    # Step 11: Output
    result = {
        "build_id": bid,
        "artifact_hash": art_hash,
        "source_hash": source_hash,
        "config_hash": config_hash,
        "format": fmt,
        "target": target,
        "quantization": quantize,
        "size_mb": round(size_mb, 2),
        "artifact_path": str(final_path),
        "imported": True,
    }

    if as_json:
        console.print(json.dumps(result, indent=2))
    else:
        console.print(f"\n[bold green]✓ Import complete[/bold green]")
        console.print(f"Build ID:      {bid[:16]}...")
        console.print(f"Artifact Hash: {art_hash[:16]}...")
        console.print(f"Source Hash:   {source_hash[:16]}...")
        console.print(f"Size:          {size_mb:.2f} MB")
        console.print(f"Artifact Path: {final_path}", overflow="fold")
        console.print(f"\n[dim]Tip: run 'mlbuild benchmark {bid[:16]}' to profile this model.[/dim]\n")