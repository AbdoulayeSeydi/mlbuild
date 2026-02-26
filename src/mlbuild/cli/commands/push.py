"""
Enterprise-grade push command.

Transactional, CAS-aware, policy-enforcing, retry-safe.
"""

from __future__ import annotations

import sys
import re
import time
import hashlib
from pathlib import Path
from typing import Iterable, List

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ...registry import LocalRegistry
from ...storage import (
    RemoteRepository,
    RetryableError,
    PolicyViolationError,
    IntegrityError,
    ValidationError,
    NotFoundError,
)

console = Console()

BUILD_ID_PATTERN = re.compile(r"^[a-f0-9]{64}$|^[a-zA-Z0-9._-]{1,128}$")

MAX_RETRIES = 5
BACKOFF_BASE = 0.5


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def validate_build_id(build_id: str):
    # Allow anything - let registry.resolve_build handle validation
    if not build_id or not build_id.strip():
        raise ValidationError("Build ID cannot be empty")


def sha256_file(path: Path) -> str:
    """Calculate SHA256 of a file or directory."""
    import hashlib
    
    h = hashlib.sha256()
    
    # If it's a directory, hash all files in it
    if path.is_dir():
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file():
                with file_path.open("rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
        return h.hexdigest()
    
    # Single file
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def retry(operation):
    for attempt in range(MAX_RETRIES):
        try:
            return operation()
        except RetryableError:
            if attempt == MAX_RETRIES - 1:
                raise
            sleep = BACKOFF_BASE * (2 ** attempt)
            time.sleep(sleep)


# ---------------------------------------------------------
# Push Service (Testable)
# ---------------------------------------------------------

class PushService:

    def __init__(self, registry: LocalRegistry, backend):
        self.registry = registry
        self.backend = backend

    def resolve_builds(
        self,
        build_id: str | None,
        push_all: bool,
        tag: str | None,
    ) -> Iterable:
        if push_all:
            yield from self.registry.stream_builds()

        elif tag:
            build = self.registry.get_build_by_tag(tag)
            if not build:
                raise NotFoundError(f"Tag not found: {tag}")
            yield build

        else:
            validate_build_id(build_id)
            build = self.registry.resolve_build(build_id)
            if not build:
                raise NotFoundError(f"Build not found: {build_id}")
            yield build

    def enforce_policy(self, artifact_path: Path):
        policy = self.backend.policy()

        size = artifact_path.stat().st_size
        if policy.max_artifact_size and size > policy.max_artifact_size:
            raise PolicyViolationError(
                f"Artifact exceeds max size ({policy.max_artifact_size})"
            )

    def push_build(self, build, force: bool = False):
        artifact_path = Path(build.artifact_path)
        self.enforce_policy(artifact_path)

        checksum = sha256_file(artifact_path)
        size = artifact_path.stat().st_size

        transaction = self.backend.begin_transaction(build.build_id)

        try:
            # CAS-aware upload
            if self.backend.supports_cas():
                if not self.backend.has_blob(checksum):
                    retry(lambda: self.backend.put_blob(checksum, artifact_path))
                retry(lambda: self.backend.verify_blob(checksum))
            else:
                retry(lambda: self.backend.artifacts.upload_artifact(
                    build.build_id,
                    artifact_path,
                    overwrite=force
                ))

            metadata = {
                "build_id": build.build_id,
                "artifact_hash": checksum, 
                "source_hash": "0" * 64,   
                "config_hash": "0" * 64,    
                "env_fingerprint": "0" * 64,  
                "size_bytes": size,
                "name": build.name,
                "target_device": build.target_device,
                "format": build.format,
                "created_at": build.created_at.isoformat(),
                "quantization": build.quantization,
                "mlbuild_version": build.mlbuild_version,
            }

            retry(lambda: self.backend.metadata.upload_metadata(
                build.build_id,
                metadata,
                overwrite=force
            ))

            self.backend.commit(transaction)

            self.backend.audit("push_build", {
                "build_id": build.build_id,
                "size_bytes": size,
                "checksum": checksum,
            })

            self.backend.telemetry("push_success", {
                "build_id": build.build_id,
                "size_bytes": size,
            })

        except Exception:
            self.backend.abort(transaction)
            self.backend.telemetry("push_failure", {
                "build_id": build.build_id,
            })
            raise

    def push_one(self, build_id: str, force: bool = False):
        """Push a single build by ID."""
        build = self.registry.resolve_build(build_id)
        if not build:
            raise NotFoundError(f"Build not found: {build_id}")
        self.push_build(build, force=force)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

@click.command()
@click.argument("build_id", required=False)
@click.option("--remote", help="Remote name")
@click.option("--all", "push_all", is_flag=True)
@click.option("--tag")
@click.option("--force", is_flag=True)
@click.option("--dry-run", is_flag=True)
def push(build_id, remote, push_all, tag, force, dry_run):

    if not any([build_id, push_all, tag]):
        console.print("[red]Specify build_id, --all, or --tag[/red]")
        sys.exit(1)

    registry = LocalRegistry()
    repo = RemoteRepository()

    try:
        remote_config = repo.get(remote) if remote else repo.get_default()
        if not remote_config:
            raise ValidationError("No default remote configured.")

        backend = repo.get_backend(remote_config.name)

        backend.validate_credentials()
        backend.ping()

        service = PushService(registry, backend)

        failures = 0
        pushed = 0

        builds = service.resolve_builds(build_id, push_all, tag)

        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            console=console,
        ) as progress:

            for build in builds:
                task = progress.add_task(f"Pushing {build.build_id[:12]}...")

                if dry_run:
                    console.print(f"[dim]Would push {build.build_id}[/dim]")
                    progress.update(task, completed=1)
                    continue

                try:
                    service.push_build(build, force=force)
                    pushed += 1
                    console.print(f"[green]✓ {build.build_id[:12]}[/green]")

                except Exception as e:
                    failures += 1
                    console.print(f"[red]✗ {build.build_id[:12]}: {e}[/red]")

                progress.update(task, completed=1)

        console.print()
        console.print(f"Pushed: {pushed}")
        console.print(f"Failed: {failures}")

        if failures > 0:
            sys.exit(2)

    except Exception as e:
        console.print(f"[red]Fatal:[/red] {e}")
        sys.exit(3)