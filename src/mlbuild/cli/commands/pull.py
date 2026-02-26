import hashlib
import shutil
import tempfile
import time
import sys  # ← ADD THIS
import click
from rich.console import Console
from pathlib import Path
from datetime import datetime
from decimal import Decimal

from ...registry import LocalRegistry
from ...core.types import Build
from ...storage import RemoteRepository  # ← ADD THIS

console = Console()


class PullResult:
    def __init__(self):
        self.pulled = 0
        self.skipped = 0
        self.failed = 0


class PullService:
    MAX_RETRIES = 3
    RETRY_BACKOFF = 1.5

    def __init__(self, backend, registry=None, console=None):
        self.backend = backend
        self.registry = registry or LocalRegistry()
        self.console = console

    # -------------------------
    # Public API
    # -------------------------

    def pull(self, build_id=None, pull_all=False, tag=None, force=False, dry_run=False):
        self.backend.ping()

        build_ids = self._resolve_build_ids(
            build_id=build_id,
            pull_all=pull_all,
            tag=tag,
        )

        result = PullResult()

        for bid in build_ids:
            try:
                if dry_run:
                    self._print(f"[dim]Would pull {bid}[/dim]")
                    continue

                self._pull_single(bid, force)
                result.pulled += 1

            except AlreadyExistsError:
                self._print(f"[yellow]⊘ Skipped {bid[:12]} (exists)[/yellow]")
                result.skipped += 1

            except Exception as e:
                self._print(f"[red]✗ Failed {bid[:12]}: {e}[/red]")
                result.failed += 1

        self._print_summary(result)
        return result

    # -------------------------
    # Build Resolution
    # -------------------------

    # -------------------------
    # Build Resolution (Git-style)
    # -------------------------

    def _resolve_build_ids(self, build_id, pull_all, tag):
        if pull_all:
            return self._stream_all_build_ids()

        if tag:
            # For tags, we'd query remote tag store
            # For now, treat as prefix
            return [self._resolve_prefix(tag)]

        return [self._resolve_prefix(build_id)]

    def _resolve_prefix(self, prefix: str):
        """
        Git-style prefix resolution.
        
        Returns full build_id if prefix is unique.
        Raises on ambiguity or no match.
        """
        if not prefix or len(prefix) < 4:
            raise ValueError("Prefix must be at least 4 characters")
        
        # If already full SHA, return as-is
        if len(prefix) == 64:
            return prefix
        
        # Find all matches in remote
        matches = []
        build_ids, _ = self.backend.metadata.list_builds(limit=1000)
        
        for bid in build_ids:
            if bid.startswith(prefix):
                matches.append(bid)
        
        if len(matches) == 0:
            raise ValueError(f"No builds match prefix: {prefix}")
        
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous prefix '{prefix}' matches {len(matches)} builds:\n" +
                "\n".join(f"  - {m[:16]}..." for m in matches[:5])
            )
        
        return matches[0]

    def _stream_all_build_ids(self):
        """Stream all remote build IDs."""
        build_ids, _ = self.backend.metadata.list_builds(limit=1000)
        for bid in build_ids:
            yield bid

    # -------------------------
    # Single Pull
    # -------------------------

    def _pull_single(self, build_id, force):
        if self.registry.exists(build_id) and not force:
            raise AlreadyExistsError()

        metadata = self._retry(
            lambda: self.backend.metadata.download_metadata(build_id)
        )

        expected_hash = metadata["artifact_hash"]

        with tempfile.TemporaryDirectory() as staging_dir:
            staging_path = Path(staging_dir) / build_id

            self._retry(
                lambda: self.backend.artifacts.download_artifact(build_id, staging_path)
            )

            actual_hash = self._compute_hash(staging_path)

            if actual_hash != expected_hash:
                raise IntegrityError(
                    f"Checksum mismatch for {build_id}"
                )

            final_path = self._commit_artifact_atomically(
                build_id,
                staging_path,
            )

            self._register_build(metadata, final_path)

        self._print(f"[green]✓ Pulled {build_id[:12]}[/green]")

    def pull_one(self, build_id: str, force: bool = False):
        """Pull a single build by ID."""
        self._pull_single(build_id, force=force)

    # -------------------------
    # Atomic Commit
    # -------------------------

    def _commit_artifact_atomically(self, build_id, staging_path):
        target_dir = Path(".mlbuild/artifacts") / build_id
        target_dir.parent.mkdir(parents=True, exist_ok=True)

        if target_dir.exists():
            shutil.rmtree(target_dir)

        shutil.move(str(staging_path), str(target_dir))
        return target_dir

    # -------------------------
    # Registry
    # -------------------------

    def _register_build(self, metadata, artifact_path):
        total_size = sum(
            f.stat().st_size
            for f in artifact_path.rglob("*")
            if f.is_file()
        )

        build = Build(
            build_id=metadata["build_id"],
            artifact_hash=metadata["artifact_hash"],
            source_hash=metadata["source_hash"],
            config_hash=metadata["config_hash"],
            env_fingerprint=metadata["env_fingerprint"],
            name=metadata.get("name"),
            notes="Pulled from remote",
            created_at=datetime.fromisoformat(metadata["created_at"]),
            source_path="remote",
            target_device=metadata["target_device"],
            format=metadata["format"],
            quantization=metadata.get("quantization", {}),
            optimizer_config={},
            backend_versions={},
            environment_data={},
            mlbuild_version=metadata["mlbuild_version"],
            python_version="unknown",
            platform="unknown",
            os_version="unknown",
            artifact_path=str(artifact_path),
            size_mb=Decimal(str(total_size / (1024 * 1024))),
        )

        with self.registry.transaction():
            self.registry.save_build(build)

    # -------------------------
    # Integrity
    # -------------------------

    def _compute_hash(self, path):
        sha = hashlib.sha256()
        for file in sorted(path.rglob("*")):
            if file.is_file():
                sha.update(file.read_bytes())
        return sha.hexdigest()

    # -------------------------
    # Retry
    # -------------------------

    def _retry(self, fn):
        for attempt in range(self.MAX_RETRIES):
            try:
                return fn()
            except Exception:
                if attempt == self.MAX_RETRIES - 1:
                    raise
                time.sleep(self.RETRY_BACKOFF ** attempt)

    # -------------------------
    # Output
    # -------------------------

    def _print(self, msg):
        if self.console:
            self.console.print(msg)

    def _print_summary(self, result):
        self._print(f"\nPulled: {result.pulled}")
        if result.skipped:
            self._print(f"Skipped: {result.skipped}")
        if result.failed:
            self._print(f"Failed: {result.failed}")


class AlreadyExistsError(Exception):
    pass


class IntegrityError(Exception):
    pass

# ==========================================================
# CLI
# ==========================================================

@click.command()
@click.argument("build_id", required=False)
@click.option("--remote", help="Remote name")
@click.option("--all", "pull_all", is_flag=True)
@click.option("--tag")
@click.option("--force", is_flag=True)
@click.option("--dry-run", is_flag=True)
def pull(build_id, remote, pull_all, tag, force, dry_run):
    """Pull builds from remote storage."""
    
    if not any([build_id, pull_all, tag]):
        console.print("[red]Specify build_id, --all, or --tag[/red]")
        sys.exit(1)
    
    try:
        repo = RemoteRepository()
        
        # Get remote config (default if not specified)
        if remote:
            backend = repo.get_backend(remote)
        else:
            default_remote = repo.get_default()
            if not default_remote:
                console.print("[red]No default remote configured[/red]")
                console.print("Add one with: mlbuild remote add <name> --backend local --path <path> --default")
                sys.exit(1)
            backend = repo.get_backend(default_remote.name)
        
        service = PullService(backend, console=console)
        
        result = service.pull(
            build_id=build_id,
            pull_all=pull_all,
            tag=tag,
            force=force,
            dry_run=dry_run,
        )
        
        sys.exit(0 if result.failed == 0 else 1)
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)