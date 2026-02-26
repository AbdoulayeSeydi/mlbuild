"""
Enterprise-grade bidirectional sync command.

Guarantees:
- Backend abstraction (no direct instantiation)
- Streamed diff (no memory blowup)
- Hash-aware divergence detection
- Deterministic policy resolution
- Concurrency lock
- Transaction-safe push/pull delegation
- Proper exit codes
- No CLI recursion
"""

import sys
import os
import time
import fcntl
import click
from dataclasses import dataclass
from rich.console import Console

from ...storage import RemoteRepository
from ...registry import LocalRegistry


console = Console()


# ==========================================================
# Concurrency Lock
# ==========================================================

class FileLock:
    def __init__(self, path: str):
        self.path = path
        self.fd = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.fd = open(self.path, "w")
        fcntl.flock(self.fd, fcntl.LOCK_EX)

    def __exit__(self, exc_type, exc, tb):
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        self.fd.close()


# ==========================================================
# Sync Policy
# ==========================================================

class SyncPolicy:
    """
    merge           - bidirectional, detect divergence
    mirror-local    - local authoritative
    mirror-remote   - remote authoritative
    reconcile       - abort on divergence
    """

    def __init__(self, mode: str):
        self.mode = mode

    def resolve(self, state: str):
        if state == "identical":
            return "skip"

        if state == "local_only":
            return "push"

        if state == "remote_only":
            return "pull"

        if state == "diverged":
            if self.mode == "mirror-local":
                return "push"
            if self.mode == "mirror-remote":
                return "pull"
            if self.mode == "reconcile":
                return "abort"
            return "conflict"

        return "skip"


# ==========================================================
# Result Object
# ==========================================================

@dataclass
class SyncResult:
    pushed: int = 0
    pulled: int = 0
    skipped: int = 0
    diverged: int = 0
    failed: int = 0

    @property
    def exit_code(self):
        return 1 if self.failed > 0 else 0


# ==========================================================
# Sync Implementation
# ==========================================================

class SyncEngine:

    def __init__(self, backend, console=None):
        self.backend = backend
        self.console = console
        self.registry = LocalRegistry()
        
        # Import services
        from .push import PushService
        from .pull import PullService
        
        self.push_service = PushService(self.registry, backend)
        self.pull_service = PullService(backend, registry=self.registry, console=console)

    # ------------------------------------------------------

    def sync(self, policy: SyncPolicy,
             push_only=False,
             pull_only=False,
             dry_run=False,
             force=False):

        result = SyncResult()
        start_time = time.time()

        with FileLock(".mlbuild/sync.lock"):

            local_iter = self._stream_local()
            remote_iter = self._stream_remote()

            for state in self._diff_stream(local_iter, remote_iter):

                action = policy.resolve(state["type"])

                if push_only and action == "pull":
                    continue

                if pull_only and action == "push":
                    continue

                if action == "skip":
                    result.skipped += 1
                    continue

                if action == "abort":
                    raise RuntimeError(
                        f"Divergence detected for {state['build_id']}"
                    )

                if action == "conflict" and not force:
                    result.diverged += 1
                    continue

                if dry_run:
                    continue

                try:
                    if action == "push":
                        self.push_service.push_one(
                            state["build_id"],
                            force=force
                        )
                        result.pushed += 1

                    elif action == "pull":
                        self.pull_service.pull_one(
                            state["build_id"],
                            force=force
                        )
                        result.pulled += 1

                except Exception as e:
                    if self.console:
                        self.console.print(
                            f"[red]✗ {state['build_id'][:12]}: {e}[/red]"
                        )
                    result.failed += 1

        duration = time.time() - start_time
        self._print_summary(result, duration)

        return result

    # ------------------------------------------------------

    def _stream_local(self):
        for build in self.registry.iter_builds_sorted():
            yield {
                "build_id": build.build_id,
                "hash": build.artifact_hash,
            }

    def _stream_remote(self):
        for page in self.backend.metadata.iter_builds():
            for item in page:
                yield item

    # ------------------------------------------------------
    # Scalable streaming diff (O(n), constant memory)
    # ------------------------------------------------------

    def _diff_stream(self, local_iter, remote_iter):

        local = next(local_iter, None)
        remote = next(remote_iter, None)

        while local or remote:

            if local and (not remote or local["build_id"] < remote["build_id"]):
                yield {"type": "local_only", **local}
                local = next(local_iter, None)

            elif remote and (not local or remote["build_id"] < local["build_id"]):
                yield {"type": "remote_only", **remote}
                remote = next(remote_iter, None)

            else:
                if local["hash"] == remote["hash"]:
                    yield {"type": "identical", **local}
                else:
                    yield {"type": "diverged", **local}

                local = next(local_iter, None)
                remote = next(remote_iter, None)

    # ------------------------------------------------------

    def _print_summary(self, result, duration):
        if not self.console:
            return

        self.console.print("\n[bold]Sync Summary[/bold]")
        self.console.print(f"Pushed: {result.pushed}")
        self.console.print(f"Pulled: {result.pulled}")
        self.console.print(f"Skipped: {result.skipped}")
        self.console.print(f"Diverged: {result.diverged}")
        self.console.print(f"Failed: {result.failed}")
        self.console.print(f"Duration: {duration:.2f}s")


# ==========================================================
# CLI
# ==========================================================

@click.command()
@click.option("--remote", help="Remote name")
@click.option("--mode",
              type=click.Choice(
                  ["merge", "mirror-local", "mirror-remote", "reconcile"]
              ),
              default="merge")
@click.option("--push-only", is_flag=True)
@click.option("--pull-only", is_flag=True)
@click.option("--dry-run", is_flag=True)
@click.option("--force", is_flag=True)
def sync(remote, mode, push_only, pull_only, dry_run, force):
    """Bidirectional sync between local and remote storage."""

    if push_only and pull_only:
        console.print(
            "[red]Cannot combine --push-only and --pull-only[/red]"
        )
        sys.exit(2)

    try:
        repo = RemoteRepository()
        
        # Get remote config (default if not specified)
        if remote:
            backend = repo.get_backend(remote)
        else:
            default_remote = repo.get_default()
            if not default_remote:
                console.print("[red]No default remote configured[/red]")
                sys.exit(1)
            backend = repo.get_backend(default_remote.name)

        engine = SyncEngine(backend, console=console)

        result = engine.sync(
            policy=SyncPolicy(mode),
            push_only=push_only,
            pull_only=pull_only,
            dry_run=dry_run,
            force=force,
        )

        sys.exit(result.exit_code)

    except Exception as e:
        console.print(f"[red]Fatal:[/red] {e}")
        sys.exit(1)