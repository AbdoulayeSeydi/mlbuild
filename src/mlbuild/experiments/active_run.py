"""
Active run persistence - stores current run ID across CLI calls.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from contextlib import contextmanager


class WorkspaceNotFoundError(RuntimeError):
    """Raised when no .mlbuild workspace can be located."""


class InvalidWorkspaceError(RuntimeError):
    """Raised when .mlbuild exists but is incomplete or corrupted."""


class ActiveRunStore:
    """
    Persist and retrieve the active run ID from a workspace.

    Infrastructure component.
    Does not perform domain validation of run existence.
    """

    ACTIVE_FILE = "active_run"
    LOCK_FILE = "active_run.lock"
    REGISTRY_FILE = "registry.db"

    def __init__(self, start_path: Optional[Path] = None) -> None:
        start_path = start_path or Path.cwd()
        self._workspace = self._discover_workspace(start_path)
        self._validate_workspace(self._workspace)

        self._path = self._workspace / self.ACTIVE_FILE
        self._lock_path = self._workspace / self.LOCK_FILE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set(self, run_id: str) -> None:
        """
        Persist active run ID atomically.

        Raises:
            ValueError: if run_id invalid
            OSError: filesystem failure
        """
        run_id = (run_id or "").strip()
        if not run_id:
            raise ValueError("run_id cannot be empty")

        with self._file_lock():
            tmp_path = self._path.with_suffix(".tmp")
            tmp_path.write_text(run_id, encoding="utf-8")
            tmp_path.replace(self._path)  # atomic on POSIX

    def get(self) -> Optional[str]:
        """
        Retrieve active run ID.
        Returns None if not set.
        """
        if not self._path.exists():
            return None

        content = self._path.read_text(encoding="utf-8").strip()
        return content or None

    def clear(self) -> None:
        """
        Remove active run marker.
        """
        with self._file_lock():
            if self._path.exists():
                self._path.unlink()

    # ------------------------------------------------------------------
    # Workspace Discovery
    # ------------------------------------------------------------------

    @classmethod
    def _discover_workspace(cls, start: Path) -> Path:
        """
        Walk upward until a valid .mlbuild directory is found.
        """
        current = start.resolve()

        while True:
            candidate = current / ".mlbuild"
            if candidate.exists() and candidate.is_dir():
                return candidate

            if current == current.parent:
                break  # reached filesystem root

            current = current.parent

        raise WorkspaceNotFoundError(
            "No .mlbuild workspace found. Run `mlbuild init`."
        )

    @classmethod
    def _validate_workspace(cls, workspace: Path) -> None:
        """
        Ensure workspace is structurally valid.
        Prevents partial/fake .mlbuild directories.
        """
        registry = workspace / cls.REGISTRY_FILE

        if not registry.exists():
            raise InvalidWorkspaceError(
                f"Incomplete workspace at {workspace}. "
                "Missing registry.db. Run `mlbuild init`."
            )

    # ------------------------------------------------------------------
    # Concurrency Protection
    # ------------------------------------------------------------------

    @contextmanager
    def _file_lock(self):
        """
        Cross-platform advisory lock using lock file.

        Lightweight protection against concurrent writes
        across multiple CLI processes.
        """
        fd = None
        try:
            fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR)

            if os.name == "posix":
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX)
            elif os.name == "nt":
                import msvcrt

                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)

            yield

        finally:
            if fd is not None:
                if os.name == "posix":
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_UN)
                elif os.name == "nt":
                    import msvcrt

                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

                os.close(fd)