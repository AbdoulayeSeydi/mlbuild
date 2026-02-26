from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import fcntl

from pathlib import Path
from typing import Dict, List, Optional, Tuple, BinaryIO

# ============================================================
# Constants / Limits
# ============================================================

MAX_ARTIFACT_SIZE = 5 * 1024 * 1024 * 1024  # 5GB hard limit
MAX_METADATA_SIZE = 5 * 1024 * 1024         # 5MB
BUILD_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


# ============================================================
# Exceptions
# ============================================================

class StorageError(Exception):
    pass


class IntegrityError(StorageError):
    pass


class ValidationError(StorageError):
    pass


class ConcurrencyError(StorageError):
    pass


# ============================================================
# Utility Functions
# ============================================================

def validate_build_id(build_id: str) -> None:
    # Allow both full IDs and prefixes
    if not re.match(r"^[a-zA-Z0-9_-]{8,128}$", build_id):
        raise ValidationError(
            "Invalid build_id. Must be 8-128 alphanumeric characters."
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()

    for member in tar.getmembers():
        member_path = (destination / member.name).resolve()
        if not str(member_path).startswith(str(destination)):
            raise StorageError("Unsafe tar member path detected.")

        if member.isdev():
            raise StorageError("Device files not allowed in archive.")

    tar.extractall(destination)


def tar_filter(member: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
    if member.isdev():
        return None
    if member.issym():
        return None
    if member.name.startswith("/"):
        return None
    return member


# ============================================================
# Local Artifact Backend
# ============================================================

class LocalArtifactBackend:

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path).resolve()
        self.artifacts_dir = self.base_path / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # File Locking (per artifact)
    # --------------------------------------------------------

    def _lock_path(self, build_id: str) -> Path:
        return self.artifacts_dir / f"{build_id}.lock"

    def _acquire_lock(self, build_id: str):
        lock_path = self._lock_path(build_id)
        lock_file = open(lock_path, "w")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            return lock_file
        except Exception:
            lock_file.close()
            raise ConcurrencyError("Failed to acquire lock.")

    # --------------------------------------------------------
    # Upload
    # --------------------------------------------------------

    def upload_artifact(
        self,
        build_id: str,
        artifact_path: Path,
        overwrite: bool = False,
    ) -> Dict[str, str]:

        validate_build_id(build_id)

        dest = self.artifacts_dir / f"{build_id}.tar.gz"
        lock_file = self._acquire_lock(build_id)

        try:
            if dest.exists() and not overwrite:
                raise FileExistsError(f"Artifact exists: {build_id}")

            with tempfile.NamedTemporaryFile(
                dir=self.artifacts_dir,
                delete=False,
                suffix=".tar.gz"
            ) as tmp_file:

                tmp_path = Path(tmp_file.name)

                with tarfile.open(tmp_path, "w:gz") as tar:
                    tar.add(
                        artifact_path,
                        arcname=artifact_path.name,
                        recursive=True,
                        filter=tar_filter,
                    )

                tmp_file.flush()
                os.fsync(tmp_file.fileno())

            size = tmp_path.stat().st_size
            if size > MAX_ARTIFACT_SIZE:
                tmp_path.unlink(missing_ok=True)
                raise StorageError("Artifact exceeds size limit.")

            checksum = sha256_file(tmp_path)

            os.replace(tmp_path, dest)

            return {
                "path": str(dest),
                "sha256": checksum,
                "size": str(size),
            }

        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    def download_artifact(
        self,
        build_id: str,
        destination: Path,
        expected_sha256: Optional[str] = None,
    ) -> None:

        validate_build_id(build_id)

        source = self.artifacts_dir / f"{build_id}.tar.gz"
        if not source.exists():
            raise FileNotFoundError(build_id)

        if expected_sha256:
            actual = sha256_file(source)
            if actual != expected_sha256:
                raise IntegrityError("Checksum mismatch.")

        destination.mkdir(parents=True, exist_ok=True)

        with tarfile.open(source, "r:gz") as tar:
            safe_extract(tar, destination)

    # --------------------------------------------------------
    # Delete
    # --------------------------------------------------------

    def delete_artifact(self, build_id: str) -> None:
        validate_build_id(build_id)

        artifact_path = self.artifacts_dir / f"{build_id}.tar.gz"

        if not artifact_path.exists():
            raise FileNotFoundError(build_id)

        artifact_path.unlink()

    # --------------------------------------------------------
    # Stable Pagination (Lexicographic)
    # --------------------------------------------------------

    def list_artifacts(
        self,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> Tuple[List[str], Optional[str]]:

        files = sorted(self.artifacts_dir.glob("*.tar.gz"))

        start_index = 0
        if cursor:
            validate_build_id(cursor)
            for i, p in enumerate(files):
                if p.stem > cursor:
                    start_index = i
                    break

        page = files[start_index:start_index + limit]
        build_ids = [p.stem for p in page]

        next_cursor = build_ids[-1] if len(page) == limit else None

        return build_ids, next_cursor


# ============================================================
# Metadata Backend (Atomic + Size Limited)
# ============================================================

class LocalMetadataBackend:

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path).resolve()
        self.metadata_dir = self.base_path / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def upload_metadata(
        self,
        build_id: str,
        metadata: Dict,
        overwrite: bool = False,
    ) -> None:

        validate_build_id(build_id)

        dest = self.metadata_dir / f"{build_id}.json"

        encoded = json.dumps(metadata, sort_keys=True).encode()
        if len(encoded) > MAX_METADATA_SIZE:
            raise StorageError("Metadata exceeds size limit.")

        if dest.exists() and not overwrite:
            raise FileExistsError(build_id)

        with tempfile.NamedTemporaryFile(
            dir=self.metadata_dir,
            delete=False,
        ) as tmp_file:

            tmp_file.write(encoded)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())

            tmp_path = Path(tmp_file.name)

        os.replace(tmp_path, dest)

    def download_metadata(self, build_id: str) -> Dict:
        validate_build_id(build_id)

        # Try exact match first
        source = self.metadata_dir / f"{build_id}.json"
        
        if not source.exists():
            # Try prefix match
            matches = list(self.metadata_dir.glob(f"{build_id}*.json"))
            if not matches:
                raise FileNotFoundError(build_id)
            if len(matches) > 1:
                raise ValueError(f"Ambiguous build ID prefix: {build_id}")
            source = matches[0]

        return json.loads(source.read_text())
    
    def delete_metadata(self, build_id: str) -> None:
        """Delete metadata."""
        validate_build_id(build_id)
        
        metadata_path = self.metadata_dir / f"{build_id}.json"
        if not metadata_path.exists():
            raise FileNotFoundError(build_id)
        
        metadata_path.unlink()
    
    def list_builds(
        self,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> Tuple[List[str], Optional[str]]:
        """List builds with cursor pagination."""
        metadata_files = sorted(self.metadata_dir.glob("*.json"))
        
        start_index = 0
        if cursor:
            validate_build_id(cursor)
            for i, p in enumerate(metadata_files):
                if p.stem > cursor:
                    start_index = i
                    break
        
        page = metadata_files[start_index:start_index + limit]
        build_ids = [p.stem for p in page]
        
        next_cursor = build_ids[-1] if len(page) == limit else None
        
        return build_ids, next_cursor
    
    def ping(self) -> None:
        """Verify storage is accessible."""
        if not self.metadata_dir.exists():
            raise StorageError(f"Metadata directory not accessible")

    def iter_builds(self):
        """Iterate builds in pages for sync."""
        build_ids, _ = self.list_builds(limit=1000)
        yield [{"build_id": bid, "hash": "0"*64} for bid in build_ids]
    
# ============================================================
# Combined Backend
# ============================================================

class LocalStorageBackend:
    """
    Combined local storage backend (artifacts + metadata).
    """
    
    def __init__(self, base_path: Path):
        self.artifacts = LocalArtifactBackend(base_path)
        self.metadata = LocalMetadataBackend(base_path)
        self._base_path = base_path
    
    def ping(self) -> None:
        """Verify both backends are accessible."""
        if not self.artifacts.artifacts_dir.exists():
            raise StorageError(f"Artifacts directory not accessible")
        if not self.metadata.metadata_dir.exists():
            raise StorageError(f"Metadata directory not accessible")
    
    def validate_credentials(self) -> None:
        """Validate access (no-op for local)."""
        pass
    
    def supports_cas(self) -> bool:
        """Content-addressable storage support."""
        return False
    
    def policy(self):
        """Return storage policy."""
        from types import SimpleNamespace
        return SimpleNamespace(
            max_artifact_size=5 * 1024 * 1024 * 1024,  # 5GB
        )
    
    def begin_transaction(self, build_id: str):
        """Begin transaction (no-op for local)."""
        return None
    
    def commit(self, transaction):
        """Commit transaction (no-op for local)."""
        pass
    
    def abort(self, transaction):
        """Abort transaction (no-op for local)."""
        pass
    
    def put_artifact(self, build_id: str, artifact_path: Path, overwrite: bool = False, atomic: bool = True):
        """Upload artifact."""
        return self.artifacts.upload_artifact(build_id, artifact_path, overwrite=overwrite)
    
    def put_metadata(self, build_id: str, metadata: Dict, transaction=None):
        """Upload metadata."""
        self.metadata.upload_metadata(build_id, metadata, overwrite=True)
    
    def audit(self, action: str, data: Dict):
        """Audit log (no-op for now)."""
        pass
    
    def telemetry(self, event: str, data: Dict):
        """Telemetry (no-op for now)."""
        pass