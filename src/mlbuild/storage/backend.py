from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple


# ============================================================
# Exception Hierarchy
# ============================================================

class StorageError(Exception):
    """Base class for all storage-related failures."""


class ConnectivityError(StorageError):
    """Raised when backend is unreachable or misconfigured."""


class RetryableError(StorageError):
    """Temporary failure, safe to retry."""


class PolicyViolationError(StorageError):
    """Operation violates backend policy."""


class IntegrityError(StorageError):
    """Checksum/data corruption detected."""


class ArtifactNotFoundError(StorageError):
    """Artifact does not exist."""


class ArtifactAlreadyExistsError(StorageError):
    """Artifact exists and overwrite=False."""


class MetadataNotFoundError(StorageError):
    """Metadata does not exist."""


class MetadataConflictError(StorageError):
    """Version or overwrite conflict in metadata."""


# ============================================================
# Artifact Storage Backend
# ============================================================

class ArtifactStorageBackend(ABC):
    """
    Binary artifact storage (S3, GCS, Azure, filesystem, etc.).

    This interface is strictly for large binary objects.
    """

    @abstractmethod
    def upload_artifact(
        self,
        build_id: str,
        artifact_path: Path,
        *,
        overwrite: bool = False,
        atomic: bool = True,
    ) -> str:
        """
        Upload artifact from filesystem.

        Args:
            build_id: Unique identifier.
            artifact_path: Directory or archive to upload.
            overwrite: If False, raises ArtifactAlreadyExistsError.
            atomic: If True, must ensure artifact is not partially visible.

        Returns:
            Canonical remote URI.

        Raises:
            FileNotFoundError
            ArtifactAlreadyExistsError
            StorageError
        """
        raise NotImplementedError

    @abstractmethod
    def upload_stream(
        self,
        build_id: str,
        stream: BinaryIO,
        *,
        overwrite: bool = False,
    ) -> str:
        """
        Stream artifact directly to backend.

        Required for large artifacts and CI pipelines.
        """
        raise NotImplementedError

    @abstractmethod
    def download_artifact(
        self,
        build_id: str,
        destination: Path,
    ) -> None:
        """
        Download artifact into destination directory.

        Raises:
            ArtifactNotFoundError
            StorageError
        """
        raise NotImplementedError

    @abstractmethod
    def delete_artifact(self, build_id: str) -> None:
        """
        Delete artifact.

        Must be idempotent.
        """
        raise NotImplementedError

    @abstractmethod
    def artifact_exists(self, build_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_artifact_size(self, build_id: str) -> int:
        """
        Returns artifact size in bytes.

        Raises:
            ArtifactNotFoundError
        """
        raise NotImplementedError

    @abstractmethod
    def list_artifacts(
        self,
        *,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> Tuple[List[str], Optional[str]]:
        """
        Cursor-based pagination.

        Returns:
            (build_ids, next_cursor)

        next_cursor is None when exhausted.
        """
        raise NotImplementedError

    @abstractmethod
    def ping(self) -> None:
        """
        Validate connectivity and credentials.

        Raises:
            ConnectivityError
        """
        raise NotImplementedError


# ============================================================
# Metadata Backend
# ============================================================

class BuildMetadataBackend(ABC):
    """
    Metadata storage for build records.

    Should typically be:
        - Postgres
        - SQLite (local mode)
        - DynamoDB
        - etc.

    Must provide strong consistency.
    """

    @abstractmethod
    def save_metadata(
        self,
        build_id: str,
        metadata: Dict,
        *,
        schema_version: int,
        overwrite: bool = False,
    ) -> None:
        """
        Persist build metadata.

        Args:
            build_id: Unique build identifier.
            metadata: JSON-serializable dict.
            schema_version: Metadata schema version.
            overwrite: If False, raise MetadataConflictError on conflict.

        Raises:
            MetadataConflictError
            StorageError
        """
        raise NotImplementedError

    @abstractmethod
    def load_metadata(
        self,
        build_id: str,
    ) -> Tuple[Dict, int]:
        """
        Retrieve metadata and schema version.

        Returns:
            (metadata_dict, schema_version)

        Raises:
            MetadataNotFoundError
        """
        raise NotImplementedError

    @abstractmethod
    def delete_metadata(self, build_id: str) -> None:
        """
        Delete metadata.

        Must be idempotent.
        """
        raise NotImplementedError

    @abstractmethod
    def list_builds(
        self,
        *,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> Tuple[List[str], Optional[str]]:
        """
        Cursor-based pagination for metadata store.
        """
        raise NotImplementedError

    @abstractmethod
    def ping(self) -> None:
        raise NotImplementedError