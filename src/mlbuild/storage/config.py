"""
Enterprise-grade remote storage configuration system.

Design goals:
- Strong typing
- Strict validation
- Deterministic behavior
- Atomic writes
- Cross-process safety
- Schema versioning
- Clear separation of concerns
"""

from __future__ import annotations

import os
import yaml
import tempfile
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from contextlib import contextmanager

# ================================
# Exceptions
# ================================


class ConfigError(Exception):
    """Base configuration error."""


class ValidationError(ConfigError):
    """Configuration validation failure."""


class NotFoundError(ConfigError):
    """Requested remote does not exist."""


class SchemaError(ConfigError):
    """Invalid or unsupported schema version."""


# ================================
# Backend Enum
# ================================


class Backend(str, Enum):
    S3 = "s3"
    LOCAL = "local"


# ================================
# Domain Model
# ================================


@dataclass(frozen=True)
class RemoteConfig:
    """
    Immutable, validated remote configuration.
    """

    name: str
    backend: Backend

    # S3
    bucket: Optional[str] = None
    prefix: str = "mlbuild/"
    region: Optional[str] = None
    endpoint_url: Optional[str] = None

    # Local
    path: Optional[Path] = None

    default: bool = False

    def __post_init__(self):
        object.__setattr__(self, "prefix", self._normalize_prefix(self.prefix))
        self._validate()

    # ----------------------------
    # Validation
    # ----------------------------

    def _validate(self) -> None:
        if not isinstance(self.backend, Backend):
            raise ValidationError(f"Invalid backend: {self.backend}")

        if not self.name or not self.name.strip():
            raise ValidationError("Remote name must be non-empty.")

        if self.backend is Backend.S3:
            if not self.bucket:
                raise ValidationError("S3 backend requires 'bucket'.")
            if self.path is not None:
                raise ValidationError("S3 backend cannot define local 'path'.")

        if self.backend is Backend.LOCAL:
            if not self.path:
                raise ValidationError("Local backend requires 'path'.")
            expanded = self.path.expanduser().resolve()
            object.__setattr__(self, "path", expanded)
            if self.bucket is not None:
                raise ValidationError("Local backend cannot define 'bucket'.")

        if not self.prefix:
            raise ValidationError("Prefix cannot be empty.")

    # ----------------------------
    # Prefix normalization
    # ----------------------------

    @staticmethod
    def _normalize_prefix(prefix: str) -> str:
        prefix = prefix.strip()
        prefix = prefix.lstrip("/")
        prefix = prefix.rstrip("/") + "/"

        if ".." in prefix:
            raise ValidationError("Prefix cannot contain '..'.")

        return prefix


# ================================
# Serialization Layer
# ================================


class RemoteConfigSchema:
    """
    Responsible ONLY for (de)serialization.
    """

    VERSION = 1

    @classmethod
    def load(cls, raw: dict) -> Dict[str, RemoteConfig]:
        if not raw:
            return {}

        version = raw.get("version")
        if version != cls.VERSION:
            raise SchemaError(
                f"Unsupported config version: {version}. "
                f"Expected version {cls.VERSION}."
            )

        remotes_raw = raw.get("remotes", {})
        remotes: Dict[str, RemoteConfig] = {}

        for name, data in remotes_raw.items():
            try:
                config = RemoteConfig(
                    name=name,
                    backend=Backend(data["backend"]),
                    bucket=data.get("bucket"),
                    prefix=data.get("prefix", "mlbuild/"),
                    region=data.get("region"),
                    endpoint_url=data.get("endpoint_url"),
                    path=Path(data["path"]) if data.get("path") else None,
                    default=data.get("default", False),
                )
                remotes[name] = config
            except Exception as e:
                raise ValidationError(
                    f"Invalid configuration for remote '{name}': {e}"
                ) from e

        cls._validate_default(remotes)
        return remotes

    @classmethod
    def dump(cls, remotes: Dict[str, RemoteConfig]) -> dict:
        cls._validate_default(remotes)

        return {
            "version": cls.VERSION,
            "remotes": {
                name: cls._to_dict(config)
                for name, config in remotes.items()
            },
        }

    @staticmethod
    def _to_dict(config: RemoteConfig) -> dict:
        data = {
            "backend": config.backend.value,
            "prefix": config.prefix,
            "default": config.default,
        }

        if config.bucket:
            data["bucket"] = config.bucket
        if config.region:
            data["region"] = config.region
        if config.endpoint_url:
            data["endpoint_url"] = config.endpoint_url
        if config.path:
            data["path"] = str(config.path)

        return data

    @staticmethod
    def _validate_default(remotes: Dict[str, RemoteConfig]) -> None:
        defaults = [r for r in remotes.values() if r.default]
        if len(defaults) > 1:
            raise ValidationError("Multiple remotes marked as default.")
        # Zero default is allowed intentionally.


# ================================
# Repository Layer
# ================================


class RemoteRepository:
    """
    Persistence + concurrency layer.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = (
            config_path
            if config_path
            else Path.cwd() / ".mlbuild" / "remotes.yaml"
        )

        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Optional[Dict[str, RemoteConfig]] = None

    # ----------------------------
    # File locking (POSIX)
    # ----------------------------

    @contextmanager
    def _locked_file(self, mode: str):
        import fcntl

        with open(self.config_path, mode) as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield f
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    # ----------------------------
    # Load
    # ----------------------------

    def load(self) -> Dict[str, RemoteConfig]:
        if self._cache is not None:
            return self._cache

        if not self.config_path.exists():
            self._cache = {}
            return self._cache

        try:
            raw = yaml.safe_load(self.config_path.read_text())
            remotes = RemoteConfigSchema.load(raw)
            self._cache = remotes
            return remotes
        except yaml.YAMLError as e:
            raise ConfigError("Malformed YAML configuration.") from e
        except OSError as e:
            raise ConfigError("Failed to read configuration file.") from e

    # ----------------------------
    # Atomic Save
    # ----------------------------

    def save(self, remotes: Dict[str, RemoteConfig]) -> None:
        data = RemoteConfigSchema.dump(remotes)

        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self.config_path.parent
        )

        try:
            with os.fdopen(tmp_fd, "w") as tmp_file:
                yaml.safe_dump(data, tmp_file, sort_keys=False)

            os.replace(tmp_path, self.config_path)
            self._cache = remotes
        except Exception:
            os.unlink(tmp_path)
            raise

    # ----------------------------
    # Operations
    # ----------------------------

    def get(self, name: str) -> RemoteConfig:
        remotes = self.load()
        if name not in remotes:
            raise NotFoundError(f"Remote '{name}' does not exist.")
        return remotes[name]

    def get_backend(self, name: str):
        """Get backend instance for a remote."""
        config = self.get(name)
        
        if config.backend == Backend.LOCAL:
            from .local import LocalStorageBackend
            return LocalStorageBackend(config.path)
        elif config.backend == Backend.S3:
            raise NotImplementedError("S3 backend not yet implemented")
        else:
            raise ValueError(f"Unknown backend: {config.backend}")

    def get_default(self) -> Optional[RemoteConfig]:
        remotes = self.load()
        for r in remotes.values():
            if r.default:
                return r
        return None

    def add(self, config: RemoteConfig) -> None:
        remotes = self.load().copy()
        remotes[config.name] = config
        self.save(remotes)

    def remove(self, name: str) -> None:
        remotes = self.load().copy()
        if name not in remotes:
            raise NotFoundError(f"Remote '{name}' does not exist.")
        del remotes[name]
        self.save(remotes)

    def set_default(self, name: str) -> None:
        remotes = self.load().copy()

        if name not in remotes:
            raise NotFoundError(f"Remote '{name}' does not exist.")

        updated = {}
        for rname, config in remotes.items():
            updated[rname] = RemoteConfig(
                **{
                    **config.__dict__,
                    "default": rname == name,
                }
            )

        self.save(updated)