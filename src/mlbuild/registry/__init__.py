"""Registry for storing builds and benchmarks."""

from .local import LocalRegistry
from .schema import SCHEMA_SQL, SCHEMA_VERSION

__all__ = ["LocalRegistry", "SCHEMA_SQL", "SCHEMA_VERSION"]