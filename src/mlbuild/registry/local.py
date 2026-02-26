"""
Enterprise-grade local SQLite registry for MLBuild.

Stored at: .mlbuild/registry.db

Invariants:
- Builds are immutable.
- build_id = SHA256(source_hash + config_hash + artifact_hash)
- ISO8601 UTC timestamps.
- Canonical JSON storage (sorted keys, compact separators).
- Tags are Docker-style aliases (1 tag -> 1 build).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Optional, Any
from datetime import datetime, timezone
from decimal import Decimal
from contextlib import contextmanager

from .schema import SCHEMA_SQL
from ..core.types import Build, Benchmark
from ..core.errors import InternalError

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts)


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


# ------------------------------------------------------------
# Registry
# ------------------------------------------------------------

class LocalRegistry:
    """
    Hardened local SQLite registry.

    - WAL mode enabled
    - Foreign keys enforced
    - Deterministic JSON storage
    - Build immutability enforced
    """

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path.cwd() / ".mlbuild" / "registry.db"

        self.db_path = Path(db_path).resolve()
        logger.debug(f"LocalRegistry using DB: {self.db_path}")

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # DON'T call _init_db() here - only verify it exists
        if not self.db_path.exists():
            raise InternalError(
                f"Registry not initialized. Run 'mlbuild init' first.\n"
                f"Expected database at: {self.db_path}"
            )

    def init_schema(self) -> None:
        """Initialize database schema. Called only by 'mlbuild init'."""
        from .schema import SCHEMA_VERSION

        with self._connect() as conn:
            # Check if schema_version table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
            )
            schema_table_exists = cursor.fetchone() is not None

            if schema_table_exists:
                # Database exists - check version
                cursor = conn.execute("SELECT version FROM schema_version WHERE id = 1")
                row = cursor.fetchone()

                if row:
                    db_version = row[0]

                    if db_version < SCHEMA_VERSION:
                        raise InternalError(
                            f"Database schema v{db_version} is outdated. "
                            f"Expected v{SCHEMA_VERSION}. "
                            f"Migration required but not yet implemented."
                        )
                    elif db_version > SCHEMA_VERSION:
                        raise InternalError(
                            f"Database schema v{db_version} is newer than MLBuild v{SCHEMA_VERSION}. "
                            f"Upgrade MLBuild."
                        )

            # Create or update schema
            try:
                conn.executescript(SCHEMA_SQL)
            except sqlite3.OperationalError as e:
                raise InternalError(f"Schema initialization failed: {e}")

            # Verify version after init
            cursor = conn.execute("SELECT version FROM schema_version WHERE id = 1")
            row = cursor.fetchone()

            if row and row[0] != SCHEMA_VERSION:
                raise InternalError(
                    f"Schema version mismatch after init: got {row[0]}, expected {SCHEMA_VERSION}"
                )

    def verify_schema(self) -> tuple[bool, str]:
        """
        Verify database schema version.
        
        Returns:
            (is_valid, message)
        """
        from .schema import SCHEMA_VERSION
        
        with self._connect() as conn:
            try:
                cursor = conn.execute(
                    "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
                )
                row = cursor.fetchone()
                
                if not row:
                    return False, "No schema version found in database"
                
                db_version = row[0]
                if db_version == SCHEMA_VERSION:
                    return True, f"Schema version {SCHEMA_VERSION} (current)"
                elif db_version < SCHEMA_VERSION:
                    return False, f"Schema outdated: DB v{db_version}, Expected v{SCHEMA_VERSION} (migration needed)"
                else:
                    return False, f"Schema too new: DB v{db_version}, Expected v{SCHEMA_VERSION} (downgrade MLBuild)"
                    
            except sqlite3.OperationalError as e:
                return False, f"Schema verification failed: {e}"

    # ------------------------------------------------------------
    # Connection Management
    # ------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            isolation_level=None,  # explicit transactions
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    # ------------------------------------------------------------
    # Build Operations
    # ------------------------------------------------------------

    def save_build(self, build: Build) -> None:
        """
        Insert immutable build.
        Raises if duplicate or conflicting.
        """
        import logging
        logger = logging.getLogger(__name__)

        # Pre-DB validation
        if len(build.build_id) != 64:
            raise InternalError("Invalid build_id length")

        if len(build.artifact_hash) != 64:
            raise InternalError("Invalid artifact_hash length")

        if build.size_mb < 0:
            raise InternalError("size_mb cannot be negative")
        
        logger.info(f"Attempting to save build: {build.build_id[:16]}...")
        logger.info(f"  artifact_hash: {build.artifact_hash[:16]}...")
        logger.info(f"  env_fingerprint: {build.env_fingerprint[:16]}...")

        with self._connect() as conn:
            try:
                conn.execute("BEGIN;")
                
                values = (
                    build.build_id,
                    build.artifact_hash,
                    build.source_hash,
                    build.config_hash,
                    build.env_fingerprint,
                    build.name,
                    build.notes,
                    build.created_at.isoformat().replace("+00:00", "Z"),
                    build.source_path,
                    build.target_device,
                    build.format,
                    _canonical_json(build.quantization),
                    _canonical_json(build.optimizer_config),
                    _canonical_json(build.backend_versions),
                    _canonical_json(build.environment_data),
                    build.mlbuild_version,
                    build.python_version,
                    build.platform,
                    build.os_version,
                    build.artifact_path,
                    int(build.size_mb * 1024 * 1024),
                )
                
                logger.info(f"Inserting {len(values)} values into builds table")
                
                conn.execute(
                    """
                    INSERT INTO builds (
                        build_id,
                        artifact_hash,
                        source_hash,
                        config_hash,
                        env_fingerprint,
                        name,
                        notes,
                        created_at,
                        source_path,
                        target_device,
                        format,
                        quantization,
                        optimizer_config,
                        backend_versions,
                        environment_data,
                        mlbuild_version,
                        python_version,
                        platform,
                        os_version,
                        artifact_path,
                        size_bytes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
                conn.execute("COMMIT;")
                logger.info("Build saved successfully!")
            except sqlite3.IntegrityError as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"SQLite IntegrityError: {exc}")
                logger.error(f"Error message: {str(exc)}")
                
                cursor = conn.execute(
                    "SELECT build_id, artifact_hash, env_fingerprint, created_at FROM builds LIMIT 5"
                )
                rows = cursor.fetchall()
                logger.error(f"Current builds in DB: {len(rows)}")
                for row in rows:
                    logger.error(f"  {row['build_id'][:16]}... | {row['artifact_hash'][:16]}... | {row['created_at']}")
                
                raise InternalError(
                    "Build already exists or violates uniqueness constraints"
                ) from exc

    def get_build(self, build_id: str) -> Optional[Build]:
        """
        Resolve exact build_id only.
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM builds
                WHERE build_id = ?
                AND deleted_at IS NULL
                """,
                (build_id,),
            ).fetchone()

            return self._row_to_build(row) if row else None

    def resolve_build(self, identifier: str) -> Optional[Build]:
            """
            Resolve by:
            1. Tag (exact match)
            2. Exact ID
            3. Unique prefix
            4. Exact name

            Raises on ambiguity.
            """
            
            with self._connect() as conn:
                # 1. Check if it's a tag
                tag_row = conn.execute(
                    """
                    SELECT build_id FROM tags
                    WHERE tag = ?
                    """,
                    (identifier,),
                ).fetchone()
                
                if tag_row:
                    build_id = tag_row['build_id']
                    return self.get_build(build_id)
            
            # 2. Exact ID
            build = self.get_build(identifier)
            if build:
                return build

            with self._connect() as conn:
                # 3. Prefix match
                rows = conn.execute(
                    """
                    SELECT * FROM builds
                    WHERE build_id LIKE ? || '%'
                    AND deleted_at IS NULL
                    """,
                    (identifier,),
                ).fetchall()

                if len(rows) == 1:
                    return self._row_to_build(rows[0])
                elif len(rows) > 1:
                    raise InternalError("Ambiguous build prefix")

                # 4. Exact name
                row = conn.execute(
                    """
                    SELECT * FROM builds
                    WHERE name = ?
                    AND deleted_at IS NULL
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (identifier,),
                ).fetchone()

                return self._row_to_build(row) if row else None
            
    def iter_builds_sorted(self):
        """Iterate builds in sorted order for sync."""
        builds = self.list_builds(limit=1000, order_by=("build_id ASC",))
        for build in builds:
            yield {
                "build_id": build.build_id,
                "hash": build.artifact_hash,
            }
            

    def exists(self, build_id: str) -> bool:
        """Check if a build exists in the registry."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM builds WHERE build_id = ?",
                (build_id,)
            ).fetchone()
            return row is not None
        
    def get_build_by_prefix(self, prefix: str) -> list:
        """
        Get builds by partial build ID prefix.
        
        Args:
            prefix: Partial build ID (e.g., "46f5637")
            
        Returns:
            List of matching Build objects
        """
        builds = self.list_builds(limit=1000) 
        return [b for b in builds if b.build_id.startswith(prefix)]

    def list_builds(
        self,
        limit: int = 50,
        offset: int = 0,
        target: str | None = None,
        name_pattern: str | None = None,
        tag: str | None = None,
        include_deleted: bool = False,
        date_from: str | None = None,
        date_to: str | None = None,
        order_by: tuple = ("created_at DESC", "build_id ASC"),
    ) -> List[Build]:
        """List builds with advanced filtering and pagination."""
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        
        if offset < 0:
            raise ValueError("offset must be non-negative")
        
        with self._connect() as conn:
            where_clauses = []
            params = []
            
            if not include_deleted:
                where_clauses.append("deleted_at IS NULL")
            
            if target:
                where_clauses.append("target_device = ?")
                params.append(target)
            
            if name_pattern:
                where_clauses.append("name LIKE ?")
                params.append(name_pattern)
            
            if tag:
                where_clauses.append(
                    "build_id IN (SELECT build_id FROM tags WHERE tag = ?)"
                )
                params.append(tag)
            
            if date_from:
                where_clauses.append("created_at >= ?")
                params.append(date_from)
            
            if date_to:
                where_clauses.append("created_at <= ?")
                params.append(date_to)
            
            where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
            order_sql = ", ".join(order_by)
            
            sql = f"""
                SELECT * FROM builds
                WHERE {where_sql}
                ORDER BY {order_sql}
                LIMIT ? OFFSET ?
            """
            
            params.extend([limit, offset])
            
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_build(r) for r in rows]
    
   
    @contextmanager
    def transaction(self):
        """Transaction context manager."""
        with self._connect() as conn:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    # ------------------------------------------------------------
    # Benchmarks
    # ------------------------------------------------------------

    def save_benchmark(self, benchmark: Benchmark) -> None:
        """Save benchmark result to database."""
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO benchmarks (
                    build_id, device_chip, runtime, measurement_type,
                    compute_unit, latency_p50_ms, latency_p95_ms, latency_p99_ms,
                    memory_peak_mb, num_runs, measured_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                benchmark.build_id,
                benchmark.device_chip,
                benchmark.runtime.value if hasattr(benchmark.runtime, 'value') else benchmark.runtime,
                benchmark.measurement_type.value if hasattr(benchmark.measurement_type, 'value') else benchmark.measurement_type,
                benchmark.compute_unit.value if hasattr(benchmark.compute_unit, 'value') else benchmark.compute_unit,
                benchmark.latency_p50_ms,
                benchmark.latency_p95_ms,
                benchmark.latency_p99_ms,
                benchmark.memory_peak_mb,
                benchmark.num_runs,
                benchmark.measured_at.isoformat(),
            ))
            conn.commit()

    # ------------------------------------------------------------
    # Tags (Docker-style alias)
    # ------------------------------------------------------------

    def add_tag(self, build_id: str, tag: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tags (tag, build_id, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(tag)
                DO UPDATE SET build_id = excluded.build_id
                """,
                (tag, build_id, _utc_now_iso()),
            )

    def stream_builds(self, limit: int = 1000):
        """Stream builds efficiently."""
        return self.list_builds(limit=limit)

    def get_build_by_tag(self, tag: str):
        """Get build by tag name."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT build_id FROM tags WHERE tag = ?",
                (tag,)
            ).fetchone()

            if not row:
                return None

            return self.get_build(row['build_id'])

    # ------------------------------------------------------------
    # Row Mapping
    # ------------------------------------------------------------

    def _row_to_build(self, row: sqlite3.Row) -> Build:
        """Convert database row to Build object with backward compatibility."""
        
        def safe_get(row, key, default=None):
            try:
                return row[key]
            except (KeyError, IndexError):
                return default
        
        return Build(
            build_id=row["build_id"],
            artifact_hash=row["artifact_hash"],
            source_hash=row["source_hash"],
            config_hash=row["config_hash"],
            env_fingerprint=safe_get(row, "env_fingerprint", ""),
            name=row["name"],
            notes=row["notes"],
            created_at=_parse_iso(row["created_at"]),
            source_path=row["source_path"],
            target_device=row["target_device"],
            format=row["format"],
            quantization=json.loads(row["quantization"]),
            optimizer_config=json.loads(row["optimizer_config"]),
            backend_versions=json.loads(row["backend_versions"]),
            environment_data=json.loads(safe_get(row, "environment_data", "{}")),
            mlbuild_version=row["mlbuild_version"],
            python_version=row["python_version"],
            platform=row["platform"],
            os_version=row["os_version"],
            artifact_path=row["artifact_path"],
            size_mb=Decimal(row["size_bytes"]) / Decimal(1024 * 1024),
        )