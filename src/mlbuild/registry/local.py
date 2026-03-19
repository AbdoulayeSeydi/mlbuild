from __future__ import annotations

"""
Local SQLite registry for MLBuild.

Stored at: .mlbuild/registry.db

Invariants:
- Builds are immutable.
- build_id = SHA256(source_hash + config_hash + artifact_hash)
- ISO8601 UTC timestamps.
- Canonical JSON storage (sorted keys, compact separators).
- Tags are Docker-style aliases (1 tag -> 1 build).
"""


import json
import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Any
from datetime import datetime, timezone
from decimal import Decimal
from contextlib import contextmanager

from .schema import SCHEMA_SQL

from ..core.types import Build, Benchmark
from ..core.errors import InternalError

if TYPE_CHECKING:
    from ..core.accuracy.config import AccuracyResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _parse_since(since: str) -> datetime | None:
    """
    Parse human-readable since strings to UTC datetime.
    Supports: 'yesterday', 'N days ago', 'YYYY-MM-DD'
    Returns None if unparseable.
    """
    import re
    from datetime import timedelta

    since = since.strip().lower()
    now = datetime.now(timezone.utc)

    if since == "yesterday":
        return (now - timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    match = re.match(r"(\d+)\s+days?\s+ago", since)
    if match:
        days = int(match.group(1))
        return (now - timedelta(days=days)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    try:
        dt = datetime.strptime(since, "%Y-%m-%d")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass

    return None

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts)


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _is_imported(build: Build) -> int:
    """Return 1 if this build was registered via mlbuild import, 0 otherwise."""
    return 1 if build.backend_versions.get("imported") == "true" else 0


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
        from .schema import SCHEMA_VERSION, SCHEMA_SQL, get_migration

        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
            )
            schema_table_exists = cursor.fetchone() is not None

            if schema_table_exists:
                cursor = conn.execute("SELECT version FROM schema_version WHERE id = 1")
                row = cursor.fetchone()

                if row:
                    db_version = row[0]

                    if db_version > SCHEMA_VERSION:
                        raise InternalError(
                            f"Database schema v{db_version} is newer than MLBuild "
                            f"v{SCHEMA_VERSION}. Upgrade MLBuild."
                        )

                    # Run all pending migrations in order
                    while db_version < SCHEMA_VERSION:
                        sql = get_migration(db_version, db_version + 1)
                        if not sql:
                            raise InternalError(
                                f"No migration path from v{db_version} to "
                                f"v{db_version + 1}. Cannot upgrade automatically."
                            )
                        logger.info(f"Migrating schema v{db_version} → v{db_version + 1}...")
                        conn.executescript(sql)
                        db_version += 1
                        logger.info(f"Migration to v{db_version} complete.")
                    return

            # Fresh install — create full schema
            try:
                conn.executescript(SCHEMA_SQL)
            except sqlite3.OperationalError as e:
                raise InternalError(f"Schema initialization failed: {e}")

            cursor = conn.execute("SELECT version FROM schema_version WHERE id = 1")
            row = cursor.fetchone()
            if row and row[0] != SCHEMA_VERSION:
                raise InternalError(
                    f"Schema version mismatch after init: got {row[0]}, "
                    f"expected {SCHEMA_VERSION}"
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
                    _is_imported(build),
                    getattr(build, "task_type", None),
                    getattr(build, "variant_id", None),
                    getattr(build, "root_build_id", None),
                    getattr(build, "parent_build_id", None),
                    getattr(build, "optimization_pass", None),
                    getattr(build, "optimization_method", None),
                    getattr(build, "weight_precision", None),
                    getattr(build, "activation_precision", None),
                    1 if getattr(build, "has_graph", False) else 0,
                    getattr(build, "graph_format", None),
                    getattr(build, "graph_path", None),
                    getattr(build, "cached_latency_p50_ms", None),
                    getattr(build, "cached_latency_p95_ms", None),
                    getattr(build, "cached_memory_peak_mb", None),
                )
                
                logger.info(f"Inserting {len(values)} values into builds table")
                
                conn.execute(
                    """
                    INSERT INTO builds (
                        build_id, artifact_hash, source_hash, config_hash,
                        env_fingerprint, name, notes, created_at, source_path,
                        target_device, format, quantization, optimizer_config,
                        backend_versions, environment_data, mlbuild_version,
                        python_version, platform, os_version, artifact_path,
                        size_bytes, imported,
                        task_type,
                        variant_id, root_build_id, parent_build_id,
                        optimization_pass, optimization_method,
                        weight_precision, activation_precision,
                        has_graph, graph_format, graph_path,
                        cached_latency_p50_ms, cached_latency_p95_ms,
                        cached_memory_peak_mb
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?
                    )
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

    def resolve_build(self, identifier: str, roots_only: bool = False) -> Optional[Build]:
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
                prefix_where = "build_id LIKE ? || '%' AND deleted_at IS NULL"
                if roots_only:
                    prefix_where += " AND parent_build_id IS NULL"
                rows = conn.execute(
                    f"""
                    SELECT * FROM builds
                    WHERE {prefix_where}
                    """,
                    (identifier,),
                ).fetchall()
                if len(rows) == 1:
                    return self._row_to_build(rows[0])
                elif len(rows) > 1:
                    raise InternalError("Ambiguous build prefix")

                # 4. Exact name
                where = "name = ? AND deleted_at IS NULL"
                if roots_only:
                    where += " AND parent_build_id IS NULL"
                row = conn.execute(
                    f"""
                    SELECT * FROM builds
                    WHERE {where}
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

    @property
    def artifacts_root(self) -> Path:
        return self.db_path.parent / "artifacts"
    

    def update_cached_benchmark(
        self,
        build_id: str,
        latency_p50_ms: float,
        latency_p95_ms: float,
        memory_peak_mb: float,
    ) -> None:
        """Cache benchmark results on the build row for quick access."""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE builds
                SET cached_latency_p50_ms = ?,
                    cached_latency_p95_ms = ?,
                    cached_memory_peak_mb = ?
                WHERE build_id = ?
                """,
                (latency_p50_ms, latency_p95_ms, memory_peak_mb, build_id),
            )

    def find_baseline(
        self,
        source_hash: str,
        format: str,
        target_device: str,
    ) -> Optional[Build]:
        """
        Find existing root build by content identity.
        Never matches variants (parent_build_id IS NULL enforced).
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM builds
                WHERE source_hash = ?
                AND format = ?
                AND target_device = ?
                AND parent_build_id IS NULL
                AND deleted_at IS NULL
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (source_hash, format, target_device),
            ).fetchone()
        return self._row_to_build(row) if row else None
    
    # ------------------------------------------------------------
    # Accuracy Checks
    # ------------------------------------------------------------

    def save_accuracy_check(self, result: AccuracyResult) -> int:
        """
        Persist an accuracy check result to the registry.
        Returns the row id.
        """
        row = result.as_db_row()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO accuracy_checks (
                    baseline_build_id,
                    candidate_build_id,
                    cosine_similarity,
                    mean_abs_error,
                    max_abs_error,
                    top1_agreement,
                    num_samples,
                    seed,
                    passed,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["baseline_build_id"],
                    row["candidate_build_id"],
                    row["cosine_similarity"],
                    row["mean_abs_error"],
                    row["max_abs_error"],
                    row["top1_agreement"],
                    row["num_samples"],
                    row["seed"],
                    row["passed"],
                    row["created_at"],
                ),
            )
            return cursor.lastrowid

    def get_accuracy_checks(
        self,
        baseline_build_id: str,
        candidate_build_id: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """
        Retrieve accuracy check history for a baseline build.

        Parameters
        ----------
        baseline_build_id
            The reference build to look up checks for.
        candidate_build_id
            If provided, filter to checks against this specific candidate.
        limit
            Max rows to return, newest first.

        Returns
        -------
        list[dict]
            Raw dicts with all accuracy_checks columns.
        """
        with self._connect() as conn:
            if candidate_build_id:
                rows = conn.execute(
                    """
                    SELECT * FROM accuracy_checks
                    WHERE baseline_build_id = ?
                    AND   candidate_build_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (baseline_build_id, candidate_build_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM accuracy_checks
                    WHERE baseline_build_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (baseline_build_id, limit),
                ).fetchall()

            return [dict(r) for r in rows]

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
        
    def get_tag_row(self, tag: str) -> dict | None:
        """Return the raw tag row or None if not found."""
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM tags WHERE tag = ?",
                (tag,)
            ).fetchone()

    def delete_tag(self, tag: str) -> None:
        """Delete a tag by name. No-op if tag doesn't exist."""
        with self._connect() as conn:
            conn.execute("DELETE FROM tags WHERE tag = ?", (tag,))

    def get_baseline_history(self, limit: int = 20) -> list:
        """
        Return tags that look like baselines — mlbuild-baseline,
        main-*, baseline-*, production-*.
        """
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT tags.tag, tags.build_id, tags.created_at,
                    builds.name, builds.format, builds.target_device,
                    builds.cached_latency_p50_ms, builds.size_bytes
                FROM tags
                LEFT JOIN builds ON tags.build_id = builds.build_id
                WHERE tags.tag = 'mlbuild-baseline'
                OR tags.tag LIKE 'main-%'
                OR tags.tag LIKE 'baseline-%'
                OR tags.tag LIKE 'production-%'
                ORDER BY tags.created_at DESC
                LIMIT ?
                """,
                (limit,)
            ).fetchall()

    def resolve_tag(self, tag_or_id: str):
        """
        Resolve a tag name or build ID prefix to a Build.

        Tries tag lookup first, then falls back to resolve_build().
        Returns (build, tag_name) where tag_name is None if resolved by ID.
        """
        build = self.get_build_by_tag(tag_or_id)
        if build is not None:
            return build, tag_or_id
        build = self.resolve_build(tag_or_id)
        return build, None

    
    # ------------------------------------------------------------
    # Command Log
    # ------------------------------------------------------------

    def save_command(self, row: dict) -> None:
        """
        Insert one row into command_log.
        Always fails silently — history logging never surfaces errors
        or affects the result of the actual command.
        """
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO command_log (
                        id, machine_id, machine_name, platform,
                        command_name, args_json, raw_command,
                        linked_build_id, linked_benchmark_id,
                        exit_code, error_message, duration_ms,
                        mlbuild_version, ran_at
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        row["id"],
                        row["machine_id"],
                        row["machine_name"],
                        row["platform"],
                        row["command_name"],
                        row["args_json"],
                        row["raw_command"],
                        row.get("linked_build_id"),
                        row.get("linked_benchmark_id"),
                        row.get("exit_code", 0),
                        row.get("error_message"),
                        row.get("duration_ms", 0),
                        row["mlbuild_version"],
                        row["ran_at"],
                    ),
                )
        except Exception:
            pass

    def get_history(
        self,
        command_name: str | None = None,
        since=None,
        build_id: str | None = None,
        limit: int = 50,
        failed_only: bool = False,
    ) -> list[dict]:
        if limit < 1 or limit > 1000:
            limit = 50

        where_clauses = []
        params: list = []

        if command_name:
            where_clauses.append("command_name = ?")
            params.append(command_name)

        if failed_only:
            where_clauses.append("exit_code != 0")

        if build_id:
            where_clauses.append("linked_build_id LIKE ? || '%'")
            params.append(build_id)

        if since is not None:
            # Accept either a datetime object or an ISO string
            if isinstance(since, datetime):
                since_str = since.isoformat().replace("+00:00", "Z")
            else:
                since_str = since
            where_clauses.append("ran_at >= ?")
            params.append(since_str)

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM command_log
                WHERE {where_sql}
                ORDER BY ran_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_command(self, short_id: str) -> bool:
        """
        Delete one row from command_log by short UUID prefix.
        Minimum 4 characters enforced.
        Returns True if deleted, False if not found.
        Raises InternalError if prefix is ambiguous.
        Touches nothing outside command_log — builds and benchmarks unaffected.
        """
        if len(short_id) < 4:
            raise InternalError(
                f"Prefix too short: '{short_id}'. Use at least 4 characters."
            )

        with self._connect() as conn:
            matches = conn.execute(
                "SELECT id, command_name, ran_at FROM command_log WHERE id LIKE ? || '%'",
                (short_id,),
            ).fetchall()

            if not matches:
                return False

            if len(matches) > 1:
                raise InternalError(
                    f"Prefix '{short_id}' matches {len(matches)} entries. "
                    f"Use a longer ID."
                )

            conn.execute(
                "DELETE FROM command_log WHERE id = ?",
                (matches[0]["id"],),
            )
            return True

    def clear_history(self) -> int:
        """
        Delete all rows from command_log.
        Returns count of deleted rows.
        Touches nothing outside command_log — builds and benchmarks unaffected.
        """
        with self._connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM command_log"
            ).fetchone()[0]
            conn.execute("DELETE FROM command_log")
            return count

    def get_distinct_machines(self) -> list[str]:
        """
        Returns distinct machine_name values from command_log.
        Used by mlbuild log to decide whether to show machine column.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT machine_name FROM command_log"
            ).fetchall()
            return [r["machine_name"] for r in rows]
        
    def get_history_by_prefix(self, prefix: str) -> list[dict]:
        """Get history entries matching an ID prefix."""
        if len(prefix) < 4:
            raise InternalError(
                f"Prefix too short: '{prefix}'. Use at least 4 characters."
            )
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM command_log WHERE id LIKE ? || '%' ORDER BY ran_at DESC",
                (prefix,),
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_history(self, full_id: str) -> bool:
        """Delete one row from command_log by full ID."""
        with self._connect() as conn:
            conn.execute("DELETE FROM command_log WHERE id = ?", (full_id,))
            return True

    def count_history(self) -> int:
        """Return total number of rows in command_log."""
        with self._connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM command_log"
            ).fetchone()[0]

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
        
        # v5: read imported flag from column, fall back to JSON for
        # any rows that predate the migration running.
        raw_imported = safe_get(row, "imported", None)
        if raw_imported is None:
            backend_versions = json.loads(safe_get(row, "backend_versions", "{}"))
            imported_flag = "true" if backend_versions.get("imported") == "true" else "false"
        else:
            imported_flag = "true" if raw_imported == 1 else "false"

        backend_versions = json.loads(row["backend_versions"])
        # Keep backend_versions consistent with the imported flag
        backend_versions["imported"] = imported_flag

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
            backend_versions=backend_versions,
            environment_data=json.loads(safe_get(row, "environment_data", "{}")),
            mlbuild_version=row["mlbuild_version"],
            python_version=row["python_version"],
            platform=row["platform"],
            os_version=row["os_version"],
            artifact_path=row["artifact_path"],
            size_mb=Decimal(row["size_bytes"]) / Decimal(1024 * 1024),
            task_type=safe_get(row, "task_type"),
            variant_id=safe_get(row, "variant_id"),
            root_build_id=safe_get(row, "root_build_id"),
            parent_build_id=safe_get(row, "parent_build_id"),
            optimization_pass=safe_get(row, "optimization_pass"),
            optimization_method=safe_get(row, "optimization_method"),
            weight_precision=safe_get(row, "weight_precision"),
            activation_precision=safe_get(row, "activation_precision"),
            has_graph=bool(safe_get(row, "has_graph", 0)),
            graph_format=safe_get(row, "graph_format"),
            graph_path=safe_get(row, "graph_path"),
            cached_latency_p50_ms=safe_get(row, "cached_latency_p50_ms"),
            cached_latency_p95_ms=safe_get(row, "cached_latency_p95_ms"),
            cached_memory_peak_mb=safe_get(row, "cached_memory_peak_mb"),
        )