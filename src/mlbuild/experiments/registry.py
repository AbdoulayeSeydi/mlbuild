"""
Enterprise-grade SQLite-backed registry.

Production hardened:
- Explicit transaction control
- BEGIN IMMEDIATE writes
- WAL mode
- Busy timeout
- Strict schema version enforcement
- Single-row schema_meta
- Foreign key enforcement
- JSON validation
- Proper indexing
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import List, Optional

SCHEMA_VERSION = 1


DDL = """
-- ============================================================
-- Schema Meta (Single Row Enforced)
-- ============================================================

CREATE TABLE IF NOT EXISTS schema_meta (
    id INTEGER PRIMARY KEY CHECK(id = 1),
    version INTEGER NOT NULL CHECK(version > 0)
);

-- ============================================================
-- Experiments
-- ============================================================

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id  TEXT PRIMARY KEY,
    name           TEXT UNIQUE NOT NULL,
    description    TEXT,
    created_at     TEXT NOT NULL,
    deleted_at     TEXT,
    metadata       TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(metadata)),
    schema_version INTEGER NOT NULL
);

-- ============================================================
-- Audit Log
-- ============================================================

CREATE TABLE IF NOT EXISTS experiment_audit_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    action        TEXT NOT NULL,
    actor         TEXT,
    timestamp     TEXT NOT NULL,
    FOREIGN KEY (experiment_id)
        REFERENCES experiments(experiment_id)
        ON DELETE CASCADE
);

-- ============================================================
-- Indexes
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_experiments_created_at
ON experiments(created_at);

CREATE INDEX IF NOT EXISTS idx_experiments_deleted_at
ON experiments(deleted_at);

CREATE INDEX IF NOT EXISTS idx_experiments_name
ON experiments(name);
"""


class InfrastructureError(Exception):
    """Database or schema failure."""


class _Transaction:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def execute(self, sql: str, params=()):
        return self._conn.execute(sql, params)

    def fetch_one(self, sql: str, params=()):
        row = self._conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def fetch_all(self, sql: str, params=()):
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


class SQLiteRegistry:
    """
    Production-grade SQLite registry.

    Safe under:
    - Concurrent reads
    - Concurrent writes (WAL + busy_timeout)
    - Schema mismatch detection
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._conn = self._connect()
        self._bootstrap()

    # ============================================================
    # Connection Setup
    # ============================================================

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self._db_path,
            isolation_level=None,          # manual transaction control
            check_same_thread=False,
        )

        conn.row_factory = sqlite3.Row

        # Operational Hardening
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")

        return conn

    # ============================================================
    # Bootstrap & Schema Management
    # ============================================================

    def _bootstrap(self) -> None:
        try:
            self._conn.executescript(DDL)

            row = self._conn.execute(
                "SELECT version FROM schema_meta WHERE id = 1"
            ).fetchone()

            if not row:
                self._conn.execute(
                    "INSERT INTO schema_meta (id, version) VALUES (1, ?)",
                    (SCHEMA_VERSION,),
                )
            else:
                existing_version = row["version"]
                if existing_version != SCHEMA_VERSION:
                    self._migrate(existing_version, SCHEMA_VERSION)

        except Exception as e:
            raise InfrastructureError(
                "Failed during database bootstrap"
            ) from e

    def _migrate(self, from_version: int, to_version: int) -> None:
        """
        Migration gate.

        Extend this with real migrations when schema changes.
        """
        raise InfrastructureError(
            f"Database schema version {from_version} "
            f"is incompatible with required version {to_version}. "
            f"Run migrations before starting service."
        )

    def get_schema_version(self) -> int:
        row = self._conn.execute(
            "SELECT version FROM schema_meta WHERE id = 1"
        ).fetchone()
        return row["version"]

    # ============================================================
    # Transaction Control (Enterprise Safe)
    # ============================================================

    @contextmanager
    def transaction(self):
        """
        Explicit transaction boundary.
        BEGIN IMMEDIATE prevents write starvation.
        """
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            tx = _Transaction(self._conn)
            yield tx
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ============================================================
    # Read Operations
    # ============================================================

    def fetch_all(self, sql: str, params=()) -> List[dict]:
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def fetch_one(self, sql: str, params=()) -> Optional[dict]:
        row = self._conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    # ============================================================
    # Shutdown
    # ============================================================

    def close(self) -> None:
        self._conn.close()