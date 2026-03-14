"""
SQLite schema for MLBuild registry (v6).

Invariants:
- artifact_hash = SHA256 of normalized CoreML artifact
- config_hash   = SHA256 of canonical JSON config
- source_hash   = SHA256 of source model file
- env_fingerprint = SHA256 of environment data
- build_id      = SHA256(source_hash + config_hash + artifact_hash + env_fingerprint + mlbuild_version)

All timestamps are ISO8601 UTC (e.g., 2026-02-14T03:42:11Z).
All JSON fields must be canonical and json_valid().
Builds are soft-deletable (deleted_at).

v5 changes:
- Added `imported` INTEGER NOT NULL DEFAULT 0 to builds table
  Populated from backend_versions JSON for existing rows on migration.

v6 changes:
- Added `task_type` TEXT NULL to builds table.
  NULL = unknown (existing builds unaffected, show as 'unknown' in log/report).
  Valid values: 'vision', 'nlp', 'audio', 'multimodal', 'unknown'.
  Populated by mlbuild build / mlbuild import via auto-detect or --task flag.
"""

# v7 changes:
# - Added optimization lineage columns: variant_id, root_build_id, parent_build_id,
#   optimization_pass, optimization_method, weight_precision, activation_precision.
# - Added graph storage columns: has_graph, graph_format, graph_path.
# - Added cached benchmark columns: cached_latency_p50_ms, cached_latency_p95_ms,
#   cached_memory_peak_mb.

from __future__ import annotations

SCHEMA_VERSION = 8

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

-- ============================================================
-- Schema version tracking (migration-safe)
-- ============================================================

CREATE TABLE IF NOT EXISTS schema_version (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    version INTEGER NOT NULL,
    applied_at TEXT NOT NULL
);

-- Insert schema version v6 (ignore if exists)
INSERT OR IGNORE INTO schema_version (id, version, applied_at)
VALUES (1, 6, datetime('now'));


-- ============================================================
-- Builds
-- ============================================================

CREATE TABLE IF NOT EXISTS builds (
    build_id TEXT PRIMARY KEY CHECK(length(build_id) = 64),

    artifact_hash     TEXT NOT NULL CHECK(length(artifact_hash) = 64),
    source_hash       TEXT NOT NULL CHECK(length(source_hash) = 64),
    config_hash       TEXT NOT NULL CHECK(length(config_hash) = 64),
    env_fingerprint   TEXT NOT NULL CHECK(length(env_fingerprint) = 64),

    name        TEXT,
    notes       TEXT,

    created_at  TEXT NOT NULL,
    deleted_at  TEXT,

    source_path TEXT NOT NULL,

    target_device TEXT NOT NULL,
    format TEXT NOT NULL CHECK (
        format IN ('coreml', 'mlpackage', 'onnx', 'tflite')
    ),

    quantization     TEXT NOT NULL CHECK (json_valid(quantization)),
    optimizer_config TEXT NOT NULL CHECK (json_valid(optimizer_config)),
    backend_versions TEXT NOT NULL CHECK (json_valid(backend_versions)),
    environment_data TEXT NOT NULL CHECK (json_valid(environment_data)),

    mlbuild_version TEXT NOT NULL,
    python_version  TEXT NOT NULL,
    platform        TEXT NOT NULL,
    os_version      TEXT NOT NULL,

    artifact_path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),

    -- v5: explicit imported flag.
    -- 1 = registered via mlbuild import, 0 = produced by mlbuild build.
    imported INTEGER NOT NULL DEFAULT 0 CHECK (imported IN (0, 1)),

    -- v6: task type detected or specified at build/import time.
    -- NULL = unknown (pre-v6 builds). Displayed as 'unknown' in log/report.
    -- Valid: 'vision' | 'nlp' | 'audio' | 'multimodal' | 'unknown'
    task_type TEXT CHECK (
        task_type IS NULL OR
        task_type IN ('vision', 'nlp', 'audio', 'multimodal', 'unknown')

    -- v7: optimization lineage
    -- NULL for all builds created by build/import (not optimize)
    variant_id        TEXT,
    root_build_id     TEXT,
    parent_build_id   TEXT,
    optimization_pass   TEXT,
    optimization_method TEXT,
    weight_precision    TEXT CHECK (
        weight_precision IS NULL OR
        weight_precision IN ('fp32', 'fp16', 'int8')
    ),
    activation_precision TEXT CHECK (
        activation_precision IS NULL OR
        activation_precision IN ('fp32', 'fp16', 'int8')
    ),

    -- v7: graph storage (ONNX stored once at graphs/{root_build_id}.onnx)
    has_graph    INTEGER NOT NULL DEFAULT 0 CHECK (has_graph IN (0, 1)),
    graph_format TEXT,
    graph_path   TEXT,

    -- v7: cached benchmark snapshot stored at registration time
    cached_latency_p50_ms REAL CHECK (cached_latency_p50_ms IS NULL OR cached_latency_p50_ms >= 0),
    cached_latency_p95_ms REAL CHECK (cached_latency_p95_ms IS NULL OR cached_latency_p95_ms >= 0),
    cached_memory_peak_mb REAL CHECK (cached_memory_peak_mb IS NULL OR cached_memory_peak_mb >= 0)
    )
);

-- Active-build uniqueness only (allows recreation after soft delete)
CREATE UNIQUE INDEX IF NOT EXISTS idx_builds_active_unique
ON builds(source_hash, config_hash, artifact_hash, env_fingerprint)
WHERE deleted_at IS NULL;


-- ============================================================
-- Benchmarks
-- ============================================================

CREATE TABLE IF NOT EXISTS benchmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    build_id TEXT NOT NULL,

    device_chip TEXT NOT NULL,
    runtime TEXT NOT NULL,

    measurement_type TEXT NOT NULL CHECK (
        measurement_type IN ('latency', 'throughput', 'memory')
    ),

    compute_unit TEXT,

    latency_p50_ms REAL CHECK (latency_p50_ms IS NULL OR latency_p50_ms >= 0),
    latency_p95_ms REAL CHECK (latency_p95_ms IS NULL OR latency_p95_ms >= 0),
    latency_p99_ms REAL CHECK (latency_p99_ms IS NULL OR latency_p99_ms >= 0),
    memory_peak_mb REAL CHECK (memory_peak_mb IS NULL OR memory_peak_mb >= 0),

    num_runs INTEGER NOT NULL CHECK (num_runs > 0),

    measured_at TEXT NOT NULL,

    FOREIGN KEY(build_id) REFERENCES builds(build_id) ON DELETE CASCADE,

    UNIQUE (
        build_id,
        device_chip,
        runtime,
        measurement_type,
        compute_unit,
        measured_at
    )
);


-- ============================================================
-- Tags
-- ============================================================

CREATE TABLE IF NOT EXISTS tags (
    tag TEXT PRIMARY KEY CHECK (
        length(tag) <= 128
        AND tag NOT LIKE '% %'
    ),
    build_id TEXT NOT NULL,
    created_at TEXT NOT NULL,

    FOREIGN KEY(build_id) REFERENCES builds(build_id) ON DELETE CASCADE
);

-- ============================================================
-- Experiments (MLflow-style tracking)
-- ============================================================

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL,
    deleted_at TEXT,
    metadata TEXT CHECK (json_valid(metadata)),
    schema_version INTEGER NOT NULL DEFAULT 1
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_active_experiment_name
    ON experiments(name)
    WHERE deleted_at IS NULL;

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    run_name TEXT,
    status TEXT NOT NULL CHECK (
        status IN ('running', 'completed', 'failed', 'cancelled')
    ),
    started_at TEXT NOT NULL,
    ended_at TEXT,

    -- Link to build (optional)
    build_id TEXT,

    -- Metadata
    params  TEXT CHECK (json_valid(params)),
    metrics TEXT CHECK (json_valid(metrics)),
    tags    TEXT CHECK (json_valid(tags)),

    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    FOREIGN KEY(build_id) REFERENCES builds(build_id) ON DELETE SET NULL
);

-- Indexes for experiments
CREATE INDEX IF NOT EXISTS idx_experiments_created
    ON experiments(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_runs_experiment
    ON runs(experiment_id);

CREATE INDEX IF NOT EXISTS idx_runs_status
    ON runs(status);

CREATE INDEX IF NOT EXISTS idx_runs_started
    ON runs(started_at DESC);

CREATE INDEX IF NOT EXISTS idx_runs_build
    ON runs(build_id);


-- ============================================================
-- Indexes
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_builds_created
    ON builds(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_builds_deleted
    ON builds(deleted_at);

CREATE INDEX IF NOT EXISTS idx_builds_target
    ON builds(target_device);

CREATE INDEX IF NOT EXISTS idx_builds_config_hash
    ON builds(config_hash);

CREATE INDEX IF NOT EXISTS idx_builds_env_fingerprint
    ON builds(env_fingerprint);

CREATE INDEX IF NOT EXISTS idx_builds_source_hash
    ON builds(source_hash);

CREATE INDEX IF NOT EXISTS idx_builds_artifact_hash
    ON builds(artifact_hash);

CREATE INDEX IF NOT EXISTS idx_builds_imported
    ON builds(imported);

-- v6: index for task_type filtering (mlbuild log --task vision)
CREATE INDEX IF NOT EXISTS idx_builds_task_type
    ON builds(task_type);

CREATE INDEX IF NOT EXISTS idx_benchmarks_build
    ON benchmarks(build_id);

CREATE INDEX IF NOT EXISTS idx_benchmarks_chip
    ON benchmarks(device_chip);

CREATE INDEX IF NOT EXISTS idx_benchmarks_measured
    ON benchmarks(measured_at DESC);

CREATE INDEX IF NOT EXISTS idx_tags_build
    ON tags(build_id);

CREATE INDEX IF NOT EXISTS idx_builds_root
    ON builds(root_build_id);

CREATE INDEX IF NOT EXISTS idx_builds_parent
    ON builds(parent_build_id);

CREATE INDEX IF NOT EXISTS idx_builds_variant_id
    ON builds(variant_id);


-- ============================================================
-- Accuracy Checks (v8)
-- ============================================================

CREATE TABLE IF NOT EXISTS accuracy_checks (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    baseline_build_id   TEXT NOT NULL,
    candidate_build_id  TEXT NOT NULL,
    cosine_similarity   REAL,
    mean_abs_error      REAL,
    max_abs_error       REAL,
    top1_agreement      REAL,
    num_samples         INTEGER NOT NULL,
    seed                INTEGER NOT NULL,
    passed              INTEGER NOT NULL CHECK (passed IN (0, 1)),
    created_at          TEXT NOT NULL,
    FOREIGN KEY(baseline_build_id) REFERENCES builds(build_id),
    FOREIGN KEY(candidate_build_id) REFERENCES builds(build_id)
);

CREATE INDEX IF NOT EXISTS idx_accuracy_baseline
    ON accuracy_checks(baseline_build_id);

CREATE INDEX IF NOT EXISTS idx_accuracy_candidate
    ON accuracy_checks(candidate_build_id);

CREATE INDEX IF NOT EXISTS idx_accuracy_pair
    ON accuracy_checks(baseline_build_id, candidate_build_id);
"""



# ============================================================
# Migration: v4 → v5
# ============================================================

MIGRATION_V4_TO_V5 = """
-- Add imported column (safe to run multiple times due to IF NOT EXISTS guard
-- handled in Python before execution).
ALTER TABLE builds ADD COLUMN imported INTEGER NOT NULL DEFAULT 0
    CHECK (imported IN (0, 1));

-- Back-fill: mark any row whose backend_versions JSON contains
-- "imported":"true" as imported=1.
UPDATE builds
SET imported = 1
WHERE json_extract(backend_versions, '$.imported') = 'true';

-- Bump schema version
UPDATE schema_version SET version = 5, applied_at = datetime('now')
WHERE id = 1;
"""


# ============================================================
# Migration: v5 → v6  [ADDED]
# ============================================================

MIGRATION_V5_TO_V6 = """
-- Add task_type column (nullable — existing builds unaffected).
-- SQLite does not support CHECK constraints in ALTER TABLE,
-- so constraint is enforced at the application layer for this column.
ALTER TABLE builds ADD COLUMN task_type TEXT;

-- No back-fill — existing builds stay NULL, displayed as 'unknown'.

-- Add index for task_type filtering.
CREATE INDEX IF NOT EXISTS idx_builds_task_type
    ON builds(task_type);

-- Bump schema version
UPDATE schema_version SET version = 6, applied_at = datetime('now')
WHERE id = 1;
"""

# ============================================================
# Migration: v6 → v7
# ============================================================

MIGRATION_V6_TO_V7 = """
ALTER TABLE builds ADD COLUMN variant_id TEXT;
ALTER TABLE builds ADD COLUMN root_build_id TEXT;
ALTER TABLE builds ADD COLUMN parent_build_id TEXT;
ALTER TABLE builds ADD COLUMN optimization_pass TEXT;
ALTER TABLE builds ADD COLUMN optimization_method TEXT;
ALTER TABLE builds ADD COLUMN weight_precision TEXT;
ALTER TABLE builds ADD COLUMN activation_precision TEXT;
ALTER TABLE builds ADD COLUMN has_graph INTEGER NOT NULL DEFAULT 0;
ALTER TABLE builds ADD COLUMN graph_format TEXT;
ALTER TABLE builds ADD COLUMN graph_path TEXT;
ALTER TABLE builds ADD COLUMN cached_latency_p50_ms REAL;
ALTER TABLE builds ADD COLUMN cached_latency_p95_ms REAL;
ALTER TABLE builds ADD COLUMN cached_memory_peak_mb REAL;

CREATE INDEX IF NOT EXISTS idx_builds_root
    ON builds(root_build_id);

CREATE INDEX IF NOT EXISTS idx_builds_parent
    ON builds(parent_build_id);

CREATE INDEX IF NOT EXISTS idx_builds_variant_id
    ON builds(variant_id);

UPDATE schema_version SET version = 7, applied_at = datetime('now')
WHERE id = 1;
"""

# ============================================================
# Migration: v7 → v8
# ============================================================

MIGRATION_V7_TO_V8 = """
CREATE TABLE IF NOT EXISTS accuracy_checks (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    baseline_build_id   TEXT NOT NULL,
    candidate_build_id  TEXT NOT NULL,
    cosine_similarity   REAL,
    mean_abs_error      REAL,
    max_abs_error       REAL,
    top1_agreement      REAL,
    num_samples         INTEGER NOT NULL,
    seed                INTEGER NOT NULL,
    passed              INTEGER NOT NULL CHECK (passed IN (0, 1)),
    created_at          TEXT NOT NULL,
    FOREIGN KEY(baseline_build_id) REFERENCES builds(build_id),
    FOREIGN KEY(candidate_build_id) REFERENCES builds(build_id)
);

CREATE INDEX IF NOT EXISTS idx_accuracy_baseline
    ON accuracy_checks(baseline_build_id);

CREATE INDEX IF NOT EXISTS idx_accuracy_candidate
    ON accuracy_checks(candidate_build_id);

CREATE INDEX IF NOT EXISTS idx_accuracy_pair
    ON accuracy_checks(baseline_build_id, candidate_build_id);

UPDATE schema_version SET version = 8, applied_at = datetime('now')
WHERE id = 1;
"""

# ============================================================
# Migration registry — ordered list of all migrations
# ============================================================

MIGRATIONS: list[tuple[int, int, str]] = [
    (4, 5, MIGRATION_V4_TO_V5),
    (5, 6, MIGRATION_V5_TO_V6),
    (6, 7, MIGRATION_V6_TO_V7),
    (7, 8, MIGRATION_V7_TO_V8),
]


def get_migration(from_version: int, to_version: int) -> str | None:
    """
    Return the SQL migration string for a given version jump, or None.

    Usage (in LocalRegistry)
    ------------------------
    sql = get_migration(current_version, current_version + 1)
    if sql:
        conn.executescript(sql)
    """
    for src, dst, sql in MIGRATIONS:
        if src == from_version and dst == to_version:
            return sql
    return None