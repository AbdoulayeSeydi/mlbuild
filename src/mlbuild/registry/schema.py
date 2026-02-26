"""
Enterprise-grade SQLite schema for MLBuild registry (v4).

Invariants:
- artifact_hash = SHA256 of normalized CoreML artifact
- config_hash   = SHA256 of canonical JSON config
- source_hash   = SHA256 of source model file
- env_fingerprint = SHA256 of environment data
- build_id      = SHA256(source_hash + config_hash + artifact_hash + env_fingerprint + mlbuild_version)

All timestamps are ISO8601 UTC (e.g., 2026-02-14T03:42:11Z).
All JSON fields must be canonical and json_valid().
Builds are soft-deletable (deleted_at).
"""

from __future__ import annotations

SCHEMA_VERSION = 4

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

-- Insert schema version v3 (ignore if exists)
INSERT OR IGNORE INTO schema_version (id, version, applied_at)
VALUES (1, 4, datetime('now'));


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
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0)
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
    params TEXT CHECK (json_valid(params)),
    metrics TEXT CHECK (json_valid(metrics)),
    tags TEXT CHECK (json_valid(tags)),
    
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

CREATE INDEX IF NOT EXISTS idx_benchmarks_build
    ON benchmarks(build_id);

CREATE INDEX IF NOT EXISTS idx_benchmarks_chip
    ON benchmarks(device_chip);

CREATE INDEX IF NOT EXISTS idx_benchmarks_measured
    ON benchmarks(measured_at DESC);

CREATE INDEX IF NOT EXISTS idx_tags_build
    ON tags(build_id);
"""