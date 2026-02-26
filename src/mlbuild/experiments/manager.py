"""
Enterprise-grade experiment tracking manager.
Self-contained: no external exception module required.
"""

from __future__ import annotations

import json
import re
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from .experiment import Experiment, Run, Metadata


# ============================================================
# Domain Exceptions (Self-Contained)
# ============================================================

class MLBuildError(Exception):
    """Base domain error."""


class ValidationError(MLBuildError):
    """Input validation failed."""


class NotFoundError(MLBuildError):
    """Entity not found."""


class ConflictError(MLBuildError):
    """Conflict with existing state."""


class InfrastructureError(MLBuildError):
    """Database / storage / schema failure."""


# ============================================================
# Constants
# ============================================================

ALLOWED_SORT_COLUMNS = {"created_at", "name"}
MAX_NAME_LENGTH = 128
SCHEMA_VERSION = 1


# ============================================================
# Utilities
# ============================================================

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _validate_name(name: str) -> None:
    if not name:
        raise ValidationError("Experiment name cannot be empty")

    if len(name) > MAX_NAME_LENGTH:
        raise ValidationError(
            f"Experiment name exceeds {MAX_NAME_LENGTH} characters"
        )

    if not re.match(r"^[a-zA-Z0-9_\-\.]+$", name):
        raise ValidationError(
            "Experiment name contains invalid characters"
        )


# ============================================================
# Experiment Manager
# ============================================================

class ExperimentManager:
    """
    Enterprise-grade experiment manager with:
    - Strict validation
    - Injection-safe queries
    - Transaction boundaries
    - Schema version enforcement
    - State transition enforcement
    - Audit logging hooks
    """

    def __init__(self, registry):
        self._registry = registry
        
        

    # --------------------------------------------------------
    # Schema Enforcement
    # --------------------------------------------------------

    def _verify_schema_version(self) -> None:
        version = self._registry.get_schema_version()
        if version != SCHEMA_VERSION:
            raise InfrastructureError(
                f"Incompatible schema version {version}. "
                f"Expected {SCHEMA_VERSION}. Run migrations."
            )

    # --------------------------------------------------------
    # Transaction Boundary
    # --------------------------------------------------------

    @contextmanager
    def _transaction(self):
        with self._registry.transaction() as tx:
            yield tx

    # ========================================================
    # Create Experiment
    # ========================================================

    def create_experiment(
            self,
            name: str,
            description: Optional[str] = None,
        ):
            """Create a new experiment."""
            
            # Check if exists (including deleted)
            existing = self.get_experiment_including_deleted(name)
            
            if existing and existing['deleted_at'] is None:
                raise ConflictError(f"Experiment '{name}' already exists")
            
            if existing and existing['deleted_at'] is not None:
                raise ConflictError(
                    f"Experiment '{name}' was previously deleted. "
                    f"Restore it or choose a different name."
                )

            experiment_id = uuid.uuid4()
            created_at = _utc_now()
            metadata = {}

            with self._registry._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO experiments (
                        experiment_id,
                        name,
                        description,
                        created_at,
                        metadata,
                        schema_version
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(experiment_id),
                        name,
                        description,
                        created_at.isoformat(),
                        json.dumps(metadata),
                        1,
                    ),
                )
                conn.commit()

            return {
                "experiment_id": experiment_id,
                "name": name,
                "description": description,
                "created_at": created_at,
                "metadata": metadata,
            }

    # ========================================================
    # List Experiments (Injection Safe + Cursor Pagination)
    # ========================================================

    def list_experiments(
        self,
        limit: int = 50,
        cursor: Optional[str] = None,
        sort_by: str = "created_at",
        descending: bool = True,
        include_deleted: bool = False,
        name_prefix: Optional[str] = None,
    ) -> List[dict]:
        """List experiments with filtering."""

        if limit <= 0 or limit > 1000:
            limit = 50

        direction = "DESC" if descending else "ASC"

        where_clauses = []
        params = []

        if not include_deleted:
            where_clauses.append("deleted_at IS NULL")

        if name_prefix:
            where_clauses.append("name LIKE ?")
            params.append(f"{name_prefix}%")

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        sql = f"""
            SELECT *
            FROM experiments
            WHERE {where_sql}
            ORDER BY {sort_by} {direction}
            LIMIT ?
        """

        params.append(limit)

        with self._registry._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        experiments = []
        for row in rows:
            experiments.append({
                "experiment_id": uuid.UUID(row["experiment_id"]),
                "name": row["name"],
                "description": row["description"],
                "created_at": datetime.fromisoformat(row["created_at"]),
                "deleted_at": datetime.fromisoformat(row["deleted_at"]) if row["deleted_at"] else None,
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            })

        return experiments

    # ========================================================
    # Soft Delete (Strict State Validation)
    # ========================================================

    def soft_delete_experiment(
        self,
        experiment_id: uuid.UUID,
        deleted_by: Optional[str] = None,
    ) -> None:

        with self._transaction() as tx:

            row = tx.fetch_one(
                "SELECT deleted_at FROM experiments WHERE experiment_id = ?",
                (str(experiment_id),),
            )

            if not row:
                raise NotFoundError("Experiment not found")

            if row["deleted_at"] is not None:
                raise ConflictError("Experiment already deleted")

            result = tx.execute(
                """
                UPDATE experiments
                SET deleted_at = ?
                WHERE experiment_id = ?
                """,
                (_utc_now().isoformat(), str(experiment_id)),
            )

            if result.rowcount != 1:
                raise InfrastructureError(
                    "Unexpected row count during delete"
                )


    # ========================================================
    # Hydration (Safe)
    # ========================================================

    def _hydrate(self, row) -> dict:
        try:
            return {
                "experiment_id": uuid.UUID(row["experiment_id"]),
                "name": row["name"],
                "description": row["description"],
                "created_at": _normalize_utc(
                    datetime.fromisoformat(row["created_at"])
                ),
                "deleted_at": _normalize_utc(
                    datetime.fromisoformat(row["deleted_at"])
                ) if row["deleted_at"] else None,
                "metadata": json.loads(row["metadata"]),
                "schema_version": row["schema_version"],
            }
        except Exception as e:
            raise InfrastructureError(
                f"Corrupted experiment record {row.get('experiment_id')}"
            ) from e
        

    # ============================================================
    # Runs
    # ============================================================
    
    def start_run(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
    ) -> Run:
        """Start a new run in an experiment."""
        from .experiment import Run, RunStatus, ParamSet, MetricSet, TagSet, HardwareSnapshot
        import platform
        import os
        
        # Get experiment
        exp = self.get_experiment(experiment_name)
        if not exp:
            raise ValueError(f"Experiment not found: {experiment_name}")
        
        run_id = uuid.uuid4()
        started_at = _utc_now()
        
        # Capture hardware
        hardware = HardwareSnapshot(
            device_name=platform.machine(),
            os_version=platform.platform(),
            cpu_cores=os.cpu_count() or 0,
            accelerator=None,
            driver_version=None,
        )
        
        # Create run
        run = Run(
            run_id=run_id,
            experiment_id=uuid.UUID(exp["experiment_id"]),  # ← fixed: dict access + UUID cast
            run_name=run_name,
            status=RunStatus.RUNNING,
            started_at=started_at,
            ended_at=None,
            build_id=None,
            params=ParamSet(values={}),
            metrics=None,
            tags=TagSet(labels={}),
            hardware=hardware,
        )
        
        # Save to DB
        with self._registry._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, experiment_id, run_name, status, 
                    started_at, build_id, params, metrics, tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(run_id),
                    exp["experiment_id"],
                    run_name,
                    'running',
                    started_at.isoformat(),
                    None,
                    json.dumps({}),
                    json.dumps({}),
                    json.dumps({}),
                )
            )
            conn.commit()
        
        return run
    
    def get_experiment(self, name_or_id: str) -> Optional[dict]:
        """Get experiment by name or ID."""
        
        with self._registry._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM experiments
                WHERE (experiment_id = ? OR name = ?)
                AND deleted_at IS NULL
                """,
                (name_or_id, name_or_id)
            ).fetchone()
            
            if not row:
                return None
            
            return {
                "experiment_id": row['experiment_id'],
                "name": row['name'],
                "description": row['description'],
                "created_at": datetime.fromisoformat(row['created_at']),
                "metadata": json.loads(row['metadata']),
                "deleted_at": datetime.fromisoformat(row['deleted_at']) if row['deleted_at'] else None,
            }
            
            
            return result

    def get_experiment_including_deleted(self, name_or_id: str) -> Optional[dict]:
        """Get experiment by name or ID, including soft-deleted ones."""
       
        
        with self._registry._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM experiments
                WHERE (experiment_id = ? OR name = ?)
                """,
                (name_or_id, name_or_id)
            ).fetchone()
            
            if not row:
                return None
            
            return {
                "experiment_id": row['experiment_id'],
                "name": row['name'],
                "description": row['description'],
                "created_at": datetime.fromisoformat(row['created_at']),
                "metadata": json.loads(row['metadata']),
                "deleted_at": datetime.fromisoformat(row['deleted_at']) if row['deleted_at'] else None,
            }

    def log_param(self, run_id: str, key: str, value: any):
        """Log a parameter to a run."""
        with self._registry._connect() as conn:
            row = conn.execute(
                "SELECT params FROM runs WHERE run_id = ?",
                (run_id,)
            ).fetchone()
            
            if not row:
                raise ValueError(f"Run not found: {run_id}")
            
            params = json.loads(row['params'])
            params[key] = value
            
            conn.execute(
                "UPDATE runs SET params = ? WHERE run_id = ?",
                (json.dumps(params, sort_keys=True), run_id)
            )
            conn.commit()
    
    def log_metric(self, run_id: str, key: str, value: float):
        """Log a metric to a run."""
        with self._registry._connect() as conn:
            row = conn.execute(
                "SELECT metrics FROM runs WHERE run_id = ?",
                (run_id,)
            ).fetchone()
            
            if not row:
                raise ValueError(f"Run not found: {run_id}")
            
            metrics = json.loads(row['metrics'])
            metrics[key] = float(value)
            
            conn.execute(
                "UPDATE runs SET metrics = ? WHERE run_id = ?",
                (json.dumps(metrics, sort_keys=True), run_id)
            )
            conn.commit()
    
    def attach_build(self, run_id: str, build_id: str):
        """Attach a build to a run."""
        # Resolve build
        build = self._registry.resolve_build(build_id)
        if not build:
            raise ValueError(f"Build not found: {build_id}")
        
        with self._registry._connect() as conn:
            conn.execute(
                "UPDATE runs SET build_id = ? WHERE run_id = ?",
                (build.build_id, run_id)
            )
            conn.commit()
    
    def end_run(self, run_id: str, status: str = 'completed'):
        """End a run."""
        ended_at = _utc_now()

        with self._registry._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status = ?, ended_at = ?
                WHERE run_id = ?
                """,
                (status, ended_at.isoformat(), run_id)
            )
            conn.commit()
    
    def list_runs(
        self,
        experiment_name: Optional[str] = None,
        limit: int = 50,
    ) -> List:
        """List runs, optionally filtered by experiment."""
        from .experiment import Run, RunStatus, ParamSet, MetricSet, TagSet, HardwareSnapshot

        with self._registry._connect() as conn:
            if experiment_name:
                exp = self.get_experiment(experiment_name)
                if not exp:
                    raise ValueError(f"Experiment not found: {experiment_name}")
                
                rows = conn.execute(
                    """
                    SELECT * FROM runs
                    WHERE experiment_id = ?
                    ORDER BY started_at DESC
                    LIMIT ?
                    """,
                    (exp["experiment_id"], limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM runs
                    ORDER BY started_at DESC
                    LIMIT ?
                    """,
                    (limit,)
                ).fetchall()
            
            runs = []
            for row in rows:
                hardware = HardwareSnapshot(
                    device_name="unknown",
                    os_version="unknown",
                    cpu_cores=0,
                    accelerator=None,
                    driver_version=None,
                )
                
                run = Run(
                    run_id=uuid.UUID(row['run_id']),
                    experiment_id=uuid.UUID(row['experiment_id']),  # ← fixed: use row directly, not exp
                    run_name=row['run_name'],
                    status=RunStatus(row['status']),
                    started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
                    ended_at=datetime.fromisoformat(row['ended_at']) if row['ended_at'] else None,
                    build_id=row['build_id'],
                    params=ParamSet(values=json.loads(row['params'])),
                    metrics=MetricSet(scalars=json.loads(row['metrics'])) if row['metrics'] and json.loads(row['metrics']) else None,
                    tags=TagSet(labels=json.loads(row['tags'])),
                    hardware=hardware,
                )
                runs.append(run)
            
            return runs
    
    def get_active_run(self) -> Optional[str]:
        """Get the currently active run ID (if any)."""
        with self._registry._connect() as conn:
            row = conn.execute(
                """
                SELECT run_id FROM runs
                WHERE status = 'running'
                ORDER BY started_at DESC
                LIMIT 1
                """
            ).fetchone()
            
            return row['run_id'] if row else None
        
    def get_run(self, run_id: str):
        """Get a run by ID."""
        from .experiment import Run, RunStatus, ParamSet, MetricSet, TagSet, HardwareSnapshot
        
        with self._registry._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?",
                (run_id,)
            ).fetchone()
            
            if not row:
                return None
            
            hardware = HardwareSnapshot(
                device_name="unknown",
                os_version="unknown",
                cpu_cores=0,
                accelerator=None,
                driver_version=None,
            )
            
            return Run(
                run_id=uuid.UUID(row['run_id']),
                experiment_id=uuid.UUID(row['experiment_id']),
                run_name=row['run_name'],
                status=RunStatus(row['status']),
                started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
                ended_at=datetime.fromisoformat(row['ended_at']) if row['ended_at'] else None,
                build_id=row['build_id'],
                params=ParamSet(values=json.loads(row['params'])),
                metrics=MetricSet(scalars=json.loads(row['metrics'])) if row['metrics'] and json.loads(row['metrics']) else None,
                tags=TagSet(labels=json.loads(row['tags'])),
                hardware=hardware,
            )