from __future__ import annotations

import csv
import io
import json
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlbuild.models.build_view import BuildView


# ── Timestamp normalization (single source of truth) ──────────

def _to_iso(dt: datetime) -> str:
    """Normalize datetime → UTC ISO8601 with Z suffix."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Sorting (never trust registry order) ──────────────────────

def _sorted_artifacts(view: "BuildView"):
    return sorted(view.artifacts, key=lambda a: getattr(a, "priority", 0))


def _sorted_benchmarks(view: "BuildView"):
    return sorted(view.benchmarks, key=lambda b: b.ran_at, reverse=True)


def _sorted_accuracy(view: "BuildView"):
    return sorted(view.accuracy_records, key=lambda a: a.ran_at, reverse=True)


def _sorted_tags(view: "BuildView"):
    return sorted(view.tags)


# ── Primary artifact (strict semantics) ───────────────────────

def _primary_artifact(view: "BuildView"):
    artifacts = _sorted_artifacts(view)

    for a in artifacts:
        if getattr(a, "priority", None) == 0:
            return a

    return artifacts[0] if artifacts else None


# ── JSON ──────────────────────────────────────────────────────

def build_view_to_json(view: "BuildView") -> str:
    from dataclasses import asdict

    def _serialize(obj: Any):
        if isinstance(obj, datetime):
            return _to_iso(obj)
        return str(obj)  # forward-compatible fallback

    raw = asdict(view)

    # Enforce deterministic ordering
    raw["artifacts"] = [asdict(a) for a in _sorted_artifacts(view)]
    raw["benchmarks"] = [asdict(b) for b in _sorted_benchmarks(view)]
    raw["accuracy_records"] = [asdict(a) for a in _sorted_accuracy(view)]
    raw["tags"] = _sorted_tags(view)

    payload = {
        "version": "1",
        "exported_at": _to_iso(datetime.now(timezone.utc)),
        "data": raw,
    }

    out = json.dumps(
        payload,
        default=_serialize,
        sort_keys=True,
        ensure_ascii=False,
        indent=2,
    )

    return out + "\n"


# ── Flat CSV ──────────────────────────────────────────────────

FLAT_CSV_COLUMNS = [
    "build_id", "id_short", "name", "created_at", "task_type",
    "format", "target", "quantize",
    "compute_unit", "device", "p50_ms", "p95_ms",
    "runs", "batch_size", "backend", "ran_at",
]


def build_view_to_flat_csv(view: "BuildView") -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")

    writer.writerow(FLAT_CSV_COLUMNS)

    primary = _primary_artifact(view)
    fmt      = getattr(primary, "format", "") if primary else ""
    target   = getattr(primary, "target", "") if primary else ""
    quantize = getattr(primary, "quantize", "") if primary else ""

    benchmarks = _sorted_benchmarks(view)

    if not benchmarks:
        writer.writerow([
            view.id,
            getattr(view, "id_short", view.id[:8]),
            view.name or "",
            _to_iso(view.created_at),
            view.task_type,
            fmt, target, quantize,
            "", "", "", "", "", "", "", "",
        ])
    else:
        for b in benchmarks:
            writer.writerow([
                view.id,
                getattr(view, "id_short", view.id[:8]),
                view.name or "",
                _to_iso(view.created_at),
                view.task_type,
                fmt, target, quantize,
                getattr(b, "compute_unit", ""),
                getattr(b, "device", ""),
                b.p50_ms if b.p50_ms is not None else "",
                b.p95_ms if b.p95_ms is not None else "",
                getattr(b, "runs", ""),
                b.batch_size if b.batch_size is not None else "",
                getattr(b, "backend", ""),
                _to_iso(b.ran_at),
            ])

    out = buf.getvalue()
    if not out.endswith("\n"):
        out += "\n"
    return out


# ── Directory CSV ─────────────────────────────────────────────

def _write_csv(path: Path, columns: list[str], rows: list[list]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def build_view_to_csv_dir(
    view: "BuildView",
    output_dir: Path,
    force: bool = False,
) -> None:
    # Ensure parent exists
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # Pre-flight
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(
            f"Directory already exists and is not empty: {output_dir}\n"
            f"Use --force to overwrite."
        )

    # Prepare temp dir (same parent for atomic move)
    tmp_dir = Path(tempfile.mkdtemp(
        prefix="mlbuild-export-",
        dir=output_dir.parent,
    ))

    backup_dir = None

    try:
        # ── Write files ──

        _write_csv(
            tmp_dir / "build.csv",
            ["id", "id_short", "name", "created_at", "source",
             "task_type", "detection_tier", "notes"],
            [[
                view.id,
                getattr(view, "id_short", view.id[:8]),
                view.name or "",
                _to_iso(view.created_at),
                view.source,
                view.task_type,
                view.detection_tier,
                view.notes or "",
            ]]
        )

        _write_csv(
            tmp_dir / "artifacts.csv",
            ["build_id", "format", "target", "quantize",
             "size_mb", "sha256", "priority"],
            [
                [
                    view.id,
                    getattr(a, "format", ""),
                    getattr(a, "target", ""),
                    getattr(a, "quantize", ""),
                    f"{getattr(a, 'size_mb', 0):.6f}",
                    getattr(a, "sha256", ""),
                    getattr(a, "priority", 0),
                ]
                for a in _sorted_artifacts(view)
            ]
        )

        _write_csv(
            tmp_dir / "benchmarks.csv",
            ["build_id", "id", "compute_unit", "device",
             "p50_ms", "p95_ms", "runs", "warmup", "batch_size",
             "input_shape", "backend", "ran_at"],
            [
                [
                    view.id,
                    getattr(b, "id", ""),
                    getattr(b, "compute_unit", ""),
                    getattr(b, "device", ""),
                    b.p50_ms if b.p50_ms is not None else "",
                    b.p95_ms if b.p95_ms is not None else "",
                    getattr(b, "runs", ""),
                    b.warmup if getattr(b, "warmup", None) is not None else "",
                    b.batch_size if b.batch_size is not None else "",
                    getattr(b, "input_shape", "") or "",
                    getattr(b, "backend", ""),
                    _to_iso(b.ran_at),
                ]
                for b in _sorted_benchmarks(view)
            ]
        )

        _write_csv(
            tmp_dir / "accuracy.csv",
            ["build_id", "compared_to", "primary_metric", "threshold",
             "cosine", "mae", "top1", "dataset", "passed", "ran_at"],
            [
                [
                    view.id,
                    getattr(a, "compared_to", ""),
                    getattr(a, "primary_metric", ""),
                    a.threshold if a.threshold is not None else "",
                    a.cosine if a.cosine is not None else "",
                    a.mae if a.mae is not None else "",
                    a.top1 if a.top1 is not None else "",
                    getattr(a, "dataset", "") or "",
                    "true" if a.passed else "false",
                    _to_iso(a.ran_at),
                ]
                for a in _sorted_accuracy(view)
            ]
        )

        _write_csv(
            tmp_dir / "tags.csv",
            ["build_id", "tag"],
            [[view.id, t] for t in _sorted_tags(view)]
        )

        # ── Safe atomic replace ──

        if output_dir.exists():
            if force:
                backup_dir = output_dir.with_name(output_dir.name + ".bak")
                output_dir.rename(backup_dir)

        tmp_dir.rename(output_dir)

        if backup_dir:
            shutil.rmtree(backup_dir)

    except Exception as e:
        # Attempt restore if needed
        if backup_dir and not output_dir.exists():
            backup_dir.rename(output_dir)

        raise RuntimeError(
            f"{e}\nTemporary data left at: {tmp_dir}"
        ) from e