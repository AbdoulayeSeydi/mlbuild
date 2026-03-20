from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING
from collections import defaultdict

from rich.console import Console
from rich.text import Text
from mlbuild.cli.formatters.utils import relative_time as _relative

if TYPE_CHECKING:
    from mlbuild.models.build_view import BuildView

# ── Layout Constants ───────────────────────────────────────────

INDENT = 2
LABEL_WIDTH = 14  # value starts at char 16 (2 + 14)
SECTION_WIDTH = 66

console = Console()


# ── Core Helpers ───────────────────────────────────────────────

def _label(text: str) -> Text:
    return Text(" " * INDENT + f"{text:<{LABEL_WIDTH}}", style="bold")


def _row(label: str, value: str, style: str = "") -> Text:
    t = _label(label)
    t.append(value, style=style)
    return t


def _dim_row(label: str, value: str) -> Text:
    return _row(label, value, "dim")


def _section(title: str) -> Text:
    prefix = f"── {title} "
    bar_len = max(0, SECTION_WIDTH - len(prefix))
    return Text(f"\n{prefix}{'─' * bar_len}", style="bold cyan")


def _no_records() -> Text:
    return Text(" " * INDENT + "no records", style="dim")


# ── Table Utility (fixed columns, no drift) ────────────────────

def _pad(val: str, width: int) -> str:
    return f"{val:<{width}}"


# ── Sections ───────────────────────────────────────────────────

def _render_build(v: "BuildView") -> list[Text]:
    lines = [_section("Build")]

    abs_time = v.created_at.strftime("%Y-%m-%d %H:%M:%S")
    rel_time = _relative(v.created_at)

    lines.append(_row("ID", v.id[:8]))
    lines.append(_row("Name", v.name or "—"))
    lines.append(_row("Created", f"{abs_time}  ({rel_time})"))
    lines.append(_row("Source", v.source))

    return lines


def _render_task(v: "BuildView") -> list[Text]:
    lines = [_section("Task")]
    lines.append(_row("Type", v.task_type))
    lines.append(_row("Detection", v.detection_tier))
    return lines


def _render_artifacts(v: "BuildView") -> list[Text]:
    lines = [_section("Artifacts")]

    if not v.artifacts:
        lines.append(_no_records())
        return lines

    artifacts = sorted(v.artifacts, key=lambda x: x.priority)

    # compute stable widths
    fmt_w = max(len(a.format) for a in artifacts)
    qt_w = max(len(a.quantize) for a in artifacts)
    sz_w = max(len(f"{a.size_mb:.2f} MB") for a in artifacts)
    tgt_w = max(len(a.target) for a in artifacts)

    for a in artifacts:
        role = "primary" if a.priority == 0 else f"p{a.priority}"

        row = (
            " " * INDENT
            + _pad(a.format, fmt_w)
            + "  "
            + _pad(a.quantize, qt_w)
            + "  "
            + _pad(f"{a.size_mb:.2f} MB", sz_w)
            + "  "
            + _pad(a.sha256[:16], 16)
            + "  "
            + _pad(a.target, tgt_w)
            + "  "
        )

        t = Text(row)
        t.append(role, style="dim")
        lines.append(t)

    return lines


def _render_benchmarks(v: "BuildView") -> list[Text]:
    lines = [_section("Benchmarks")]

    if not v.benchmarks:
        lines.append(_no_records())
        return lines

    # group deterministically
    groups: dict[str, list] = defaultdict(list)
    for b in v.benchmarks:
        groups[b.compute_unit].append(b)

    for cu in sorted(groups.keys()):
        rows = sorted(groups[cu], key=lambda x: x.ran_at, reverse=True)

        # dynamic label width (prevents truncation)
        label_w = max(len(f"{cu} ({b.device})") for b in rows)

        for b in rows:
            label = f"{cu} ({b.device})"

            p50 = f"p50 {b.p50_ms:.2f}ms" if b.p50_ms is not None else "p50 —"
            p95 = f"p95 {b.p95_ms:.2f}ms" if b.p95_ms is not None else "p95 —"

            runs = f"runs {b.runs}"
            batch = f"batch {b.batch_size}"
            age = _relative(b.ran_at)

            row = (
                " " * INDENT
                + _pad(label, label_w)
                + "  "
                + _pad(p50, 14)
                + "  "
                + _pad(p95, 14)
                + "  "
                + _pad(runs, 10)
                + "  "
                + _pad(batch, 8)
                + "  "
                + age
            )

            lines.append(Text(row))

    return lines


def _render_accuracy(v: "BuildView") -> list[Text]:
    lines = [_section("Accuracy")]

    if not v.accuracy_records:
        lines.append(_no_records())
        return lines

    for a in sorted(v.accuracy_records, key=lambda x: x.ran_at, reverse=True):
        cosine = f"cosine {a.cosine:.3f}" if a.cosine is not None else "cosine —"
        mae = f"MAE {a.mae:.4f}" if a.mae is not None else "MAE —"
        top1 = f"top-1 {a.top1 * 100:.1f}%" if a.top1 is not None else "top-1 —"

        thresh = (
            f"threshold {a.threshold:.3f} ({a.primary_metric})"
            if a.threshold is not None
            else f"metric {a.primary_metric}"
        )

        result = "PASSED" if a.passed else "FAILED"
        result_style = "dim green" if a.passed else "dim red"

        row = (
            f"  vs {a.compared_to:<8}  "
            f"{cosine}   {mae}   {top1}   {thresh}   "
        )

        t = Text(row)
        t.append(result, style=result_style)
        lines.append(t)

    return lines


def _render_tags(v: "BuildView") -> list[Text]:
    lines = [_section("Tags")]

    if not v.tags:
        lines.append(_no_records())
        return lines

    lines.append(Text(" " * INDENT + "   ".join(sorted(v.tags))))
    return lines


def _render_notes(v: "BuildView") -> list[Text]:
    lines = [_section("Notes")]

    if not v.notes:
        lines.append(_no_records())
        return lines

    lines.append(Text(" " * INDENT + v.notes))
    return lines


# ── Public API ─────────────────────────────────────────────────

def render_inspect(v: "BuildView", short: bool = False) -> None:
    sections = [
        _render_build(v),
        _render_task(v),
        _render_artifacts(v),
        _render_benchmarks(v),
    ]

    if not short:
        sections.extend([
            _render_accuracy(v),
            _render_tags(v),
            _render_notes(v),
        ])

    for section in sections:
        for line in section:
            console.print(line)

    console.print()