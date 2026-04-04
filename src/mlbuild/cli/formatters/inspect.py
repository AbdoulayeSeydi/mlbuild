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

INDENT       = 2
LABEL_WIDTH  = 14   # value starts at char 16 (2 + 14)
SECTION_WIDTH = 66

console = Console()


# ── Core Helpers  (unchanged) ──────────────────────────────────

def _label(text: str) -> Text:
    return Text(" " * INDENT + f"{text:<{LABEL_WIDTH}}", style="bold")


def _row(label: str, value: str, style: str = "") -> Text:
    t = _label(label)
    t.append(value, style=style)
    return t


def _dim_row(label: str, value: str) -> Text:
    return _row(label, value, "dim")


def _section(title: str) -> Text:
    prefix  = f"── {title} "
    bar_len = max(0, SECTION_WIDTH - len(prefix))
    return Text(f"\n{prefix}{'─' * bar_len}", style="bold cyan")


def _no_records() -> Text:
    return Text(" " * INDENT + "no records", style="dim")


def _pad(val: str, width: int) -> str:
    return f"{val:<{width}}"


# ── Helpers for v2 fields ──────────────────────────────────────

def _has_v2(v: "BuildView") -> bool:
    """True when the BuildView carries ModelProfile data (Step 4+)."""
    return getattr(v, "subtype", "none") not in ("none", None, "")


def _format_subtype(subtype: str) -> str:
    """Human-friendly subtype label."""
    return {
        "detection":           "detection",
        "segmentation":        "segmentation  [dim](placeholder — not yet implemented)[/dim]",
        "timeseries":          "time-series",
        "recommendation":      "recommendation",
        "generative_stateful": "generative (stateful)",
        "multimodal":          "multimodal",
        "none":                "—",
    }.get(subtype or "none", subtype or "—")


def _format_execution(execution: str, state_optional: bool) -> str:
    """Human-friendly execution mode label."""
    label = {
        "standard":            "standard",
        "stateful":            "stateful",
        "partially_stateful":  "partially stateful",
        "kv_cache":            "kv-cache",
        "multi_input":         "multi-input",
        "streaming":           "streaming  [dim](placeholder — not yet implemented)[/dim]",
    }.get(execution or "standard", execution or "standard")

    if state_optional:
        label += "  [dim](state optional)[/dim]"
    return label


def _format_role(role: str) -> tuple[str, str]:
    """
    Return (label, style) for an input role.
    Unknown roles get a yellow warning style.
    """
    if role in ("unknown_float", "unknown_int", "unknown_structured"):
        return f"⚠  {role}", "yellow"
    return role, "dim"


# ── Sections ───────────────────────────────────────────────────

def _render_build(v: "BuildView") -> list[Text]:
    lines    = [_section("Build")]
    abs_time = v.created_at.strftime("%Y-%m-%d %H:%M:%S")
    rel_time = _relative(v.created_at)

    lines.append(_row("ID",      v.id[:8]))
    lines.append(_row("Name",    v.name or "—"))
    lines.append(_row("Created", f"{abs_time}  ({rel_time})"))
    lines.append(_row("Source",  v.source))

    return lines


def _render_task(v: "BuildView") -> list[Text]:
    """
    Task section — v1 rows always present.
    v2 rows (Subtype, Execution, NMS) added when ModelProfile data is available.
    """
    lines = [_section("Task")]

    lines.append(_row("Type",      v.task_type))
    lines.append(_row("Detection", v.detection_tier))

    # v2 — ModelProfile rows (additive, guarded)
    if _has_v2(v):
        subtype   = getattr(v, "subtype",        "none")
        execution = getattr(v, "execution_mode",  "standard")
        nms       = getattr(v, "nms_inside",      False)
        st_opt    = getattr(v, "state_optional",  False)

        lines.append(_row("Subtype",   _format_subtype(subtype)))
        lines.append(_row("Execution", _format_execution(execution, st_opt)))

        if nms:
            lines.append(_dim_row("NMS",   "inside graph  (final-detection validation)"))
        elif subtype == "detection":
            lines.append(_dim_row("NMS",   "outside graph  (raw-prediction validation)"))

    return lines


def _render_artifacts(v: "BuildView") -> list[Text]:
    """
    Artifacts section — unchanged column layout.
    v2: unknown input_roles shown with yellow glyph after the artifact table.
    """
    lines = [_section("Artifacts")]

    if not v.artifacts:
        lines.append(_no_records())
        _render_input_roles_inline(v, lines)
        return lines

    artifacts = sorted(v.artifacts, key=lambda x: x.priority)

    fmt_w = max(len(a.format)   for a in artifacts)
    qt_w  = max(len(a.quantize) for a in artifacts)
    sz_w  = max(len(f"{a.size_mb:.2f} MB") for a in artifacts)
    tgt_w = max(len(a.target)   for a in artifacts)

    for a in artifacts:
        role = "primary" if a.priority == 0 else f"p{a.priority}"

        row = (
            " " * INDENT
            + _pad(a.format,            fmt_w) + "  "
            + _pad(a.quantize,          qt_w)  + "  "
            + _pad(f"{a.size_mb:.2f} MB", sz_w) + "  "
            + _pad(a.sha256[:16],       16)    + "  "
            + _pad(a.target,            tgt_w) + "  "
        )
        t = Text(row)
        t.append(role, style="dim")
        lines.append(t)

    _render_input_roles_inline(v, lines)
    return lines


def _render_input_roles_inline(v: "BuildView", lines: list[Text]) -> None:
    """
    Append input role lines to an existing lines list.

    Known roles render dimmed.  Unknown roles (unknown_float, unknown_int,
    unknown_structured) render in yellow with a ⚠ glyph, because they signal
    that benchmark inputs for those tensors may not reflect real-world values.

    This is additive — only appended when input_roles is non-empty.
    """
    input_roles: dict = getattr(v, "input_roles", {})
    if not input_roles:
        return

    has_unknown = any(
        r in ("unknown_float", "unknown_int", "unknown_structured")
        for r in input_roles.values()
    )

    header_style = "yellow" if has_unknown else "dim"
    lines.append(Text(" " * INDENT + "Input roles:", style=header_style))

    name_w = max(len(n) for n in input_roles) if input_roles else 8

    for tensor_name, role in sorted(input_roles.items()):
        label, style = _format_role(role)
        t = Text(" " * (INDENT + 2) + f"{tensor_name:<{name_w}}  ")
        t.append(label, style=style)
        lines.append(t)


def _render_benchmarks(v: "BuildView") -> list[Text]:
    """
    Benchmarks section — unchanged column layout.
    v2: benchmark_caveats rendered as dimmed warning lines after the table.
    """
    lines = [_section("Benchmarks")]

    if not v.benchmarks:
        lines.append(_no_records())
        _render_benchmark_caveats(v, lines)
        return lines

    groups: dict[str, list] = defaultdict(list)
    for b in v.benchmarks:
        groups[b.compute_unit].append(b)

    for cu in sorted(groups.keys()):
        rows    = sorted(groups[cu], key=lambda x: x.ran_at, reverse=True)
        label_w = max(len(f"{cu} ({b.device})") for b in rows)

        for b in rows:
            label = f"{cu} ({b.device})"
            p50   = f"p50 {b.p50_ms:.2f}ms"   if b.p50_ms is not None else "p50 —"
            p95   = f"p95 {b.p95_ms:.2f}ms"   if b.p95_ms is not None else "p95 —"
            runs  = f"runs {b.runs}"
            batch = f"batch {b.batch_size}"
            age   = _relative(b.ran_at)

            row = (
                " " * INDENT
                + _pad(label, label_w) + "  "
                + _pad(p50,  14)       + "  "
                + _pad(p95,  14)       + "  "
                + _pad(runs, 10)       + "  "
                + _pad(batch, 8)       + "  "
                + age
            )
            lines.append(Text(row))

    _render_benchmark_caveats(v, lines)
    return lines


def _render_benchmark_caveats(v: "BuildView", lines: list[Text]) -> None:
    """
    Append benchmark caveat lines to an existing lines list.

    Caveats are machine-readable limitation notes populated at build time.
    They render dimmed so they don't dominate the output but are visible
    when scrolling. A pipeline reading raw benchmark numbers should check
    BuildView.benchmark_caveats before presenting latency as authoritative.

    This is additive — only appended when benchmark_caveats is non-empty.
    """
    caveats: list = getattr(v, "benchmark_caveats", [])
    if not caveats:
        return

    lines.append(Text(""))   # blank line before caveats
    for caveat in caveats:
        t = Text(" " * INDENT)
        t.append(f"⚠  {caveat}", style="dim yellow")
        lines.append(t)


def _render_accuracy(v: "BuildView") -> list[Text]:
    lines = [_section("Accuracy")]

    if not v.accuracy_records:
        lines.append(_no_records())
        return lines

    for a in sorted(v.accuracy_records, key=lambda x: x.ran_at, reverse=True):
        cosine = f"cosine {a.cosine:.3f}"          if a.cosine    is not None else "cosine —"
        mae    = f"MAE {a.mae:.4f}"                if a.mae       is not None else "MAE —"
        top1   = f"top-1 {a.top1 * 100:.1f}%"     if a.top1      is not None else "top-1 —"
        thresh = (
            f"threshold {a.threshold:.3f} ({a.primary_metric})"
            if a.threshold is not None
            else f"metric {a.primary_metric}"
        )

        result       = "PASSED" if a.passed else "FAILED"
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


# ── Public API  (unchanged signature) ─────────────────────────

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