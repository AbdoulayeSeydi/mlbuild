"""
mlbuild report <build_id>

Generates a self-contained HTML performance report for a build.
Pulls from registry metadata + benchmark history.

Options:
  --output PATH     Output file path (default: mlbuild_report_<id>_<ts>.html)
  --open            Open in browser after generation
  --format html     Output format (html only; pdf requires weasyprint)
"""

import sys
import click
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from ...registry import LocalRegistry
from ...core.errors import MLBuildError

console = Console(width=None)


# ============================================================
# Registry helper — queries benchmarks table directly
# ============================================================

def _get_benchmarks(registry: LocalRegistry, build_id: str) -> list:
    """
    Fetch all benchmark rows for a build, newest first.
    Works around the missing get_benchmarks() registry method.
    """
    with registry._connect() as conn:
        rows = conn.execute(
            """
            SELECT device_chip, runtime, compute_unit,
                   latency_p50_ms, latency_p95_ms, latency_p99_ms,
                   memory_peak_mb, num_runs, measured_at
            FROM benchmarks
            WHERE build_id = ?
            ORDER BY measured_at DESC
            """,
            (build_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def _get_sibling_builds(registry: LocalRegistry, build) -> list:
    """
    Find builds that share the same base name (e.g. mobilenet-fp32 → mobilenet-*).
    Used for the comparison context section.
    """
    if not build.name:
        return []
    # Extract base name: strip trailing -fp32 / -fp16 / -int8 / -int4
    import re
    base = re.sub(r"[-_](fp32|fp16|fp8|int8|int4|q4|q8)$", "", build.name, flags=re.I)
    all_builds = registry.list_builds(limit=200)
    siblings = [
        b for b in all_builds
        if b.build_id != build.build_id
        and b.name
        and (b.name.startswith(base) or base in b.name)
    ]
    return siblings[:5]  # cap at 5 siblings


# ============================================================
# Task helpers
# ============================================================

def _display_task(build) -> str:
    """Return task_type for display, falling back to 'unknown' for pre-v6 NULL rows."""
    return getattr(build, "task_type", None) or "unknown"


def _describe_inputs(build) -> str:
    """
    Return a human-readable description of the synthetic inputs used for
    benchmarking. Uses task_inputs.describe_inputs() when available;
    falls back to a static string per task.
    """
    task = _display_task(build)
    try:
        from ...core.task_inputs import TaskInputFactory, ModelInfo as InputModelInfo
        from ...core.task_detection import TaskType
        info = InputModelInfo(
            op_types=[], input_shapes={}, input_names=[], metadata={}, num_nodes=0,
        )
        factory = TaskInputFactory(info)
        resolved = TaskType(task) if task in [t.value for t in TaskType] else TaskType.UNKNOWN
        return factory.describe_inputs(resolved)
    except Exception:
        fallbacks = {
            "vision":     "Synthetic 224×224 RGB image (float32, NCHW)",
            "nlp":        "Synthetic token sequence (seq_len=128, int32)",
            "audio":      "Synthetic waveform (16 kHz, 1s) or mel spectrogram (80 bins)",
            "multimodal": "Synthetic mixed inputs (image + token sequence)",
            "unknown":    "Synthetic zero-valued tensors (task unknown)",
        }
        return fallbacks.get(task, "Synthetic inputs")


# ============================================================
# HTML generation
# ============================================================

def _render_html(build, benchmarks: list, siblings: list) -> str:
    """
    Render a self-contained HTML report. No external dependencies.
    Monospace + industrial aesthetic — appropriate for an ML infra tool.
    """

    # ── Derived values ────────────────────────────────────────
    quant_type = build.quantization_type or "unknown"
    quant_dict = build.quantization or {}
    size_mb = float(build.size_mb)
    created = build.created_at.strftime("%Y-%m-%d %H:%M UTC") if build.created_at else "—"
    build_short = build.build_id[:16]
    task_value = _display_task(build)
    input_desc = _describe_inputs(build)

    # Latest benchmark
    bench = benchmarks[0] if benchmarks else None

    throughput = None
    if bench and bench["latency_p50_ms"] and bench["latency_p50_ms"] > 0:
        throughput = 1000.0 / bench["latency_p50_ms"]

    # Sibling comparison rows
    sibling_rows = ""
    if siblings:
        for s in siblings:
            s_quant = s.quantization_type or "?"
            s_size = float(s.size_mb)
            size_delta = ((s_size - size_mb) / size_mb * 100) if size_mb > 0 else 0
            if abs(size_delta) < 0.1:
                delta_class = "neutral"
                delta_str = "≈ same size"
            elif size_delta < 0:
                delta_class = "green"
                delta_str = f"{size_delta:+.1f}%"
            else:
                delta_class = "red"
                delta_str = f"{size_delta:+.1f}%"
            sibling_rows += f"""
            <tr>
                <td>{s.name or s.build_id[:12]}</td>
                <td>{s.format or '—'}</td>
                <td>{s_quant.upper()}</td>
                <td>{s_size:.2f} MB</td>
                <td class="{delta_class}">{delta_str} vs this</td>
            </tr>"""

    # Benchmark history rows
    bench_history_rows = ""
    for i, b in enumerate(benchmarks[:5]):
        ts = b["measured_at"][:16].replace("T", " ") if b["measured_at"] else "—"
        tput = f"{1000/b['latency_p50_ms']:.1f}" if b["latency_p50_ms"] else "—"
        bench_history_rows += f"""
            <tr {'class="latest"' if i == 0 else ''}>
                <td>{ts}</td>
                <td>{b['device_chip'] or '—'}</td>
                <td>{b['compute_unit'] or '—'}</td>
                <td>{b['latency_p50_ms']:.3f} ms</td>
                <td>{b['latency_p95_ms']:.3f} ms</td>
                <td>{b['latency_p99_ms']:.3f} ms</td>
                <td>{tput} inf/s</td>
                <td>{b['memory_peak_mb']:.2f} MB</td>
                <td>{b['num_runs']}</td>
            </tr>"""

    if not bench_history_rows:
        bench_history_rows = '<tr><td colspan="9" class="empty">No benchmark data. Run <code>mlbuild benchmark ' + build_short + '</code></td></tr>'

    # Quantization detail rows
    quant_rows = ""
    for k, v in quant_dict.items():
        quant_rows += f"<tr><td>{k}</td><td>{v}</td></tr>"
    if not quant_rows:
        quant_rows = '<tr><td colspan="2" class="empty">No quantization metadata</td></tr>'

    # Recommendations
    recs = []
    if not benchmarks:
        recs.append(("run-benchmark", "No benchmark data found.",
                     f"Run <code>mlbuild benchmark {build_short}</code> to collect latency metrics."))
    if quant_type == "fp32" and siblings:
        int8_sibling = next((s for s in siblings if "int8" in (s.quantization_type or "")), None)
        if int8_sibling:
            recs.append(("compare", "INT8 variant available.",
                         f"Run <code>mlbuild compare-quantization {build_short} {int8_sibling.build_id[:16]}</code> to evaluate the tradeoff."))
        else:
            recs.append(("quantize", "No INT8 variant found.",
                         "Consider building an INT8 version with <code>mlbuild build --quantize int8</code> for edge deployment."))
    if bench and bench["compute_unit"] == "CPU_ONLY" and build.format == "coreml":
        recs.append(("compute-unit", "CoreML running on CPU only.",
                     f"Try <code>mlbuild benchmark {build_short} --compute-unit ALL</code> to leverage ANE/GPU."))
    if bench and bench["memory_peak_mb"] > 100:
        recs.append(("memory", f"Peak RSS {bench['memory_peak_mb']:.1f} MB is high.",
                     "Consider quantization to reduce runtime memory footprint."))
    if not recs:
        recs.append(("ok", "No issues detected.", "Build looks healthy for deployment."))

    rec_html = ""
    for tag, title, detail in recs:
        icon = {"ok": "✓", "run-benchmark": "→", "compare": "⟷",
                "quantize": "⬇", "compute-unit": "⚡", "memory": "⚠"}.get(tag, "•")
        rec_class = "rec-ok" if tag == "ok" else "rec-action"
        rec_html += f"""
        <div class="rec {rec_class}">
            <span class="rec-icon">{icon}</span>
            <div>
                <strong>{title}</strong>
                <p>{detail}</p>
            </div>
        </div>"""

    # ── HTML template ─────────────────────────────────────────
    bench_section = f"""
        <div class="metric-grid">
            <div class="metric">
                <div class="metric-label">p50 latency</div>
                <div class="metric-value">{bench['latency_p50_ms']:.3f}<span class="unit">ms</span></div>
            </div>
            <div class="metric">
                <div class="metric-label">p95 latency</div>
                <div class="metric-value">{bench['latency_p95_ms']:.3f}<span class="unit">ms</span></div>
            </div>
            <div class="metric">
                <div class="metric-label">p99 latency</div>
                <div class="metric-value">{bench['latency_p99_ms']:.3f}<span class="unit">ms</span></div>
            </div>
            <div class="metric">
                <div class="metric-label">throughput</div>
                <div class="metric-value">{throughput:.1f}<span class="unit">inf/s</span></div>
            </div>
            <div class="metric">
                <div class="metric-label">peak memory</div>
                <div class="metric-value">{bench['memory_peak_mb']:.2f}<span class="unit">MB</span></div>
            </div>
            <div class="metric">
                <div class="metric-label">runs</div>
                <div class="metric-value">{bench['num_runs']}<span class="unit">samples</span></div>
            </div>
            <div class="metric">
                <div class="metric-label">device</div>
                <div class="metric-value-sm">{bench['device_chip'] or '—'}</div>
            </div>
            <div class="metric">
                <div class="metric-label">compute unit</div>
                <div class="metric-value-sm">{bench['compute_unit'] or '—'}</div>
            </div>
        </div>
    """ if bench else '<p class="empty-state">No benchmark data. Run <code>mlbuild benchmark ' + build_short + '</code></p>'

    sibling_section = f"""
        <table>
            <thead>
                <tr>
                    <th>Build</th><th>Format</th><th>Quant</th>
                    <th>Size</th><th>Δ Size</th>
                </tr>
            </thead>
            <tbody>
                <tr class="current-row">
                    <td><strong>{build.name or build_short}</strong> ← this</td>
                    <td>{build.format or '—'}</td>
                    <td>{quant_type.upper()}</td>
                    <td>{size_mb:.2f} MB</td>
                    <td class="neutral">—</td>
                </tr>
                {sibling_rows}
            </tbody>
        </table>
    """ if siblings else '<p class="empty-state">No related builds found in registry.</p>'

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MLBuild Report — {build.name or build_short}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

  :root {{
    --bg:        #0d0f12;
    --bg2:       #141720;
    --bg3:       #1c2030;
    --border:    #2a2f3f;
    --border2:   #363d52;
    --text:      #c8d0e0;
    --text-dim:  #6b7592;
    --text-head: #e8edf8;
    --accent:    #4f8ef7;
    --accent2:   #7eb8f7;
    --green:     #3dd68c;
    --red:       #f76f6f;
    --yellow:    #f7c948;
    --mono:      'IBM Plex Mono', monospace;
    --sans:      'IBM Plex Sans', sans-serif;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    line-height: 1.6;
    min-height: 100vh;
  }}

  /* ── Header ── */
  .header {{
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
    padding: 28px 40px 24px;
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 24px;
  }}

  .header-left {{}}

  .wordmark {{
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 500;
    color: var(--text-dim);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 10px;
  }}

  .build-name {{
    font-family: var(--mono);
    font-size: 26px;
    font-weight: 600;
    color: var(--text-head);
    letter-spacing: -0.02em;
    margin-bottom: 8px;
  }}

  .build-meta {{
    display: flex;
    gap: 18px;
    flex-wrap: wrap;
    align-items: center;
  }}

  .badge {{
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 3px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }}

  .badge-format  {{ background: #1a2540; color: var(--accent2); border: 1px solid #2a3f6f; }}
  .badge-quant   {{ background: #1a2d20; color: var(--green);   border: 1px solid #2a4f35; }}
  .badge-device  {{ background: #2a2010; color: var(--yellow);  border: 1px solid #4f3a10; }}
  .badge-task    {{ background: #2a1a40; color: #c084fc;        border: 1px solid #4f2a6f; }}

  .header-right {{
    text-align: right;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-dim);
    line-height: 2;
    white-space: nowrap;
  }}

  .header-right strong {{
    color: var(--text);
    font-weight: 500;
  }}

  /* ── Layout ── */
  .main {{
    max-width: 1100px;
    margin: 0 auto;
    padding: 36px 40px 60px;
    display: flex;
    flex-direction: column;
    gap: 32px;
  }}

  /* ── Section ── */
  .section {{
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
  }}

  .section-header {{
    padding: 14px 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
    background: var(--bg3);
  }}

  .section-title {{
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
    color: var(--text-dim);
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }}

  .section-dot {{
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent);
    flex-shrink: 0;
  }}

  .section-body {{ padding: 20px; }}

  /* ── Metric grid ── */
  .metric-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 16px;
  }}

  .metric {{
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 14px 16px;
  }}

  .metric-label {{
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-dim);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 6px;
  }}

  .metric-value {{
    font-family: var(--mono);
    font-size: 22px;
    font-weight: 600;
    color: var(--text-head);
    letter-spacing: -0.02em;
    line-height: 1.2;
  }}

  .metric-value-sm {{
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 500;
    color: var(--text-head);
    line-height: 1.4;
    word-break: break-all;
  }}

  .unit {{
    font-size: 12px;
    font-weight: 400;
    color: var(--text-dim);
    margin-left: 3px;
  }}

  /* ── Tables ── */
  table {{
    width: 100%;
    border-collapse: collapse;
    font-family: var(--mono);
    font-size: 12px;
  }}

  thead tr {{
    border-bottom: 1px solid var(--border2);
  }}

  th {{
    text-align: left;
    padding: 10px 12px;
    color: var(--text-dim);
    font-weight: 500;
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    white-space: nowrap;
  }}

  td {{
    padding: 9px 12px;
    border-bottom: 1px solid var(--border);
    color: var(--text);
    vertical-align: middle;
  }}

  tr:last-child td {{ border-bottom: none; }}

  tr.latest td {{ background: rgba(79,142,247,0.04); }}
  tr.current-row td {{ background: rgba(79,142,247,0.06); }}

  .green  {{ color: var(--green); }}
  .red    {{ color: var(--red); }}
  .neutral {{ color: var(--text-dim); }}

  /* ── Build info grid ── */
  .info-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
  }}

  .info-row {{
    display: contents;
  }}

  .info-key {{
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-dim);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 9px 16px;
    border-bottom: 1px solid var(--border);
    border-right: 1px solid var(--border);
    background: var(--bg3);
  }}

  .info-val {{
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text);
    padding: 9px 16px;
    border-bottom: 1px solid var(--border);
    word-break: break-all;
  }}

  .info-key:nth-last-child(2),
  .info-val:last-child {{
    border-bottom: none;
  }}

  /* ── Recommendations ── */
  .rec {{
    display: flex;
    gap: 14px;
    align-items: flex-start;
    padding: 14px 16px;
    border-radius: 4px;
    margin-bottom: 10px;
    border: 1px solid;
  }}

  .rec:last-child {{ margin-bottom: 0; }}

  .rec-ok     {{ background: rgba(61,214,140,0.05); border-color: rgba(61,214,140,0.2); }}
  .rec-action {{ background: rgba(247,201,72,0.05); border-color: rgba(247,201,72,0.2); }}

  .rec-icon {{
    font-family: var(--mono);
    font-size: 16px;
    font-weight: 600;
    color: var(--accent);
    margin-top: 1px;
    flex-shrink: 0;
    width: 20px;
  }}

  .rec-ok .rec-icon   {{ color: var(--green); }}
  .rec-action .rec-icon {{ color: var(--yellow); }}

  .rec strong {{
    display: block;
    font-size: 13px;
    color: var(--text-head);
    margin-bottom: 3px;
  }}

  .rec p {{
    font-size: 12px;
    color: var(--text-dim);
    line-height: 1.5;
  }}

  .rec code {{
    font-family: var(--mono);
    font-size: 11px;
    background: var(--bg3);
    border: 1px solid var(--border2);
    padding: 1px 6px;
    border-radius: 3px;
    color: var(--accent2);
  }}

  /* ── Empty states ── */
  .empty-state {{
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text-dim);
    padding: 20px 0;
    text-align: center;
  }}

  .empty-state code {{
    background: var(--bg3);
    border: 1px solid var(--border2);
    padding: 2px 8px;
    border-radius: 3px;
    color: var(--accent2);
  }}

  td.empty {{
    color: var(--text-dim);
    text-align: center;
    padding: 20px;
  }}

  td.empty code {{
    background: var(--bg3);
    border: 1px solid var(--border2);
    padding: 2px 8px;
    border-radius: 3px;
    color: var(--accent2);
  }}

  /* ── Footer ── */
  .footer {{
    text-align: center;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-dim);
    letter-spacing: 0.08em;
    padding: 20px 0 0;
    border-top: 1px solid var(--border);
  }}
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <div class="wordmark">MLBuild · Performance Report</div>
    <div class="build-name">{build.name or build_short}</div>
    <div class="build-meta">
      <span class="badge badge-format">{(build.format or 'unknown').upper()}</span>
      <span class="badge badge-quant">{quant_type.upper()}</span>
      <span class="badge badge-device">{build.target_device or 'unknown'}</span>
      <span class="badge badge-task">{task_value}</span>
    </div>
  </div>
  <div class="header-right">
    <div><strong>Build ID</strong><br>{build.build_id[:32]}<br>{build.build_id[32:]}</div>
    <div style="margin-top:10px"><strong>Created</strong><br>{created}</div>
    <div style="margin-top:10px"><strong>Size</strong><br>{size_mb:.3f} MB</div>
  </div>
</div>

<div class="main">

  <!-- Benchmark Summary -->
  <div class="section">
    <div class="section-header">
      <div class="section-dot"></div>
      <div class="section-title">Latest Benchmark</div>
    </div>
    <div class="section-body">
      {bench_section}
    </div>
  </div>

  <!-- Build Info -->
  <div class="section">
    <div class="section-header">
      <div class="section-dot" style="background:var(--green)"></div>
      <div class="section-title">Build Info</div>
    </div>
    <div class="info-grid">
      <div class="info-key">Artifact Path</div>
      <div class="info-val">{build.artifact_path or '—'}</div>
      <div class="info-key">Format</div>
      <div class="info-val">{build.format or '—'}</div>
      <div class="info-key">Target Device</div>
      <div class="info-val">{build.target_device or '—'}</div>
      <div class="info-key">Quantization</div>
      <div class="info-val">{quant_type.upper()}</div>
      <div class="info-key">Task</div>
      <div class="info-val">{task_value}</div>
      <div class="info-key">Benchmark Inputs</div>
      <div class="info-val">{input_desc}</div>
      <div class="info-key">Size</div>
      <div class="info-val">{size_mb:.4f} MB ({int(size_mb * 1024 * 1024):,} bytes)</div>
      <div class="info-key">Created</div>
      <div class="info-val">{created}</div>
      <div class="info-key">Build ID</div>
      <div class="info-val">{build.build_id}</div>
    </div>
  </div>

  <!-- Quantization Details -->
  <div class="section">
    <div class="section-header">
      <div class="section-dot" style="background:var(--yellow)"></div>
      <div class="section-title">Quantization Details</div>
    </div>
    <div class="section-body">
      <table>
        <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
        <tbody>{quant_rows}</tbody>
      </table>
    </div>
  </div>

  <!-- Benchmark History -->
  <div class="section">
    <div class="section-header">
      <div class="section-dot" style="background:var(--accent2)"></div>
      <div class="section-title">Benchmark History</div>
      <span style="font-family:var(--mono);font-size:10px;color:var(--text-dim);margin-left:auto">last {min(len(benchmarks),5)} of {len(benchmarks)} runs</span>
    </div>
    <div class="section-body" style="padding:0">
      <table>
        <thead>
          <tr>
            <th>Measured At</th><th>Device</th><th>Compute</th>
            <th>p50</th><th>p95</th><th>p99</th>
            <th>Throughput</th><th>Memory</th><th>Runs</th>
          </tr>
        </thead>
        <tbody>{bench_history_rows}</tbody>
      </table>
    </div>
  </div>

  <!-- Related Builds -->
  <div class="section">
    <div class="section-header">
      <div class="section-dot" style="background:#a78bfa"></div>
      <div class="section-title">Related Builds</div>
    </div>
    <div class="section-body" style="padding: {'0' if siblings else '20px'}">
      {sibling_section}
    </div>
  </div>

  <!-- Recommendations -->
  <div class="section">
    <div class="section-header">
      <div class="section-dot" style="background:var(--yellow)"></div>
      <div class="section-title">Recommendations</div>
    </div>
    <div class="section-body">
      {rec_html}
    </div>
  </div>

  <div class="footer">
    Generated by MLBuild · {generated_at}
  </div>

</div>
</body>
</html>"""


# ============================================================
# CLI command
# ============================================================

@click.command()
@click.argument("build_id")
@click.option("--output", "-o", default=None,
              help="Output file path (default: mlbuild_report_<id>_<ts>.html)")
@click.option("--open", "open_browser", is_flag=True,
              help="Open report in browser after generation")
@click.option("--format", "fmt", default="html",
              type=click.Choice(["html", "pdf"]),
              help="Output format (pdf requires weasyprint)")
def report(build_id: str, output: str, open_browser: bool, fmt: str):
    """
    Generate a performance report for a build.

    Produces a self-contained HTML file with benchmark history,
    build metadata, quantization details, and deployment recommendations.

    Examples:
        mlbuild report e67a6a87
        mlbuild report e67a6a87 --open
        mlbuild report e67a6a87 --output ~/Desktop/mobilenet_report.html
        mlbuild report e67a6a87 --format pdf
    """
    try:
        registry = LocalRegistry()

        # ── Resolve build ──────────────────────────────────────
        build = registry.resolve_build(build_id)
        if not build:
            console.print(f"[red]Build not found: {build_id}[/red]")
            sys.exit(2)

        console.print(f"\n[bold]Generating report[/bold] for {build.name or build.build_id[:16]}...")
        console.print(f"  Task:   {_display_task(build)}")
        console.print(f"  Inputs: {_describe_inputs(build)}")

        # ── Fetch data ─────────────────────────────────────────
        benchmarks = _get_benchmarks(registry, build.build_id)
        siblings = _get_sibling_builds(registry, build)

        console.print(f"  Benchmarks found: {len(benchmarks)}")
        console.print(f"  Related builds:   {len(siblings)}")

        # ── Render HTML ────────────────────────────────────────
        html = _render_html(build, benchmarks, siblings)

        # ── Output path ────────────────────────────────────────
        if output:
            out_path = Path(output)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = Path(f"mlbuild_report_{build.build_id[:8]}_{ts}.html")

        # ── Write HTML ─────────────────────────────────────────
        out_path.write_text(html, encoding="utf-8")
        console.print(f"\n[green]✓ Report written to:[/green] {out_path.resolve()}")

        # ── Optional PDF ───────────────────────────────────────
        if fmt == "pdf":
            pdf_path = out_path.with_suffix(".pdf")
            try:
                import weasyprint
                weasyprint.HTML(filename=str(out_path)).write_pdf(str(pdf_path))
                console.print(f"[green]✓ PDF written to:[/green] {pdf_path.resolve()}")
                out_path = pdf_path
            except ImportError:
                console.print(
                    "[yellow]⚠ weasyprint not installed — HTML only.[/yellow]\n"
                    "  Install with: pip install weasyprint"
                )

        # ── Open browser ───────────────────────────────────────
        if open_browser:
            import webbrowser
            webbrowser.open(out_path.resolve().as_uri())
            console.print("[dim]Opened in browser.[/dim]")

        console.print()

        # ── Cloud sync ────────────────────────────────────────
        try:
            from ...cloud.sync import push
            push("command_activity", {
                "command": "report",
                "args": {
                    "build_id": build.build_id,
                    "build_name": build.name,
                    "format": fmt,
                    "output_path": str(out_path.resolve()),
                    "benchmark_count": len(benchmarks),
                    "sibling_count": len(siblings),
                },
                "exit_code": 0,
                "machine_id": None,
                "machine_name": None,
            })
        except Exception:
            pass

        console.print()

    except MLBuildError as e:
        console.print(f"\n[red]Error:[/red] {e}\n")
        sys.exit(2)
    except Exception as e:
        console.print(f"\n[red]Unexpected error:[/red] {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(2)