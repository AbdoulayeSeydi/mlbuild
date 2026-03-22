"""
Profile command — model profiling.

CoreML NeuralNetwork : Per-layer cumulative timing
CoreML MLProgram     : Full-model timing + op breakdown
TFLite (standard)    : Full-model timing + tensor/op summary
TFLite (--deep)      : 6-feature deep profiling (no device required)
CoreML (--deep)      : 5-feature deep profiling (NeuralNetwork only)

TFLite deep profiling features (--deep flag):
  1. Per-op timing       Real hardware timing via TFLite built-in profiler
  2. Memory flow         Activation memory at each layer boundary
  3. Bottleneck          COMPUTE vs MEMORY bound per op
  4. Cold start          Load → first → stable decomposition + sparkline
  5. Quant sensitivity   fp32 vs int8 divergence (requires --int8-build)
  6. Fusion detection    Fused kernels + missed fusion opportunities

CoreML deep profiling features (--deep flag, NeuralNetwork only):
  1. Per-layer timing    Incremental p50 via sliced subgraph benchmarking
  2. Memory flow         Estimated activation memory from weight dimensions
  3. Bottleneck          COMPUTE vs MEMORY bound per layer
  4. Cold start          Load → first → stable decomposition + sparkline
  5. Fusion detection    Conv+Activation, BN+Scale, etc. from spec layer types

Standard flags (all formats):
  --cold-start         Cold start decomposition
  --memory             Peak RSS profiling
  --analyze-warmup     Warmup stability analysis
"""

import sys
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from pathlib import Path

from ...registry import LocalRegistry
from ...profiling import CumulativeLayerProfiler

# --- PATCH: task-aware imports ---
from ...core.task_detection import TaskType
from ...core.task_inputs import TaskInputFactory, ModelInfo as InputModelInfo
from ...core.task_validation import (
    TaskOutputValidator,
    StrictOutputConfig,
    format_validation_warning,
    should_exit_on_validation,
)

console = Console(width=None)


# --- PATCH: shared helper (mirrors build.py / benchmark.py) ---
def _read_global_strict() -> bool:
    """Read [validation] strict_output from .mlbuild/config.toml if present."""
    try:
        config_path = Path(".mlbuild/config.toml")
        if not config_path.exists():
            return False
        import tomllib
        with config_path.open("rb") as f:
            data = tomllib.load(f)
        return bool(data.get("validation", {}).get("strict_output", False))
    except Exception:
        return False


# --- PATCH: build task-aware synthetic inputs for a single inference pass ---
def _build_profile_inputs(artifact_path: Path, fmt: str, task: TaskType, seq_len: int) -> dict:
    """
    Return a name→ndarray dict of synthetic inputs appropriate for the task.

    Falls back to zeros on any failure so profiling never silently breaks.
    seq_len is only used for NLP (single representative pass, not the full ladder).
    """
    try:
        import numpy as np

        if fmt in ("coreml", "mlpackage") or artifact_path.is_dir():
            import coremltools as ct
            spec = ct.models.MLModel(str(artifact_path)).get_spec()
            input_specs = [
                (inp.name, tuple(max(d, 1) for d in inp.type.multiArrayType.shape))
                for inp in spec.description.input
            ]
            input_names  = [n for n, _ in input_specs]
            input_shapes = {n: s for n, s in input_specs}

        elif fmt == "tflite":
            import tensorflow as tf
            interp = tf.lite.Interpreter(model_path=str(artifact_path))
            interp.allocate_tensors()
            input_names  = [d["name"] for d in interp.get_input_details()]
            input_shapes = {d["name"]: tuple(d["shape"]) for d in interp.get_input_details()}

        else:
            return {}

        info = InputModelInfo(
            op_types=[],
            input_shapes=input_shapes,
            input_names=input_names,
            metadata={},
            num_nodes=0,
        )

        factory = TaskInputFactory(info)
        schemas = factory.build_input_schemas(task)

        # For NLP, override sequence dimension with the representative seq_len
        if task == TaskType.NLP:
            from ...core.task_inputs import get_nlp_seq_lens
            schemas = factory.build_input_schemas(task, seq_len=seq_len)

        return {name: arr for name, arr in schemas.items()}

    except Exception:
        # Graceful fallback — zeros, same shapes as before
        import numpy as np
        try:
            return {
                name: np.zeros(shape, dtype=np.float32)
                for name, shape in input_shapes.items()
            }
        except Exception:
            return {}


# ============================================================
# TFLite deep profiling display helpers
# ============================================================

def _display_op_timing(op_timing, total_ms: float, top: int):
    console.print()
    console.print(Rule("[bold cyan]1 · Per-Op Timing[/bold cyan]"))
    console.print(f"[dim]Total model time: {total_ms:.3f} ms   Showing top {top} by execution time[/dim]\n")

    t = Table(show_header=True, header_style="bold cyan", show_lines=False)
    t.add_column("#",        style="dim", width=4)
    t.add_column("Op Name",              max_width=36)
    t.add_column("Type",                 max_width=22)
    t.add_column("Time (ms)", justify="right", width=10)
    t.add_column("% Total",  justify="right", width=10)
    t.add_column("Fused",    justify="center", width=6)

    for row in op_timing[:top]:
        bar = "█" * int(row.pct_total / 2)
        c = "red" if row.pct_total > 20 else "yellow" if row.pct_total > 10 else "green"
        fused = "[green]✓[/green]" if row.is_fused else "[dim]—[/dim]"
        t.add_row(
            str(row.index), row.name[:36], row.op_type[:22],
            f"[{c}]{row.time_ms:.3f}[/{c}]",
            f"{row.pct_total:.1f}%  {bar}",
            fused,
        )

    console.print(t)
    if len(op_timing) > top:
        console.print(f"[dim]... {len(op_timing) - top} more ops (use --top N)[/dim]")


def _display_memory_flow(memory_flow, peak_mb: float, peak_op: str, top: int):
    console.print()
    console.print(Rule("[bold cyan]2 · Tensor Memory Flow[/bold cyan]"))
    console.print(f"[dim]Peak activation: {peak_mb:.3f} MB at {peak_op}[/dim]\n")

    t = Table(show_header=True, header_style="bold cyan", show_lines=False)
    t.add_column("Op",       max_width=30)
    t.add_column("Type",     max_width=20)
    t.add_column("Input MB",  justify="right", width=10)
    t.add_column("Output MB", justify="right", width=10)
    t.add_column("Act. MB",   justify="right", width=10)
    t.add_column("Peak",      justify="center", width=6)

    sorted_flow = sorted(memory_flow, key=lambda r: r.activation_mb, reverse=True)
    for row in sorted_flow[:top]:
        peak_str = "[bold red]★[/bold red]" if row.is_memory_peak else "[dim]—[/dim]"
        c = "red" if row.is_memory_peak else "yellow" if row.activation_mb > peak_mb * 0.5 else "default"
        t.add_row(
            row.op_name[:30], row.op_type[:20],
            f"{row.input_mb:.3f}", f"{row.output_mb:.3f}",
            f"[{c}]{row.activation_mb:.3f}[/{c}]",
            peak_str,
        )

    console.print(t)


def _display_bottlenecks(bottlenecks, compute_bound: int, memory_bound: int, top: int):
    console.print()
    console.print(Rule("[bold cyan]3 · Bottleneck Classification[/bold cyan]"))
    console.print(f"[dim]Ridge point: 38 FLOPs/byte (Apple M1). Above = compute-bound, below = memory-bound.[/dim]\n")

    total_cls = compute_bound + memory_bound
    if total_cls > 0:
        console.print(
            f"  [green]COMPUTE-BOUND[/green]: {compute_bound} ops ({compute_bound/total_cls*100:.0f}%)  "
            f"[yellow]MEMORY-BOUND[/yellow]: {memory_bound} ops ({memory_bound/total_cls*100:.0f}%)  "
            f"[dim]UNKNOWN: {len(bottlenecks)-total_cls}[/dim]\n"
        )

    t = Table(show_header=True, header_style="bold cyan", show_lines=False)
    t.add_column("Op",        min_width=20, max_width=26, no_wrap=True)
    t.add_column("Type",      min_width=16, max_width=20, no_wrap=True)
    t.add_column("FLOPs",     justify="right", min_width=7)
    t.add_column("Bytes",     justify="right", min_width=9)
    t.add_column("AI",        justify="right", min_width=4)
    t.add_column("Bound",     justify="center", min_width=8)
    t.add_column("ms",        justify="right", min_width=6)

    for row in bottlenecks[:top]:
        if row.classification == "COMPUTE":
            b = "[green]COMPUTE[/green]"
        elif row.classification == "MEMORY":
            b = "[yellow]MEMORY[/yellow]"
        else:
            b = "[dim]UNKNOWN[/dim]"

        flops_s = f"{row.flops/1e6:.2f}M" if row.flops >= 1e6 else f"{row.flops:.0f}"
        bytes_s = f"{row.bytes_moved/1024:.1f}KB" if row.bytes_moved >= 1024 else f"{row.bytes_moved:.0f}B"

        t.add_row(
            row.op_name[:30], row.op_type[:18],
            flops_s, bytes_s, f"{row.arithmetic_intensity:.1f}",
            b, f"{row.time_ms:.3f}",
        )
    console.print(t)

    if memory_bound > compute_bound and memory_bound > 0:
        console.print(Panel(
            "Model is predominantly [yellow]memory-bound[/yellow]. "
            "Quantization, pruning, or smaller feature maps will have more impact than kernel tuning.",
            title="[yellow]Optimization Insight[/yellow]", border_style="yellow",
        ))
    elif compute_bound > memory_bound and compute_bound > 0:
        console.print(Panel(
            "Model is predominantly [green]compute-bound[/green]. "
            "Operator fusion, GPU/NPU delegation, or FP16 kernels will have most impact.",
            title="[green]Optimization Insight[/green]", border_style="green",
        ))


def _display_cold_start_decomposition(cs):
    console.print()
    console.print(Rule("[bold cyan]4 · Cold Start Decomposition[/bold cyan]"))
    console.print()

    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("Phase",        width=26)
    t.add_column("Time (ms)",    justify="right", width=12)
    t.add_column("Notes",        style="dim")

    t.add_row("Model load",        f"{cs.load_time_ms:.2f}",       "interpreter + allocate_tensors")
    t.add_row("First inference",   f"{cs.first_inference_ms:.2f}", "XNNPACK graph build + cache fill")
    t.add_row("Second inference",  f"{cs.second_inference_ms:.2f}", "partially warm")
    t.add_row("Stable p50",        f"{cs.stable_p50_ms:.3f}",      f"from run {cs.stable_start_run} onward")

    tax_ms = max(cs.cold_start_tax_ms, 0.0)
    tax_pct = max(cs.cold_start_tax_pct, 0.0)
    c = "red" if tax_pct > 100 else "yellow" if tax_pct > 30 else "green"
    t.add_row(
        "Cold start tax",
        f"[{c}]+{tax_ms:.2f} ms ({tax_pct:.0f}%)[/{c}]",
        "first_inference − stable_p50",
    )
    console.print(t)

    from ...profiling.cold_start import sparkline
    curve = cs.warmup_curve_ms
    n = min(50, len(curve))
    spark = sparkline(curve[:n], width=50)
    mn, mx = min(curve[:n]), max(curve[:n])
    console.print()
    console.print(f"[bold]Warmup Curve[/bold] [dim](first {n} runs)[/dim]")
    console.print(f"  [dim]max {mx:6.1f}ms[/dim] ┤{spark}")
    console.print(f"  [dim]min {mn:6.1f}ms[/dim] ┤{'▔' * len(spark)}")
    console.print()

    if cs.cold_start_tax_pct > 100:
        console.print(Panel(
            f"First inference is [red]{cs.cold_start_tax_pct:.0f}%[/red] above stable. "
            "XNNPACK builds its optimized execution plan on the first call. "
            "Pre-warm with 1–2 dummy inferences before showing results to users.",
            title="[yellow]Cold Start Recommendation[/yellow]", border_style="yellow",
        ))
    elif cs.cold_start_tax_pct > 30:
        console.print(Panel(
            f"Cold start adds {cs.cold_start_tax_ms:.1f} ms. "
            "Consider pre-warming the model at app launch.",
            border_style="dim",
        ))


def _display_quant_sensitivity(rows):
    console.print()
    console.print(Rule("[bold cyan]5 · Quantization Sensitivity Map[/bold cyan]"))
    console.print("[dim]Output divergence between fp32 and int8 on 50 synthetic inputs.[/dim]\n")

    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("Layer",       max_width=30)
    t.add_column("MSE",         justify="right", width=12)
    t.add_column("MAE",         justify="right", width=12)
    t.add_column("Max Error",   justify="right", width=12)
    t.add_column("Cosine Sim",  justify="right", width=12)
    t.add_column("Sensitivity", justify="center", width=12)

    for row in rows:
        if row.sensitivity == "LOW":
            s = "[green]LOW[/green]"
        elif row.sensitivity == "MEDIUM":
            s = "[yellow]MEDIUM[/yellow]"
        else:
            s = "[red]HIGH[/red]"
        c = "green" if row.cosine_similarity > 0.99 else "yellow" if row.cosine_similarity > 0.95 else "red"
        t.add_row(
            row.op_name[:30],
            f"{row.mse:.6f}", f"{row.mae:.6f}", f"{row.max_abs_error:.6f}",
            f"[{c}]{row.cosine_similarity:.4f}[/{c}]",
            s,
        )
    console.print(t)
    console.print()
    console.print(Panel(
        "Cosine sim > 0.99 = output ranking preserved (generally safe to deploy).\n"
        "HIGH sensitivity = quantization hurts these outputs most — consider "
        "keeping sensitive layers in fp32 (selective/mixed quantization).",
        title="[dim]Interpretation[/dim]", border_style="dim",
    ))


def _display_fusion_map(fusion_groups, unfused_opportunities):
    console.print()
    console.print(Rule("[bold cyan]6 · Op Fusion Detection[/bold cyan]"))
    console.print()

    if fusion_groups:
        console.print(f"[bold]{len(fusion_groups)} fused kernel group(s) detected:[/bold]\n")
        t = Table(show_header=True, header_style="bold cyan")
        t.add_column("Group", width=6)
        t.add_column("Op Indices", width=16)
        t.add_column("Pattern", max_width=30)
        t.add_column("Note", style="dim")
        for g in fusion_groups:
            t.add_row(
                str(g.group_id),
                ", ".join(str(i) for i in g.op_indices),
                " + ".join(g.op_types),
                g.note,
            )
        console.print(t)
    else:
        console.print("[dim]No fused kernel groups detected.[/dim]")

    console.print()
    if unfused_opportunities:
        console.print(f"[bold yellow]Missed Fusion Opportunities ({len(unfused_opportunities)}):[/bold yellow]\n")
        for opp in unfused_opportunities:
            console.print(f"  [yellow]•[/yellow] {opp}")
        console.print()
        console.print(Panel(
            "These op sequences could potentially be fused. "
            "Reordering layers or adjusting activation placement may allow TFLite to fuse them automatically.",
            title="[yellow]Optimization Hint[/yellow]", border_style="yellow",
        ))
    else:
        console.print("[green]✓ No obvious missed fusion opportunities.[/green]")


# ============================================================
# CoreML deep profiling display helpers
# ============================================================

def _display_coreml_layer_timing(timing_rows, total_ms: float, top: int):
    console.print()
    console.print(Rule("[bold cyan]1 · Per-Layer Timing[/bold cyan]"))
    console.print(f"[dim]Total model time: {total_ms:.3f} ms   Showing top {top} by incremental time[/dim]\n")

    t = Table(show_header=True, header_style="bold cyan", show_lines=False)
    t.add_column("#",          style="dim", width=4)
    t.add_column("Layer Name", max_width=32)
    t.add_column("Type",       max_width=18)
    t.add_column("Inc (ms)",   justify="right", width=9)
    t.add_column("Cum (ms)",   justify="right", width=9)
    t.add_column("% Total",    justify="right", width=10)
    t.add_column("Param MB",   justify="right", width=9)

    for row in timing_rows[:top]:
        bar = "█" * int(row.pct_total / 2)
        c = "red" if row.pct_total > 20 else "yellow" if row.pct_total > 10 else "green"
        t.add_row(
            str(row.index), row.name[:32], row.layer_type[:18],
            f"[{c}]{row.incremental_p50_ms:.3f}[/{c}]",
            f"{row.cumulative_p50_ms:.3f}",
            f"{row.pct_total:.1f}%  {bar}",
            f"{row.param_mb:.2f}",
        )
    console.print(t)
    if len(timing_rows) > top:
        console.print(f"[dim]... {len(timing_rows) - top} more layers (use --top N)[/dim]")


def _display_coreml_memory_flow(memory_rows, peak_mb: float, peak_layer: str, top: int):
    console.print()
    console.print(Rule("[bold cyan]2 · Layer Memory Flow (estimated)[/bold cyan]"))
    console.print(
        f"[dim]Peak activation: ~{peak_mb:.3f} MB at {peak_layer}   "
        f"Sizes estimated from weight dims, not live tensors[/dim]\n"
    )
    t = Table(show_header=True, header_style="bold cyan", show_lines=False)
    t.add_column("Layer",         max_width=30)
    t.add_column("Type",          max_width=18)
    t.add_column("Param MB",      justify="right", width=9)
    t.add_column("Act. MB (est)", justify="right", width=12)
    t.add_column("Peak",          justify="center", width=6)

    for row in memory_rows[:top]:
        peak_str = "[bold red]★[/bold red]" if row.is_memory_peak else "[dim]—[/dim]"
        c = "red" if row.is_memory_peak else "yellow" if row.activation_mb > peak_mb * 0.5 else "default"
        t.add_row(
            row.name[:30], row.layer_type[:18],
            f"{row.param_mb:.3f}",
            f"[{c}]{row.activation_mb:.3f}[/{c}]",
            peak_str,
        )
    console.print(t)


def _display_coreml_bottlenecks(bottleneck_rows, compute_bound: int, memory_bound: int, top: int):
    console.print()
    console.print(Rule("[bold cyan]3 · Bottleneck Classification[/bold cyan]"))
    console.print(f"[dim]Ridge point: 38 FLOPs/byte (Apple M1). Above = compute-bound, below = memory-bound.[/dim]\n")

    total_cls = compute_bound + memory_bound
    if total_cls > 0:
        console.print(
            f"  [green]COMPUTE-BOUND[/green]: {compute_bound} layers ({compute_bound/total_cls*100:.0f}%)  "
            f"[yellow]MEMORY-BOUND[/yellow]: {memory_bound} layers ({memory_bound/total_cls*100:.0f}%)  "
            f"[dim]UNKNOWN: {len(bottleneck_rows)-total_cls}[/dim]\n"
        )

    t = Table(show_header=True, header_style="bold cyan", show_lines=False)
    t.add_column("Layer",   min_width=20, max_width=26, no_wrap=True)
    t.add_column("Type",    min_width=14, max_width=18, no_wrap=True)
    t.add_column("FLOPs",   justify="right", min_width=7)
    t.add_column("Bytes",   justify="right", min_width=9)
    t.add_column("AI",      justify="right", min_width=4)
    t.add_column("Bound",   justify="center", min_width=8)
    t.add_column("Inc ms",  justify="right", min_width=7)

    for row in bottleneck_rows[:top]:
        b = ("[green]COMPUTE[/green]" if row.classification == "COMPUTE"
             else "[yellow]MEMORY[/yellow]" if row.classification == "MEMORY"
             else "[dim]UNKNOWN[/dim]")
        flops_s = f"{row.flops/1e6:.2f}M" if row.flops >= 1e6 else f"{row.flops:.0f}"
        bytes_s = f"{row.bytes_moved/1024:.1f}KB" if row.bytes_moved >= 1024 else f"{row.bytes_moved:.0f}B"
        t.add_row(
            row.name[:26], row.layer_type[:18],
            flops_s, bytes_s, f"{row.arithmetic_intensity:.1f}",
            b, f"{row.incremental_p50_ms:.3f}",
        )
    console.print(t)

    if memory_bound > compute_bound and memory_bound > 0:
        console.print(Panel(
            "Model is predominantly [yellow]memory-bound[/yellow]. "
            "Quantization, pruning, or smaller feature maps will have more impact than kernel tuning.",
            title="[yellow]Optimization Insight[/yellow]", border_style="yellow",
        ))
    elif compute_bound > memory_bound and compute_bound > 0:
        console.print(Panel(
            "Model is predominantly [green]compute-bound[/green]. "
            "Operator fusion, ANE delegation, or FP16 will have most impact.",
            title="[green]Optimization Insight[/green]", border_style="green",
        ))


def _display_coreml_fusion(fusion_groups, unfused_opportunities):
    console.print()
    console.print(Rule("[bold cyan]5 · Op Fusion Detection[/bold cyan]"))
    console.print()

    if fusion_groups:
        console.print(f"[bold]{len(fusion_groups)} fusible pattern(s) detected:[/bold]\n")
        t = Table(show_header=True, header_style="bold cyan")
        t.add_column("Group",   width=6)
        t.add_column("Indices", width=12)
        t.add_column("Pattern", max_width=28)
        t.add_column("Note",    style="dim")
        for g in fusion_groups:
            t.add_row(
                str(g.group_id),
                ", ".join(str(i) for i in g.layer_indices),
                " + ".join(g.layer_types),
                g.note,
            )
        console.print(t)
    else:
        console.print("[dim]No fusible patterns detected.[/dim]")

    console.print()
    if unfused_opportunities:
        console.print(f"[bold yellow]Unfused Opportunities ({len(unfused_opportunities)}):[/bold yellow]\n")
        for opp in unfused_opportunities:
            console.print(f"  [yellow]•[/yellow] {opp}")
    else:
        console.print("[green]✓ No obvious missed fusion opportunities.[/green]")


def _display_mlprogram_deep_note():
    console.print(Panel(
        "This model is [bold]MLProgram[/bold] format.\n\n"
        "MLProgram models are compiled by the ANE toolchain at load time — "
        "the graph is fused and optimised before any layer runs. "
        "Per-layer slicing is not possible without Xcode Instruments.\n\n"
        "[bold]Available:[/bold]  ④ Cold start decomposition\n"
        "[bold]Unavailable without Xcode:[/bold]  ① timing  ② memory  ③ bottleneck  ⑤ fusion",
        title="[yellow]MLProgram — Partial Deep Profile[/yellow]",
        border_style="yellow",
    ))


# ============================================================
# TFLite standard profiling (no --deep)
# ============================================================

def _profile_tflite_standard(build, runs: int, warmup: int):
    from ...backends.tflite.benchmark_runner import TFLiteBenchmarkRunner

    console.print(f"[dim]Format: TFLite[/dim]")
    console.print(f"[dim]Artifact: {Path(build.artifact_path).name}[/dim]\n")

    runner = TFLiteBenchmarkRunner()
    metrics = runner.benchmark(
        model_path=Path(build.artifact_path),
        runs=runs,
        warmup=warmup,
    )

    console.print("[bold]Full Model Timing[/bold]\n")
    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    for k, v in [
        ("p50",        f"{metrics['p50_ms']:.3f} ms"),
        ("p95",        f"{metrics['p95_ms']:.3f} ms"),
        ("p99",        f"{metrics['p99_ms']:.3f} ms"),
        ("mean",       f"{metrics['mean_ms']:.3f} ms"),
        ("std",        f"{metrics['std_ms']:.3f} ms"),
        ("min",        f"{metrics['min_ms']:.3f} ms"),
        ("max",        f"{metrics['max_ms']:.3f} ms"),
        ("throughput", f"{metrics['throughput_inf_per_sec']:.1f} inf/s"),
        ("memory RSS", f"{metrics['memory_rss_mb']:.2f} MB"),
        ("failures",   str(metrics['failures'])),
    ]:
        t.add_row(k, v)
    console.print(t)

    console.print("\n[bold]Model Structure[/bold]\n")
    try:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=str(build.artifact_path))
        interp.allocate_tensors()
        in_d = interp.get_input_details()
        out_d = interp.get_output_details()

        st = Table(show_header=True, header_style="bold cyan")
        st.add_column("Property")
        st.add_column("Value", justify="right")
        st.add_row("Total tensors", str(len(interp.get_tensor_details())))
        st.add_row("Input tensors", str(len(in_d)))
        st.add_row("Output tensors", str(len(out_d)))
        for i, d in enumerate(in_d):
            st.add_row(f"Input[{i}] shape", str(tuple(d["shape"])))
            st.add_row(f"Input[{i}] dtype", str(d["dtype"].__name__))
        for i, d in enumerate(out_d):
            st.add_row(f"Output[{i}] shape", str(tuple(d["shape"])))
            st.add_row(f"Output[{i}] dtype", str(d["dtype"].__name__))
        console.print(st)
    except Exception as e:
        console.print(f"[yellow]Structure unavailable: {e}[/yellow]")

    console.print(Panel(
        "Run [bold]mlbuild profile <build_id> --deep[/bold] for full per-layer analysis:\n"
        "  • Per-op timing (real hardware, not estimates)\n"
        "  • Tensor memory flow at each layer\n"
        "  • COMPUTE vs MEMORY bottleneck classification\n"
        "  • Cold start decomposition with sparkline\n"
        "  • Op fusion detection",
        title="[cyan]Deep Profiling Available[/cyan]",
        border_style="cyan",
    ))


# ============================================================
# Shared display helpers (cold start + memory)
# ============================================================

def _display_cold_start(result, title: str = "Cold Start Analysis"):
    console.print()
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]"))
    console.print()

    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    t.add_column("Signal")

    t.add_row("Model load time",  f"{result.load_time_ms:.1f} ms", "[dim]disk → memory + JIT compile[/dim]")

    c_first = "red" if result.warmup_ratio > 5 else "yellow" if result.warmup_ratio > 2 else "green"
    t.add_row("First inference", f"{result.first_inference_ms:.2f} ms",
              f"[{c_first}]{result.warmup_ratio:.1f}x slower than stable[/{c_first}]")

    t.add_row("Stable p50", f"{result.stable_p50_ms:.3f} ms", "[dim]median after warmup[/dim]")
    t.add_row("Stable p95", f"{result.stable_p95_ms:.3f} ms", "")
    t.add_row("Stable p99", f"{result.stable_p99_ms:.3f} ms", "")

    stable_str = f"run {result.stable_start_run}" if result.stable_start_run > 0 else "run 1 (immediate)"
    t.add_row("Stable starts at", stable_str, "[dim]CV < 5% window[/dim]")

    overhead_ms = result.first_inference_ms - result.stable_p50_ms
    overhead_pct = (overhead_ms / result.stable_p50_ms * 100) if result.stable_p50_ms > 0 else 0
    c_oh = "red" if overhead_pct > 100 else "yellow" if overhead_pct > 30 else "green"
    t.add_row("Cold start overhead", f"{overhead_ms:.2f} ms",
              f"[{c_oh}]+{overhead_pct:.0f}% above stable[/{c_oh}]")

    t.add_row("Warmup ratio", f"{result.warmup_ratio:.2f}x", "")

    drift_str = "[red]⚠ throttling detected[/red]" if result.thermal_drift_detected else "[green]stable[/green]"
    t.add_row("Thermal drift", f"{result.thermal_normalized_slope:+.4f} rel/run", drift_str)

    console.print(t)

    from ...profiling.cold_start import sparkline
    n = min(50, len(result.warmup_curve_ms))
    spark = sparkline(result.warmup_curve_ms[:n], width=50)
    mn, mx = min(result.warmup_curve_ms[:n]), max(result.warmup_curve_ms[:n])
    console.print()
    console.print(f"[bold]Warmup Curve[/bold] [dim](first {n} runs)[/dim]")
    console.print(f"  [dim]max {mx:6.1f}ms[/dim] ┤{spark}")
    console.print(f"  [dim]min {mn:6.1f}ms[/dim] ┤{'▔' * len(spark)}")
    console.print(f"  [dim]       run 1{'':>{len(spark)-8}}run {n}[/dim]")
    if result.stable_start_run > 0:
        pos = int(result.stable_start_run / n * len(spark))
        console.print(f"  [dim]{'':>14}{' '*pos}↑ stable from run {result.stable_start_run}[/dim]")
    console.print()


def _display_memory_report(report):
    console.print()
    console.print(Rule("[bold cyan]Memory Profile[/bold cyan]"))
    console.print()

    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    t.add_column("Notes")

    t.add_row("Baseline RSS",  f"{report.baseline_rss_mb:.1f} MB", "[dim]before inference[/dim]")
    t.add_row("Peak RSS",      f"{report.peak_rss_mb:.1f} MB",     "[dim]mid-inference spike[/dim]")
    t.add_row("Final RSS",     f"{report.final_rss_mb:.1f} MB",    "[dim]after inference[/dim]")

    c = "red" if report.peak_delta_mb > 200 else "yellow" if report.peak_delta_mb > 50 else "green"
    t.add_row("Peak spike", f"[{c}]+{report.peak_delta_mb:.1f} MB[/{c}]", "[dim]peak - baseline[/dim]")

    if report.tracemalloc_available:
        t.add_row("Python heap Δ",       f"+{report.heap_delta_mb:.1f} MB",    "[dim]tracemalloc[/dim]")
        t.add_row("Driver overhead est.", f"~{report.non_python_rss_delta_mb:.1f} MB",
                  "[dim]spike - Python heap (approx)[/dim]")
    else:
        t.add_row("Python heap Δ", "N/A", "[dim]tracemalloc unavailable[/dim]")

    t.add_row("Samples", str(report.num_samples), f"[dim]@ {report.poll_interval_ms}ms intervals[/dim]")
    t.add_row("Duration", f"{report.duration_ms:.0f} ms", "")
    console.print(t)

    if report.final_rss_mb < report.baseline_rss_mb:
        console.print(f"[dim]Note: Final RSS below baseline — framework released cached buffers (expected).[/dim]")
    console.print()


# ============================================================
# CoreML display helpers (standard profiling)
# ============================================================

def _display_neuralnetwork_profile(result, top):
    profiles = result['layers']
    console.print(f"[bold]Top {top} Slowest Layers (Incremental Time)[/bold]\n")
    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("Layer", style="dim")
    t.add_column("Type")
    t.add_column("Inc Time (ms)", justify="right")
    t.add_column("Cum Time (ms)", justify="right")
    t.add_column("Param MB", justify="right")
    for p in sorted(profiles, key=lambda p: p['incremental_time']['p50_ms'], reverse=True)[:top]:
        t.add_row(p['name'][:30], p['type'][:20],
                  f"{p['incremental_time']['p50_ms']:.3f}",
                  f"{p['cumulative_time']['p50_ms']:.3f}",
                  f"{p['param_memory_mb']:.2f}")
    console.print(t)
    total = profiles[-1]['cumulative_time']['p50_ms'] if profiles else 0
    console.print(f"\n[bold]Total Model Time:[/bold] {total:.3f} ms\n")


def _display_mlprogram_profile(result):
    console.print(Panel(result['warning'], title="[yellow]MLProgram Profiling Limitation[/yellow]",
                        border_style="yellow"))
    console.print()
    timing = result['full_model_timing']
    console.print("[bold]Full Model Timing[/bold]\n")
    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    for k, v in [("p50", timing['p50_ms']), ("p95", timing['p95_ms']),
                 ("p99", timing['p99_ms']), ("mean", timing['mean_ms']),
                 ("std", timing['std_ms'])]:
        t.add_row(k, f"{v:.3f} ms")
    console.print(t)

    bd = result['operation_breakdown']
    console.print(f"\n[bold]Operation Breakdown[/bold] — {bd['total_operations']} ops\n")
    ot = Table(show_header=True, header_style="bold cyan")
    ot.add_column("Op Type")
    ot.add_column("Count", justify="right")
    ot.add_column("% Total", justify="right")
    total = bd['total_operations']
    for op, cnt in sorted(bd['operation_counts'].items(), key=lambda x: x[1], reverse=True)[:15]:
        ot.add_row(op, str(cnt), f"{cnt/total*100:.1f}%")
    console.print(ot)
    console.print("\n[dim]For detailed per-layer profiling, use Xcode Instruments[/dim]\n")


def _display_warmup_analysis_coreml(artifact_path: Path):
    try:
        from ...profiling.warmup_analyzer import EnterpriseWarmupAnalyzer
        from ...visualization.charts import create_warmup_curve_chart
    except ImportError as e:
        console.print(f"[yellow]Warmup analyzer unavailable: {e}[/yellow]")
        return

    console.print()
    console.print(Rule("[bold cyan]Warmup Stability Analysis[/bold cyan]"))
    console.print()

    analyzer = EnterpriseWarmupAnalyzer(artifact_path)
    ws = analyzer.analyze(num_runs=100)

    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    for k, v in [
        ("Load Time",                f"{ws.load_time_ms:.3f} ms"),
        ("First Inference",          f"{ws.first_inference_ms:.3f} ms"),
        ("Stable State (p50)",       f"{ws.stable_mean_ms:.3f} ms"),
        ("Time to Stable",           f"{ws.time_to_stable_run or 'N/A'} runs"),
        ("Warmup Ratio",             f"{ws.warmup_ratio:.2f}x"),
        ("Coefficient of Variation", f"{ws.coefficient_of_variation:.3f}"),
    ]:
        t.add_row(k, v)
    if ws.throttling_detected:
        t.add_row("Thermal Throttling", "[red]DETECTED[/red]")
        t.add_row("Throttling Slope", f"{ws.throttling_slope_ms_per_run:.4f} ms/run")
    else:
        t.add_row("Thermal Throttling", "[green]Not detected[/green]")
    console.print(t)

    try:
        console.print()
        console.print(create_warmup_curve_chart(ws.latencies_ms[:50]))
    except Exception:
        pass
    console.print()


# ============================================================
# CLI
# ============================================================

@click.command()
@click.argument('build_id')
@click.option('--runs',            default=50,  type=int, help='Benchmark runs (default: 50)')
@click.option('--warmup',          default=10,  type=int, help='Warmup runs (default: 10)')
@click.option('--top',             default=15,  type=int, help='Show top N ops (default: 15)')
@click.option('--deep',            is_flag=True,          help='Deep profile (TFLite: 6 features, CoreML NeuralNetwork: 5 features)')
@click.option('--int8-build',      default=None,          help='TFLite --deep: int8 build ID for quant sensitivity')
@click.option('--analyze-warmup',  is_flag=True,          help='Warmup stability analysis')
@click.option('--cold-start',      is_flag=True,          help='Cold start decomposition')
@click.option('--memory',          is_flag=True,          help='Peak RSS memory profiling')
@click.option('--cold-start-runs', default=60,  type=int, help='Runs for cold start (default: 60)')
@click.option('--quant-samples',   default=50,  type=int, help='Samples for quant sensitivity (default: 50)')
# --- PATCH: --task flag ---
@click.option(
    '--task',
    type=click.Choice(['vision', 'nlp', 'audio', 'unknown']),
    default=None,
    help='Override task type. Falls back to registry build record if omitted.',
)
# --- PATCH: --seq-len flag (NLP representative pass, not full ladder) ---
@click.option(
    '--seq-len',
    'seq_len',
    default=128,
    type=int,
    help='NLP representative sequence length for profiling (default: 128).',
)
# --- PATCH: --strict-output flag ---
@click.option(
    '--strict-output',
    'strict_output',
    is_flag=True,
    default=False,
    help='Hard-fail on output validation warnings (overrides config.toml).',
)
def profile(
    build_id, runs, warmup, top, deep, int8_build,
    analyze_warmup, cold_start, memory, cold_start_runs, quant_samples,
    task, seq_len, strict_output,  # --- PATCH ---
):
    """
    Profile model performance. Supports CoreML and TFLite.

    TFLite deep profiling (add --deep):

      1. Per-op timing         Real hardware timing, not estimates
      2. Memory flow           Activation memory at each layer boundary
      3. Bottleneck            COMPUTE vs MEMORY bound classification
      4. Cold start            Load → first → stable decomposition
      5. Quant sensitivity     fp32 vs int8 divergence (--int8-build)
      6. Fusion detection      Fused kernels + missed fusion opportunities

    CoreML deep profiling (add --deep, NeuralNetwork format only):

      1. Per-layer timing      Incremental p50 via sliced subgraph benchmarking
      2. Memory flow           Estimated activation memory from weight dimensions
      3. Bottleneck            COMPUTE vs MEMORY bound per layer
      4. Cold start            Load → first → stable decomposition
      5. Fusion detection      Conv+Activation, BN+Scale, etc.

    Examples:

      mlbuild profile <build_id>
      mlbuild profile <build_id> --deep
      mlbuild profile <build_id> --deep --int8-build <int8_id>
      mlbuild profile <build_id> --deep --top 20
      mlbuild profile <build_id> --cold-start
      mlbuild profile <build_id> --memory
      mlbuild profile <build_id> --cold-start --memory
      mlbuild profile <build_id> --task nlp --seq-len 256
    """
    console.print(f"\n[bold]Model Profiling[/bold]")
    console.print(f"Build: [cyan]{build_id[:16]}...[/cyan]\n")

    registry = LocalRegistry()
    build = registry.resolve_build(build_id)

    if not build:
        console.print(f"[red]Build not found: {build_id}[/red]")
        raise SystemExit(1)

    artifact_path = Path(build.artifact_path)

    # --- PATCH: resolve task (flag → registry → unknown) ---
    if task:
        resolved_task = TaskType(task)
    elif getattr(build, 'task_type', None):
        resolved_task = TaskType(build.task_type)
    else:
        resolved_task = TaskType.UNKNOWN

    # --- PATCH: resolve StrictOutputConfig + validator ---
    strict_cfg = StrictOutputConfig.from_command(
        strict_flag=strict_output,
        global_strict=_read_global_strict(),
    )
    validator = TaskOutputValidator(config=strict_cfg)

    # --- PATCH: print Task line in header ---
    console.print(f"[dim]Task: {resolved_task.value}[/dim]")

    # --- PATCH: build task-aware synthetic inputs once (used by memory profiling) ---
    profile_inputs = _build_profile_inputs(artifact_path, build.format, resolved_task, seq_len)

    # ── Cold start (standard, any format) ────────────────────
    if cold_start and not deep:
        console.print(f"[cyan]Running cold start analysis ({cold_start_runs} runs)...[/cyan]")
        try:
            from ...profiling.cold_start import ColdStartProfiler
            cs_result = ColdStartProfiler(
                model_path=artifact_path, backend=build.format, num_runs=cold_start_runs,
            ).profile()
            _display_cold_start(cs_result)
        except Exception as e:
            console.print(f"[red]Cold start failed: {e}[/red]")
            import traceback; traceback.print_exc()

    # ── Memory profiling ──────────────────────────────────────
    if memory:
        console.print(f"[cyan]Running memory profiling ({runs} runs)...[/cyan]")
        try:
            from ...profiling.memory_profiler import BasicMemoryProfiler
            mem_profiler = BasicMemoryProfiler(poll_interval_ms=2.0)

            if build.format == "tflite":
                mem_profiler.start()
                from ...backends.tflite.benchmark_runner import TFLiteBenchmarkRunner
                TFLiteBenchmarkRunner().benchmark(model_path=artifact_path, runs=runs, warmup=warmup)
                mem_profiler.stop()
            else:
                # --- PATCH: use task-aware inputs instead of random zeros ---
                import coremltools as ct
                mem_profiler.start()
                model = ct.models.MLModel(str(artifact_path))
                outputs = {}
                for _ in range(runs):
                    try:
                        outputs = model.predict(profile_inputs) or {}
                    except Exception:
                        pass
                mem_profiler.stop()

                # --- PATCH: validate outputs from the last inference pass ---
                if outputs:
                    import numpy as np
                    np_outputs = {
                        k: (v if isinstance(v, np.ndarray) else np.array(v))
                        for k, v in outputs.items()
                    }
                    val_result = validator.validate(np_outputs, resolved_task)
                    warn_str = format_validation_warning(val_result)
                    if warn_str:
                        console.print(warn_str)
                    if should_exit_on_validation(val_result, strict_cfg):
                        console.print("[red]✗ Output validation failed (strict mode)[/red]")
                        sys.exit(1)

            _display_memory_report(mem_profiler.report())
        except Exception as e:
            console.print(f"[red]Memory profiling failed: {e}[/red]")
            import traceback; traceback.print_exc()

    # ── TFLite deep profiling ─────────────────────────────────
    if build.format == "tflite" and deep:
        from ...backends.tflite.deep_profiler import TFLiteDeepProfiler

        console.print(f"[bold]TFLite Deep Profile[/bold] [dim]— {artifact_path.name}[/dim]\n")

        int8_path = None
        if int8_build:
            int8_b = registry.resolve_build(int8_build)
            if int8_b:
                int8_path = Path(int8_b.artifact_path)
                console.print(f"[dim]INT8 model: {int8_b.name or int8_build[:12]}[/dim]\n")
            else:
                console.print(f"[yellow]INT8 build not found: {int8_build} — skipping quant sensitivity[/yellow]\n")

        # --- PATCH: pass task-aware inputs to deep profiler ---
        profiler = TFLiteDeepProfiler(artifact_path, inputs=profile_inputs or None)

        # 1. Per-op timing
        console.print("[cyan]① Profiling per-op timing...[/cyan]")
        op_timing, total_ms = [], 0.0
        try:
            op_timing, total_ms = profiler.profile_op_timing(runs=runs, warmup=warmup)
            _display_op_timing(op_timing, total_ms, top)
        except Exception as e:
            console.print(f"[yellow]Per-op timing: {e}[/yellow]")

        # 2. Memory flow
        console.print("[cyan]② Analyzing memory flow...[/cyan]")
        try:
            mf, peak_mb, peak_op = profiler.profile_memory_flow()
            _display_memory_flow(mf, peak_mb, peak_op, top)
        except Exception as e:
            console.print(f"[yellow]Memory flow: {e}[/yellow]")

        # 3. Bottleneck classification
        console.print("[cyan]③ Classifying bottlenecks...[/cyan]")
        try:
            bn = profiler.classify_bottlenecks(op_timing)
            cb = sum(1 for b in bn if b.classification == "COMPUTE")
            mb = sum(1 for b in bn if b.classification == "MEMORY")
            _display_bottlenecks(bn, cb, mb, top)
        except Exception as e:
            console.print(f"[yellow]Bottleneck classification: {e}[/yellow]")

        # 4. Cold start decomposition
        console.print("[cyan]④ Decomposing cold start...[/cyan]")
        try:
            cs = profiler.profile_cold_start(runs=cold_start_runs)
            _display_cold_start_decomposition(cs)
        except Exception as e:
            console.print(f"[yellow]Cold start decomposition: {e}[/yellow]")

        # 5. Quant sensitivity
        if int8_path:
            console.print(f"[cyan]⑤ Computing quant sensitivity ({quant_samples} samples)...[/cyan]")
            try:
                sens = profiler.profile_quant_sensitivity(
                    fp32_path=artifact_path, int8_path=int8_path, num_samples=quant_samples,
                )
                _display_quant_sensitivity(sens)
            except Exception as e:
                console.print(f"[yellow]Quant sensitivity: {e}[/yellow]")
        else:
            console.print("[dim]⑤ Quant sensitivity: skipped (pass --int8-build <id> to enable)[/dim]")

        # 6. Fusion detection
        console.print("[cyan]⑥ Detecting op fusion...[/cyan]")
        try:
            fg, unfused = profiler.detect_fusion()
            _display_fusion_map(fg, unfused)
        except Exception as e:
            console.print(f"[yellow]Fusion detection: {e}[/yellow]")

        # --- PATCH: validate outputs from deep profiler's inference pass ---
        try:
            raw_outputs = profiler.last_outputs  # populated by profile_op_timing if supported
            if raw_outputs:
                import numpy as np
                np_outputs = {
                    k: (v if isinstance(v, np.ndarray) else np.array(v))
                    for k, v in raw_outputs.items()
                }
                val_result = validator.validate(np_outputs, resolved_task)
                warn_str = format_validation_warning(val_result)
                if warn_str:
                    console.print(warn_str)
                if should_exit_on_validation(val_result, strict_cfg):
                    console.print("[red]✗ Output validation failed (strict mode)[/red]")
                    sys.exit(1)
        except AttributeError:
            pass  # profiler doesn't expose last_outputs yet — no-op

        console.print()
        return

    # ── TFLite standard ───────────────────────────────────────
    if build.format == "tflite":
        _profile_tflite_standard(build, runs, warmup)

        if analyze_warmup:
            console.print("[cyan]Running TFLite warmup analysis...[/cyan]")
            try:
                from ...profiling.cold_start import ColdStartProfiler
                cs_result = ColdStartProfiler(
                    model_path=artifact_path, backend="tflite", num_runs=cold_start_runs,
                ).profile()
                _display_cold_start(cs_result, title="Warmup Analysis")
            except Exception as e:
                console.print(f"[red]TFLite warmup analysis failed: {e}[/red]")
        return

    # ── CoreML ────────────────────────────────────────────────
    if build.format != "coreml":
        console.print(f"[red]Profiling not supported for format: {build.format}[/red]")
        return

    # ── CoreML deep profiling ─────────────────────────────────
    if deep:
        from ...backends.coreml.deep_profiler import CoreMLDeepProfiler

        console.print(f"[bold]CoreML Deep Profile[/bold] [dim]— {artifact_path.name}[/dim]\n")

        # --- PATCH: pass task-aware inputs to deep profiler ---
        deep_profiler = CoreMLDeepProfiler(artifact_path, inputs=profile_inputs or None)

        if deep_profiler.model_type == "mlProgram":
            _display_mlprogram_deep_note()
            console.print("[cyan]④ Decomposing cold start...[/cyan]")
            try:
                result = deep_profiler.profile(
                    num_runs=runs, warmup_runs=warmup, cold_start_runs=cold_start_runs,
                )
                _display_cold_start(result.cold_start)
            except Exception as e:
                console.print(f"[red]Cold start failed: {e}[/red]")
                import traceback; traceback.print_exc()
            return

        console.print("[cyan]Profiling NeuralNetwork model (sliced subgraph timing)...[/cyan]")
        console.print("[dim]Each layer slice is benchmarked separately — this may take a moment.[/dim]\n")

        try:
            result = deep_profiler.profile(
                num_runs=runs, warmup_runs=warmup, cold_start_runs=cold_start_runs,
            )
        except Exception as e:
            console.print(f"[red]CoreML deep profile failed: {e}[/red]")
            import traceback; traceback.print_exc()
            return

        console.print("[cyan]① Per-layer timing[/cyan]")
        try:
            _display_coreml_layer_timing(result.layer_timing, result.total_time_ms, top)
        except Exception as e:
            console.print(f"[yellow]Layer timing display: {e}[/yellow]")

        console.print("[cyan]② Memory flow[/cyan]")
        try:
            _display_coreml_memory_flow(
                result.memory_flow, result.peak_memory_mb, result.peak_memory_layer, top
            )
        except Exception as e:
            console.print(f"[yellow]Memory flow display: {e}[/yellow]")

        console.print("[cyan]③ Bottleneck classification[/cyan]")
        try:
            _display_coreml_bottlenecks(
                result.bottlenecks, result.compute_bound_layers,
                result.memory_bound_layers, top,
            )
        except Exception as e:
            console.print(f"[yellow]Bottleneck display: {e}[/yellow]")

        console.print("[cyan]④ Cold start decomposition[/cyan]")
        try:
            _display_cold_start(result.cold_start)
        except Exception as e:
            console.print(f"[yellow]Cold start display: {e}[/yellow]")

        console.print("[cyan]⑤ Fusion detection[/cyan]")
        try:
            _display_coreml_fusion(result.fusion_groups, result.unfused_opportunities)
        except Exception as e:
            console.print(f"[yellow]Fusion display: {e}[/yellow]")

        # --- PATCH: validate outputs from CoreML deep profiler ---
        try:
            raw_outputs = getattr(result, 'last_outputs', None)
            if raw_outputs:
                import numpy as np
                np_outputs = {
                    k: (v if isinstance(v, np.ndarray) else np.array(v))
                    for k, v in raw_outputs.items()
                }
                val_result = validator.validate(np_outputs, resolved_task)
                warn_str = format_validation_warning(val_result)
                if warn_str:
                    console.print(warn_str)
                if should_exit_on_validation(val_result, strict_cfg):
                    console.print("[red]✗ Output validation failed (strict mode)[/red]")
                    sys.exit(1)
        except Exception:
            pass

        console.print()
        return

    # ── CoreML standard (no --deep) ───────────────────────────
    profiler = CumulativeLayerProfiler(artifact_path)
    console.print(f"[dim]Model type: {profiler.model_type}[/dim]")
    console.print(f"[dim]Total operations: {len(profiler.layers)}[/dim]\n")

    result = profiler.profile(num_runs=runs, warmup_runs=warmup)

    if result['profiling_mode'] == 'cumulative_slicing':
        _display_neuralnetwork_profile(result, top)
    elif result['profiling_mode'] == 'full_model_only':
        _display_mlprogram_profile(result)

    if analyze_warmup:
        _display_warmup_analysis_coreml(artifact_path)