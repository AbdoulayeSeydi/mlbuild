"""
Profile command: Production-grade layer profiling.
"""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from pathlib import Path

from ...registry import LocalRegistry
from ...profiling import CumulativeLayerProfiler

console = Console()


@click.command()
@click.argument('build_id')
@click.option('--runs', default=50, type=int, help='Profiling runs')
@click.option('--warmup', default=10, type=int, help='Warmup runs')
@click.option('--top', default=15, type=int, help='Show top N slowest layers')
@click.option('--analyze-warmup', is_flag=True, help='Analyze warmup stability')
def profile(build_id: str, runs: int, warmup: int, top: int, analyze_warmup: bool):
    """
    Profile CoreML model.
    
    - NeuralNetwork: Per-layer cumulative timing
    - MLProgram: Full-model timing + operation breakdown
    
    Examples:
        mlbuild profile febc5f7d
    """
    console.print(f"\n[bold]Model Profiling[/bold]")
    console.print(f"Build: {build_id[:16]}...\n")
    
    # Get build from registry
    registry = LocalRegistry()
    builds = [b for b in registry.list_builds(limit=1000) if b.build_id.startswith(build_id)]
    
    if not builds:
        console.print(f"[red]Build not found: {build_id}[/red]")
        return
    
    build = builds[0]
    
    if build.format != "coreml":
        console.print(f"[red]Profiling only works for CoreML builds[/red]")
        return
    
    # Profile the model
    profiler = CumulativeLayerProfiler(Path(build.artifact_path))
    
    console.print(f"[dim]Model type: {profiler.model_type}[/dim]")
    console.print(f"[dim]Total operations: {len(profiler.layers)}[/dim]\n")
    
    result = profiler.profile(num_runs=runs, warmup_runs=warmup)
    
    # Display based on profiling mode
    if result['profiling_mode'] == 'cumulative_slicing':
        _display_neuralnetwork_profile(result, top)
    elif result['profiling_mode'] == 'full_model_only':
        _display_mlprogram_profile(result)

    # Warmup analysis
    if analyze_warmup:
        console.print(f"\n[bold]Warmup Stability Analysis[/bold]\n")
        
        from ...profiling.warmup_analyzer import EnterpriseWarmupAnalyzer
        
        analyzer = EnterpriseWarmupAnalyzer(Path(build.artifact_path))
        warmup_stats = analyzer.analyze(num_runs=100)
        
        warmup_table = Table(show_header=True, header_style="bold cyan")
        warmup_table.add_column("Metric")
        warmup_table.add_column("Value", justify="right")
        
        warmup_table.add_row("Load Time", f"{warmup_stats.load_time_ms:.3f} ms")
        warmup_table.add_row("First Inference", f"{warmup_stats.first_inference_ms:.3f} ms")
        warmup_table.add_row("Stable State (p50)", f"{warmup_stats.stable_mean_ms:.3f} ms")
        warmup_table.add_row("Time to Stable", f"{warmup_stats.time_to_stable_run or 'N/A'} runs")
        warmup_table.add_row("Warmup Ratio", f"{warmup_stats.warmup_ratio:.2f}x")
        warmup_table.add_row("Coefficient of Variation", f"{warmup_stats.coefficient_of_variation:.3f}")
        
        if warmup_stats.throttling_detected:
            warmup_table.add_row("⚠️ Thermal Throttling", "DETECTED", style="red bold")
            warmup_table.add_row("Throttling Slope", f"{warmup_stats.throttling_slope_ms_per_run:.4f} ms/run")
        else:
            warmup_table.add_row("Thermal Throttling", "Not detected", style="green")
        
        console.print(warmup_table)
        # Warmup curve visualization
        from ...visualization.charts import create_warmup_curve_chart
        
        warmup_curve = create_warmup_curve_chart(warmup_stats.latencies_ms[:50])  # First 50 runs
        console.print()
        console.print(warmup_curve)
        console.print()


def _display_neuralnetwork_profile(result, top):
    """Display NeuralNetwork cumulative profiling results."""
    profiles = result['layers']
    
    console.print(f"[bold]Top {top} Slowest Layers (Incremental Time)[/bold]\n")
    
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Layer", style="dim")
    table.add_column("Type")
    table.add_column("Inc Time (ms)", justify="right")
    table.add_column("Cum Time (ms)", justify="right")
    table.add_column("Param MB", justify="right")
    
    # Sort by incremental time
    sorted_profiles = sorted(
        profiles,
        key=lambda p: p['incremental_time']['p50_ms'],
        reverse=True
    )
    
    for profile in sorted_profiles[:top]:
        inc_time = profile['incremental_time']['p50_ms']
        cum_time = profile['cumulative_time']['p50_ms']
        
        table.add_row(
            profile['name'][:30],
            profile['type'][:20],
            f"{inc_time:.3f}",
            f"{cum_time:.3f}",
            f"{profile['param_memory_mb']:.2f}",
        )
    
    console.print(table)
    
    # Total
    total_time = profiles[-1]['cumulative_time']['p50_ms'] if profiles else 0
    console.print(f"\n[bold]Total Model Time:[/bold] {total_time:.3f} ms\n")


def _display_mlprogram_profile(result):
    """Display MLProgram full-model profiling results."""
    
    # Warning
    console.print(Panel(
        result['warning'],
        title="[yellow]MLProgram Profiling Limitation[/yellow]",
        border_style="yellow"
    ))
    console.print()
    
    # Full model timing
    timing = result['full_model_timing']
    
    console.print("[bold]Full Model Timing[/bold]\n")
    
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    
    table.add_row("p50", f"{timing['p50_ms']:.3f} ms")
    table.add_row("p95", f"{timing['p95_ms']:.3f} ms")
    table.add_row("p99", f"{timing['p99_ms']:.3f} ms")
    table.add_row("mean", f"{timing['mean_ms']:.3f} ms")
    table.add_row("std", f"{timing['std_ms']:.3f} ms")
    
    console.print(table)
    
    # Operation breakdown
    breakdown = result['operation_breakdown']
    
    console.print(f"\n[bold]Operation Breakdown[/bold]")
    console.print(f"Total Operations: {breakdown['total_operations']}\n")
    
    # Top operation types
    op_table = Table(show_header=True, header_style="bold cyan")
    op_table.add_column("Operation Type")
    op_table.add_column("Count", justify="right")
    op_table.add_column("% of Total", justify="right")
    
    sorted_ops = sorted(
        breakdown['operation_counts'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    total = breakdown['total_operations']
    
    for op_type, count in sorted_ops[:15]:
        percent = (count / total * 100) if total > 0 else 0
        op_table.add_row(op_type, str(count), f"{percent:.1f}%")
    
    console.print(op_table)
    
    # Compute-heavy operations
    if breakdown['compute_heavy_operations']:
        console.print(f"\n[bold yellow]Compute-Heavy Operations[/bold yellow]")
        for op, count in breakdown['compute_heavy_operations'].items():
            console.print(f"  • {op}: {count} ops")
    
    console.print("\n[dim]For detailed per-layer profiling, use Xcode Instruments[/dim]\n")