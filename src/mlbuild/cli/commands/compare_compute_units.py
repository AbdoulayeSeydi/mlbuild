"""
Compare same model across different compute units (CoreML) or thread counts (TFLite).
"""

import click
from rich.console import Console
from rich.table import Table
from pathlib import Path

from ...benchmark.runner import CoreMLBenchmarkRunner, ComputeUnit
from ...registry import LocalRegistry

console = Console()


@click.command()
@click.argument('build_id')
@click.option('--runs', default=50, type=int, help='Benchmark runs per compute unit')
@click.option('--warmup', default=10, type=int, help='Warmup runs')
def compare_compute_units(build_id: str, runs: int, warmup: int):
    """
    Compare model across compute units (CoreML) or thread counts (TFLite).

    Examples:
        mlbuild compare-compute-units febc5f7d
        mlbuild compare-compute-units febc5f7d --runs 100
    """
    console.print(f"\n[bold]Comparing Compute Units[/bold]")
    console.print(f"Build: {build_id[:16]}...")
    console.print(f"Runs per unit: {runs}\n")

    registry = LocalRegistry()
    builds = registry.list_builds(limit=1000)
    matching_builds = [b for b in builds if b.build_id.startswith(build_id)]

    if not matching_builds:
        console.print(f"[red]Build not found: {build_id}[/red]")
        return

    if len(matching_builds) > 1:
        console.print(f"[yellow]Multiple builds match '{build_id}':[/yellow]")
        for b in matching_builds[:5]:
            console.print(f"  {b.build_id[:16]}... - {b.name or '(unnamed)'}")
        return

    build = matching_builds[0]
    model_path = Path(build.artifact_path)

    # -------------------------------------------------------
    # TFLite: compare thread counts instead of compute units
    # -------------------------------------------------------
    if build.format == "tflite":
        console.print("[dim]TFLite detected — comparing thread counts (1, 2, 4)[/dim]\n")

        from ...backends.tflite.benchmark_runner import TFLiteBenchmarkRunner

        results = {}
        for num_threads in [1, 2, 4]:
            label = f"{num_threads} thread{'s' if num_threads > 1 else ''}"
            console.print(f"⏱  Benchmarking {label}...")
            try:
                runner = TFLiteBenchmarkRunner(num_threads=num_threads)
                metrics = runner.benchmark(
                    model_path=model_path,
                    runs=runs,
                    warmup=warmup,
                )
                results[label] = metrics
            except Exception as e:
                console.print(f"[red]Failed ({label}): {e}[/red]")

        if not results:
            console.print("[red]No successful benchmarks[/red]")
            return

        console.print("\n[bold]Thread Count Comparison[/bold]\n")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Threads", style="dim")
        table.add_column("p50 (ms)", justify="right")
        table.add_column("p95 (ms)", justify="right")
        table.add_column("p99 (ms)", justify="right")
        table.add_column("Throughput (FPS)", justify="right")
        table.add_column("Memory (MB)", justify="right")

        for label, m in results.items():
            throughput = 1000.0 / m["p50_ms"] if m["p50_ms"] > 0 else 0
            table.add_row(
                label,
                f"{m['p50_ms']:.3f}",
                f"{m['p95_ms']:.3f}",
                f"{m['p99_ms']:.3f}",
                f"{throughput:.1f}",
                f"{m['memory_rss_mb']:.1f}",
            )

        console.print(table)

        fastest_label = min(results, key=lambda k: results[k]["p50_ms"])
        slowest_label = max(results, key=lambda k: results[k]["p50_ms"])
        speedup = results[slowest_label]["p50_ms"] / results[fastest_label]["p50_ms"]

        console.print(f"\n[bold green]Fastest:[/bold green] {fastest_label} ({results[fastest_label]['p50_ms']:.3f} ms)")
        console.print(f"[bold red]Slowest:[/bold red] {slowest_label} ({results[slowest_label]['p50_ms']:.3f} ms)")
        console.print(f"[bold]Speedup:[/bold] {speedup:.2f}x\n")
        return

    # -------------------------------------------------------
    # CoreML: compare CPU / GPU / Neural Engine
    # -------------------------------------------------------
    results = {}
    for cu in [ComputeUnit.CPU_ONLY, ComputeUnit.CPU_AND_GPU, ComputeUnit.ALL]:
        console.print(f"⏱  Benchmarking {cu.value}...")
        try:
            runner = CoreMLBenchmarkRunner(
                model_path=model_path,
                compute_unit=cu,
                warmup_runs=warmup,
                benchmark_runs=runs,
            )
            result, _ = runner.run(build_id=build_id, return_raw=True)
            results[cu.value] = result
        except Exception as e:
            console.print(f"[red]Failed ({cu.value}): {e}[/red]")

    if not results:
        console.print("[red]No successful benchmarks[/red]")
        return

    console.print("\n[bold]Compute Unit Comparison[/bold]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Compute Unit", style="dim")
    table.add_column("p50 (ms)", justify="right")
    table.add_column("p95 (ms)", justify="right")
    table.add_column("p99 (ms)", justify="right")
    table.add_column("Throughput (FPS)", justify="right")
    table.add_column("Memory (MB)", justify="right")

    for cu_name, result in results.items():
        throughput = 1000.0 / result.latency_p50 if result.latency_p50 > 0 else 0
        table.add_row(
            cu_name,
            f"{result.latency_p50:.3f}",
            f"{result.latency_p95:.3f}",
            f"{result.latency_p99:.3f}",
            f"{throughput:.1f}",
            f"{result.memory_peak_mb:.1f}",
        )

    console.print(table)

    fastest = min(results.items(), key=lambda x: x[1].latency_p50)
    slowest = max(results.items(), key=lambda x: x[1].latency_p50)
    speedup = slowest[1].latency_p50 / fastest[1].latency_p50

    console.print(f"\n[bold green]Fastest:[/bold green] {fastest[0]} ({fastest[1].latency_p50:.3f} ms)")
    console.print(f"[bold red]Slowest:[/bold red] {slowest[0]} ({slowest[1].latency_p50:.3f} ms)")
    console.print(f"[bold]Speedup:[/bold] {speedup:.2f}x\n")