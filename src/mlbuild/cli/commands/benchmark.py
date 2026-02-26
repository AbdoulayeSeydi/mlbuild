"""
Benchmark command: Profile model performance on-device.
"""

import click
from rich.console import Console
from rich.table import Table
from pathlib import Path
import json

from ...registry import LocalRegistry
from ...benchmark.runner import (
    CoreMLBenchmarkRunner,
    ComputeUnit,
)
from ...core.errors import MLBuildError

console = Console()


@click.command()
@click.argument("build_id")
@click.option("--runs", default=100, help="Number of benchmark runs")
@click.option("--warmup", default=20, help="Number of warmup runs")
@click.option("--compute-unit", 
              type=click.Choice(["CPU_ONLY", "CPU_AND_GPU", "ALL"]),
              default="ALL",
              help="Compute unit to use")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def benchmark(build_id: str, runs: int, warmup: int, compute_unit: str, as_json: bool):
    """
    Benchmark a build on the current device.
    
    Examples:
        mlbuild benchmark <build_id>
        mlbuild benchmark <build_id> --runs 1000
        mlbuild benchmark <build_id> --compute-unit CPU_ONLY
    """
    try:
        # Resolve build
        registry = LocalRegistry()
        build = registry.resolve_build(build_id)
        
        if not build:
            console.print(f"[red]Build not found: {build_id}[/red]")
            raise click.Abort()
        
        # Check artifact exists
        artifact_path = Path(build.artifact_path)
        if not artifact_path.exists():
            console.print(f"[red]Artifact not found: {artifact_path}[/red]")
            raise click.Abort()
        
        # Display what we're benchmarking
        if not as_json:
            console.print(f"\n[bold]Benchmarking:[/bold] {build.name or build.build_id[:16]}")
            console.print(f"Target: {build.target_device}")
            console.print(f"Quantization: {build.quantization_type}")
            console.print(f"Compute Unit: {compute_unit}\n")
        
        # Run benchmark
        cu_enum = ComputeUnit[compute_unit]
        
        runner = CoreMLBenchmarkRunner(
            model_path=artifact_path,
            compute_unit=cu_enum,
            warmup_runs=warmup,
            benchmark_runs=runs,
            ci_mode=False,
        )
        
        result, raw_latencies = runner.run(
            build_id=build.build_id,
            return_raw=True,
        )
        
        # Save to registry
        from ...core.types import Benchmark
        from datetime import datetime, timezone
        
        bench = Benchmark(
            build_id=build.build_id,
            device_chip=result.chip,
            runtime="coreml",
            measurement_type="latency",
            compute_unit=result.compute_unit,
            latency_p50_ms=result.latency_p50,
            latency_p95_ms=result.latency_p95,
            latency_p99_ms=result.latency_p99,
            memory_peak_mb=result.memory_peak_mb,
            num_runs=result.num_runs,
            measured_at=datetime.now(timezone.utc),
        )
        
        registry.save_benchmark(bench)
        
        # Display results
        if as_json:
            data = {
                "build_id": result.build_id,
                "device_chip": result.chip,
                "runtime": "coreml",
                "compute_unit": result.compute_unit,
                "latency_p50_ms": result.latency_p50,
                "latency_p95_ms": result.latency_p95,
                "latency_p99_ms": result.latency_p99,
                "latency_mean_ms": result.latency_mean,
                "latency_std_ms": result.latency_std,
                "p50_ci_low": result.p50_ci_low,
                "p50_ci_high": result.p50_ci_high,
                "autocorr_lag1": result.autocorr_lag1,
                "thermal_drift_ratio": result.thermal_drift_ratio,
                "memory_peak_mb": result.memory_peak_mb,
                "num_runs": result.num_runs,
                "failures": result.failures,
            }
            console.print(json.dumps(data, indent=2))
        else:
            # Rich table
            table = Table(title="Benchmark Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right", style="green")
            
            table.add_row("Device", result.chip)
            table.add_row("Runs", f"{result.num_runs}")
            table.add_row("Failures", f"{result.failures}")
            table.add_row("", "")
            table.add_row("Latency (p50)", f"{result.latency_p50:.3f} ms")
            table.add_row("Latency (p95)", f"{result.latency_p95:.3f} ms")
            table.add_row("Latency (p99)", f"{result.latency_p99:.3f} ms")
            table.add_row("Latency (mean)", f"{result.latency_mean:.3f} ms")
            table.add_row("Latency (std)", f"{result.latency_std:.3f} ms")
            table.add_row("", "")
            table.add_row("95% CI (p50)", f"[{result.p50_ci_low:.3f}, {result.p50_ci_high:.3f}]")
            table.add_row("", "")
            table.add_row("Autocorrelation", f"{result.autocorr_lag1:.3f}")
            table.add_row("Thermal drift", f"{result.thermal_drift_ratio:.3f}")
            table.add_row("", "")
            table.add_row("Memory (peak)", f"{result.memory_peak_mb:.2f} MB")
            
            console.print()
            console.print(table)
            console.print()
            console.print("[green]✓ Benchmark saved to registry[/green]\n")
    
    except MLBuildError as e:
        console.print(f"\n[red]Error:[/red] {e}\n")
        raise click.Abort()