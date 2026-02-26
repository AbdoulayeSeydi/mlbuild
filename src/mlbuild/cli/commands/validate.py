"""
Validate command: CI/CD constraint enforcement.

Validates builds against performance constraints.
Exit code 0 = pass, 1 = fail (for CI integration).
"""

import sys
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from pathlib import Path

from ...registry import LocalRegistry
from ...benchmark.runner import CoreMLBenchmarkRunner, ComputeUnit

console = Console()


@click.command()
@click.argument('build_id')
@click.option('--max-latency', type=float, help='Maximum p50 latency in ms')
@click.option('--max-p95', type=float, help='Maximum p95 latency in ms')
@click.option('--max-memory', type=float, help='Maximum memory usage in MB')
@click.option('--max-size', type=float, help='Maximum model size in MB')
@click.option('--runs', default=50, type=int, help='Benchmark runs')
@click.option('--warmup', default=10, type=int, help='Warmup runs')
@click.option('--compute-unit', default='all', type=click.Choice(['all', 'cpu', 'gpu']))
@click.option('--ci', is_flag=True, help='CI mode (suppress output, exit codes only)')
def validate(
    build_id: str,
    max_latency: float,
    max_p95: float,
    max_memory: float,
    max_size: float,
    runs: int,
    warmup: int,
    compute_unit: str,
    ci: bool,
):
    """
    Validate build against performance constraints.
    
    Exit codes:
        0 = All constraints passed
        1 = One or more constraints failed
    
    Examples:
        mlbuild validate <build> --max-latency 10 --max-memory 500
        mlbuild validate <build> --max-latency 10 --ci
    """
    if not ci:
        console.print(f"\n[bold]Constraint Validation[/bold]")
        console.print(f"Build: {build_id[:16]}...\n")
    
    # Get build from registry
    registry = LocalRegistry()
    build = registry.resolve_build(build_id)
    
    if not build:
        if not ci:
            console.print(f"[red]Build not found: {build_id}[/red]")
        sys.exit(1)
    
    violations = []
    
    # Size constraint (no benchmark needed)
    if max_size is not None:
        size_mb = float(build.size_mb)
        if size_mb > max_size:
            violations.append({
                'constraint': 'size',
                'limit': max_size,
                'actual': size_mb,
                'unit': 'MB',
            })
    
    # Performance constraints (need benchmark)
    needs_benchmark = any([max_latency, max_p95, max_memory])
    
    if needs_benchmark:
        if build.format != "coreml":
            if not ci:
                console.print(f"[yellow]Benchmarking only works for CoreML builds[/yellow]")
            # Can still check size
            if violations:
                _report_violations(violations, ci)
                sys.exit(1)
            sys.exit(0)
        
        if not ci:
            console.print(f"[dim]Running benchmark...[/dim]\n")
        
        # Map compute unit
        cu_map = {
            'all': ComputeUnit.ALL,
            'cpu': ComputeUnit.CPU_ONLY,
            'gpu': ComputeUnit.CPU_AND_GPU,
        }
        
        runner = CoreMLBenchmarkRunner(
            model_path=Path(build.artifact_path),
            compute_unit=cu_map[compute_unit],
            warmup_runs=warmup,
            benchmark_runs=runs,
        )
        
        result, _ = runner.run(build_id=build.build_id, return_raw=True)
        
        # Check latency constraints
        if max_latency is not None and result.latency_p50 > max_latency:
            violations.append({
                'constraint': 'latency_p50',
                'limit': max_latency,
                'actual': result.latency_p50,
                'unit': 'ms',
            })
        
        if max_p95 is not None and result.latency_p95 > max_p95:
            violations.append({
                'constraint': 'latency_p95',
                'limit': max_p95,
                'actual': result.latency_p95,
                'unit': 'ms',
            })
        
        if max_memory is not None and result.memory_peak_mb > max_memory:
            violations.append({
                'constraint': 'memory',
                'limit': max_memory,
                'actual': result.memory_peak_mb,
                'unit': 'MB',
            })
    
    # Report results
    if violations:
        _report_violations(violations, ci)
        sys.exit(1)
    else:
        if not ci:
            console.print("[bold green]✓ All constraints passed[/bold green]\n")
        sys.exit(0)


def _report_violations(violations, ci_mode):
    """Report constraint violations."""
    if ci_mode:
        # CI mode: minimal output
        print(f"FAILED: {len(violations)} constraint(s) violated")
        for v in violations:
            print(f"  {v['constraint']}: {v['actual']:.2f}{v['unit']} > {v['limit']}{v['unit']}")
        return
    
    # Interactive mode: rich output
    console.print(Panel(
        f"[bold red]{len(violations)} Constraint Violation(s)[/bold red]",
        border_style="red"
    ))
    console.print()
    
    table = Table(show_header=True, header_style="bold red")
    table.add_column("Constraint")
    table.add_column("Limit", justify="right")
    table.add_column("Actual", justify="right")
    table.add_column("Violation", justify="right")
    
    for v in violations:
        limit_str = f"{v['limit']:.2f} {v['unit']}"
        actual_str = f"{v['actual']:.2f} {v['unit']}"
        over = v['actual'] - v['limit']
        over_pct = (over / v['limit']) * 100
        violation_str = f"+{over:.2f} ({over_pct:.1f}% over)"
        
        table.add_row(
            v['constraint'],
            limit_str,
            actual_str,
            violation_str,
        )
    
    console.print(table)
    console.print()