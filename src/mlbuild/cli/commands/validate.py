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

# --- PATCH: task-aware imports ---
from ...core.task_detection import TaskType
from ...core.task_validation import (
    TaskOutputValidator,
    StrictOutputConfig,
    format_validation_warning,
    should_exit_on_validation,
)

console = Console()


# --- PATCH: shared helper ---
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


def _benchmark_build(build, runs, warmup, compute_unit):
    """Route to correct benchmark runner based on build format."""
    if build.format == "tflite":
        from ...backends.tflite.benchmark_runner import TFLiteBenchmarkRunner

        runner = TFLiteBenchmarkRunner()
        metrics = runner.benchmark(
            model_path=Path(build.artifact_path),
            runs=runs,
            warmup=warmup,
        )
        # Return a simple namespace matching the fields we need
        class _Result:
            latency_p50 = metrics["p50_ms"]
            latency_p95 = metrics["p95_ms"]
            latency_p99 = metrics["p99_ms"]
            memory_peak_mb = max(metrics["memory_rss_mb"], 0.0)
            outputs = {}  # TFLite runner doesn't expose outputs here
        return _Result()

    else:
        cu_map = {
            "all": ComputeUnit.ALL,
            "cpu": ComputeUnit.CPU_ONLY,
            "gpu": ComputeUnit.CPU_AND_GPU,
        }
        runner = CoreMLBenchmarkRunner(
            model_path=Path(build.artifact_path),
            compute_unit=cu_map[compute_unit],
            warmup_runs=warmup,
            benchmark_runs=runs,
        )
        result, _ = runner.run(build_id=build.build_id, return_raw=True)
        result.outputs = getattr(result, 'outputs', {})
        return result


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
# --- PATCH: --task flag ---
@click.option(
    '--task',
    type=click.Choice(['vision', 'nlp', 'audio', 'unknown']),
    default=None,
    help='Override task type. Falls back to registry build record if omitted.',
)
# --- PATCH: --strict-output flag ---
@click.option(
    '--strict-output',
    'strict_output',
    is_flag=True,
    default=False,
    help='Hard-fail on output validation warnings (overrides config.toml).',
)
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
    task: str,
    strict_output: bool,
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

    registry = LocalRegistry()
    build = registry.resolve_build(build_id)

    if not build:
        if not ci:
            console.print(f"[red]Build not found: {build_id}[/red]")
        sys.exit(1)

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
    if not ci:
        console.print(f"[dim]Task: {resolved_task.value}[/dim]\n")

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

    # Performance constraints
    needs_benchmark = any([max_latency, max_p95, max_memory])

    if needs_benchmark:
        if not ci:
            console.print(f"[dim]Running benchmark ({build.format})...[/dim]\n")

        try:
            result = _benchmark_build(build, runs, warmup, compute_unit)

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

            # --- PATCH: output validation from the benchmark inference pass ---
            raw_outputs = getattr(result, 'outputs', {})
            if raw_outputs:
                import numpy as np
                np_outputs = {
                    k: (v if isinstance(v, np.ndarray) else np.array(v))
                    for k, v in raw_outputs.items()
                }
                val_result = validator.validate(np_outputs, resolved_task)
                warn_str = format_validation_warning(val_result)
                if warn_str:
                    if ci:
                        print(warn_str)
                    else:
                        console.print(warn_str)
                if should_exit_on_validation(val_result, strict_cfg):
                    if ci:
                        print("FAILED: output validation (strict mode)")
                    else:
                        console.print("[red]✗ Output validation failed (strict mode)[/red]")
                    sys.exit(1)

        except Exception as e:
            if not ci:
                console.print(f"[red]Benchmark failed: {e}[/red]")
            sys.exit(1)

    # Report results
    if violations:
        _report_violations(violations, ci)
        sys.exit(1)
    else:
        if ci:
            print("✓ All constraints passed")
        else:
            console.print("[bold green]✓ All constraints passed[/bold green]\n")
        sys.exit(0)


def _report_violations(violations, ci_mode):
    if ci_mode:
        print(f"FAILED: {len(violations)} constraint(s) violated")
        for v in violations:
            print(f"  {v['constraint']}: {v['actual']:.2f}{v['unit']} > {v['limit']}{v['unit']}")
        return

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
        table.add_row(v['constraint'], limit_str, actual_str, violation_str)

    console.print(table)
    console.print()