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
from types import SimpleNamespace

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

# --- ADD THESE IMPORTS AT TOP ---
from ...core.budget import load_budget, merge_constraints, constraint_origin

console = Console(width=None)


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
    except Exception as e:
        console.print(f"[yellow]Warning: invalid config.toml: {e}[/yellow]")
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
        # FIX: use SimpleNamespace instead of fake class
        return SimpleNamespace(
            latency_p50=metrics["p50_ms"],
            latency_p95=metrics["p95_ms"],
            latency_p99=metrics["p99_ms"],
            memory_peak_mb=max(metrics["memory_rss_mb"], 0.0),
            outputs={},
        )

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
@click.option('--dataset', default=None, type=click.Path(exists=True, path_type=Path), help='Dataset for accuracy check.')
@click.option('--baseline-id', default=None, help='Baseline build ID (default: root build).')
@click.option('--cosine-threshold', default=0.99, type=float, show_default=True)
@click.option('--top1-threshold', default=0.99, type=float, show_default=True)
@click.option('--accuracy-samples', default=200, type=int, show_default=True)
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
    dataset: Path = None,
    baseline_id: str = None,
    cosine_threshold: float = 0.99,
    top1_threshold: float = 0.99,
    accuracy_samples: int = 200,
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
    # FIX: integrate budget constraints
    try:
        budget = load_budget()
    except Exception as e:
        if not ci:
            console.print(f"[red]Invalid budget file:[/red] {e}")
        sys.exit(1)

    explicit = {
        "max_latency_ms": max_latency,
        "max_p95_ms":     max_p95,
        "max_memory_mb":  max_memory,
        "max_size_mb":    max_size,
    }

    merged = merge_constraints(explicit, budget)

    max_latency = merged["max_latency_ms"]
    max_p95     = merged["max_p95_ms"]
    max_memory  = merged["max_memory_mb"]
    max_size    = merged["max_size_mb"]

    if not ci and any(v is not None for v in budget.values()):
        console.print("[dim]Using budget constraints from .mlbuild/budget.toml[/dim]\n")

    if not ci:
        console.print(f"\n[bold]Constraint Validation[/bold]")
        console.print(f"Build: {build_id[:16]}...\n")

    # FIX: safe registry handling
    try:
        registry = LocalRegistry()
        build = registry.resolve_build(build_id)
    except Exception as e:
        if ci:
            print(f"STATUS: FAIL\nERROR: registry: {e}")
        else:
            console.print(f"[red]Registry error:[/red] {e}")
        sys.exit(1)

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
        # FIX: safe size handling
        if build.size_mb is None:
            if ci:
                print("STATUS: FAIL\nERROR: missing size")
            else:
                console.print("[red]Build missing size information[/red]")
            sys.exit(1)

        size_mb = float(build.size_mb)
        if size_mb > max_size:
            violations.append({
                'constraint': 'max_size_mb',
                'limit': max_size,
                'actual': size_mb,
                'unit': 'MB',
                'origin': constraint_origin("max_size_mb", explicit, budget),
            })

    # Performance constraints
    needs_benchmark = any([max_latency, max_p95, max_memory])

    if needs_benchmark:
        if not ci:
            console.print(f"[dim]Running benchmark ({build.format})...[/dim]\n")

        try:
            result = _benchmark_build(build, runs, warmup, compute_unit)

            # FIX: structured constraint checks
            checks = [
                ("max_latency_ms", result.latency_p50, max_latency, "ms"),
                ("max_p95_ms", result.latency_p95, max_p95, "ms"),
                ("max_memory_mb", result.memory_peak_mb, max_memory, "MB"),
            ]

            for key, actual, limit, unit in checks:
                if limit is None:
                    continue
                if actual > limit:
                    violations.append({
                        "constraint": key,
                        "limit": limit,
                        "actual": actual,
                        "unit": unit,
                        "origin": constraint_origin(key, explicit, budget),
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
            if ci:
                print(f"STATUS: FAIL\nERROR: benchmark: {e}")
            else:
                console.print(f"[red]Benchmark failed:[/red] {e}")
            sys.exit(1)

    # Accuracy check (only if --dataset provided)
    accuracy_result = None
    if dataset is not None:
        from ...validation.accuracy_validator import AccuracyValidator

        acc_baseline = None
        if baseline_id:
            acc_baseline = registry.resolve_build(baseline_id)
            if acc_baseline is None and not ci:
                console.print(f"[yellow]Warning: baseline not found, skipping accuracy[/yellow]")
        elif build.root_build_id and build.root_build_id != build.build_id:
            acc_baseline = registry.resolve_build(build.root_build_id)

        if acc_baseline is None:
            if not ci:
                console.print("[dim]Accuracy check skipped — build is root baseline[/dim]")
        else:
            if not ci:
                console.print(f"[dim]Running accuracy check ({accuracy_samples} samples)...[/dim]\n")
            validator = AccuracyValidator(
                build=build,
                baseline=acc_baseline,
                dataset=dataset,
                cosine_threshold=cosine_threshold,
                top1_threshold=top1_threshold,
                max_samples=accuracy_samples,
                registry=registry,
            )
            accuracy_result = validator.validate()
            if accuracy_result.skipped:
                if not ci:
                    console.print(f"[yellow]Accuracy check skipped: {accuracy_result.skip_reason}[/yellow]")
            else:
                for v in accuracy_result.violations:
                    violations.append(v.as_violation_dict())

    # Report results
    if violations:
        _report_violations(violations, ci)

        # ── Cloud sync ────────────────────────────────────────
        try:
            from ...cloud.sync import push_accuracy
            push_accuracy(
                check_type="validate",
                baseline_build_id=None,
                candidate_build_id=build.build_id,
                cosine_similarity=getattr(accuracy_result, 'cosine_similarity', None) if accuracy_result else None,
                top1_agreement=getattr(accuracy_result, 'top1_agreement', None) if accuracy_result else None,
                kl_divergence=None,
                js_divergence=None,
                rmse=None,
                mae=None,
                max_error=None,
                samples=None,
                seed=None,
                passed=False,
            )
        except Exception:
            pass

        sys.exit(1)
    else:
        if not ci and accuracy_result and not accuracy_result.skipped:
            cos = f"{accuracy_result.cosine_similarity:.4f}" if accuracy_result.cosine_similarity is not None else "—"
            top1 = f"{accuracy_result.top1_agreement:.4f}" if accuracy_result.top1_agreement is not None else "—"
            console.print(f"[dim]Accuracy: cosine={cos}  top1={top1}[/dim]")
        if ci:
            print("STATUS: PASS")
        else:
            console.print("[bold green]✓ All constraints passed[/bold green]\n")

        # ── Cloud sync ────────────────────────────────────────
        try:
            from ...cloud.sync import push_accuracy
            push_accuracy(
                check_type="validate",
                baseline_build_id=None,
                candidate_build_id=build.build_id,
                cosine_similarity=getattr(accuracy_result, 'cosine_similarity', None) if accuracy_result else None,
                top1_agreement=getattr(accuracy_result, 'top1_agreement', None) if accuracy_result else None,
                kl_divergence=None,
                js_divergence=None,
                rmse=None,
                mae=None,
                max_error=None,
                samples=None,
                seed=None,
                passed=True,
            )
        except Exception:
            pass

        sys.exit(0)


def _report_violations(violations, ci_mode):
    if ci_mode:
        print(f"STATUS: FAIL")
        print(f"VIOLATIONS: {len(violations)}")
        for v in violations:
            print(f"  {v['constraint']}: {v['actual']:.2f}{v['unit']} > {v['limit']}{v['unit']}")
        return

    console.print(Panel(
        f"[bold red]{len(violations)} Constraint Violation(s)[/bold red]",
        border_style="red"
    ))
    console.print()

    table = Table(show_header=True, header_style="bold red")
    table.add_column("Constraint", no_wrap=True)
    table.add_column("Limit", justify="right", no_wrap=True)
    table.add_column("Actual", justify="right", no_wrap=True)
    table.add_column("Violation", justify="right", no_wrap=True)
    table.add_column("Source", justify="right", no_wrap=True)

    for v in violations:
        limit_str = f"{v['limit']:.2f} {v['unit']}"
        actual_str = f"{v['actual']:.2f} {v['unit']}"
        over = v['actual'] - v['limit']
        over_pct = (over / v['limit']) * 100
        if not v['unit']:
            violation_str = f"{over:+.4f} ({abs(over_pct):.1f}% below threshold)"
        else:
            violation_str = f"+{over:.2f} ({over_pct:.1f}% over)"
        table.add_row(
            v['constraint'],
            limit_str,
            actual_str,
            violation_str,
            v.get('origin', 'unknown')
        )

    console.print(table)
    console.print()