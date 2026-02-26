"""
Compare benchmark results between two builds.
Regression detection for CI/CD.

Exit codes:
  0 = no regression
  1 = regression detected  
  2 = error
"""

import json
import sys
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ...registry import LocalRegistry
from ...benchmark.runner import detect_regression, CoreMLBenchmarkRunner, ComputeUnit
from ...core.errors import MLBuildError

import numpy as np

console = Console()


def _pct_change(baseline_val: float, candidate_val: float) -> float:
    """Compute percent change, handling zero baseline."""
    if baseline_val == 0:
        return 0.0
    return round(((candidate_val - baseline_val) / baseline_val) * 100, 2)


@click.command()
@click.argument("baseline_id")
@click.argument("candidate_id")
@click.option("--threshold", default=5.0, help="Regression threshold in percent (default: 5%)")
@click.option("--metric",
              type=click.Choice(["p50", "p95", "p99", "mean"]),
              default="p50",
              help="Latency metric to compare")
@click.option("--compute-unit",
              type=click.Choice(["CPU_ONLY", "CPU_AND_GPU", "ALL"]),
              default="ALL")
@click.option("--runs", default=100, type=int)
@click.option("--warmup", default=20, type=int)
@click.option("--json", "as_json", is_flag=True, help="Output results as JSON")
@click.option("--ci", is_flag=True, help="CI mode: strict exit codes")
def compare(
    baseline_id: str,
    candidate_id: str,
    threshold: float,
    metric: str,
    compute_unit: str,
    runs: int,
    warmup: int,
    as_json: bool,
    ci: bool,
):
    """
    Compare two builds and detect performance regressions.

    Exit codes (in --ci mode):
        0 = no regression
        1 = regression detected
        2 = error

    Examples:
        mlbuild compare mobilenet-fp32 mobilenet-fp16-v2
        mlbuild compare mobilenet-fp32 mobilenet-fp16-v2 --threshold 10
        mlbuild compare mobilenet-fp32 mobilenet-fp16-v2 --ci
        mlbuild compare mobilenet-fp32 mobilenet-fp16-v2 --json
    """
    try:
        registry = LocalRegistry()

        # Resolve builds
        baseline = registry.resolve_build(baseline_id)
        candidate = registry.resolve_build(candidate_id)

        if not baseline:
            console.print(f"[red]Baseline not found: {baseline_id}[/red]")
            sys.exit(2)

        if not candidate:
            console.print(f"[red]Candidate not found: {candidate_id}[/red]")
            sys.exit(2)

        if not as_json:
            console.print(f"\n[bold]Regression Detection[/bold]")
            console.print(f"Baseline:  {baseline.name or baseline_id} ({baseline.quantization_type})")
            console.print(f"Candidate: {candidate.name or candidate_id} ({candidate.quantization_type})")
            console.print(f"Threshold: {threshold}%")
            console.print(f"Metric:    {metric}\n")

        cu_enum = ComputeUnit[compute_unit]

        # Benchmark baseline
        if not as_json:
            console.print("[cyan]Benchmarking baseline...[/cyan]")

        baseline_runner = CoreMLBenchmarkRunner(
            model_path=baseline.artifact_path,
            compute_unit=cu_enum,
            warmup_runs=warmup,
            benchmark_runs=runs,
        )
        import coremltools as ct
        baseline_runner.model = ct.models.MLModel(
            str(baseline.artifact_path),
            compute_units=cu_enum.to_coreml(),
        )
        spec = baseline_runner.model.get_spec()
        baseline_runner.inputs = {
            i.name: tuple(i.type.multiArrayType.shape)
            for i in spec.description.input
        }
        baseline_result, baseline_latencies = baseline_runner.run(
            build_id=baseline.build_id,
            return_raw=True,
        )

        # Benchmark candidate
        if not as_json:
            console.print("[cyan]Benchmarking candidate...[/cyan]")

        candidate_runner = CoreMLBenchmarkRunner(
            model_path=candidate.artifact_path,
            compute_unit=cu_enum,
            warmup_runs=warmup,
            benchmark_runs=runs,
        )
        candidate_result, candidate_latencies = candidate_runner.run(
            build_id=candidate.build_id,
            return_raw=True,
        )

        # Detect regression using the primary metric
        regression = detect_regression(
            baseline=baseline_result,
            candidate=candidate_result,
            baseline_latencies=baseline_latencies,
            candidate_latencies=candidate_latencies,
            threshold_percent=threshold,
        )

        # Determine regression using the selected metric
        baseline_val = getattr(baseline_result, f"latency_{metric}_ms" if metric != "mean" else "latency_mean_ms", baseline_result.latency_p50)
        candidate_val = getattr(candidate_result, f"latency_{metric}_ms" if metric != "mean" else "latency_mean_ms", candidate_result.latency_p50)
        change_pct = _pct_change(baseline_val, candidate_val)
        regression_detected = change_pct > threshold

        if as_json:
            output = {
                "baseline": {
                    "build_id": baseline.build_id,
                    "name": baseline.name,
                    "quantization": baseline.quantization_type,
                    "p50_ms": float(baseline_result.latency_p50),
                    "p95_ms": float(baseline_result.latency_p95),
                    "p99_ms": float(getattr(baseline_result, "latency_p99", 0.0)),
                    "memory_mb": float(getattr(baseline_result, "memory_peak_mb", 0.0)),
                },
                "candidate": {
                    "build_id": candidate.build_id,
                    "name": candidate.name,
                    "quantization": candidate.quantization_type,
                    "p50_ms": float(candidate_result.latency_p50),
                    "p95_ms": float(candidate_result.latency_p95),
                    "p99_ms": float(getattr(candidate_result, "latency_p99", 0.0)),
                    "memory_mb": float(getattr(candidate_result, "memory_peak_mb", 0.0)),
                },
                "change": {
                    "p50": _pct_change(baseline_result.latency_p50, candidate_result.latency_p50),
                    "p95": _pct_change(baseline_result.latency_p95, candidate_result.latency_p95),
                    "p99": _pct_change(
                        getattr(baseline_result, "latency_p99", 0.0),
                        getattr(candidate_result, "latency_p99", 0.0),
                    ),
                    "memory": _pct_change(
                        getattr(baseline_result, "memory_peak_mb", 0.0),
                        getattr(candidate_result, "memory_peak_mb", 0.0),
                    ),
                },
                "regression_detected": regression_detected,
                "threshold": float(threshold),
                "metric": metric,
                "statistical": {
                    "p_value": float(regression.p_value),
                    "significant": bool(regression.p_value < 0.05),
                },
            }
            console.print(json.dumps(output, indent=2))

        else:
            # Rich table output
            table = Table(title="Benchmark Comparison", show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Baseline", justify="right")
            table.add_column("Candidate", justify="right")
            table.add_column("Change", justify="right")

            def fmt_change(pct: float) -> str:
                if pct > threshold:
                    return f"[red]{pct:+.2f}%[/red]"
                elif pct > 0:
                    return f"[yellow]{pct:+.2f}%[/yellow]"
                else:
                    return f"[green]{pct:+.2f}%[/green]"

            table.add_row(
                "p50 latency (ms)",
                f"{baseline_result.latency_p50:.2f}",
                f"{candidate_result.latency_p50:.2f}",
                fmt_change(_pct_change(baseline_result.latency_p50, candidate_result.latency_p50)),
            )
            table.add_row(
                "p95 latency (ms)",
                f"{baseline_result.latency_p95:.2f}",
                f"{candidate_result.latency_p95:.2f}",
                fmt_change(_pct_change(baseline_result.latency_p95, candidate_result.latency_p95)),
            )

            p99_base = getattr(baseline_result, "latency_p99", None)
            p99_cand = getattr(candidate_result, "latency_p99", None)
            if p99_base is not None and p99_cand is not None:
                table.add_row(
                    "p99 latency (ms)",
                    f"{p99_base:.2f}",
                    f"{p99_cand:.2f}",
                    fmt_change(_pct_change(p99_base, p99_cand)),
                )

            mem_base = getattr(baseline_result, "memory_peak_mb", None)
            mem_cand = getattr(candidate_result, "memory_peak_mb", None)
            if mem_base is not None and mem_cand is not None:
                table.add_row(
                    "Peak memory (MB)",
                    f"{mem_base:.1f}",
                    f"{mem_cand:.1f}",
                    fmt_change(_pct_change(mem_base, mem_cand)),
                )

            console.print(table)

            # Statistical analysis
            console.print(f"\n[bold]Statistical Analysis:[/bold]")
            console.print(f"  p-value:   {regression.p_value:.4f} {'[green](significant)[/green]' if regression.p_value < 0.05 else '(not significant)'}")
            console.print(f"  Change ({metric}): {change_pct:+.2f}%")
            console.print(f"  Threshold: {threshold}%")
            console.print()

            # Verdict panel
            if regression_detected:
                console.print(Panel.fit(
                    f"[bold red]⚠ REGRESSION DETECTED[/bold red]\n"
                    f"Candidate is {change_pct:+.1f}% slower than baseline on {metric}\n"
                    f"(threshold: {threshold}%, p={regression.p_value:.4f})",
                    border_style="red",
                ))
            else:
                console.print(Panel.fit(
                    f"[bold green]✓ NO REGRESSION[/bold green]\n"
                    f"Change: {change_pct:+.1f}% on {metric} (within {threshold}% threshold)",
                    border_style="green",
                ))

        if ci:
            sys.exit(1 if regression_detected else 0)

    except MLBuildError as e:
        console.print(f"\n[red]Error:[/red] {e}\n")
        sys.exit(2)

    except Exception as e:
        console.print(f"\n[red]Unexpected error:[/red] {e}\n")
        sys.exit(2)