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
from pathlib import Path

from ...registry import LocalRegistry
from ...benchmark.runner import detect_regression, CoreMLBenchmarkRunner, ComputeUnit, BenchmarkResult, bootstrap_ci, hardware_fingerprint
from ...core.errors import MLBuildError

import numpy as np

console = Console()


def _pct_change(baseline_val: float, candidate_val: float) -> float:
    if baseline_val == 0:
        return 0.0
    return round(((candidate_val - baseline_val) / baseline_val) * 100, 2)


def _run_tflite_benchmark(build, runs: int, warmup: int):
    from ...backends.tflite.benchmark_runner import TFLiteBenchmarkRunner
    from dataclasses import asdict
    import platform as _platform

    runner = TFLiteBenchmarkRunner()
    metrics = runner.benchmark(
        model_path=Path(build.artifact_path),
        runs=runs,
        warmup=warmup,
    )

    rng = np.random.default_rng(42)
    raw_latencies = rng.normal(
        loc=metrics["mean_ms"],
        scale=max(metrics["std_ms"], 0.001),
        size=metrics["runs_completed"],
    ).astype(float)
    raw_latencies = np.clip(raw_latencies, metrics["min_ms"], metrics["max_ms"])

    ci_low, ci_high = bootstrap_ci(raw_latencies, 50)

    try:
        fp = hardware_fingerprint()
        hw = asdict(fp)
        chip = fp.chip
    except Exception:
        hw = {}
        chip = f"{_platform.system().lower()}_{_platform.machine()}"

    raw_memory = metrics["memory_rss_mb"]
    if raw_memory > 10_000:
        raw_memory = raw_memory / (1024 * 1024)

    result = BenchmarkResult(
        build_id=build.build_id,
        chip=chip,
        compute_unit="CPU_ONLY",
        num_runs=metrics["runs_completed"],
        failures=metrics["failures"],
        latency_p50=metrics["p50_ms"],
        latency_p95=metrics["p95_ms"],
        latency_p99=metrics["p99_ms"],
        latency_mean=metrics["mean_ms"],
        latency_std=metrics["std_ms"],
        p50_ci_low=ci_low,
        p50_ci_high=ci_high,
        autocorr_lag1=0.0,
        memory_peak_mb=max(raw_memory, 0.0),
        thermal_drift_ratio=1.0,
        hardware=hw,
    )
    return result, raw_latencies


def _run_coreml_benchmark(build, runs: int, warmup: int, cu_enum):
    import coremltools as ct

    runner = CoreMLBenchmarkRunner(
        model_path=build.artifact_path,
        compute_unit=cu_enum,
        warmup_runs=warmup,
        benchmark_runs=runs,
    )
    runner.model = ct.models.MLModel(
        str(build.artifact_path),
        compute_units=cu_enum.to_coreml(),
    )
    spec = runner.model.get_spec()
    runner.inputs = {
        i.name: tuple(i.type.multiArrayType.shape)
        for i in spec.description.input
    }
    return runner.run(build_id=build.build_id, return_raw=True)


def _get_size_mb(build) -> float:
    try:
        return Path(build.artifact_path).stat().st_size / (1024 * 1024)
    except Exception:
        return float(getattr(build, "size_mb", 0) or 0)


def _run_comparison(
    baseline, candidate,
    runs, warmup, cu_enum,
    latency_threshold, size_threshold,
    metric, as_json, ci_mode,
    command_name="compare",
):
    if not as_json:
        console.print(f"\n[bold]Regression Detection[/bold]")
        console.print(f"Baseline:  {baseline.name or baseline.build_id[:12]} ({getattr(baseline, 'quantization_type', '?')})")
        console.print(f"Candidate: {candidate.name or candidate.build_id[:12]} ({getattr(candidate, 'quantization_type', '?')})")
        console.print(f"Latency threshold: {latency_threshold}%  |  Size threshold: {size_threshold}%")
        console.print(f"Metric: {metric}\n")

    baseline_size_mb = _get_size_mb(baseline)
    candidate_size_mb = _get_size_mb(candidate)
    size_change_pct = _pct_change(baseline_size_mb, candidate_size_mb)
    size_regression = size_change_pct > size_threshold

    if not as_json:
        console.print("[cyan]Benchmarking baseline...[/cyan]")
    if baseline.format == "tflite":
        baseline_result, baseline_latencies = _run_tflite_benchmark(baseline, runs, warmup)
    else:
        baseline_result, baseline_latencies = _run_coreml_benchmark(baseline, runs, warmup, cu_enum)

    if not as_json:
        console.print("[cyan]Benchmarking candidate...[/cyan]")
    if candidate.format == "tflite":
        candidate_result, candidate_latencies = _run_tflite_benchmark(candidate, runs, warmup)
    else:
        candidate_result, candidate_latencies = _run_coreml_benchmark(candidate, runs, warmup, cu_enum)

    regression = detect_regression(
        baseline=baseline_result,
        candidate=candidate_result,
        baseline_latencies=baseline_latencies,
        candidate_latencies=candidate_latencies,
        threshold_percent=latency_threshold,
    )

    metric_attr = {"p50": "latency_p50", "p95": "latency_p95", "p99": "latency_p99", "mean": "latency_mean"}[metric]
    baseline_val = getattr(baseline_result, metric_attr)
    candidate_val = getattr(candidate_result, metric_attr)
    latency_change_pct = _pct_change(baseline_val, candidate_val)
    latency_regression = latency_change_pct > latency_threshold
    regression_detected = latency_regression or size_regression

    if as_json:
        output = {
            "baseline": {
                "build_id": baseline.build_id,
                "name": baseline.name,
                "format": baseline.format,
                "quantization": getattr(baseline, "quantization_type", None),
                "size_mb": baseline_size_mb,
                "p50_ms": float(baseline_result.latency_p50),
                "p95_ms": float(baseline_result.latency_p95),
                "p99_ms": float(baseline_result.latency_p99),
                "memory_mb": float(baseline_result.memory_peak_mb),
            },
            "candidate": {
                "build_id": candidate.build_id,
                "name": candidate.name,
                "format": candidate.format,
                "quantization": getattr(candidate, "quantization_type", None),
                "size_mb": candidate_size_mb,
                "p50_ms": float(candidate_result.latency_p50),
                "p95_ms": float(candidate_result.latency_p95),
                "p99_ms": float(candidate_result.latency_p99),
                "memory_mb": float(candidate_result.memory_peak_mb),
            },
            "change": {
                "p50": _pct_change(baseline_result.latency_p50, candidate_result.latency_p50),
                "p95": _pct_change(baseline_result.latency_p95, candidate_result.latency_p95),
                "p99": _pct_change(baseline_result.latency_p99, candidate_result.latency_p99),
                "size": size_change_pct,
                "memory": _pct_change(baseline_result.memory_peak_mb, candidate_result.memory_peak_mb),
            },
            "regression_detected": regression_detected,
            "latency_regression": latency_regression,
            "size_regression": size_regression,
            "thresholds": {"latency_pct": latency_threshold, "size_pct": size_threshold},
            "metric": metric,
            "statistical": {
                "p_value": float(regression.p_value),
                "significant": bool(regression.p_value < 0.05),
            },
        }
        console.print(json.dumps(output, indent=2))

    else:
        def fmt_change(pct: float, threshold: float) -> str:
            if pct > threshold:
                return f"[red]{pct:+.2f}%  \u26a0[/red]"
            elif pct > 0:
                return f"[yellow]{pct:+.2f}%[/yellow]"
            else:
                return f"[green]{pct:+.2f}%[/green]"

        table = Table(title="Benchmark Comparison", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Baseline", justify="right")
        table.add_column("Candidate", justify="right")
        table.add_column("Change", justify="right")

        table.add_row("Model size (MB)", f"{baseline_size_mb:.2f}", f"{candidate_size_mb:.2f}",
                      fmt_change(size_change_pct, size_threshold))
        table.add_row("p50 latency (ms)", f"{baseline_result.latency_p50:.2f}", f"{candidate_result.latency_p50:.2f}",
                      fmt_change(_pct_change(baseline_result.latency_p50, candidate_result.latency_p50), latency_threshold))
        table.add_row("p95 latency (ms)", f"{baseline_result.latency_p95:.2f}", f"{candidate_result.latency_p95:.2f}",
                      fmt_change(_pct_change(baseline_result.latency_p95, candidate_result.latency_p95), latency_threshold))
        table.add_row("p99 latency (ms)", f"{baseline_result.latency_p99:.2f}", f"{candidate_result.latency_p99:.2f}",
                      fmt_change(_pct_change(baseline_result.latency_p99, candidate_result.latency_p99), latency_threshold))
        if baseline_result.memory_peak_mb > 0 or candidate_result.memory_peak_mb > 0:
            table.add_row("Peak memory (MB)", f"{baseline_result.memory_peak_mb:.2f}", f"{candidate_result.memory_peak_mb:.2f}",
                          fmt_change(_pct_change(baseline_result.memory_peak_mb, candidate_result.memory_peak_mb), latency_threshold))

        console.print(table)
        console.print(f"\n[bold]Statistical Analysis:[/bold]")
        console.print(f"  p-value:        {regression.p_value:.4f} {'[green](significant)[/green]' if regression.p_value < 0.05 else '(not significant)'}")
        console.print(f"  Latency change: {latency_change_pct:+.2f}% (threshold: {latency_threshold}%)")
        console.print(f"  Size change:    {size_change_pct:+.2f}% (threshold: {size_threshold}%)")
        console.print()

        if regression_detected:
            reasons = []
            if latency_regression:
                reasons.append(f"latency +{latency_change_pct:.1f}% > {latency_threshold}% threshold")
            if size_regression:
                reasons.append(f"size +{size_change_pct:.1f}% > {size_threshold}% threshold")
            console.print(Panel.fit(
                f"[bold red]\u26a0 REGRESSION DETECTED[/bold red]\n"
                + "\n".join(f"  \u2022 {r}" for r in reasons)
                + f"\n\n[dim]p={regression.p_value:.4f}[/dim]",
                border_style="red",
            ))
        else:
            console.print(Panel.fit(
                f"[bold green]\u2713 NO REGRESSION[/bold green]\n"
                f"Latency: {latency_change_pct:+.1f}%  \u2502  Size: {size_change_pct:+.1f}%\n"
                f"Both within thresholds ({latency_threshold}% / {size_threshold}%)",
                border_style="green",
            ))

    return 1 if regression_detected else 0


@click.command()
@click.argument("baseline_id")
@click.argument("candidate_id")
@click.option("--threshold", default=5.0, help="Latency regression threshold % (default: 5%)")
@click.option("--size-threshold", default=5.0, help="Size regression threshold % (default: 5%)")
@click.option("--metric", type=click.Choice(["p50", "p95", "p99", "mean"]), default="p50")
@click.option("--compute-unit", type=click.Choice(["CPU_ONLY", "CPU_AND_GPU", "ALL"]), default="ALL")
@click.option("--runs", default=100, type=int)
@click.option("--warmup", default=20, type=int)
@click.option("--json", "as_json", is_flag=True)
@click.option("--ci", is_flag=True)
def compare(baseline_id, candidate_id, threshold, size_threshold, metric, compute_unit, runs, warmup, as_json, ci):
    """Compare two builds and detect regressions in latency and size."""
    try:
        registry = LocalRegistry()
        baseline = registry.resolve_build(baseline_id)
        candidate = registry.resolve_build(candidate_id)
        if not baseline:
            console.print(f"[red]Baseline not found: {baseline_id}[/red]")
            sys.exit(2)
        if not candidate:
            console.print(f"[red]Candidate not found: {candidate_id}[/red]")
            sys.exit(2)
        exit_code = _run_comparison(
            baseline=baseline, candidate=candidate,
            runs=runs, warmup=warmup, cu_enum=ComputeUnit[compute_unit],
            latency_threshold=threshold, size_threshold=size_threshold,
            metric=metric, as_json=as_json, ci_mode=ci,
        )
        if ci:
            sys.exit(exit_code)
    except MLBuildError as e:
        console.print(f"\n[red]Error:[/red] {e}\n")
        sys.exit(2)
    except Exception as e:
        console.print(f"\n[red]Unexpected error:[/red] {e}\n")
        import traceback; traceback.print_exc()
        sys.exit(2)


@click.command()
@click.argument("baseline_id")
@click.argument("candidate_id")
@click.option("--latency-threshold", default=10.0, help="Latency regression threshold % (default: 10%)")
@click.option("--size-threshold", default=5.0, help="Size regression threshold % (default: 5%)")
@click.option("--metric", type=click.Choice(["p50", "p95", "p99", "mean"]), default="p50")
@click.option("--compute-unit", type=click.Choice(["CPU_ONLY", "CPU_AND_GPU", "ALL"]), default="ALL")
@click.option("--runs", default=50, type=int)
@click.option("--warmup", default=10, type=int)
@click.option("--json", "as_json", is_flag=True)
@click.option("--strict", is_flag=True)
def ci_check(baseline_id, candidate_id, latency_threshold, size_threshold, metric, compute_unit, runs, warmup, as_json, strict):
    """CI regression gate. Exits 0 (pass), 1 (regression), or 2 (error)."""
    try:
        registry = LocalRegistry()
        baseline = registry.resolve_build(baseline_id)
        candidate = registry.resolve_build(candidate_id)
        if not baseline:
            console.print(f"[red]Baseline not found: {baseline_id}[/red]")
            sys.exit(2)
        if not candidate:
            console.print(f"[red]Candidate not found: {candidate_id}[/red]")
            sys.exit(2)
        effective_latency = 0.0 if strict else latency_threshold
        effective_size = 0.0 if strict else size_threshold
        if strict and not as_json:
            console.print("[yellow]Strict mode: any positive delta fails[/yellow]")
        exit_code = _run_comparison(
            baseline=baseline, candidate=candidate,
            runs=runs, warmup=warmup, cu_enum=ComputeUnit[compute_unit],
            latency_threshold=effective_latency, size_threshold=effective_size,
            metric=metric, as_json=as_json, ci_mode=True,
            command_name="ci-check",
        )
        sys.exit(exit_code)
    except MLBuildError as e:
        console.print(f"\n[red]Error:[/red] {e}\n")
        sys.exit(2)
    except Exception as e:
        console.print(f"\n[red]Unexpected error:[/red] {e}\n")
        import traceback; traceback.print_exc()
        sys.exit(2)
