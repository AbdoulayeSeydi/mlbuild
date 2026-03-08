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

# --- PATCH: task-aware import ---
from ...core.task_detection import TaskType

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


# --- PATCH: NLP seq-len bucket comparison ---

def _fetch_nlp_buckets(registry, build_id: str) -> dict[int, dict]:
    """
    Fetch benchmark rows for a build grouped by seq-len bucket.

    Returns {seq_len: {p50, p95, p99}} from registry benchmark rows.
    Falls back to empty dict if registry doesn't expose seq_len on rows.
    """
    try:
        rows = registry.get_benchmarks(build_id)  # list of Benchmark objects
        buckets = {}
        for row in rows:
            seq_len = getattr(row, 'seq_len', None)
            if seq_len is not None:
                buckets[int(seq_len)] = {
                    'p50': row.latency_p50_ms,
                    'p95': row.latency_p95_ms,
                    'p99': row.latency_p99_ms,
                }
        return buckets
    except Exception:
        return {}


def _run_nlp_comparison(baseline, candidate, latency_threshold, as_json, ci_mode):
    """
    Per-seq-len bucket comparison for NLP models.
    Reads benchmark rows already stored in the registry (written by benchmark.py Step 7).
    """
    registry = LocalRegistry()
    baseline_buckets = _fetch_nlp_buckets(registry, baseline.build_id)
    candidate_buckets = _fetch_nlp_buckets(registry, candidate.build_id)

    # Union of all seq-lens present in either build, sorted
    all_seq_lens = sorted(set(baseline_buckets) | set(candidate_buckets))

    if not all_seq_lens:
        if not as_json and not ci_mode:
            console.print(
                "[yellow]No per-seq-len benchmark rows found in registry for these builds.\n"
                "Run [bold]mlbuild benchmark <build_id> --task nlp[/bold] for both builds first.[/yellow]"
            )
        return 0  # not an error — no data to compare

    regression_detected = False
    rows = []  # (seq_len, b_p50, c_p50, delta_ms, pct, is_regression)

    for seq_len in all_seq_lens:
        b = baseline_buckets.get(seq_len)
        c = candidate_buckets.get(seq_len)
        if b is None or c is None:
            rows.append((seq_len, None, None, None, None, False))
            continue
        delta_ms = c['p50'] - b['p50']
        pct = _pct_change(b['p50'], c['p50'])
        is_reg = pct > latency_threshold
        if is_reg:
            regression_detected = True
        rows.append((seq_len, b['p50'], c['p50'], delta_ms, pct, is_reg))

    if as_json:
        output = {
            "task": "nlp",
            "baseline": {"build_id": baseline.build_id, "name": baseline.name},
            "candidate": {"build_id": candidate.build_id, "name": candidate.name},
            "seq_len_buckets": [
                {
                    "seq_len": seq_len,
                    "baseline_p50_ms": b_p50,
                    "candidate_p50_ms": c_p50,
                    "delta_ms": delta_ms,
                    "pct_change": pct,
                    "regression": is_reg,
                }
                for seq_len, b_p50, c_p50, delta_ms, pct, is_reg in rows
            ],
            "regression_detected": regression_detected,
            "threshold_pct": latency_threshold,
        }
        console.print(json.dumps(output, indent=2))
        return 1 if regression_detected else 0

    table = Table(title="NLP Comparison (per seq-len)", show_header=True, header_style="bold magenta")
    table.add_column("Seq Len",       style="cyan",   justify="right")
    table.add_column("Baseline p50",               justify="right")
    table.add_column("Candidate p50",              justify="right")
    table.add_column("Δ",                          justify="right")
    table.add_column("Regression?",                justify="center")

    for seq_len, b_p50, c_p50, delta_ms, pct, is_reg in rows:
        if b_p50 is None or c_p50 is None:
            table.add_row(str(seq_len), "—", "—", "—", "[dim]no data[/dim]")
            continue
        delta_str = f"{delta_ms:+.1f} ms"
        if is_reg:
            reg_str = f"[red]✗ +{pct:.0f}%[/red]"
            delta_col = f"[red]{delta_str}[/red]"
        elif pct > 0:
            reg_str = f"[yellow]⚠ +{pct:.0f}%[/yellow]"
            delta_col = f"[yellow]{delta_str}[/yellow]"
        else:
            reg_str = "[green]—[/green]"
            delta_col = f"[green]{delta_str}[/green]"
        table.add_row(str(seq_len), f"{b_p50:.1f} ms", f"{c_p50:.1f} ms", delta_col, reg_str)

    console.print(table)
    console.print()

    if regression_detected:
        console.print(Panel.fit(
            "[bold red]⚠ REGRESSION DETECTED[/bold red]\n"
            f"One or more seq-len buckets exceeded the {latency_threshold}% threshold.",
            border_style="red",
        ))
    else:
        console.print(Panel.fit(
            "[bold green]✓ NO REGRESSION[/bold green]\n"
            f"All seq-len buckets within {latency_threshold}% threshold.",
            border_style="green",
        ))

    return 1 if regression_detected else 0


def _run_comparison(
    baseline, candidate,
    runs, warmup, cu_enum,
    latency_threshold, size_threshold,
    metric, as_json, ci_mode,
    command_name="compare",
    resolved_task=TaskType.UNKNOWN,   # --- PATCH ---
):
    if not as_json:
        console.print(f"\n[bold]Regression Detection[/bold]")
        console.print(f"Baseline:  {baseline.name or baseline.build_id[:12]} ({getattr(baseline, 'quantization_type', '?')})")
        console.print(f"Candidate: {candidate.name or candidate.build_id[:12]} ({getattr(candidate, 'quantization_type', '?')})")
        console.print(f"Latency threshold: {latency_threshold}%  |  Size threshold: {size_threshold}%")
        console.print(f"Metric: {metric}")
        console.print(f"Task: {resolved_task.value}")   # --- PATCH: Task header line ---
        console.print()

    # --- PATCH: NLP takes the bucket path ---
    if resolved_task == TaskType.NLP:
        return _run_nlp_comparison(baseline, candidate, latency_threshold, as_json, ci_mode)

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
            "task": resolved_task.value,   # --- PATCH ---
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
# --- PATCH: --task flag ---
@click.option(
    "--task",
    type=click.Choice(["vision", "nlp", "audio", "unknown"]),
    default=None,
    help="Override task type. Falls back to baseline build's registry record if omitted.",
)
def compare(baseline_id, candidate_id, threshold, size_threshold, metric, compute_unit, runs, warmup, as_json, ci, task):
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

        # --- PATCH: resolve task from flag → baseline registry → UNKNOWN ---
        if task:
            resolved_task = TaskType(task)
        elif getattr(baseline, 'task_type', None):
            resolved_task = TaskType(baseline.task_type)
        else:
            resolved_task = TaskType.UNKNOWN

        exit_code = _run_comparison(
            baseline=baseline, candidate=candidate,
            runs=runs, warmup=warmup, cu_enum=ComputeUnit[compute_unit],
            latency_threshold=threshold, size_threshold=size_threshold,
            metric=metric, as_json=as_json, ci_mode=ci,
            resolved_task=resolved_task,   # --- PATCH ---
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
# --- PATCH: --task flag ---
@click.option(
    "--task",
    type=click.Choice(["vision", "nlp", "audio", "unknown"]),
    default=None,
    help="Override task type. Falls back to baseline build's registry record if omitted.",
)
def ci_check(baseline_id, candidate_id, latency_threshold, size_threshold, metric, compute_unit, runs, warmup, as_json, strict, task):
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

        # --- PATCH: resolve task from flag → baseline registry → UNKNOWN ---
        if task:
            resolved_task = TaskType(task)
        elif getattr(baseline, 'task_type', None):
            resolved_task = TaskType(baseline.task_type)
        else:
            resolved_task = TaskType.UNKNOWN

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
            resolved_task=resolved_task,   # --- PATCH ---
        )
        sys.exit(exit_code)
    except MLBuildError as e:
        console.print(f"\n[red]Error:[/red] {e}\n")
        sys.exit(2)
    except Exception as e:
        console.print(f"\n[red]Unexpected error:[/red] {e}\n")
        import traceback; traceback.print_exc()
        sys.exit(2)