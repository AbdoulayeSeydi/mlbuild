"""
Production-grade quantization comparison.

Compares FP32 / FP16 / INT8 builds with:
- Deterministic baseline selection
- Strict registry resolution
- Isolated benchmark failures
- Hardware-aware reporting
- Tail latency metrics
- Accuracy loss estimation
- Machine-readable JSON output
"""

import json
import click
from pathlib import Path
from rich.console import Console
from rich.table import Table

from ...registry import LocalRegistry
from ...benchmark.runner import CoreMLBenchmarkRunner, ComputeUnit

console = Console()


# ============================================================
# Helpers
# ============================================================

def _precision_rank(qtype: str) -> int:
    order = {"fp32": 0, "fp16": 1, "int8": 2}
    return order.get((qtype or "").lower(), 99)


def _resolve_build(registry: LocalRegistry, prefix: str):
    """
    Resolve build by prefix with strict collision detection.
    """
    matches = registry.get_build_by_prefix(prefix)

    if len(matches) == 0:
        raise ValueError(f"No build found for prefix: {prefix}")

    if len(matches) > 1:
        ids = ", ".join(b.build_id[:12] for b in matches)
        raise ValueError(
            f"Ambiguous prefix '{prefix}'. Matches: {ids}"
        )

    return matches[0]


def _compute_unit_from_flag(flag: str) -> ComputeUnit:
    mapping = {
        "all": ComputeUnit.ALL,    
        "cpu": ComputeUnit.CPU_ONLY,
        "ane": ComputeUnit.ALL,      
        "gpu": ComputeUnit.CPU_AND_GPU,
    }
    return mapping[flag]


def _estimate_accuracy_loss(baseline_build, candidate_build, registry):
    """
    Estimate accuracy loss between two builds.
    
    Uses calibration data for testing.
    """
    from ...backends.coreml.exporter import ModelIngestion
    from ...loaders import load_model
    from ...core.calibration import CalibrationConfig, CalibrationDataset, PreprocessingConfig
    from ...core.accuracy import estimate_model_accuracy
    
    try:
        # Load source model to get input shape
        ir = load_model(baseline_build.source_path)
        _, _, shape_tuples = ModelIngestion.extract_input_specs(ir)
        
        # Generate test samples (use different seed than calibration)
        cal_config = CalibrationConfig(
            sample_count=50,  # 50 test samples
            input_shape=tuple(shape_tuples[0]),
            preprocessing=PreprocessingConfig(),
            seed=123,  # Different seed for testing
        )
        
        cal_dataset = CalibrationDataset(cal_config)
        test_samples = list(cal_dataset.generate_synthetic())
        
        # Estimate accuracy
        metrics = estimate_model_accuracy(
            Path(baseline_build.artifact_path),
            Path(candidate_build.artifact_path),
            test_samples,
        )
        
        return metrics
        
    except Exception as e:
        console.print(f"[yellow]Accuracy estimation failed: {e}[/yellow]")
        return None


# ============================================================
# CLI
# ============================================================

@click.command()
@click.argument("build_ids", nargs=-1, required=True)
@click.option("--runs", default=50, type=int)
@click.option("--warmup", default=10, type=int)
@click.option(
    "--compute-unit",
    default="all",
    type=click.Choice(["all", "cpu", "ane", "gpu"]),
    help="Compute backend",
)
@click.option(
    "--baseline",
    default=None,
    help="Explicit baseline build ID prefix",
)
@click.option(
    "--json-output",
    is_flag=True,
    help="Emit machine-readable JSON",
)
def compare_quantization(
    build_ids,
    runs,
    warmup,
    compute_unit,
    baseline,
    json_output,
):
    """
    Compare quantization builds safely and deterministically.

    Example:
        mlbuild compare-quantization <fp32> <fp16> <int8>
    """

    if len(build_ids) < 2:
        console.print("[red]Need at least 2 build IDs[/red]")
        return

    registry = LocalRegistry()

    # --------------------------------------------------------
    # Resolve builds strictly
    # --------------------------------------------------------
    builds = []
    for prefix in build_ids:
        try:
            build = _resolve_build(registry, prefix)
            builds.append(build)
        except Exception as e:
            console.print(f"[red]{e}[/red]")
            return

    # --------------------------------------------------------
    # Deterministic sorting by quantization precision
    # --------------------------------------------------------
    builds.sort(
        key=lambda b: _precision_rank(
            (b.quantization or {}).get("type", "")
        )
    )

    # --------------------------------------------------------
    # Baseline selection
    # --------------------------------------------------------
    if baseline:
        baseline_build = _resolve_build(registry, baseline)
    else:
        # Default: lowest precision rank (FP32 preferred)
        baseline_build = builds[0]

    # --------------------------------------------------------
    # Benchmark builds
    # --------------------------------------------------------
    results = []
    compute_enum = _compute_unit_from_flag(compute_unit)

    for build in builds:

        if build.format != "coreml":
            console.print(
                f"[yellow]Skipping non-CoreML build: {build.build_id}[/yellow]"
            )
            continue

        console.print(
            f"⏱ Benchmarking {build.name or build.build_id[:12]}..."
        )

        try:
            runner = CoreMLBenchmarkRunner(
                model_path=Path(build.artifact_path),
                compute_unit=compute_enum,
                warmup_runs=warmup,
                benchmark_runs=runs,
            )

            result, _ = runner.run(
                build_id=build.build_id,
                return_raw=True,
            )

            results.append({
                "build": build,
                "benchmark": result,
            })

        except Exception as e:
            console.print(
                f"[red]Benchmark failed for {build.build_id}: {e}[/red]"
            )
            continue

    if not results:
        console.print("[red]No successful benchmarks[/red]")
        return

    # Ensure baseline exists in successful results
    baseline_result = next(
        (r for r in results if r["build"].build_id == baseline_build.build_id),
        None,
    )

    if baseline_result is None:
        console.print("[red]Baseline build failed benchmarking[/red]")
        return

    baseline_latency = baseline_result["benchmark"].latency_p50
    baseline_size = float(baseline_result["build"].size_mb)

    # --------------------------------------------------------
    # Accuracy Estimation (if baseline is FP32)
    # --------------------------------------------------------
    accuracy_results = {}
    
    if baseline_build.quantization.get('type') == 'fp32':
        console.print("\n[dim]Estimating accuracy loss...[/dim]")
        
        for r in results:
            b = r["build"]
            
            if b.build_id == baseline_build.build_id:
                continue  # Skip baseline
            
            console.print(f"  Comparing {b.name or b.build_id[:12]}...")
            metrics = _estimate_accuracy_loss(baseline_build, b, registry)
            if metrics:
                accuracy_results[b.build_id] = metrics

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------
    if json_output:
        payload = []

        for r in results:
            b = r["build"]
            bench = r["benchmark"]

            size_mb = float(b.size_mb)
            latency = bench.latency_p50

            size_delta = (
                (size_mb - baseline_size) / baseline_size * 100
            )
            latency_delta = (
                (latency - baseline_latency) / baseline_latency * 100
            )

            item = {
                "build_id": b.build_id,
                "quantization": (b.quantization or {}).get("type"),
                "size_mb": size_mb,
                "latency_p50": latency,
                "latency_p95": bench.latency_p95,
                "latency_p99": bench.latency_p99,
                "memory_mb": bench.memory_peak_mb,
                "size_delta_pct": size_delta,
                "latency_delta_pct": latency_delta,
                "compute_unit": compute_unit,
            }
            
            if b.build_id in accuracy_results:
                acc = accuracy_results[b.build_id]
                item["accuracy"] = {
                    "mse": acc.mse,
                    "mae": acc.mae,
                    "max_error": acc.max_error,
                    "relative_error_pct": acc.relative_error,
                }
            
            payload.append(item)

        print(json.dumps(payload, indent=2))
        return

    # --------------------------------------------------------
    # Rich table output
    # --------------------------------------------------------
    console.print(
        f"\n[bold]Quantization Comparison "
        f"(compute={compute_unit})[/bold]\n"
    )

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Build")
    table.add_column("Quant")
    table.add_column("Size (MB)", justify="right")
    table.add_column("p50 (ms)", justify="right")
    table.add_column("p95 (ms)", justify="right")
    table.add_column("p99 (ms)", justify="right")
    table.add_column("Memory (MB)", justify="right")
    table.add_column("Δ vs Baseline", justify="right")
    
    if accuracy_results:
        table.add_column("Accuracy Loss", justify="right")

    for r in results:
        b = r["build"]
        bench = r["benchmark"]

        size_mb = float(b.size_mb)
        latency = bench.latency_p50

        size_delta = (
            (size_mb - baseline_size) / baseline_size * 100
        )
        latency_delta = (
            (latency - baseline_latency) / baseline_latency * 100
        )

        delta_text = ""

        if b.build_id == baseline_build.build_id:
            delta_text = "BASELINE"
        else:
            latency_str = (
                f"[red]+{latency_delta:.1f}% slower[/red]"
                if latency_delta > 0
                else f"[green]{abs(latency_delta):.1f}% faster[/green]"
            )

            size_str = (
                f"[red]+{size_delta:.1f}% larger[/red]"
                if size_delta > 0
                else f"[green]{abs(size_delta):.1f}% smaller[/green]"
            )

            delta_text = f"{latency_str}\n{size_str}"

        # Build row
        row_data = [
            b.name or b.build_id[:12],
            (b.quantization or {}).get("type", "unknown").upper(),
            f"{size_mb:.2f}",
            f"{bench.latency_p50:.3f}",
            f"{bench.latency_p95:.3f}",
            f"{bench.latency_p99:.3f}",
            f"{bench.memory_peak_mb:.2f}",
            delta_text,
        ]
        
        # Add accuracy column if available
        if accuracy_results:
            if b.build_id in accuracy_results:
                acc = accuracy_results[b.build_id]
                acc_text = f"MSE: {acc.mse:.2e}\nMAE: {acc.mae:.2e}\nRel: {acc.relative_error:.2f}%"
                row_data.append(acc_text)
            else:
                row_data.append("BASELINE")
        
        table.add_row(*row_data)

    console.print(table)
    console.print()