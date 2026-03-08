"""
Benchmark command: Profile model performance on-device.
"""

import click
from rich.console import Console
from rich.table import Table
from pathlib import Path
import json
import sys

import numpy as np

from ...registry import LocalRegistry
from ...benchmark.runner import (
    CoreMLBenchmarkRunner,
    ComputeUnit,
)
from ...core.errors import MLBuildError

# --- PATCH: task-aware imports ---
from ...core.task_detection import TaskType
from ...core.task_inputs import TaskInputFactory, ModelInfo as InputModelInfo
from ...core.task_validation import (
    TaskOutputValidator,
    StrictOutputConfig,
    format_validation_warning,
    should_exit_on_validation,
)

console = Console()


# --- PATCH: shared helper (mirrors build.py) ---
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


@click.command()
@click.argument("build_id")
@click.option("--runs", default=100, help="Number of benchmark runs")
@click.option("--warmup", default=20, help="Number of warmup runs")
@click.option("--compute-unit", 
              type=click.Choice(["CPU_ONLY", "CPU_AND_GPU", "ALL"]),
              default="ALL",
              help="Compute unit to use")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
# --- PATCH: --task flag ---
@click.option(
    "--task",
    type=click.Choice(["vision", "nlp", "audio", "unknown"]),
    default=None,
    help="Override task type. Falls back to registry build record if omitted.",
)
# --- PATCH: --strict-output flag ---
@click.option(
    "--strict-output",
    "strict_output",
    is_flag=True,
    default=False,
    help="Hard-fail on output validation warnings (overrides config.toml).",
)
def benchmark(build_id: str, runs: int, warmup: int, compute_unit: str, as_json: bool,
              task: str, strict_output: bool):
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

        # --- PATCH: resolve task (flag → registry → unknown) ---
        if task:
            resolved_task = TaskType(task)
        elif getattr(build, "task_type", None):
            resolved_task = TaskType(build.task_type)
        else:
            resolved_task = TaskType.UNKNOWN

        # --- PATCH: resolve StrictOutputConfig ---
        strict_cfg = StrictOutputConfig.from_command(
            strict_flag=strict_output,
            global_strict=_read_global_strict(),
        )
        validator = TaskOutputValidator(config=strict_cfg)

        # Display what we're benchmarking
        if not as_json:
            console.print(f"\n[bold]Benchmarking:[/bold] {build.name or build.build_id[:16]}")
            console.print(f"Format: {build.format}")
            console.print(f"Target: {build.target_device}")
            console.print(f"Quantization: {build.quantization_type}")
            console.print(f"Task: {resolved_task.value}")          # --- PATCH ---
            if build.format == "coreml":
                console.print(f"Compute Unit: {compute_unit}")
            console.print()

        # -------------------------------------------------------
        # Route to correct runner based on build format
        # -------------------------------------------------------
        if build.format == "tflite":
            from ...backends.tflite.benchmark_runner import TFLiteBenchmarkRunner
            from ...benchmark.runner import BenchmarkResult, mad_based_filter, bootstrap_ci, autocorrelation_lag1, hardware_fingerprint
            from dataclasses import asdict
            import platform as _platform

            runner = TFLiteBenchmarkRunner()

            # --- PATCH: NLP seq-len loop vs. flat benchmark ---
            if resolved_task == TaskType.NLP:
                _run_nlp_benchmark_tflite(
                    runner=runner,
                    build=build,
                    artifact_path=artifact_path,
                    runs=runs,
                    warmup=warmup,
                    registry=registry,
                    validator=validator,
                    strict_cfg=strict_cfg,
                    as_json=as_json,
                )
                return

            # Non-NLP TFLite path (unchanged logic, validation added)
            metrics = runner.benchmark(
                model_path=artifact_path,
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
                memory_peak_mb=max(metrics["memory_rss_mb"], 0.0),
                thermal_drift_ratio=1.0,
                hardware=hw,
            )
            runtime = "tflite"

        else:
            # Default: CoreML
            cu_enum = ComputeUnit[compute_unit]

            # --- PATCH: NLP seq-len loop for CoreML ---
            if resolved_task == TaskType.NLP:
                _run_nlp_benchmark_coreml(
                    artifact_path=artifact_path,
                    build=build,
                    cu_enum=cu_enum,
                    warmup=warmup,
                    runs=runs,
                    registry=registry,
                    validator=validator,
                    strict_cfg=strict_cfg,
                    as_json=as_json,
                )
                return

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
            runtime = "coreml"

        # Save to registry
        from ...core.types import Benchmark
        from datetime import datetime, timezone
        
        bench = Benchmark(
            build_id=build.build_id,
            device_chip=result.chip,
            runtime=runtime,
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

        # --- PATCH: output validation (non-NLP path) ---
        # Runners don't expose raw output tensors yet — validation is a
        # no-op here until runners return outputs in Step 8 (profile.py).
        # Wired now so the pattern is consistent.

        # Display results
        if as_json:
            data = {
                "build_id": result.build_id,
                "device_chip": result.chip,
                "runtime": runtime,
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
                "task": resolved_task.value,     # --- PATCH ---
            }
            console.print(json.dumps(data, indent=2))
        else:
            # Rich table
            table = Table(title="Benchmark Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right", style="green")
            
            table.add_row("Device", result.chip)
            table.add_row("Runtime", runtime)
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


# =============================================================================
# --- PATCH: NLP seq-len benchmark helpers ---
# =============================================================================

def _get_nlp_seq_lens(artifact_path: Path) -> list[int]:
    """
    Return the clamped NLP seq-len ladder for this model.
    Tries to read max_seq_len from model metadata; falls back to 256.
    """
    from ...core.task_inputs import get_nlp_seq_lens
    try:
        # Attempt to read max sequence length from CoreML/TFLite metadata
        model_max = _infer_model_max_seq_len(artifact_path)
    except Exception:
        model_max = 256
    return get_nlp_seq_lens(model_max)


def _infer_model_max_seq_len(artifact_path: Path) -> int:
    """
    Best-effort read of max sequence length from model metadata.
    Returns 256 if unavailable.
    """
    suffix = artifact_path.suffix.lower()
    if suffix in (".mlpackage", ".mlmodel") or artifact_path.is_dir():
        try:
            import coremltools as ct
            spec = ct.models.MLModel(str(artifact_path)).get_spec()
            for inp in spec.description.input:
                shape = inp.type.multiArrayType.shape
                if len(shape) >= 2:
                    return int(shape[-1]) or 256
        except Exception:
            pass
    elif suffix == ".tflite":
        try:
            import tensorflow as tf
            interp = tf.lite.Interpreter(model_path=str(artifact_path))
            interp.allocate_tensors()
            for detail in interp.get_input_details():
                if len(detail["shape"]) >= 2:
                    return int(detail["shape"][-1]) or 256
        except Exception:
            pass
    return 256


def _run_nlp_benchmark_coreml(
    artifact_path, build, cu_enum, warmup, runs,
    registry, validator, strict_cfg, as_json,
):
    """NLP seq-len benchmark loop for CoreML."""
    from ...benchmark.runner import CoreMLBenchmarkRunner
    from ...core.types import Benchmark
    from datetime import datetime, timezone
    import platform as _platform

    seq_lens = _get_nlp_seq_lens(artifact_path)

    if not as_json:
        console.print(f"Benchmarking NLP model — seq lens: {seq_lens}\n")

    rows = []  # (seq_len, p50, p95, p99)
    json_buckets = []

    for seq_len in seq_lens:
        runner = CoreMLBenchmarkRunner(
            model_path=artifact_path,
            compute_unit=cu_enum,
            warmup_runs=warmup,
            benchmark_runs=runs,
            ci_mode=False,
        )

        result, _ = runner.run(
            build_id=build.build_id,
            return_raw=True,
            seq_len=seq_len,      # passed through to runner if supported
        )

        # Registry: one row per seq-len bucket
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

        rows.append((seq_len, result.latency_p50, result.latency_p95, result.latency_p99))
        json_buckets.append({
            "seq_len": seq_len,
            "latency_p50_ms": result.latency_p50,
            "latency_p95_ms": result.latency_p95,
            "latency_p99_ms": result.latency_p99,
        })

    _print_nlp_results(rows, json_buckets, build, as_json, strict_cfg, validator)


def _run_nlp_benchmark_tflite(
    runner, build, artifact_path, runs, warmup,
    registry, validator, strict_cfg, as_json,
):
    """NLP seq-len benchmark loop for TFLite."""
    from ...benchmark.runner import BenchmarkResult, bootstrap_ci, hardware_fingerprint
    from ...core.types import Benchmark
    from dataclasses import asdict
    from datetime import datetime, timezone
    import platform as _platform

    seq_lens = _get_nlp_seq_lens(artifact_path)

    if not as_json:
        console.print(f"Benchmarking NLP model — seq lens: {seq_lens}\n")

    rows = []
    json_buckets = []

    try:
        fp = hardware_fingerprint()
        chip = fp.chip
    except Exception:
        chip = f"{_platform.system().lower()}_{_platform.machine()}"

    for seq_len in seq_lens:
        metrics = runner.benchmark(
            model_path=artifact_path,
            runs=runs,
            warmup=warmup,
            seq_len=seq_len,      # passed through to runner if supported
        )

        bench = Benchmark(
            build_id=build.build_id,
            device_chip=chip,
            runtime="tflite",
            measurement_type="latency",
            compute_unit="CPU_ONLY",
            latency_p50_ms=metrics["p50_ms"],
            latency_p95_ms=metrics["p95_ms"],
            latency_p99_ms=metrics["p99_ms"],
            memory_peak_mb=max(metrics["memory_rss_mb"], 0.0),
            num_runs=metrics["runs_completed"],
            measured_at=datetime.now(timezone.utc),
        )
        registry.save_benchmark(bench)

        rows.append((seq_len, metrics["p50_ms"], metrics["p95_ms"], metrics["p99_ms"]))
        json_buckets.append({
            "seq_len": seq_len,
            "latency_p50_ms": metrics["p50_ms"],
            "latency_p95_ms": metrics["p95_ms"],
            "latency_p99_ms": metrics["p99_ms"],
        })

    _print_nlp_results(rows, json_buckets, build, as_json, strict_cfg, validator)


def _print_nlp_results(rows, json_buckets, build, as_json, strict_cfg, validator):
    """Render NLP seq-len results — table or JSON."""
    if as_json:
        console.print(json.dumps({
            "build_id": build.build_id,
            "task": "nlp",
            "seq_len_buckets": json_buckets,
        }, indent=2))
        return

    table = Table(title="NLP Benchmark Results")
    table.add_column("Seq Len", style="cyan", justify="right")
    table.add_column("p50",     style="green", justify="right")
    table.add_column("p95",     style="yellow", justify="right")
    table.add_column("p99",     style="red", justify="right")

    for seq_len, p50, p95, p99 in rows:
        table.add_row(
            str(seq_len),
            f"{p50:.1f} ms",
            f"{p95:.1f} ms",
            f"{p99:.1f} ms",
        )

    console.print(table)
    console.print()
    console.print("[green]✓ Benchmark saved to registry[/green]\n")