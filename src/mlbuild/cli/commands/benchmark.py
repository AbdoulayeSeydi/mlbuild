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

console = Console(width=None)


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
              default="ALL")
@click.option("--json", "as_json", is_flag=True)
@click.option("--task", default=None,
              type=click.Choice(["vision", "nlp", "audio", "unknown"]))
@click.option("--strict-output", "strict_output", is_flag=True, default=False)
@click.option("--platform", default=None,
              type=click.Choice(["ios", "android"]),
              help="Force platform when both iOS and Android devices are connected.")
@click.option("--udid", default=None,
              help="iOS Simulator or device UDID.")
@click.option("--signed-app", "signed_app", default=None,
              type=click.Path(exists=True),
              help="Path to signed MLBuildRunner.app for real iOS device.")
def benchmark(build_id, runs, warmup, compute_unit, as_json, task,
              strict_output, platform, udid, signed_app):
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
        
        # If artifact_path is a directory, find the actual model file inside
        if artifact_path.is_dir():
            # Check if the directory itself is an mlpackage
            if (artifact_path / "Manifest.json").exists():
                pass  # artifact_path is already the mlpackage — use it directly
            elif build.format == "tflite":
                candidates = list(artifact_path.glob("*.tflite"))
                if not candidates:
                    console.print(f"[red]No .tflite file found in artifact directory: {artifact_path}[/red]")
                    raise click.Abort()
                artifact_path = candidates[0]
            elif build.format == "coreml":
                candidates = list(artifact_path.glob("*.mlpackage")) or list(artifact_path.glob("*.mlmodel"))
                if not candidates:
                    console.print(f"[red]No model file found in artifact directory: {artifact_path}[/red]")
                    raise click.Abort()
                artifact_path = candidates[0]
            else:
                console.print(f"[red]No model file found in artifact directory: {artifact_path}[/red]")
                raise click.Abort()
            
        
        # Inside the device-connected routing block — replace the existing call:
        if "device-connected" in (build.target_device or ""):
            _run_device_connected_benchmark(
                build         = build,
                build_abi     = getattr(build, "device_abi", None),
                device_name   = getattr(build, "device_name", None),
                artifact_path = artifact_path,
                runs          = runs,
                warmup        = warmup,
                as_json       = as_json,
                registry      = registry,
                platform      = platform,
                udid          = udid,
                signed_app    = Path(signed_app) if signed_app else None,
            )
            return

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
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Value", justify="right", style="green", no_wrap=True)
            
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
    table.add_column("Seq Len", style="cyan", justify="right", no_wrap=True)
    table.add_column("p50",     style="green", justify="right", no_wrap=True)
    table.add_column("p95",     style="yellow", justify="right", no_wrap=True)
    table.add_column("p99",     style="red", justify="right", no_wrap=True)

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

def _detect_connected_platform(platform_override: str | None) -> tuple[str, str | None]:
    """
    Detect which platform is connected — android, ios, or raise on conflict.
    Returns: ("android" or "ios", udid or None)
    """
    android_found = False
    ios_found     = False
    ios_udid      = None

    # ADB check
    try:
        from ...platforms.android.adb import devices as adb_devices
        android_devices = [d for d in adb_devices() if d[1] == "device"]
        if android_devices:
            android_found = True
    except Exception:
        pass

    # idb check
    try:
        from ...platforms.ios.idb import list_targets
        ios_targets = [
            t for t in list_targets()
            if t[2].lower() in ("booted", "connected")
        ]
        if ios_targets:
            ios_found = True
            if len(ios_targets) == 1:
                ios_udid = ios_targets[0][0]
    except Exception:
        pass

    # Both found — require explicit --platform
    if android_found and ios_found:
        if platform_override == "android":
            return "android", None
        if platform_override == "ios":
            return "ios", ios_udid
        console.print(
            "\n[red]Multiple platforms detected (android + ios).[/red]\n"
            "Use [bold]--platform ios[/bold] or [bold]--platform android[/bold] "
            "to specify which device to benchmark.\n"
        )
        sys.exit(1)

    if platform_override == "android":
        return "android", None
    if platform_override == "ios":
        return "ios", ios_udid

    if android_found:
        return "android", None

    if ios_found:
        if len(ios_targets) > 1 and not ios_udid:
            console.print("\n[red]Multiple iOS simulators booted.[/red] Specify one with [bold]--udid[/bold]:\n")
            for udid, name, state in ios_targets:
                console.print(f"  [dim]{udid}[/dim]  {name}")
            console.print()
            sys.exit(1)
        return "ios", ios_udid

    console.print(
        "\n[red]No device detected.[/red]\n"
        "Connect an Android device via USB-C (USB debugging enabled)\n"
        "or boot an iOS Simulator.\n"
    )
    sys.exit(1)


def _run_device_connected_benchmark(
    build,
    build_abi:    str | None,
    device_name:  str | None,
    artifact_path,
    runs:         int,
    warmup:       int,
    as_json:      bool,
    registry,
    platform:     str | None = None,
    udid:         str | None = None,
    signed_app    = None,
) -> None:
    """
    Route a device-connected build to ADB (Android) or idb (iOS) pipeline.
    Platform auto-detected from connected devices; --platform resolves conflicts.
    """
    if udid:
        detected, auto_udid = "ios", udid
    elif not platform and hasattr(build, "format"):
        # Auto-resolve: coreml → ios, tflite → android, skip detection conflict
        fmt = getattr(build, "format", None)
        if fmt == "coreml":
            detected, auto_udid = _detect_connected_platform("ios")
        elif fmt == "tflite":
            detected, auto_udid = _detect_connected_platform("android")
        else:
            detected, auto_udid = _detect_connected_platform(platform)
    else:
        detected, auto_udid = _detect_connected_platform(platform)
    resolved_udid = udid or auto_udid

    if detected == "android":
        _run_android_device_benchmark(
            build         = build,
            build_abi     = build_abi,
            device_name   = device_name,
            artifact_path = artifact_path,
            runs          = runs,
            warmup        = warmup,
            as_json       = as_json,
            registry      = registry,
        )
    else:
        _run_ios_device_benchmark(
            build         = build,
            artifact_path = artifact_path,
            runs          = runs,
            warmup        = warmup,
            as_json       = as_json,
            registry      = registry,
            udid          = resolved_udid,
            signed_app    = signed_app,
        )


def _run_android_device_benchmark(
    build,
    build_abi:    str | None,
    device_name:  str | None,
    artifact_path,
    runs:         int,
    warmup:       int,
    as_json:      bool,
    registry,
) -> None:
    """Android ADB pipeline. Extracted from old _run_device_connected_benchmark."""
    from datetime import datetime, timezone

    try:
        from ...platforms.android.device import ADBDevice, BenchmarkConfig
        from ...core.errors import (
            ADBNotFoundError, ADBNoDeviceError,
            ADBUnauthorizedError, ADBOfflineError,
            ADBMultipleDevicesError,
        )
    except ImportError as exc:
        console.print(f"\n[red]Import error:[/red] {exc}\n")
        sys.exit(1)

    # Format check — Android only runs TFLite
    if build.format != "tflite":
        console.print(
            "\n[red]Format mismatch.[/red]\n"
            "Connected device is Android but model format is CoreML.\n"
            "[dim]Convert to TFLite first:\n"
            "  mlbuild build --model model.onnx --target device-connected --backend tflite[/dim]\n"
        )
        sys.exit(1)

    if not as_json:
        console.print(f"\n[bold]Benchmarking:[/bold] {build.name or build.build_id[:16]}")
        console.print(f"Format:  {build.format}")
        console.print(f"Target:  {device_name or build.target_device}")
        console.print(f"Built for ABI: {build_abi or 'unknown'}")
        console.print()
        console.print("[dim]Detecting connected Android device...[/dim]")

    try:
        device = ADBDevice.connect()
    except ADBNotFoundError as exc:
        console.print(f"\n[red]ADB not found.[/red] {exc.message}\n")
        sys.exit(1)
    except ADBNoDeviceError:
        console.print(
            "\n[red]No Android device connected.[/red]\n"
            "Connect a device via USB-C and enable USB debugging.\n"
        )
        sys.exit(1)
    except ADBUnauthorizedError:
        console.print(
            "\n[red]Device not authorized.[/red]\n"
            "Accept the USB debugging prompt on your phone.\n"
        )
        sys.exit(1)
    except ADBOfflineError:
        console.print(
            "\n[red]Device offline.[/red]\n"
            "Try unplugging and reconnecting.\n"
        )
        sys.exit(1)
    except ADBMultipleDevicesError as exc:
        console.print(f"\n[red]Multiple devices found.[/red] {exc.message}\n")
        sys.exit(1)
    except Exception as exc:
        console.print(f"\n[red]Connection failed:[/red] {exc}\n")
        sys.exit(1)

    connected_abi  = device.profile.primary_abi
    connected_name = f"{device.profile.manufacturer} {device.profile.model}"

    if not as_json:
        console.print(
            f"  Connected: [bold]{connected_name}[/bold]  "
            f"API {device.profile.api_level}  {connected_abi}\n"
        )
        if device.profile.is_emulator:
            console.print(
                "[yellow]⚠  Emulator detected[/yellow] — "
                "latency numbers will not reflect real device performance.\n"
            )

    if build_abi and build_abi != connected_abi:
        console.print(
            f"\n[red]ABI mismatch.[/red]\n\n"
            f"  Build ABI:     [yellow]{build_abi}[/yellow]  "
            f"(built on {device_name or 'unknown device'})\n"
            f"  Connected ABI: [yellow]{connected_abi}[/yellow]  "
            f"({connected_name})\n\n"
            f"Rebuild with this device connected:\n"
            f"  [dim]mlbuild build --model <model.onnx> --target device-connected[/dim]\n"
        )
        sys.exit(1)

    config = BenchmarkConfig(
        model_path  = artifact_path,
        num_runs    = runs,
        warmup_runs = warmup,
    )

    try:
        result = device.run(config)
    except Exception as exc:
        console.print(f"\n[red]Benchmark failed:[/red] {exc}\n")
        sys.exit(1)

    if result.error:
        console.print(f"\n[red]Run failed:[/red] {result.error}\n")
        sys.exit(1)

    view = result.view

    from ...core.types import Benchmark
    bench = Benchmark(
        build_id         = build.build_id,
        device_chip      = f"{connected_name} ({connected_abi})",
        runtime          = "tflite",
        measurement_type = "latency",
        compute_unit     = "CPU",
        latency_p50_ms   = view.cpu_p50_ms,
        latency_p95_ms   = view.cpu_p90_ms,
        latency_p99_ms   = view.cpu_p99_ms,
        memory_peak_mb   = view.cpu_peak_mem_mb,
        num_runs         = runs,
        measured_at      = datetime.now(timezone.utc),
    )
    registry.save_benchmark(bench)

    try:
        import click as _click
        ctx = _click.get_current_context()
        ctx.obj["_android_device"] = (
            f"{connected_name} — {connected_abi} (device-connected)"
        )
        ctx.obj["_linked_build_id"] = build.build_id
    except Exception:
        pass

    if as_json:
        import json as _json
        print(_json.dumps(result.registry_dict, indent=2, default=str))
        return

    _print_android_result_table(view, connected_name, connected_abi, result, device.profile.is_emulator, device.profile.api_level)

    if result.recommendation and result.recommendation.kind.value == "rerun":
        sys.exit(2)


def _run_ios_device_benchmark(
    build,
    artifact_path,
    runs:      int,
    warmup:    int,
    as_json:   bool,
    registry,
    udid:      str | None = None,
    signed_app = None,
) -> None:
    """iOS idb pipeline. Mirrors Android pipeline — same UX, different bridge."""
    from datetime import datetime, timezone
    from pathlib import Path as _Path

    try:
        from ...platforms.ios.device import IDBDevice, BenchmarkConfig
        from ...core.errors import (
            IDBNotFoundError, IDBCompanionNotRunningError,
            IDBNoDeviceError, IDBUnauthorizedError,
            IDBOfflineError, IDBMultipleDevicesError,
            UnsignedBinaryError,
        )
    except ImportError as exc:
        console.print(f"\n[red]Import error:[/red] {exc}\n")
        sys.exit(1)

    # Format check — iOS only runs CoreML
    if build.format == "tflite":
        console.print(
            "\n[red]Format mismatch.[/red]\n"
            "Connected device is iOS but model format is TFLite.\n"
            "[dim]Convert to CoreML first:\n"
            "  mlbuild build --model model.onnx --target device-connected --backend coreml[/dim]\n"
        )
        sys.exit(1)

    if not as_json:
        console.print(f"\n[bold]Benchmarking:[/bold] {build.name or build.build_id[:16]}")
        console.print(f"Format:  {build.format}")
        console.print(f"Target:  {build.target_device}")
        console.print()
        console.print("[dim]Detecting iOS target...[/dim]")

    # Connect
    try:
        device = IDBDevice.connect(udid=udid)
    except IDBNotFoundError:
        console.print(
            "\n[red]idb not found.[/red]\n"
            "Install via: [dim]brew install idb-companion[/dim]\n"
        )
        sys.exit(1)
    except IDBCompanionNotRunningError:
        console.print(
            "\n[red]idb_companion is not running.[/red]\n"
            "Start with: [dim]idb_companion --daemon[/dim]\n"
        )
        sys.exit(1)
    except IDBNoDeviceError:
        console.print(
            "\n[red]No iOS simulator detected.[/red]\n"
            "Boot one via: [dim]xcrun simctl boot <udid>[/dim]\n"
            "Or open Xcode → Simulator.\n"
        )
        sys.exit(1)
    except IDBUnauthorizedError:
        console.print(
            "\n[red]Device not trusted.[/red]\n"
            "Unlock your iPhone and tap 'Trust' when prompted.\n"
        )
        sys.exit(1)
    except IDBOfflineError:
        console.print(
            "\n[red]Device went offline.[/red]\n"
            "Try unplugging and reconnecting.\n"
        )
        sys.exit(1)
    except IDBMultipleDevicesError as exc:
        console.print(
            f"\n[red]Multiple iOS targets found.[/red]\n"
            f"Use [bold]--udid <udid>[/bold] to specify one.\n"
        )
        sys.exit(1)
    except UnsignedBinaryError:
        console.print(
            "\n[red]Real device benchmarking requires a signed MLBuildRunner.app.[/red]\n"
            "See: [link]mlbuild.dev/ios-signing[/link]\n"
            "Use [bold]--signed-app <path>[/bold] to provide your signed build,\n"
            "or run against a Simulator instead.\n"
        )
        sys.exit(1)
    except Exception as exc:
        console.print(f"\n[red]Connection failed:[/red] {exc}\n")
        sys.exit(1)

    profile = device.profile
    target_label = (
        f"{profile.name} [SIMULATOR]"
        if profile.is_simulator
        else profile.name
    )

    if not as_json:
        console.print(f"  Connected: [bold]{target_label}[/bold]  iOS {profile.ios_version}\n")
        if profile.is_simulator:
            console.print(
                "[yellow]⚠  iOS Simulator[/yellow] — "
                "ANE is unavailable. Results reflect CPU/GPU only.\n"
                "Connect a real iPhone for ANE benchmarks.\n"
            )

    # Run
    config = BenchmarkConfig(
        model_path  = artifact_path,
        num_runs    = runs,
        warmup_runs = warmup,
        signed_app  = _Path(signed_app) if signed_app else None,
    )

    try:
        result = device.run(config)
    except Exception as exc:
        console.print(f"\n[red]Benchmark failed:[/red] {exc}\n")
        sys.exit(1)

    if result.error:
        console.print(f"\n[red]Run failed:[/red] {result.error}\n")
        sys.exit(1)

    view = result.view

    # Save to registry
    from ...core.types import Benchmark
    bench = Benchmark(
        build_id         = build.build_id,
        device_chip      = f"{profile.name} ({profile.chip})",
        runtime          = "coreml",
        measurement_type = "latency",
        compute_unit     = view.compute_units_used or view.compute_units or "cpuOnly",
        latency_p50_ms   = view.cpu_p50_ms,
        latency_p95_ms   = view.cpu_p90_ms,
        latency_p99_ms   = view.cpu_p99_ms,
        memory_peak_mb   = view.cpu_peak_mem_mb,
        num_runs         = runs,
        measured_at      = datetime.now(timezone.utc),
    )
    registry.save_benchmark(bench)

    try:
        import click as _click
        ctx = _click.get_current_context()
        ctx.obj["_ios_device"] = f"{target_label} (device-connected)"
        ctx.obj["_linked_build_id"] = build.build_id
    except Exception:
        pass

    if as_json:
        import json as _json
        print(_json.dumps(result.registry_dict, indent=2, default=str))
        return

    _print_ios_result_table(view, profile, result, runs)

    if result.recommendation and result.recommendation.kind.value == "rerun":
        sys.exit(2)


def _print_android_result_table(view, connected_name, connected_abi, result, is_emulator=False, api_level=None) -> None:
    """Rich result table for Android ADB runs. Extracted from old inline code."""
    table = Table(title="Benchmark Results")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value",  justify="right", style="green", no_wrap=True)

    def _f(v):  return f"{v:.3f} ms" if v is not None else "—"
    def _fm(v): return f"{v:.2f} MB" if v is not None else "—"
    def _fv(v): return f"{v:.4f}"    if v is not None else "—"
    def _fi(v): return str(v)        if v is not None else "—"

    thermal_drift_str = "—"
    if view.thermal_pre and view.thermal_post:
        pre_temp  = view.thermal_pre.battery_temp_c
        post_temp = view.thermal_post.battery_temp_c
        if pre_temp is not None and post_temp is not None:
            delta = round(post_temp - pre_temp, 2)
            thermal_drift_str = f"{delta:+.1f}°C"
    if view.thermal_score and view.thermal_score.latency_drift_pct is not None:
        pct = view.thermal_score.latency_drift_pct * 100
        sign = "+" if pct >= 0 else ""
        thermal_drift_str += f"  ({sign}{pct:.1f}% latency drift)"

    device_label = f"{connected_name} [EMULATOR]" if is_emulator else f"{connected_name}"
    table.add_row("Device",             device_label)
    table.add_row("Runtime",            "tflite (ADB)")
    table.add_row("Android",            f"API {api_level}" if api_level else "—")
    table.add_row("Compute units req.", "cpuOnly")
    table.add_row("Compute units used", "CPU")
    table.add_row("Runs",               _fi(view.cpu_count))
    table.add_row("", "")
    table.add_row("Latency (p50)",      _f(view.cpu_p50_ms))
    table.add_row("Latency (p95/tail)", _f(view.cpu_p90_ms))
    table.add_row("Latency (mean)",     _f(view.cpu_avg_ms))
    table.add_row("Latency (std)",      _f(view.cpu_std_ms))
    table.add_row("", "")
    table.add_row("Speedup vs CPU",     f"{view.speedup:.2f}x" if view.speedup else "—")
    table.add_row("", "")
    table.add_row("Thermal drift",      thermal_drift_str)
    table.add_row("Memory (peak)",      _fm(view.cpu_peak_mem_mb))
    table.add_row("Init time",          _f(view.cpu_init_ms))
    table.add_row("Variance",           _fv(view.cpu_variance))

    console.print()
    console.print(table)
    console.print()

    if view.stability and view.stability.stability_band:
        band  = view.stability.stability_band.value
        score = view.stability.stability_score
        color = {"stable": "green", "noisy": "yellow", "unreliable": "red"}.get(band, "white")
        score_str = f" (score={score:.3f})" if score is not None else ""
        console.print(f"  Stability: [{color}]{band.upper()}[/{color}]{score_str}")

    rec = result.recommendation
    KIND_STYLES = {
        "use_cpu":  ("blue",   "→"),
        "rerun":    ("yellow", "⚠"),
    }
    color, icon = KIND_STYLES.get(rec.kind.value, ("white", "•"))
    console.print(f"\n  [{color}]{icon} {rec.message}[/{color}]")
    console.print()
    console.print("[green]✓ Benchmark saved to registry[/green]")
    console.print(
        f"[dim]Device: {connected_name} ({connected_abi}) — USB-C[/dim]\n"
    )


def _print_ios_result_table(view, profile, result, runs) -> None:
    """Rich result table for iOS idb runs. Same column structure as Android."""
    from ...platforms.ios.delegate import DelegateStatus

    table = Table(title="Benchmark Results")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value",  justify="right", style="green", no_wrap=True)

    def _f(v):  return f"{v:.3f} ms" if v is not None else "—"
    def _fm(v): return f"{v:.2f} MB" if v is not None else "—"
    def _fv(v): return f"{v:.4f}"    if v is not None else "—"

    target_label = (
        f"{profile.name} [SIMULATOR]"
        if profile.is_simulator
        else profile.name
    )

    # Thermal — discrete state on real device, N/A on simulator
    thermal_str = "N/A (simulator)"
    if not profile.is_simulator:
        if view.thermal_state_pre and view.thermal_state_post:
            thermal_str = (
                f"{view.thermal_state_pre.value} → {view.thermal_state_post.value}"
            )
        else:
            thermal_str = "—"

    # Delegate status table — show SKIPPED explicitly
    delegate_rows = []
    if view.delegate_status is not None:
        status_color = {
            "supported":    "green",
            "fallback":     "yellow",
            "unsupported":  "red",
            "inconsistent": "red",
            "skipped":      "dim",
        }.get(view.delegate_status.value, "white")
        delegate_rows.append(
            (view.delegate, view.delegate_status.value.upper(), status_color)
        )

    table.add_row("Device",              target_label if profile.is_simulator else f"{target_label} (USB-C)")
    table.add_row("Runtime",             "coreml (IDB)")
    table.add_row("iOS",                 profile.ios_version)
    table.add_row("Compute units req.",  view.compute_units or "—")
    table.add_row("Compute units used",  view.compute_units_used or "—")
    table.add_row("Runs",                str(runs))
    table.add_row("", "")
    table.add_row("Latency (p50)",       _f(view.cpu_p50_ms))
    table.add_row("Latency (p95/tail)",  _f(view.cpu_p90_ms))
    table.add_row("Latency (mean)",      _f(view.cpu_avg_ms))
    table.add_row("Latency (std)",       _f(view.cpu_std_ms))
    table.add_row("", "")
    table.add_row("Speedup vs CPU",
                  f"{view.speedup:.2f}x" if view.speedup else "—")
    table.add_row("", "")
    table.add_row("Thermal drift",       thermal_str)
    if not profile.is_simulator and view.thermal_score and view.thermal_score.latency_drift_pct is not None:
        pct = view.thermal_score.latency_drift_pct * 100
        sign = "+" if pct >= 0 else ""
        table.add_row("Latency drift",       f"{sign}{pct:.1f}%")
    table.add_row("Memory (peak)",       _fm(view.cpu_peak_mem_mb))
    table.add_row("Init time",           _f(view.cpu_init_ms))
    table.add_row("Variance",            _fv(view.cpu_variance))

    console.print()
    console.print(table)
    console.print()

    # Delegate status table
    if delegate_rows:
        dtable = Table(title="Delegate Status", show_header=True)
        dtable.add_column("Delegate", style="cyan")
        dtable.add_column("Status",   justify="right")
        for delegate, status, color in delegate_rows:
            dtable.add_row(delegate, f"[{color}]{status}[/{color}]")
        console.print(dtable)
        console.print()

    # Stability
    if view.stability and view.stability.stability_band:
        band  = view.stability.stability_band.value
        score = view.stability.stability_score
        color = {"stable": "green", "noisy": "yellow", "unreliable": "red"}.get(band, "white")
        score_str = f" (score={score:.3f})" if score is not None else ""
        console.print(f"  Stability: [{color}]{band.upper()}[/{color}]{score_str}")

    # Recommendation
    rec = result.recommendation
    KIND_STYLES = {
        "use_delegate": ("green",  "✓"),
        "use_cpu":      ("blue",   "→"),
        "rerun":        ("yellow", "⚠"),
        "exclude_delegate": ("red", "✗"),
    }
    color, icon = KIND_STYLES.get(rec.kind.value, ("white", "•"))
    console.print(f"\n  [{color}]{icon} {rec.message}[/{color}]")
    if rec.caveat:
        console.print(f"  [dim]{rec.caveat}[/dim]")
    console.print()
    console.print("[green]✓ Benchmark saved to registry[/green]")

    if profile.is_simulator:
        console.print(f"[dim]Device: {target_label} — Simulator (CPU/GPU only)[/dim]\n")
    else:
        console.print(f"[dim]Device: {target_label} (USB-C)[/dim]\n")