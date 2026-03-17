"""
MLBuild CLI entry point with lazy command loading.

Design: Commands are loaded on-demand to prevent import errors
from breaking the entire CLI.
"""

import click
import os
import sys
from pathlib import Path

# Suppress coremltools version warnings on startup
_devnull = open(os.devnull, 'w')
_old_stderr = sys.stderr
sys.stderr = _devnull
try:
    import coremltools
except Exception:
    pass
finally:
    sys.stderr = _old_stderr
    _devnull.close()

from rich.console import Console
import platform as _platform

def _is_macos() -> bool:
    return _platform.system() == "Darwin"

def _require_macos(command_name: str) -> None:
    """Print clear error if CoreML command runs on non-macOS."""
    if not _is_macos():
        Console().print(
            f"\n[red]{command_name} requires macOS (CoreML is Apple-only).[/red]\n"
            f"[dim]TFLite commands work on all platforms.[/dim]\n"
        )
        import sys; sys.exit(1)
from .commands.remote import remote
from .commands.push import push
from .commands.pull import pull
from .commands.sync import sync
from .commands.compare import ci_check
from .commands.report import report


console = Console()


@click.group()
@click.version_option(version="0.2.0", prog_name="mlbuild")
@click.option(
    "--strict-output",
    is_flag=True,
    default=False,
    help="Globally enable strict output validation (promotes warnings to failures).",
)
@click.pass_context
def cli(ctx, strict_output):
    """
    MLBuild - Deterministic build system for CoreML models.

    Track model performance and prevent regressions in CI.
    """
    import time
    ctx.ensure_object(dict)
    ctx.obj["strict_output"] = strict_output
    ctx.obj["_start_ms"] = time.monotonic() * 1000

    def _record():
        try:
            import uuid
            import json
            from datetime import datetime, timezone

            raw_args = sys.argv[1:]
            if not raw_args:
                return

            command_name = raw_args[0].lstrip("-")

            if command_name in _EXCLUDED_FROM_HISTORY:
                return

            duration_ms = int(time.monotonic() * 1000 - ctx.obj["_start_ms"])

            args_dict = {}
            i = 1
            while i < len(raw_args):
                arg = raw_args[i]
                if arg.startswith("--"):
                    key = arg.lstrip("-")
                    if i + 1 < len(raw_args) and not raw_args[i + 1].startswith("--"):
                        val = raw_args[i + 1]
                        if len(val) < 200:
                            args_dict[key] = val
                        i += 2
                    else:
                        args_dict[key] = True
                        i += 1
                else:
                    i += 1

            from ..core.machine import get_machine_info
            machine = get_machine_info()

            from .. import __version__ as mlbuild_version
            from ..registry.local import LocalRegistry
            from datetime import datetime, timezone

            row = {
                "id":                  str(uuid.uuid4()),
                "machine_id":          machine["machine_id"],
                "machine_name":        machine["machine_name"],
                "platform":            machine["platform"],
                "command_name":        command_name,
                "args_json":           json.dumps(args_dict, sort_keys=True),
                "raw_command":         "mlbuild " + " ".join(raw_args),
                "linked_build_id":     ctx.obj.get("_linked_build_id"),
                "linked_benchmark_id": ctx.obj.get("_linked_benchmark_id"),
                "exit_code":           ctx.obj.get("_exit_code", 0),
                "error_message":       ctx.obj.get("_error_message"),
                "duration_ms":         duration_ms,
                "mlbuild_version":     mlbuild_version,
                "ran_at":              datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }

            registry = LocalRegistry()
            registry.save_command(row)

        except Exception:
            pass

    ctx.call_on_close(_record)


# --- PATCH: helper for commands to merge global + local strict flag ---
def _resolve_strict(ctx: click.Context, command_flag: bool) -> bool:
    """
    Command-level --strict-output always wins if set.
    Falls back to global flag from ctx.obj, then False.
    """
    if command_flag:
        return True
    obj = ctx.find_root().obj or {}
    return bool(obj.get("strict_output", False))

# ------------------------------------------------------------
# Auto-instrumentation — fires after every command completes
# ------------------------------------------------------------

# Commands that are non-mutating introspection only.
# Logging these creates noise in history.
_EXCLUDED_FROM_HISTORY = frozenset({
    "history",
    "doctor",
    "version",
    "help",
})


# Lazy-load commands to prevent import errors from breaking the CLI
@cli.command()
@click.option(
    "--model",
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "--backend",
    default="coreml",
    type=click.Choice(["coreml", "tflite"]), 
    help="Backend to use for conversion"
)
@click.option(
    "--target",
    required=True,
    type=click.Choice([
        # Apple
        "apple_a17", "apple_a16", "apple_a15", 
        "apple_m3", "apple_m2", "apple_m1",
        # Android
        "android_arm64", "android_arm32", "android_x86",
        # Edge
        "raspberry_pi", "coral_tpu", "generic_linux",
    ]),  
)
@click.option("--name")
@click.option(
    "--quantize",
    type=click.Choice(["fp32", "fp16", "int8"]),
    default="fp32",
)
@click.option("--notes")
def build(model, backend, target, name, quantize, notes):
    """Build model using specified backend."""
    from .commands.build import build as build_cmd
    from ..backends.registry import BackendRegistry

    try:
        backend_inst = BackendRegistry.get_backend(backend)
    except (ValueError, RuntimeError) as e:
        console.print(f"\n[red]Backend Error:[/red] {e}\n")
        sys.exit(1)

    ctx = click.get_current_context()
    ctx.invoke(build_cmd, model=model, backend=backend, target=target, name=name, quantize=quantize, notes=notes)


@cli.command()
@click.argument('build_id', required=False)
@click.option('--limit', default=50, type=int)
@click.option('--offset', default=0, type=int)
@click.option('--json', 'as_json', is_flag=True)
@click.option('--csv', 'csv_path', type=str)
@click.option('--show-hashes', is_flag=True)
@click.option('--show-notes', is_flag=True)
@click.option('--full-id', is_flag=True)
@click.option('--full-hashes', is_flag=True)
@click.option('--target', default=None)
@click.option('--name', default=None)
@click.option('--tag', default=None)
@click.option('--date-from', default=None)
@click.option('--date-to', default=None)
@click.option('--task', default=None,
              type=click.Choice(['vision', 'nlp', 'audio', 'multimodal', 'unknown']))
@click.option('--format', 'fmt', default=None,
              type=click.Choice(['coreml', 'tflite']))
@click.option('--roots-only', is_flag=True, default=False)
@click.option('--source', default=None)
@click.option('--tree', is_flag=True, default=False)
def log(build_id, limit, offset, as_json, csv_path, show_hashes, show_notes,
        full_id, full_hashes, target, name, tag, date_from, date_to, task,
        fmt, roots_only, source, tree):
    """Show build history."""
    from .commands.log import log as log_cmd
    ctx = click.get_current_context()
    ctx.invoke(
        log_cmd,
        build_id=build_id,
        limit=limit,
        offset=offset,
        as_json=as_json,
        csv_path=csv_path,
        show_hashes=show_hashes,
        show_notes=show_notes,
        full_id=full_id,
        full_hashes=full_hashes,
        target=target,
        name=name,
        tag=tag,
        date_from=date_from,
        date_to=date_to,
        task=task,
        fmt=fmt,
        roots_only=roots_only,
        source=source,
        tree=tree,
    )


cli.add_command(remote)
cli.add_command(push)
cli.add_command(pull)
cli.add_command(sync)
cli.add_command(ci_check, name="ci-check")
cli.add_command(report)


@cli.command()
@click.argument('build_a')
@click.argument('build_b')
@click.option('--json', 'as_json', is_flag=True)
@click.option('--ignore-size', is_flag=True)
@click.option('--ignore-quant', is_flag=True)
@click.option('--deep', is_flag=True)
def diff(build_a, build_b, as_json, ignore_size, ignore_quant, deep):
    """Compare two builds."""
    from .commands.diff import diff as diff_cmd
    ctx = click.get_current_context()
    ctx.invoke(
        diff_cmd,
        build_a=build_a,
        build_b=build_b,
        as_json=as_json,
        ignore_size=ignore_size,
        ignore_quant=ignore_quant,
        deep=deep,
    )

@cli.command()
@click.argument('build_id')
@click.option('--runs', default=100, type=int)
@click.option('--warmup', default=20, type=int)
@click.option('--compute-unit', 
              type=click.Choice(['CPU_ONLY', 'CPU_AND_GPU', 'ALL']),
              default='ALL')
@click.option('--json', 'as_json', is_flag=True)
# --- PATCH: task + strict-output forwarded ---
@click.option('--task', default=None,
              type=click.Choice(['vision', 'nlp', 'audio', 'unknown']))
@click.option('--strict-output', 'strict_output', is_flag=True, default=False)
def benchmark(build_id, runs, warmup, compute_unit, as_json, task, strict_output):
    """Benchmark a build on current device. Supports CoreML and TFLite formats."""
    from .commands.benchmark import benchmark as benchmark_cmd
    ctx = click.get_current_context()
    ctx.invoke(
        benchmark_cmd,
        build_id=build_id,
        runs=runs,
        warmup=warmup,
        compute_unit=compute_unit,
        as_json=as_json,
        task=task,
        strict_output=_resolve_strict(ctx, strict_output),
    )

@cli.command()
@click.argument('build_id')
@click.option('--runs', default=50, type=int)
@click.option('--warmup', default=10, type=int)
def compare_compute_units(build_id, runs, warmup):
    """Compare model performance across CPU/GPU/Neural Engine."""
    from .commands.compare_compute_units import compare_compute_units as cmd
    ctx = click.get_current_context()
    ctx.invoke(cmd, build_id=build_id, runs=runs, warmup=warmup)

@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output results in JSON format")
@click.option("--soft", is_flag=True, help="Do not exit with error code on missing optional tools")
def doctor(as_json, soft):
    """Check MLBuild environment."""
    from .commands.doctor import doctor as doctor_cmd
    ctx = click.get_current_context()
    ctx.invoke(doctor_cmd, as_json=as_json, soft=soft)

@cli.command()
@click.argument('build_ids', nargs=-1, required=True)
@click.option('--runs', default=50, type=int)
@click.option('--warmup', default=10, type=int)
@click.option('--compute-unit', default='all', type=click.Choice(['all', 'cpu', 'ane', 'gpu']))
@click.option('--baseline', default=None)
@click.option('--json-output', is_flag=True)
def compare_quantization(build_ids, runs, warmup, compute_unit, baseline, json_output):
    """Compare different quantization levels (FP32/FP16/INT8)."""
    from .commands.compare_quantization import compare_quantization as cmd
    ctx = click.get_current_context()
    ctx.invoke(cmd, build_ids=build_ids, runs=runs, warmup=warmup,
               compute_unit=compute_unit, baseline=baseline, json_output=json_output)

@cli.command()
@click.argument('build_id')
@click.option('--runs',            default=50,  type=int)
@click.option('--warmup',          default=10,  type=int)
@click.option('--top',             default=15,  type=int)
@click.option('--deep',            is_flag=True)
@click.option('--int8-build',      default=None)
@click.option('--analyze-warmup',  is_flag=True)
@click.option('--cold-start',      is_flag=True)
@click.option('--memory',          is_flag=True)
@click.option('--cold-start-runs', default=60,  type=int)
@click.option('--quant-samples',   default=50,  type=int)
# --- PATCH: task + strict-output forwarded ---
@click.option('--task', default=None,
              type=click.Choice(['vision', 'nlp', 'audio', 'unknown']))
@click.option('--seq-len', 'seq_len', default=128, type=int)
@click.option('--strict-output', 'strict_output', is_flag=True, default=False)
def profile(build_id, runs, warmup, top, deep, int8_build, analyze_warmup,
            cold_start, memory, cold_start_runs, quant_samples,
            task, seq_len, strict_output):
    """Profile model performance. Use --deep for full per-layer analysis (TFLite)."""
    from .commands.profile import profile as cmd
    ctx = click.get_current_context()
    ctx.invoke(
        cmd,
        build_id=build_id, runs=runs, warmup=warmup, top=top,
        deep=deep, int8_build=int8_build,
        analyze_warmup=analyze_warmup, cold_start=cold_start,
        memory=memory, cold_start_runs=cold_start_runs,
        quant_samples=quant_samples,
        task=task,
        seq_len=seq_len,
        strict_output=_resolve_strict(ctx, strict_output),
    )

@cli.command()
@click.argument('build_id')
@click.option('--max-latency', type=float)
@click.option('--max-p95', type=float)
@click.option('--max-memory', type=float)
@click.option('--max-size', type=float)
@click.option('--runs', default=50, type=int)
@click.option('--warmup', default=10, type=int)
@click.option('--compute-unit', default='all', type=click.Choice(['all', 'cpu', 'gpu']))
@click.option('--ci', is_flag=True)
@click.option('--dataset', default=None, type=click.Path(exists=True, path_type=Path))
@click.option('--baseline-id', default=None)
@click.option('--cosine-threshold', default=0.99, type=float)
@click.option('--top1-threshold', default=0.99, type=float)
@click.option('--accuracy-samples', default=200, type=int)
# --- PATCH: task + strict-output forwarded ---
@click.option('--task', default=None,
              type=click.Choice(['vision', 'nlp', 'audio', 'unknown']))
@click.option('--strict-output', 'strict_output', is_flag=True, default=False)
def validate(build_id, max_latency, max_p95, max_memory, max_size, runs, warmup,
             compute_unit, ci, dataset, baseline_id, cosine_threshold,
             top1_threshold, accuracy_samples, task, strict_output):
    """Validate build against constraints (CI-ready)."""
    from .commands.validate import validate as validate_cmd
    ctx = click.get_current_context()
    ctx.invoke(
        validate_cmd,
        build_id=build_id,
        max_latency=max_latency,
        max_p95=max_p95,
        max_memory=max_memory,
        max_size=max_size,
        runs=runs,
        warmup=warmup,
        compute_unit=compute_unit,
        ci=ci,
        task=task,
        strict_output=_resolve_strict(ctx, strict_output),
        dataset=dataset,
        baseline_id=baseline_id,
        cosine_threshold=cosine_threshold,
        top1_threshold=top1_threshold,
        accuracy_samples=accuracy_samples,
    )

@cli.command()
@click.argument('baseline_id')
@click.argument('candidate_id')
@click.option('--threshold', default=5.0, type=float)
@click.option('--size-threshold', default=5.0, type=float, help='Size regression threshold % (default: 5%)')
@click.option('--metric', 
              type=click.Choice(['p50', 'p95', 'p99', 'mean']),
              default='p50')
@click.option('--compute-unit',
              type=click.Choice(['CPU_ONLY', 'CPU_AND_GPU', 'ALL']),
              default='ALL')
@click.option('--runs', default=100, type=int)
@click.option('--warmup', default=20, type=int)
@click.option('--json', 'as_json', is_flag=True)
@click.option('--ci', is_flag=True)
# --- PATCH: task forwarded ---
@click.option('--task', default=None,
              type=click.Choice(['vision', 'nlp', 'audio', 'unknown']))
@click.option('--use-cached', is_flag=True, default=False,
              help='Use cached benchmark results from registry instead of re-running.')
@click.option('--check-accuracy', is_flag=True, default=False,
              help='Run output divergence check after latency comparison.')
@click.option('--accuracy-samples', default=32, type=int, show_default=True)
@click.option('--accuracy-seed', default=42, type=int, show_default=True)
@click.option('--cosine-threshold', default=0.99, type=float, show_default=True)
@click.option('--top1-threshold', default=0.99, type=float, show_default=True)
@click.option('--accuracy-mae-threshold', default=None, type=float)
def compare(baseline_id, candidate_id, threshold, size_threshold, metric,
            compute_unit, runs, warmup, as_json, ci, task, use_cached,
            check_accuracy, accuracy_samples, accuracy_seed,
            cosine_threshold, top1_threshold, accuracy_mae_threshold):
    """Compare two builds and detect regressions in latency and model size. Supports CoreML and TFLite."""
    from .commands.compare import compare as compare_cmd
    ctx = click.get_current_context()
    exit_code = ctx.invoke(
        compare_cmd,
        baseline_id=baseline_id,
        candidate_id=candidate_id,
        threshold=threshold,
        size_threshold=size_threshold,
        metric=metric,
        compute_unit=compute_unit,
        runs=runs,
        warmup=warmup,
        as_json=as_json,
        ci=ci,
        task=task,
        use_cached=use_cached,
        check_accuracy=check_accuracy,
        accuracy_samples=accuracy_samples,
        accuracy_seed=accuracy_seed,
        cosine_threshold=cosine_threshold,
        top1_threshold=top1_threshold,
        accuracy_mae_threshold=accuracy_mae_threshold,
    )
    sys.exit(exit_code if exit_code is not None else 0)

@cli.command()
@click.option('--force', is_flag=True, help='Reinitialize even if already exists')
def init(force):
    """Initialize MLBuild workspace."""
    import logging
    from pathlib import Path
    from ..registry import LocalRegistry

    logger = logging.getLogger(__name__)

    mlbuild_dir = Path.cwd() / ".mlbuild"

    if mlbuild_dir.exists() and not force:
        console.print("\n[yellow]⚠️  .mlbuild directory already exists[/yellow]")
        console.print(f"   Location: {mlbuild_dir}")
        console.print("\n   Use --force to reinitialize\n")
        return

    console.print("\n[bold]Initializing MLBuild workspace...[/bold]\n")

    # Create directory structure
    mlbuild_dir.mkdir(exist_ok=True)
    (mlbuild_dir / "artifacts").mkdir(exist_ok=True)
    (mlbuild_dir / "temp").mkdir(exist_ok=True)

    # Create .gitignore
    gitignore_path = mlbuild_dir / ".gitignore"
    gitignore_path.write_text("artifacts/\ntemp/\n*.db-journal\n*.db-wal\n*.db-shm\n")

    # Touch the DB file so LocalRegistry.__init__ passes its existence check,
    # then call init_schema() to build the actual schema.
    db_path = mlbuild_dir / "registry.db"
    db_path.touch()
    registry = LocalRegistry(db_path=db_path)
    registry.init_schema()

    console.print(f"✓ Created .mlbuild directory")
    console.print(f"✓ Created artifacts directory")
    console.print(f"✓ Created .gitignore")
    console.print(f"✓ Initialized registry database")
    console.print(f"\n[bold green]MLBuild workspace initialized![/bold green]")
    console.print(f"\nNext steps:")
    console.print(f"  1. Run: mlbuild doctor")
    console.print(f"  2. Build: mlbuild build --model <model.onnx> --target <device>\n")


from .commands.tag import tag as tag_group
cli.add_command(tag_group)

from .commands.experiment import experiment as experiment_group
cli.add_command(experiment_group)

from .commands.run import run as run_group
cli.add_command(run_group)

from .commands.optimize import optimize as optimize_command
cli.add_command(optimize_command)

from .commands.explore import explore as explore_command
cli.add_command(explore_command)

from .commands.ci import ci as ci_command
cli.add_command(ci_command)

from mlbuild.cli.commands.accuracy import accuracy_command
cli.add_command(accuracy_command)

from .commands.history import history as history_group
cli.add_command(history_group)

if __name__ == "__main__":
    cli()