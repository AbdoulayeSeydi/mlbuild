"""
MLBuild CLI entry point with lazy command loading.

Design: Commands are loaded on-demand to prevent import errors
from breaking the entire CLI.
"""

import sys
import click
from rich.console import Console
from .commands.remote import remote
from .commands.push import push
from .commands.pull import pull
from .commands.sync import sync

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="mlbuild")
def cli():
    """
    MLBuild - Deterministic build system for CoreML models.
    
    Track model performance and prevent regressions in CI.
    """
    pass


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
    type=click.Choice(["coreml", "onnxruntime"]),
    help="Backend to use for conversion"
)
@click.option(
    "--target",
    required=True,
    type=click.Choice(["apple_a17", "apple_a16", "apple_a15", "apple_m3", "apple_m2", "apple_m1"]),
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
def log(build_id, limit, offset, as_json, csv_path, show_hashes, show_notes, full_id, full_hashes, target, name, tag, date_from, date_to):
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
    )


cli.add_command(remote)
cli.add_command(push)
cli.add_command(pull)
cli.add_command(sync)


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
def benchmark(build_id, runs, warmup, compute_unit, as_json):
    """Benchmark a build on current device."""
    from .commands.benchmark import benchmark as benchmark_cmd
    ctx = click.get_current_context()
    ctx.invoke(
        benchmark_cmd,
        build_id=build_id,
        runs=runs,
        warmup=warmup,
        compute_unit=compute_unit,
        as_json=as_json,
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
def compare_quantization(build_ids, runs, warmup):
    """Compare different quantization levels (FP32/FP16/INT8)."""
    from .commands.compare_quantization import compare_quantization as cmd
    ctx = click.get_current_context()
    ctx.invoke(cmd, build_ids=build_ids, runs=runs, warmup=warmup)

@cli.command()
@click.argument('build_id')
@click.option('--runs', default=50, type=int)
@click.option('--warmup', default=10, type=int)
@click.option('--top', default=15, type=int)
@click.option('--analyze-warmup', is_flag=True)
def profile(build_id, runs, warmup, top, analyze_warmup):
    """Profile model layer-by-layer."""
    from .commands.profile import profile as profile_cmd
    ctx = click.get_current_context()
    ctx.invoke(profile_cmd, build_id=build_id, runs=runs, warmup=warmup, top=top, analyze_warmup=analyze_warmup)

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
def validate(build_id, max_latency, max_p95, max_memory, max_size, runs, warmup, compute_unit, ci):
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
    )

@cli.command()
@click.argument('baseline_id')
@click.argument('candidate_id')
@click.option('--threshold', default=5.0, type=float)
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
def compare(baseline_id, candidate_id, threshold, metric, compute_unit, runs, warmup, as_json, ci):
    """Detect performance regressions between two builds."""
    from .commands.compare import compare as compare_cmd
    ctx = click.get_current_context()
    ctx.invoke(
        compare_cmd,
        baseline_id=baseline_id,
        candidate_id=candidate_id,
        threshold=threshold,
        metric=metric,
        compute_unit=compute_unit,
        runs=runs,
        warmup=warmup,
        as_json=as_json,
        ci=ci,
    )


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

if __name__ == "__main__":
    cli()