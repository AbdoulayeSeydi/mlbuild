"""
mlbuild accuracy <baseline> <candidate>

Runs an output divergence check between two registered builds and
saves the result to the registry as an immutable audit artifact.

Usage
-----
mlbuild accuracy <baseline> <candidate>
mlbuild accuracy 3f36810e b8aa1ef6 --samples 64
mlbuild accuracy 3f36810e b8aa1ef6 --cosine-threshold 0.995
mlbuild accuracy 3f36810e b8aa1ef6 --mae-threshold 0.005
mlbuild accuracy 3f36810e b8aa1ef6 --json
mlbuild accuracy 3f36810e b8aa1ef6 --ci
mlbuild accuracy 3f36810e b8aa1ef6 --no-save

Exit codes
----------
0  passed
1  failed (accuracy gates)
2  error  (bad args, missing build, incompatible runners)
"""

from __future__ import annotations

import json
import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _resolve_or_exit(registry, identifier: str, label: str):
    build = registry.resolve_build(identifier)
    if build is None:
        console.print(f"[red]Error:[/red] {label} build '{identifier}' not found.")
        sys.exit(2)
    return build


def _make_runner(build):
    """Instantiate the correct runner for a build's format."""
    from mlbuild.benchmark.runner import CoreMLBenchmarkRunner, TFLiteBenchmarkRunner

    if build.format == "coreml":
        return CoreMLBenchmarkRunner(build.artifact_path)
    elif build.format == "tflite":
        return TFLiteBenchmarkRunner(build.artifact_path)
    else:
        console.print(f"[red]Error:[/red] Unsupported format '{build.format}' for accuracy check.")
        sys.exit(2)


def _format_metric(value: float | None, decimals: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f}"


def _passed_label(passed: bool) -> str:
    return "[green]PASS[/green]" if passed else "[red]FAIL[/red]"


# ----------------------------------------------------------------
# Rich output
# ----------------------------------------------------------------

def _print_summary(
    baseline,
    candidate,
    result,
    row_id: int | None,
    saved: bool,
) -> None:
    console.print()

    # Header panel
    status = "PASS" if result.passed else "FAIL"
    color  = "green" if result.passed else "red"
    console.print(
        Panel(
            f"[bold {color}]{status}[/bold {color}]  "
            f"[dim]{baseline.build_id[:16]}[/dim] → "
            f"[dim]{candidate.build_id[:16]}[/dim]",
            title="[bold]mlbuild accuracy[/bold]",
            border_style=color,
            expand=False,
        )
    )

    # Build info table
    info = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    info.add_column("key",   style="dim")
    info.add_column("value")

    info.add_row("baseline",  f"{baseline.name}  [dim]{baseline.build_id[:16]}[/dim]")
    info.add_row(
        "",
        f"[dim]{baseline.format}  "
        f"{getattr(baseline, 'optimization_method', None) or 'fp32'}  "
        f"{baseline.size_mb:.1f} MB[/dim]",
    )
    info.add_row("candidate", f"{candidate.name}  [dim]{candidate.build_id[:16]}[/dim]")
    info.add_row(
        "",
        f"[dim]{candidate.format}  "
        f"{getattr(candidate, 'optimization_method', None) or 'fp32'}  "
        f"{candidate.size_mb:.1f} MB[/dim]",
    )
    info.add_row("samples",   str(result.num_samples))
    info.add_row("seed",      str(result.seed))

    console.print(info)

    # Metrics table
    metrics = Table(box=box.SIMPLE_HEAD, show_header=True, padding=(0, 2))
    metrics.add_column("metric",            style="bold")
    metrics.add_column("value",             justify="right")
    metrics.add_column("threshold",         justify="right", style="dim")
    metrics.add_column("gate",              justify="center")

    # cosine_similarity
    metrics.add_row(
        "cosine_similarity",
        _format_metric(result.cosine_similarity, 6),
        "≥ 0.990000",
        _passed_label(result.cosine_similarity >= 0.99),
    )

    # mean_abs_error — informational
    metrics.add_row(
        "mean_abs_error",
        _format_metric(result.mean_abs_error, 6),
        "—",
        "[dim]info[/dim]",
    )

    # max_abs_error — diagnostic only
    metrics.add_row(
        "max_abs_error",
        _format_metric(result.max_abs_error, 6),
        "—",
        "[dim]diag[/dim]",
    )

    # top1_agreement — only shown when active
    if result.top1_agreement is not None:
        metrics.add_row(
            "top1_agreement",
            _format_metric(result.top1_agreement, 4),
            "≥ 0.9900",
            _passed_label(result.top1_agreement >= 0.99),
        )

    console.print(metrics)

    # Failure reasons
    if result.failure_reasons:
        console.print("[bold red]Failures:[/bold red]")
        for reason in result.failure_reasons:
            console.print(f"  [red]✗[/red] {reason}")
        console.print()

    # Persistence note
    if saved and row_id is not None:
        console.print(f"[dim]saved to registry  id={row_id}[/dim]")
    else:
        console.print("[dim]not saved  (--no-save)[/dim]")

    console.print()


# ----------------------------------------------------------------
# Command
# ----------------------------------------------------------------

@click.command("accuracy")
@click.argument("baseline_id")
@click.argument("candidate_id")
@click.option("--samples",          default=32,   show_default=True, help="Number of random input samples.")
@click.option("--seed",             default=42,   show_default=True, help="RNG seed for reproducibility.")
@click.option("--cosine-threshold", default=0.99, show_default=True, type=float, help="Minimum cosine similarity to pass.")
@click.option("--top1-threshold",   default=0.99, show_default=True, type=float, help="Minimum top-1 agreement to pass (classifiers only).")
@click.option("--mae-threshold",    default=None, type=float,         help="Optional MAE gate (informational by default).")
@click.option("--no-save",          is_flag=True, default=False,      help="Skip persisting result to registry.")
@click.option("--json",  "as_json", is_flag=True, default=False,      help="Print result as JSON and exit.")
@click.option("--ci",               is_flag=True, default=False,      help="Exit 1 on failure, 0 on pass (for CI pipelines).")
@click.pass_context
def accuracy_command(
    ctx,
    baseline_id: str,
    candidate_id: str,
    samples: int,
    seed: int,
    cosine_threshold: float,
    top1_threshold: float,
    mae_threshold: float | None,
    no_save: bool,
    as_json: bool,
    ci: bool,
) -> None:
    """Check output divergence between two builds.

    Runs SAMPLES random inferences through both BASELINE_ID and CANDIDATE_ID,
    computes cosine similarity, MAE, max absolute error, and top-1 agreement,
    then saves the result to the registry as an immutable audit record.
    """
    from mlbuild.registry.local import LocalRegistry
    from mlbuild.core.accuracy.config import AccuracyConfig
    from mlbuild.core.accuracy.checker import run_accuracy_check

    try:
        registry = LocalRegistry()
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(2)

    baseline  = _resolve_or_exit(registry, baseline_id,  "baseline")
    candidate = _resolve_or_exit(registry, candidate_id, "candidate")

    if baseline.format != candidate.format:
        console.print(
            f"[red]Error:[/red] format mismatch — "
            f"baseline is '{baseline.format}', candidate is '{candidate.format}'."
        )
        sys.exit(2)

    config = AccuracyConfig(
        samples=samples,
        seed=seed,
        cosine_threshold=cosine_threshold,
        top1_threshold=top1_threshold,
        mae_threshold=mae_threshold,
    )

    if not as_json:
        console.print(
            f"[dim]Running accuracy check  "
            f"{baseline.build_id[:16]} → {candidate.build_id[:16]}  "
            f"({samples} samples, seed={seed})[/dim]"
        )

    try:
        baseline_runner  = _make_runner(baseline)
        candidate_runner = _make_runner(candidate)

        result = run_accuracy_check(
            baseline_runner,
            candidate_runner,
            config=config,
            baseline_build_id=baseline.build_id,
            candidate_build_id=candidate.build_id,
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(2)
    except Exception as exc:
        console.print(f"[red]Error:[/red] Accuracy check failed: {exc}")
        sys.exit(2)

    # Persist — always, unless --no-save
    row_id = None
    saved  = False
    if not no_save:
        try:
            row_id = registry.save_accuracy_check(result)
            saved  = True
        except Exception as exc:
            console.print(f"[yellow]Warning:[/yellow] Failed to save result: {exc}")

    # JSON output
    if as_json:
        out = {
            "baseline_build_id":  result.baseline_build_id,
            "candidate_build_id": result.candidate_build_id,
            "cosine_similarity":  result.cosine_similarity,
            "mean_abs_error":     result.mean_abs_error,
            "max_abs_error":      result.max_abs_error,
            "top1_agreement":     result.top1_agreement,
            "num_samples":        result.num_samples,
            "seed":               result.seed,
            "passed":             result.passed,
            "failure_reasons":    list(result.failure_reasons),
            "registry_row_id":    row_id,
        }
        click.echo(json.dumps(out, indent=2))
        sys.exit(0 if result.passed else 1)

    # Rich summary
    _print_summary(baseline, candidate, result, row_id, saved)

    # CI / default exit code
    if ci or True:  # always exit with meaningful code
        sys.exit(0 if result.passed else 1)