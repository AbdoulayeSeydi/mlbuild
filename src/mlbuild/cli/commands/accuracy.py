"""
mlbuild accuracy <baseline> <candidate>

Runs an output divergence check between two registered builds and
saves the result to the registry as an immutable audit artifact.

Usage
-----
mlbuild accuracy <baseline> <candidate>
mlbuild accuracy 3f36810e b8aa1ef6 --samples 64
mlbuild accuracy 3f36810e b8aa1ef6 --profile strict
mlbuild accuracy 3f36810e b8aa1ef6 --cosine-threshold 0.995
mlbuild accuracy 3f36810e b8aa1ef6 --mae-threshold 0.005
mlbuild accuracy 3f36810e b8aa1ef6 --rmse-threshold 0.01
mlbuild accuracy 3f36810e b8aa1ef6 --dataset ./inputs.npz
mlbuild accuracy 3f36810e b8aa1ef6 --task vision
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

console = Console(width=None)


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


def _threshold_label(value: float | None, threshold: float | None, mode: str = "min") -> str:
    """
    Render threshold string.
    mode='min' means value must be >= threshold (cosine, top1).
    mode='max' means value must be <= threshold (mae, rmse).
    """
    if threshold is None:
        return "—"
    prefix = "≥" if mode == "min" else "≤"
    return f"{prefix} {threshold:.6f}"


# ----------------------------------------------------------------
# Rich output
# ----------------------------------------------------------------

def _print_cross_format_notice(baseline, candidate) -> None:
    console.print(
        Panel(
            f"[bold cyan]cross-backend evaluation mode enabled[/bold cyan]\n"
            f"baseline:  [dim]{baseline.format}[/dim]  "
            f"candidate: [dim]{candidate.format}[/dim]\n"
            f"[dim]metrics are comparable — argmax and direction are "
            f"format-independent[/dim]",
            border_style="cyan",
            expand=False,
        )
    )
    console.print()


def _print_summary(
    baseline,
    candidate,
    result,
    config,
    context,
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
    if context.task_type:
        info.add_row("task_type", context.task_type)
    if context.dataset_path:
        info.add_row("dataset", str(context.dataset_path))
    if config.profile:
        info.add_row("profile", config.profile)

    console.print(info)

    # Metrics table
    metrics = Table(box=box.SIMPLE_HEAD, show_header=True, padding=(0, 2))
    metrics.add_column("metric",    style="bold")
    metrics.add_column("value",     justify="right")
    metrics.add_column("threshold", justify="right", style="dim")
    metrics.add_column("gate",      justify="center")

    # cosine_similarity — primary gate
    metrics.add_row(
        "cosine_similarity",
        _format_metric(result.cosine_similarity, 6),
        _threshold_label(result.cosine_similarity, config.cosine_threshold, "min"),
        _passed_label(result.cosine_similarity >= config.cosine_threshold),
    )

    # mean_abs_error — informational or gated
    mae_gate = config.mae_threshold is not None
    metrics.add_row(
        "mean_abs_error",
        _format_metric(result.mean_abs_error, 6),
        _threshold_label(result.mean_abs_error, config.mae_threshold, "max"),
        _passed_label(result.mean_abs_error <= config.mae_threshold)
        if mae_gate else "[dim]info[/dim]",
    )

    # rmse — optional gate
    rmse_gate = config.rmse_threshold is not None
    metrics.add_row(
        "rmse",
        _format_metric(result.rmse, 6),
        _threshold_label(result.rmse, config.rmse_threshold, "max"),
        _passed_label(result.rmse <= config.rmse_threshold)
        if rmse_gate else "[dim]info[/dim]",
    )

    # max_abs_error — diagnostic only
    metrics.add_row(
        "max_abs_error",
        _format_metric(result.max_abs_error, 6),
        "—",
        "[dim]diag[/dim]",
    )

    # error percentiles — diagnostic only, single row
    metrics.add_row(
        "error_p50/p95/p99",
        f"{_format_metric(result.error_p50, 6)}  "
        f"{_format_metric(result.error_p95, 6)}  "
        f"{_format_metric(result.error_p99, 6)}",
        "—",
        "[dim]diag[/dim]",
    )

    # top1_agreement — conditional gate, classifiers only
    if result.top1_agreement is not None:
        metrics.add_row(
            "top1_agreement",
            _format_metric(result.top1_agreement, 4),
            _threshold_label(result.top1_agreement, config.top1_threshold, "min"),
            _passed_label(result.top1_agreement >= config.top1_threshold),
        )

    # kl_divergence — diagnostic only, classifiers only
    if result.kl_divergence is not None:
        metrics.add_row(
            "kl_divergence",
            _format_metric(result.kl_divergence, 6),
            "—",
            "[dim]diag[/dim]",
        )

    # js_divergence — diagnostic only, classifiers only
    if result.js_divergence is not None:
        metrics.add_row(
            "js_divergence",
            _format_metric(result.js_divergence, 6),
            "[dim]≤ ln2 (0.693)[/dim]",
            "[dim]diag[/dim]",
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
@click.option("--samples",          default=32,    show_default=True,  help="Number of input samples.")
@click.option("--seed",             default=42,    show_default=True,  help="RNG seed for reproducibility.")
@click.option("--profile",          default=None,  type=click.Choice(["strict", "default", "loose"]), help="Threshold preset. Explicit flags override.")
@click.option("--cosine-threshold", default=None,  type=float,         help="Minimum cosine similarity to pass.")
@click.option("--top1-threshold",   default=None,  type=float,         help="Minimum top-1 agreement to pass (classifiers only).")
@click.option("--mae-threshold",    default=None,  type=float,         help="Optional MAE gate (informational by default).")
@click.option("--rmse-threshold",   default=None,  type=float,         help="Optional RMSE gate (informational by default).")
@click.option("--dataset",          default=None,  type=click.Path(),  help="Path to .npz or .npy file of real inputs.")
@click.option("--task",             default=None,  type=click.Choice(["vision", "nlp", "audio", "multimodal", "unknown"]), help="Override task type for sampling strategy.")
@click.option("--no-save",          is_flag=True,  default=False,      help="Skip persisting result to registry.")
@click.option("--json",  "as_json", is_flag=True,  default=False,      help="Print result as JSON and exit.")
@click.option("--ci",               is_flag=True,  default=False,      help="Exit 1 on failure, 0 on pass (for CI pipelines).")
@click.pass_context
def accuracy_command(
    ctx,
    baseline_id: str,
    candidate_id: str,
    samples: int,
    seed: int,
    profile: str | None,
    cosine_threshold: float | None,
    top1_threshold: float | None,
    mae_threshold: float | None,
    rmse_threshold: float | None,
    dataset: str | None,
    task: str | None,
    no_save: bool,
    as_json: bool,
    ci: bool,
) -> None:
    """Check output divergence between two builds.

    Runs SAMPLES inferences through BASELINE_ID and CANDIDATE_ID,
    computes cosine similarity, MAE, RMSE, max error, percentile breakdown,
    top-1 agreement, and KL/JS divergence (classifiers only), then saves
    the result to the registry as an immutable audit record.

    Cross-format comparison (e.g. CoreML vs TFLite) is supported.
    task_type is read from the baseline build; use --task to override.
    """
    from mlbuild.registry.local import LocalRegistry
    from mlbuild.core.accuracy.config import AccuracyConfig, EvaluationContext
    from mlbuild.core.accuracy.checker import run_accuracy_check

    try:
        registry = LocalRegistry()
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(2)

    baseline  = _resolve_or_exit(registry, baseline_id,  "baseline")
    candidate = _resolve_or_exit(registry, candidate_id, "candidate")

    # Cross-format: inform, don't block
    cross_format = baseline.format != candidate.format
    if cross_format and not as_json:
        _print_cross_format_notice(baseline, candidate)

    # Build config — profile seeds defaults, explicit flags override
    config = AccuracyConfig.from_cli(
        cosine_threshold=cosine_threshold or 0.99,
        top1_threshold=top1_threshold   or 0.99,
        mae_threshold=mae_threshold,
        rmse_threshold=rmse_threshold,
        samples=samples,
        seed=seed,
        profile=profile,
    )

    # Build context — task_type precedence: --task > build.task_type > None
    resolved_task = task or getattr(baseline, "task_type", None)
    context = EvaluationContext(
        baseline_build_id=baseline.build_id,
        candidate_build_id=candidate.build_id,
        task_type=resolved_task,
        dataset_path=dataset,
        cross_format=cross_format,
    )

    if not as_json:
        console.print(
            f"[dim]Running accuracy check  "
            f"{baseline.build_id[:16]} → {candidate.build_id[:16]}  "
            f"({samples} samples, seed={seed}"
            f"{f', task={resolved_task}' if resolved_task else ''})[/dim]"
        )

    try:
        baseline_runner  = _make_runner(baseline)
        candidate_runner = _make_runner(candidate)

        result = run_accuracy_check(
            baseline_runner,
            candidate_runner,
            config=config,
            context=context,
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
            "rmse":               result.rmse,
            "top1_agreement":     result.top1_agreement,
            "kl_divergence":      result.kl_divergence,
            "js_divergence":      result.js_divergence,
            "error_p50":          result.error_p50,
            "error_p95":          result.error_p95,
            "error_p99":          result.error_p99,
            "num_samples":        result.num_samples,
            "seed":               result.seed,
            "passed":             result.passed,
            "failure_reasons":    list(result.failure_reasons),
            "registry_row_id":    row_id,
        }
        click.echo(json.dumps(out, indent=2))
        sys.exit(0 if result.passed else 1)

    # Rich summary
    _print_summary(baseline, candidate, result, config, context, row_id, saved)

    sys.exit(0 if result.passed else 1)