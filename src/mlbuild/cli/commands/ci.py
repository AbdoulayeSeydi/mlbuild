"""
mlbuild ci — full CI orchestration command.

Usage:
    mlbuild ci --model mobilenet.onnx --baseline main-mobilenet
    mlbuild ci --build <build_id> --baseline main-mobilenet
    mlbuild ci --model mobilenet.onnx --baseline main-mobilenet --latency-regression 15
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


@click.command(name="ci")
@click.option(
    "--model",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="ONNX model path. Runs explore to produce candidates.",
)
@click.option(
    "--build",
    "build_id",
    default=None,
    help="Existing build ID to use as candidate. Mutually exclusive with --model.",
)
@click.option(
    "--baseline",
    required=True,
    help="Tag name or build ID of the baseline to compare against.",
)
@click.option(
    "--target",
    default="auto",
    help="Device target for explore (default: auto-detect).",
)
@click.option(
    "--latency-regression",
    "latency_regression_pct",
    default=None,
    type=float,
    help="Max allowed latency regression % (overrides config.toml).",
)
@click.option(
    "--size-regression",
    "size_regression_pct",
    default=None,
    type=float,
    help="Max allowed size regression % (overrides config.toml).",
)
@click.option(
    "--latency-budget",
    "latency_budget_ms",
    default=None,
    type=float,
    help="Hard latency cap in ms, independent of baseline.",
)
@click.option(
    "--size-budget",
    "size_budget_mb",
    default=None,
    type=float,
    help="Hard size cap in MB, independent of baseline.",
)
@click.option(
    "--dataset",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Calibration data for accuracy check (images dir, .npy dir, or .npz).",
)
@click.option(
    "--cosine-threshold",
    default=None,
    type=float,
    help="Minimum cosine similarity for accuracy check.",
)
@click.option(
    "--top1-threshold",
    default=None,
    type=float,
    help="Minimum top-1 agreement for accuracy check.",
)
@click.option(
    "--fail-on-missing-baseline",
    is_flag=True,
    default=False,
    help="Exit 1 if baseline tag not found (default: warn and exit 0).",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    default=False,
    help="Print JSON report to stdout.",
)
def ci(
    model: Optional[Path],
    build_id: Optional[str],
    baseline: str,
    target: str,
    latency_regression_pct: Optional[float],
    size_regression_pct: Optional[float],
    latency_budget_ms: Optional[float],
    size_budget_mb: Optional[float],
    dataset: Optional[Path],
    cosine_threshold: Optional[float],
    top1_threshold: Optional[float],
    fail_on_missing_baseline: bool,
    output_json: bool,
):
    """Run full CI performance check against a registered baseline."""

    # Validate mutual exclusion
    if model and build_id:
        raise click.UsageError("--model and --build are mutually exclusive")
    if not model and not build_id:
        raise click.UsageError("Either --model or --build is required")

    from pathlib import Path as _Path
    workspace_root = _Path.cwd()

    from ...registry.local import LocalRegistry
    from ...core.ci.thresholds import ThresholdConfig
    from ...core.ci.runner import CIRunner, CIError

    registry = LocalRegistry()

    # Load thresholds from config.toml then apply CLI overrides
    thresholds = ThresholdConfig.from_workspace(workspace_root).apply_overrides(
        latency_regression_pct=latency_regression_pct,
        size_regression_pct=size_regression_pct,
        latency_budget_ms=latency_budget_ms,
        size_budget_mb=size_budget_mb,
        cosine_threshold=cosine_threshold,
        top1_threshold=top1_threshold,
    )

    runner = CIRunner(
        registry=registry,
        thresholds=thresholds,
        workspace_root=workspace_root,
    )

    if not output_json:
        console.print(f"\n[bold]MLBuild CI[/bold]")
        console.print(f"Baseline:  {baseline}")
        if model:
            console.print(f"Model:     {model.name}")
        else:
            console.print(f"Build:     {build_id[:16]}...")
        console.print()

    try:
        report = runner.run(
            baseline_ref=baseline,
            model_path=model,
            build_id=build_id,
            target=target,
            dataset=dataset,
            fail_on_missing_baseline=fail_on_missing_baseline,
        )
    except CIError as e:
        if output_json:
            import json
            print(json.dumps({"result": "fail", "message": str(e)}))
        else:
            console.print(f"[red]CI Failed: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        logger.exception("CI command failed")
        if output_json:
            import json
            print(json.dumps({"result": "error", "message": str(e)}))
        else:
            console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(2)

    if output_json:
        print(report.to_json())
    else:
        console.print(report.to_text())

    # result can be "pass", "fail", or "skip"
    if report.result == "pass" or report.result == "skip":
        sys.exit(0)
    else:
        sys.exit(1)
