"""
mlbuild ci — full CI orchestration command.

Usage:
    mlbuild ci --model mobilenet.onnx --baseline main-mobilenet
    mlbuild ci --build <build_id> --baseline main-mobilenet
    mlbuild ci --model mobilenet.onnx --baseline main-mobilenet --latency-regression 15
"""

from __future__ import annotations

import json
import sys
import logging
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)

# Exit code contract:
#   0 = pass / skip
#   1 = threshold violation (CI fail)
#   2 = user / config error (bad flags, missing file, registry down)
#   3 = unexpected internal crash

_EXIT_FAIL = 1
_EXIT_CONFIG = 2
_EXIT_CRASH = 3


def _die(output_json: bool, status: str, message: str, code: int) -> None:
    """Emit a structured error and exit."""
    if output_json:
        print(json.dumps({"status": status, "message": message}))
    else:
        console.print(f"[red]{message}[/red]")
    sys.exit(code)


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

    # ── mutual exclusion ────────────────────────────────────────────────────
    if model and build_id:
        raise click.UsageError("--model and --build are mutually exclusive")
    if not model and not build_id:
        raise click.UsageError("Either --model or --build is required")

    workspace_root = Path.cwd()

    from ...registry.local import LocalRegistry
    from ...core.budget import load_budget
    from ...core.ci.thresholds import ThresholdConfig
    from ...core.ci.runner import CIRunner, CIError

    # ── registry ────────────────────────────────────────────────────────────
    try:
        registry = LocalRegistry()
    except Exception as e:
        logger.exception("Registry init failed")
        _die(output_json, "error", f"Registry error: {e}", _EXIT_CONFIG)

    # ── thresholds (config.toml + CLI overrides) ────────────────────────────
    try:
        thresholds = ThresholdConfig.from_workspace(workspace_root).apply_overrides(
            latency_regression_pct=latency_regression_pct,
            size_regression_pct=size_regression_pct,
            latency_budget_ms=latency_budget_ms,
            size_budget_mb=size_budget_mb,
            cosine_threshold=cosine_threshold,
            top1_threshold=top1_threshold,
        )
    except Exception as e:
        _die(output_json, "error", f"Invalid config.toml: {e}", _EXIT_CONFIG)

    # ── budget (fills gaps; CLI flags already take priority via apply_overrides) ──
    try:
        budget = load_budget()
    except Exception as e:
        _die(output_json, "error", f"Invalid budget file: {e}", _EXIT_CONFIG)

    budget_overrides = {}
    if budget.get("max_latency_ms") is not None and latency_budget_ms is None:
        budget_overrides["latency_budget_ms"] = budget["max_latency_ms"]
    if budget.get("max_p95_ms") is not None:
        budget_overrides["p95_budget_ms"] = budget["max_p95_ms"]
    if budget.get("max_memory_mb") is not None:
        budget_overrides["memory_budget_mb"] = budget["max_memory_mb"]
    if budget.get("max_size_mb") is not None and size_budget_mb is None:
        budget_overrides["size_budget_mb"] = budget["max_size_mb"]
    if budget_overrides:
        thresholds = thresholds.apply_overrides(**budget_overrides)

    # ── header ───────────────────────────────────────────────────────────────
    if not output_json:
        console.print(f"\n[bold]MLBuild CI[/bold]")
        console.print(f"  Baseline : {baseline}")
        if model:
            console.print(f"  Model    : {model.name}")
        else:
            console.print(f"  Build    : {build_id[:16]}...")
        if any(v is not None for v in budget.values()):
            console.print(
                "[dim]  Budget constraints loaded from .mlbuild/budget.toml[/dim]"
            )
        console.print()

    # ── run ──────────────────────────────────────────────────────────────────
    runner = CIRunner(
        registry=registry,
        thresholds=thresholds,
        workspace_root=workspace_root,
    )

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
            print(json.dumps({"status": "fail", "message": str(e)}))
        else:
            console.print(f"[red]CI failed:[/red] {e}")
        sys.exit(_EXIT_FAIL)
    except Exception as e:
        logger.exception("CI command crashed")
        if output_json:
            print(json.dumps({"status": "error", "message": str(e)}))
        else:
            console.print(f"[red]Unexpected error:[/red] {e}")
        sys.exit(_EXIT_CRASH)

    # ── output ───────────────────────────────────────────────────────────────
    if output_json:
        print(report.to_json())
    else:
        console.print(report.to_text())

    sys.exit(_EXIT_FAIL if report.result == "fail" else 0)