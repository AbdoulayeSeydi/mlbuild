"""
mlbuild optimize <build_id> [--method fp16|int8] [--skip-benchmark] [--json]
"""

from __future__ import annotations

from pathlib import Path

import json
from typing import List

from rich.console import Console
from rich.table import Table
from rich import box

from ...registry.local import LocalRegistry
from ...optimize.optimizer import optimize as run_optimize, prune as run_prune, OptimizeError

console = Console()


def optimize_cmd(
    build_id: str,
    opt_pass: str,
    method: str | None,
    skip_benchmark: bool,
    output_json: bool,
    calibration_data: "Path | None" = None,
    sparsity: float = 0.5,
):
    """
    Generate optimized variants of a registered build.

    This is an application-layer command invoked by the CLI.
    It contains user-facing output but does not terminate the process.
    """

    if opt_pass == "prune":
        return _run_prune_cmd(
            build_id=build_id,
            sparsity=sparsity,
            skip_benchmark=skip_benchmark,
            output_json=output_json,
        )

    if opt_pass != "quantize":
        raise ValueError(f"Unsupported optimization pass: {opt_pass}")

    registry = LocalRegistry()

    source = registry.resolve_build(build_id)
    if source is None:
        if output_json:
            print(json.dumps({"error": f"Build not found: {build_id}"}))
        else:
            console.print(f"[red]Build not found: {build_id}[/red]")
        sys.exit(1)

    if not output_json:
        _print_header(source, method)

    variants = run_optimize(
        build_id=source.build_id,
        registry=registry,
        method=method,
        benchmark=not (skip_benchmark or output_json),
        calibration_data=calibration_data,
    )

    if output_json:
        import json
        print(json.dumps(_json_output(source, variants), indent=2))
        return

    _print_table(source, variants)

    console.print(f"[green]✓[/green] {len(variants)} variant(s) registered.\n")

    return variants


def _run_prune_cmd(
    build_id: str,
    sparsity: float,
    skip_benchmark: bool,
    output_json: bool,
) -> None:
    from ...optimize.optimizer import prune as run_prune

    registry = LocalRegistry()
    source = registry.resolve_build(build_id)
    if source is None:
        if output_json:
            print(json.dumps({"error": f"Build not found: {build_id}"}))
        else:
            console.print(f"[red]Build not found: {build_id}[/red]")
        sys.exit(1)

    if not output_json:
        console.print(f"\n[bold]Pruning:[/bold] {source.name or source.build_id[:16]}")
        console.print(f"Format:   {source.format}")
        console.print(f"Sparsity: {sparsity:.0%}")
        console.print(f"Source:   {source.build_id[:16]}...\n")

    try:
        variants = run_prune(
            build_id=source.build_id,
            registry=registry,
            sparsity=sparsity,
            benchmark=not (skip_benchmark or output_json),
        )
    except OptimizeError as e:
        if output_json:
            print(json.dumps({"error": str(e)}))
        else:
            console.print(f"[red]Pruning failed: {e}[/red]")
        sys.exit(1)

    if output_json:
        print(json.dumps(_json_output(source, variants), indent=2))
        return

    _print_table(source, variants)
    console.print(f"[green]✓[/green] {len(variants)} pruned variant(s) registered.\n")


def _print_header(source, method: str | None) -> None:
    console.print(
        f"\n[bold]Optimizing[/bold]: {source.name or source.build_id[:16]}"
    )
    console.print(f"Format:  {source.format}")
    console.print(f"Method:  {method or 'auto (fp16 + int8)'}")
    console.print(f"Source:  {source.build_id[:16]}...\n")


def _json_output(source, variants) -> dict:
    return {
        "source_build_id": source.build_id,
        "variants": [
            {
                "build_id": v.build_id,
                "variant_id": v.variant_id,
                "method": v.optimization_method,
                "weight_precision": v.weight_precision,
                "size_mb": round(float(v.size_mb), 3),
                "parent_build_id": v.parent_build_id,
                "root_build_id": v.root_build_id,
                "artifact_path": v.artifact_path,
                "cached_latency_p50_ms": v.cached_latency_p50_ms,
            }
            for v in variants
        ],
    }


def _print_table(source, variants: List) -> None:

    source_size = float(source.size_mb)

    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold",
    )

    table.add_column("Method", style="cyan", no_wrap=True)
    table.add_column("Variant ID", style="dim")
    table.add_column("Build ID")
    table.add_column("Size", justify="right")
    table.add_column("Size Δ", justify="right")
    table.add_column("p50 Latency", justify="right")

    for v in variants:

        variant_size = float(v.size_mb)

        reduction = (
            (1 - variant_size / source_size) * 100
            if source_size > 0
            else 0
        )

        reduction_str = (
            f"[green]-{reduction:.1f}%[/green]"
            if reduction > 0
            else f"{reduction:.1f}%"
        )

        latency_str = (
            f"{v.cached_latency_p50_ms:.2f}ms"
            if v.cached_latency_p50_ms
            else "—"
        )

        _method_display = {
            "int8_static": "int8 (static)",
        }
        table.add_row(
            _method_display.get(v.optimization_method, v.optimization_method) or "—",
            v.variant_id or "—",
            v.build_id[:16] + "...",
            f"{variant_size:.2f} MB",
            reduction_str,
            latency_str,
        )

    console.print(table)


import click
import logging
import sys

logger = logging.getLogger(__name__)


@click.command(name="optimize")
@click.argument("build_id")
@click.option("--pass", "opt_pass", type=click.Choice(["quantize", "prune"]), required=True)
@click.option("--method", type=click.Choice(["fp16", "int8"]), required=False)
@click.option("--sparsity", default=0.5, type=float, show_default=True, help="Pruning sparsity (0.0-1.0).")
@click.option("--skip-benchmark", is_flag=True, default=False)
@click.option("--output-json", is_flag=True, default=False)
@click.option(
    "--calibration-data",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Calibration data for static INT8 (images dir, .npy dir, or .npz file).",
)
def optimize(build_id, opt_pass, method, sparsity, skip_benchmark, output_json, calibration_data):
    """Generate optimized variants of a registered build."""
    try:
        optimize_cmd(
            build_id=build_id,
            opt_pass=opt_pass,
            method=method,
            skip_benchmark=skip_benchmark,
            output_json=output_json,
            calibration_data=calibration_data,
            sparsity=sparsity,
        )
    except click.ClickException:
        raise
    except Exception as exc:
        logger.exception("optimize command failed")
        raise click.ClickException(str(exc)) from exc