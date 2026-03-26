"""
Convert command: convert model files to MLBuild-compatible formats.

Supports:
    PyTorch (.pt/.pth) → ONNX, CoreML, TFLite
    ONNX               → CoreML, TFLite
    TF SavedModel      → TFLite

Args only — all conversion logic lives in mlbuild.convert.service.
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from mlbuild.convert.service import run_convert
from mlbuild.convert.types import ConvertResult, ConvertStatus
from mlbuild.core.errors import ConvertError, ConversionCancelled

console = Console(width=None)
logger = logging.getLogger("mlbuild.cli.convert")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _parse_shape(shape_str: str) -> tuple:
    return tuple(int(x.strip()) for x in shape_str.split(","))


def _get_valid_targets() -> list[str]:
    """Pull dynamically from backend instead of hardcoding."""
    try:
        from mlbuild.backends.coreml.exporter import TARGET_MAPPING
        return sorted(TARGET_MAPPING.keys())
    except Exception:
        return []


def _print_error(e: ConvertError):
    console.print("\n[bold red]Conversion failed[/bold red]")
    console.print(f"  Stage: {getattr(e, 'stage', 'unknown')}")
    console.print(f"  Code:  {getattr(e, 'error_code', 'unknown')}")
    console.print(f"  Msg:   {e}")


def _print_result(result: ConvertResult):
    console.print()
    console.print(f"[bold]Run ID:[/bold] {result.run.run_id}")

    if result.status == ConvertStatus.CANCELLED:
        console.print("[yellow]Conversion cancelled.[/yellow]")
        return

    steps = result.steps

    if len(steps) == 1:
        step = steps[0]
        _print_step(step, 1, 1)
    else:
        _print_table(result)

    # --- Warnings with context ---
    warnings_found = False
    for i, step in enumerate(steps):
        for w in (step.conversion.warnings or []):
            if not warnings_found:
                console.print("\n[bold yellow]Warnings:[/bold yellow]")
                warnings_found = True
            console.print(f"  Step {i+1}: {w}")

    # --- Summary ---
    console.print()
    console.print(f"Total Duration: {result.total_duration_seconds:.2f}s")
    console.print(f"Cache Hits: {len(result.cache_hits)}")

    if result.final_build_id:
        console.print(f"Build ID: {result.final_build_id[:16]}...")
        console.print(f"[dim]Next:[/dim] mlbuild benchmark {result.final_build_id[:12]}")
    elif result.final_path:
        console.print(f"[dim]Output:[/dim] {result.final_path}")


def _print_step(step, i: int, total: int):
    status = "[cyan]↩[/cyan]" if step.cache.hit else "[green]✓[/green]"
    strategy = (step.conversion.metadata or {}).get("strategy")

    console.print(
        f"{status} Step {i}/{total}  "
        f"{step.src_format} → {step.dst_format}"
    )
    console.print(f"  Path       {step.output_path.name}")
    console.print(f"  Size       {step.file_size_mb:.2f} MB")
    console.print(
        f"  Duration   {'cached' if step.cache.hit else f'{step.duration_seconds:.2f}s'}"
    )

    if strategy:
        console.print(f"  Strategy   {strategy}")

    if step.build_id:
        console.print(f"  Build ID   {step.build_id[:16]}...")


def _print_table(result: ConvertResult):
    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("Step")
    table.add_column("Path")
    table.add_column("Size")
    table.add_column("Duration")
    table.add_column("Strategy")
    table.add_column("Build ID")

    for i, step in enumerate(result.steps):
        strategy = (step.conversion.metadata or {}).get("strategy") or "—"

        table.add_row(
            f"{'↩' if step.cache.hit else '✓'} {i+1}/{len(result.steps)} "
            f"{step.src_format}→{step.dst_format}",
            step.output_path.name,
            f"{step.file_size_mb:.2f} MB",
            "cached" if step.cache.hit else f"{step.duration_seconds:.2f}s",
            strategy,
            (step.build_id[:12] + "...") if step.build_id else "—",
        )

    console.print(table)


def _print_json(result: ConvertResult):
    def serialize(obj):
        if isinstance(obj, Path):
            return str(obj)
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

    print(json.dumps(result, default=serialize, indent=2))


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

@click.command("convert")
@click.option("--model", required=True, type=Path)
@click.option("--to", "target_format",
              required=True,
              type=click.Choice(["onnx", "coreml", "tflite"]))
@click.option("--target", default=None)
@click.option("--input-shape", default="1,3,224,224")
@click.option("--quantize", default="fp32",
              type=click.Choice(["fp32", "fp16"]))
@click.option("--load-mode", default="auto",
              type=click.Choice(["auto", "jit", "eager"]))
@click.option("--opset", default=None, type=int)
@click.option("--keep-intermediate", is_flag=True, default=False)
@click.option("--name", default=None)
@click.option("--notes", default=None)
@click.option("--no-register", is_flag=True, default=False)
@click.option("--debug", is_flag=True, default=False)
@click.option("--timeout", default=300, type=int)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--json-output", is_flag=True, default=False,
              help="Output result as JSON")
def convert(
    model: Path,
    target_format: str,
    target: Optional[str],
    input_shape: str,
    quantize: str,
    load_mode: str,
    opset: Optional[int],
    keep_intermediate: bool,
    name: Optional[str],
    notes: Optional[str],
    no_register: bool,
    debug: bool,
    timeout: int,
    dry_run: bool,
    json_output: bool,
):
    """Convert a model to a different format via MLBuild."""

    # --- Minimal parsing only (NO business logic) ---
    try:
        shape = _parse_shape(input_shape)
    except Exception:
        raise click.UsageError("Invalid --input-shape format")

    if timeout <= 0:
        raise click.UsageError("--timeout must be > 0")

    # Fail fast on invalid target usage
    if target and target_format != "coreml":
        raise click.UsageError("--target is only valid for coreml")

    exit_code = 0

    try:
        result: ConvertResult = run_convert(
            model_path=model,
            target_format=target_format,
            target=target,
            input_shape=shape,
            quantize=quantize,
            load_mode=load_mode,
            opset=opset,
            keep_intermediate=keep_intermediate,
            name=name,
            notes=notes,
            no_register=no_register,
            debug=debug,
            timeout=timeout,
            dry_run=dry_run,
        )

        if json_output:
            _print_json(result)
        elif not dry_run:
            _print_result(result)

        if result.status != ConvertStatus.SUCCESS:
            exit_code = 1

    except ConversionCancelled:
        console.print("\n[yellow]Conversion cancelled.[/yellow]")
        exit_code = 130

    except ConvertError as e:
        _print_error(e)
        exit_code = 1

    except Exception as e:
        logger.exception("Unexpected CLI failure")
        console.print("\n[bold red]Unexpected error[/bold red]")
        console.print(f"  {e}")
        if debug:
            traceback.print_exc()
        else:
            console.print("[dim]Run with --debug for stack trace[/dim]")
        exit_code = 1

    sys.exit(exit_code)