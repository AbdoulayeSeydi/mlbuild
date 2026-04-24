"""
mlbuild explore <onnx_file> [--target auto] [--name NAME]
                             [--backends coreml,tflite] [--fast]
                             [--output-json]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List

import click
from rich.console import Console
from rich.table import Table
from rich import box

from ...core.device import detect_device, device_label
from ...explore.explorer import SUPPORTED_BACKENDS
from ...registry.local import LocalRegistry
from ...explore.explorer import explore as core_explore

console = Console(width=None)
logger = logging.getLogger(__name__)


# -----------------------------
# CLI
# -----------------------------

@click.command(name="explore")
@click.argument("model", type=click.Path(exists=True, path_type=Path))
@click.option("--target", default="auto", help="Target device (default: auto)")
@click.option("--name", default=None, help="Experiment name")
@click.option("--backends", default="coreml,tflite", help="Comma separated backends")
@click.option("--fast", is_flag=True, default=False)
@click.option("--output-json", is_flag=True, default=False)
@click.option("--calibration-data",       default=None, type=click.Path(exists=True, path_type=Path), help="Calibration data for static INT8 (images dir, .npy dir, or .npz).")
@click.option("--check-accuracy",         is_flag=True, default=False, help="Run output divergence check for each variant.")
@click.option("--accuracy-samples",       default=32,   show_default=True, type=int,   help="Samples per accuracy check.")
@click.option("--accuracy-seed",          default=42,   show_default=True, type=int,   help="RNG seed.")
@click.option("--cosine-threshold",       default=0.99, show_default=True, type=float, help="Minimum cosine similarity.")
@click.option("--top1-threshold",         default=0.99, show_default=True, type=float, help="Minimum top-1 agreement.")
def explore(
    model: Path,
    target: str,
    name: Optional[str],
    backends: str,
    fast: bool,
    output_json: bool,
    calibration_data: Optional[Path],
    check_accuracy: bool,
    accuracy_samples: int,
    accuracy_seed: int,
    cosine_threshold: float,
    top1_threshold: float,
) -> None:
    """Sweep optimization variants for a model."""

    if not model.is_file():
        raise click.ClickException(f"Model must be a file: {model}")

    try:
        result = _run_explore(
            model=model,
            target=target,
            name=name,
            backends_raw=backends,
            fast=fast,
            calibration_data=calibration_data,
            check_accuracy=check_accuracy,
            accuracy_samples=accuracy_samples,
            accuracy_seed=accuracy_seed,
            accuracy_cosine_threshold=cosine_threshold,
            accuracy_top1_threshold=top1_threshold,
        )

        if output_json:
            console.print_json(data=_to_json(result))
            return

        console.print(f"\n[bold]Exploring:[/bold] {model.name}")
        console.print(f"Device:    {device_label(result.target)}")
        console.print(f"Backends:  {', '.join(b.backend for b in result.backends)}")
        console.print(
            f"Mode:      {'fast (fp16, 20 runs)' if result.fast_mode else 'full (fp16 + int8, 100 runs)'}"
        )
        if calibration_data:
            console.print(f"INT8:      static calibration ({calibration_data.name})")
        console.print()

        _print_table(result)

        console.print(
            f"\n[green]✓[/green] {result.total_registered} variant(s) registered. "
            f"Run [dim]mlbuild log --name {result.name}[/dim] to view all.\n"
        )

        # ── Cloud sync ────────────────────────────────────────
        try:
            from ...cloud.sync import push
            variants = []
            winner = None
            for backend_result in result.backends:
                for v in backend_result.variants:
                    variants.append({
                        "build_id": v.build_id,
                        "backend": backend_result.backend,
                        "method": v.method,
                        "size_mb": round(v.size_mb, 3),
                        "latency_p50_ms": v.latency_p50_ms,
                        "latency_delta_pct": v.latency_delta_pct,
                        "size_delta_pct": v.size_delta_pct,
                        "verdict": v.verdict,
                        "accuracy_passed": v.accuracy_passed,
                        "accuracy_cosine": v.accuracy_cosine,
                    })
                    if v.verdict == "recommended":
                        winner = v.build_id
            push("explorations", {
                "exploration_type": "explore",
                "base_build_id": str(model),
                "base_build_name": model.stem,
                "variants": variants,
                "winner_build_id": winner,
                "optimization_pass": "fast" if result.fast_mode else "full",
            })
        except Exception:
            pass

    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


# -----------------------------
# Orchestration layer
# -----------------------------

def _run_explore(
    *,
    model: Path,
    target: str,
    name: Optional[str],
    backends_raw: str,
    fast: bool,
    calibration_data: Optional[Path] = None,
    check_accuracy: bool = False,
    accuracy_samples: int = 32,
    accuracy_seed: int = 42,
    accuracy_cosine_threshold: float = 0.99,
    accuracy_top1_threshold: float = 0.99,
):
    """Orchestration wrapper between CLI and core explore."""

    registry = LocalRegistry()

    # resolve device
    if target == "auto":
        device = detect_device()
        target = device.target

    # parse backend list
    backends = _parse_backends(backends_raw)

    experiment_name = name or model.stem

    return core_explore(
        onnx_path=model,
        target=target,
        name=experiment_name,
        backends=backends,
        fast=fast,
        registry=registry,
        calibration_data=calibration_data,
        check_accuracy=check_accuracy,
        accuracy_samples=accuracy_samples,
        accuracy_seed=accuracy_seed,
        accuracy_cosine_threshold=accuracy_cosine_threshold,
        accuracy_top1_threshold=accuracy_top1_threshold,
    )


def _parse_backends(raw: str) -> List[str]:
    backends = [b.strip() for b in raw.split(",") if b.strip()]

    invalid = [b for b in backends if b not in SUPPORTED_BACKENDS]

    if invalid:
        raise click.ClickException(
            f"Unknown backends: {invalid}. Supported: {', '.join(SUPPORTED_BACKENDS)}"
        )

    return backends


# -----------------------------
# Rendering
# -----------------------------

def _print_table(result) -> None:

    VERDICT_STYLE = {
        "baseline": "[dim]",
        "recommended": "[green]",
        "aggressive": "[yellow]",
        "skip": "[red]",
    }

    for backend_result in result.backends:

        console.print(f"[bold]{backend_result.backend.upper()}[/bold]")

        table = Table(box=box.SIMPLE_HEAD)
        table.add_column("Verdict", style="bold", no_wrap=True)
        table.add_column("Method", style="cyan", no_wrap=True)
        table.add_column("Size", justify="right", no_wrap=True)
        table.add_column("p50 Latency", justify="right", no_wrap=True)
        table.add_column("vs Baseline", justify="right", no_wrap=True)
        table.add_column("Accuracy", justify="center", no_wrap=True)

        for v in backend_result.variants:

            style = VERDICT_STYLE.get(v.verdict, "[white]")

            latency = (
                f"{v.latency_p50_ms:.2f}ms"
                if v.latency_p50_ms is not None
                else "—"
            )

            delta = _format_delta(v)

            if v.accuracy_passed is None:
                acc_cell = "[dim]—[/dim]"
            elif v.accuracy_passed:
                cos = f"{v.accuracy_cosine:.4f}" if v.accuracy_cosine is not None else "?"
                acc_cell = f"[green]✓ {cos}[/green]"
            else:
                acc_cell = f"[red]✗ {v.accuracy_failure or 'failed'}[/red]"

            table.add_row(
                f"{style}{v.verdict}[/]",
                v.method,
                f"{v.size_mb:.2f} MB",
                latency,
                delta,
                acc_cell,
            )

        console.print(table)
        console.print()


def _format_delta(v) -> str:

    if v.verdict == "baseline":
        return "—"

    if v.latency_delta_pct is None or v.size_delta_pct is None:
        return "—"

    lat_arrow = "↓" if v.latency_delta_pct < 0 else "↑"
    size_arrow = "↓" if v.size_delta_pct < 0 else "↑"

    lat = abs(v.latency_delta_pct)
    size = abs(v.size_delta_pct)

    return f"{lat_arrow}{lat:.0f}% lat  {size_arrow}{size:.0f}% size"


# -----------------------------
# JSON serialization
# -----------------------------

def _to_json(result) -> dict:

    return {
        "name": result.name,
        "source_path": result.source_path,
        "target": result.target,
        "fast_mode": result.fast_mode,
        "backends": [
            {
                "backend": b.backend,
                "variants": [
                    {
                        "build_id": v.build_id,
                        "method": v.method,
                        "size_mb": round(v.size_mb, 3),
                        "latency_p50_ms": v.latency_p50_ms,
                        "verdict": v.verdict,
                        "latency_delta_pct": (
                            round(v.latency_delta_pct, 1)
                            if v.latency_delta_pct is not None
                            else None
                        ),
                        "size_delta_pct": (
                            round(v.size_delta_pct, 1)
                            if v.size_delta_pct is not None
                            else None
                        ),
                        "accuracy": None if v.accuracy_passed is None else {
                            "passed": v.accuracy_passed,
                            "cosine_similarity": v.accuracy_cosine,
                            "top1_agreement": v.accuracy_top1,
                            "failure": v.accuracy_failure,
                        },
                    }
                    for v in b.variants
                ],
            }
            for b in result.backends
        ],
    }