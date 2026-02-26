"""
Build command: Convert ONNX → CoreML and register artifact.

Infrastructure guarantees:
- Deterministic build ID (includes environment fingerprint)
- Atomic artifact promotion
- Integrity verification after move
- Full reproducibility tracking
"""

from __future__ import annotations

import os
import sys
import shutil
import hashlib
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal
from xml.parsers.expat import model

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from mlbuild import __version__ as MLBUILD_VERSION

from ...loaders import load_model
from ...backends.coreml import CoreMLExporter
from ...core.hash import (
    compute_artifact_hash,
    compute_config_hash,
    compute_source_hash,
)
from ...core.environment import (
    collect_environment,
    hash_environment,
    validate_reproducibility,
)
from ...core.calibration import (
    CalibrationConfig,
    CalibrationDataset,
    PreprocessingConfig,
)
from ...registry import LocalRegistry
from ...core.types import Build
from ...core.errors import MLBuildError

console = Console()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _structured_build_id(
    source_hash: str,
    config_hash: str,
    artifact_hash: str,
    env_fingerprint: str,
    mlbuild_version: str,
) -> str:
    """
    Compute deterministic build ID.
    
    Build ID = SHA256(source_hash || config_hash || artifact_hash || env_fingerprint || version)
    """
    return hashlib.sha256(
        b"\x00".join([
            bytes.fromhex(source_hash),
            bytes.fromhex(config_hash),
            bytes.fromhex(artifact_hash),
            bytes.fromhex(env_fingerprint),
            mlbuild_version.encode('utf-8'),
        ])
    ).hexdigest()


def _directory_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

@click.command()
@click.argument('build_id')
@click.option('--runs', default=100, type=int, help='Number of benchmark runs')
@click.option('--warmup', default=20, type=int, help='Number of warmup runs')
@click.option(
    '--compute-unit',
    type=click.Choice(['CPU_ONLY', 'CPU_AND_GPU', 'ALL']),
    default='ALL',
    help='Compute unit'
)
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def benchmark(build_id, runs, warmup, compute_unit, as_json):
    """Benchmark a build on current device."""
    from ..commands.benchmark import benchmark as benchmark_cmd

    ctx = click.get_current_context()

    ctx.invoke(
        benchmark_cmd,
        build_id=build_id,
        runs=runs,
        warmup=warmup,
        compute_unit=compute_unit,
        as_json=as_json,
    )


@click.command()
@click.option(
    "--model",
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--target",
    required=True,
    type=click.Choice(["apple_a17", "apple_a16", "apple_a15", "apple_m3", "apple_m2", "apple_m1"]),
)
@click.option("--name")
@click.option(
    "--quantize",
    type=click.Choice(["fp32", "fp16"]),
    default="fp32",
)
@click.option("--notes")
def build(model: Path, backend: str, target: str, name: str, quantize: str, notes: str):

    # ============================================================
    # RESOLVE BACKEND
    # ============================================================
    from ...backends.registry import BackendRegistry

    try:
        backend_inst = BackendRegistry.get_backend(backend)
    except (ValueError, RuntimeError) as e:
        console.print(f"\n[red]Backend Error:[/red] {e}\n")
        sys.exit(1)

    # If backend has its own build method, use it
    if backend != "coreml":
        try:
            build_obj = backend_inst.build(
                model_path=model,
                target=target,
                quantize=quantize,
                name=name,
                notes=notes,
            )
            return
        except Exception as e:
            console.print(f"\n[bold red]Build failed[/bold red]: {e}\n")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # ============================================================
    # ENFORCE DETERMINISM (DO NOT JUST WARN)
    # ============================================================
    import numpy as np
    import torch
    
    # Set environment variables
    os.environ["PYTHONHASHSEED"] = "0"
    
    # Set random seeds
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    
    # Set deterministic algorithms (may impact performance)
    torch.use_deterministic_algorithms(True, warn_only=True)

    console.print("[dim]Deterministic mode enabled (PYTHONHASHSEED=0, seeds set)[/dim]\n")

    console.print(f"\n[bold]Building:[/bold] {os.path.basename(model)}")
    console.print(f"Target: {target}")
    console.print(f"Quantization: {quantize}\n")

    try:
        # -------------------------------------------------------------
        # Step 1: Collect environment ONCE (never recompute)
        # -------------------------------------------------------------
        env_data = collect_environment()
        env_fingerprint = hash_environment(env_data)
        
        # -------------------------------------------------------------
        # Step 2: Validate reproducibility (WARN but allow non-critical)
        # -------------------------------------------------------------
        is_reproducible, warnings = validate_reproducibility()
        if warnings:
            console.print("\n[yellow]⚠️  Reproducibility warnings:[/yellow]")
            for warning in warnings:
                console.print(f"    {warning}")
            
            # Only check for CRITICAL warnings
            has_critical = any("[CRITICAL]" in w for w in warnings)
            if has_critical:
                console.print("\n[bold red]CRITICAL: Build cannot proceed[/bold red]")
                sys.exit(1)

            console.print("\n[bold yellow]WARNING: Build may not be reproducible[/bold yellow]")
            console.print("Run: export PYTHONHASHSEED=0")
            console.print("Or add to ~/.zshrc: echo 'export PYTHONHASHSEED=0' >> ~/.zshrc\n")

        # -------------------------------------------------------------
        # Step 3: Load model
        # -------------------------------------------------------------
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
            task = p.add_task("Loading ONNX...", total=None)
            ir = load_model(str(model))
            p.update(task, completed=True)

        # -------------------------------------------------------------
        # Step 4: Hash source
        # -------------------------------------------------------------
        source_hash = compute_source_hash(model)

        # -------------------------------------------------------------
        # Step 5: Build configuration
        # -------------------------------------------------------------
        config = {
            "target": target,
            "quantization": {"type": quantize},
            "optimizer": {"compute_units": "ALL"},
        }

        # Extract coremltools version from environment
        try:
            import coremltools as ct
            coremltools_version = ct.__version__
        except ImportError:
            coremltools_version = "unknown"
        
        config_hash = compute_config_hash(config, coremltools_version=coremltools_version)

        # -------------------------------------------------------------
        # Step 5.5: Generate INT8 calibration data if needed
        # -------------------------------------------------------------
        calibration_data = None
        if quantize == "int8":
            console.print("[dim]Generating INT8 calibration data...[/dim]")
            
            # Get input shape from IR
            from ...backends.coreml.exporter import ModelIngestion
            _, _, shape_tuples = ModelIngestion.extract_input_specs(ir)
            
            # Create calibration config
            cal_config = CalibrationConfig(
                sample_count=100,  # 100 calibration samples
                input_shape=tuple(shape_tuples[0]),  # Use model's input shape
                preprocessing=PreprocessingConfig(),  # No preprocessing for now
                seed=42,
            )
            
            # Generate synthetic calibration data
            cal_dataset = CalibrationDataset(cal_config)
            calibration_samples = cal_dataset.generate_synthetic()
            
            # Compute fingerprint for reproducibility
            cal_fingerprint = cal_dataset.compute_fingerprint()
            
            console.print(f"  Calibration samples: {cal_fingerprint.sample_count}")
            console.print(f"  Calibration hash: {cal_fingerprint.data_hash[:16]}...")
            console.print()
            
            calibration_data = list(calibration_samples)

        # -------------------------------------------------------------
        # Step 6: Convert to CoreML
        # -------------------------------------------------------------
        with tempfile.TemporaryDirectory() as tmp_root:

            tmp_root = Path(tmp_root)

            with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
                task = p.add_task("Converting to CoreML...", total=None)

                exporter = CoreMLExporter(target=target)
                mlpackage_path, conversion_metadata = exporter.export(
                    ir,
                    output_dir=tmp_root,
                    quantization=quantize,
                    calibration_data=calibration_data,
                )

                # Log the conversion metadata
                console.print(f"\n[dim]Conversion metadata:[/dim]")
                if "onnx_opset" in conversion_metadata:
                    console.print(f"  ONNX Opset: {conversion_metadata['onnx_opset']}")
                if "ir_version" in conversion_metadata:
                    console.print(f"  IR Version: {conversion_metadata['ir_version']}")
                if "graph_nodes_before" in conversion_metadata:
                    console.print(f"  Graph nodes: {conversion_metadata['graph_nodes_before']} → {conversion_metadata.get('graph_nodes_after', '?')}")
                if conversion_metadata.get("dynamic_dimensions"):
                    console.print(f"  Dynamic dims: {len(conversion_metadata['dynamic_dimensions'])} frozen")
                console.print()

                p.update(task, completed=True)

            # ---------------------------------------------------------
            # Step 7: Compute artifact hash
            # ---------------------------------------------------------
            artifact_hash = compute_artifact_hash(mlpackage_path)

            # ---------------------------------------------------------
            # Step 8: Compute build ID (INCLUDES ENVIRONMENT)
            # ---------------------------------------------------------
            build_id = _structured_build_id(
                source_hash,
                config_hash,
                artifact_hash,
                env_fingerprint,
                MLBUILD_VERSION,
            )

            # ---------------------------------------------------------
            # Step 9: Prepare final artifact location
            # ---------------------------------------------------------
            artifacts_root = Path(".mlbuild/artifacts").resolve()
            final_dir = artifacts_root / artifact_hash

            artifacts_root.mkdir(parents=True, exist_ok=True)

            # ---------------------------------------------------------
            # Step 10: Promote artifact atomically
            # ---------------------------------------------------------
            if final_dir.exists():
                # Verify integrity before trusting existing artifact
                existing_hash = compute_artifact_hash(final_dir)
                if existing_hash != artifact_hash:
                    raise RuntimeError(
                        "Artifact hash mismatch — possible corruption"
                    )

                console.print(f"[yellow]Reusing existing artifact {artifact_hash[:12]}[/yellow]")

            else:
                # Move entire directory atomically
                shutil.move(str(mlpackage_path), str(final_dir))

                # Post-move verification
                verified_hash = compute_artifact_hash(final_dir)
                if verified_hash != artifact_hash:
                    shutil.rmtree(final_dir, ignore_errors=True)
                    raise RuntimeError("Artifact integrity verification failed")

            # ---------------------------------------------------------
            # Step 11: Compute exact byte size
            # ---------------------------------------------------------
            size_bytes = _directory_size_bytes(final_dir)
            size_mb = Decimal(size_bytes) / Decimal(1024 * 1024)

            # ---------------------------------------------------------
            # Step 12: Create Build object with environment data
            # ---------------------------------------------------------
            # Extract framework versions from env_data structure
            backend_versions = {}
            if "numpy" in env_data and env_data["numpy"].get("installed"):
                backend_versions["numpy"] = env_data["numpy"]["version"]
            if "torch" in env_data and env_data["torch"].get("installed"):
                backend_versions["torch"] = env_data["torch"]["version"]
            if "tensorflow" in env_data and env_data["tensorflow"].get("installed"):
                backend_versions["tensorflow"] = env_data["tensorflow"]["version"]
            if "onnxruntime" in env_data and env_data["onnxruntime"].get("installed"):
                backend_versions["onnxruntime"] = env_data["onnxruntime"]["version"]
            
            # Add coremltools
            try:
                import coremltools as ct
                backend_versions["coremltools"] = ct.__version__
            except ImportError:
                backend_versions["coremltools"] = "unknown"
            
            # Add ONNX version from IR
            backend_versions["onnx"] = ir.metadata.get("framework_version", "unknown")
            
            build_obj = Build(
                build_id=build_id,
                artifact_hash=artifact_hash,
                source_hash=source_hash,
                config_hash=config_hash,
                env_fingerprint=env_fingerprint,
                name=name,
                notes=notes,
                created_at=datetime.now(timezone.utc),
                source_path=str(Path(model).resolve()),
                target_device=target,
                format="coreml",
                quantization=config["quantization"],
                optimizer_config=config["optimizer"],
                backend_versions=backend_versions,  # Use extracted versions
                environment_data=env_data,  # Full environment snapshot
                mlbuild_version=MLBUILD_VERSION,
                python_version=env_data["python"]["version"],
                platform=env_data["hardware"]["cpu"]["system"],
                os_version=env_data["hardware"]["cpu"]["release"],
                artifact_path=str(final_dir),
                size_mb=size_mb,
            )

            # ---------------------------------------------------------
            # Step 13: Registry transaction
            # ---------------------------------------------------------
            registry = LocalRegistry()
            try:
                registry.save_build(build_obj)
            except Exception:
                # Roll back artifact if registry fails
                shutil.rmtree(final_dir, ignore_errors=True)
                raise

        # -------------------------------------------------------------
        # Success output
        # -------------------------------------------------------------
        console.print(f"\n[bold green]✓ Build complete[/bold green]")
        console.print(f"Build ID:      {build_id[:16]}...")
        console.print(f"Artifact Hash: {artifact_hash[:16]}...")
        console.print(f"Source Hash:   {source_hash[:16]}...")
        console.print(f"Config Hash:   {config_hash[:16]}...")
        console.print(f"Env Fingerprint: {env_fingerprint[:16]}...")
        console.print(f"Size:          {size_mb:.2f} MB")
        console.print(f"Artifact Path: {final_dir}")
        console.print()
        
        if not is_reproducible:
            console.print("[yellow]⚠️  Build may not be reproducible. See warnings above.[/yellow]\n")

    except MLBuildError as e:
        console.print("\n[bold red]Build failed[/bold red]\n")
        console.print(e.format())
        sys.exit(e.exit_code.value)

    except Exception as e:
        console.print(f"\n[bold red]Build failed[/bold red]: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)