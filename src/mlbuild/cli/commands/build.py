"""
Build command: Convert ONNX → CoreML / TFLite and register artifact.

Infrastructure guarantees:
- Deterministic build ID (includes environment fingerprint)
- Atomic artifact promotion
- Integrity verification after move
- Full reproducibility tracking
- ModelProfile stored alongside task_type (Step 4)

Step 4 additions
----------------
- --force-domain / --force-subtype / --force-execution CLI flags
- --task deprecated → maps to --force-domain with a warning
- --dynamic-sweep accepted (logs not-yet-implemented)
- build_profile() called after detect_task()
- _assert_profile_consistency() migration guard active
- benchmark_caveats populated from ExecutionMode + Subtype
- input_roles populated from build_input_schemas_with_roles()
- model_profile_json serialized and stored on Build
- Subtype + ExecutionMode shown in success output
"""

from __future__ import annotations

import json
import logging
import os
import sys
import shutil
import hashlib
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, NamedTuple, Optional

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

# ── Task detection (v1 + v2 imports) ─────────────────────────────────────────
from ...core.task_detection import (
    detect_task,
    detection_warning,
    build_profile,
    _assert_profile_consistency,
    ModelInfo,
    TensorInfo,
    TaskType,
    Domain,
    Subtype,
    ExecutionMode,
    ModelProfile,
)

# ── Input schema builder (v2) ─────────────────────────────────────────────────
from ...core.task_inputs import build_input_schemas_with_roles

# ── Output validation config ──────────────────────────────────────────────────
from ...core.task_validation import StrictOutputConfig

logger   = logging.getLogger(__name__)
console  = Console(width=None)


# ============================================================
# Detection result container
# ============================================================

class _Detection(NamedTuple):
    """
    Returned by _detect_task_from_onnx().
    Bundles ModelInfo, DetectionResult, and ModelProfile
    so all three travel together through the build pipeline.
    """
    info:    ModelInfo
    result:  object          # DetectionResult (kept as object to avoid circular import noise)
    profile: ModelProfile


# ============================================================
# Helpers
# ============================================================

def _structured_build_id(
    source_hash:     str,
    config_hash:     str,
    artifact_hash:   str,
    env_fingerprint: str,
    mlbuild_version: str,
) -> str:
    """
    Compute deterministic build ID.
    SHA256(source_hash || config_hash || artifact_hash || env_fingerprint || version)
    """
    return hashlib.sha256(
        b"\x00".join([
            bytes.fromhex(source_hash),
            bytes.fromhex(config_hash),
            bytes.fromhex(artifact_hash),
            bytes.fromhex(env_fingerprint),
            mlbuild_version.encode("utf-8"),
        ])
    ).hexdigest()


def _directory_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _read_global_strict() -> bool:
    """Read [validation] strict_output from .mlbuild/config.toml if present."""
    try:
        config_path = Path(".mlbuild/config.toml")
        if not config_path.exists():
            return False
        import tomllib
        with config_path.open("rb") as f:
            data = tomllib.load(f)
        return bool(data.get("validation", {}).get("strict_output", False))
    except Exception:
        return False


def _apply_force_overrides(
    profile:        ModelProfile,
    force_domain:   Optional[str],
    force_subtype:  Optional[str],
    force_execution: Optional[str],
) -> ModelProfile:
    """
    Apply --force-domain / --force-subtype / --force-execution overrides.

    Any override sets confidence=1.0 and confidence_tier="graph" to signal
    that the profile was explicitly specified, not inferred.
    Logs the override so consumers can distinguish forced vs auto-detected profiles.
    """
    if not any([force_domain, force_subtype, force_execution]):
        return profile

    try:
        domain    = Domain(force_domain)          if force_domain    else profile.domain
        subtype   = Subtype(force_subtype)         if force_subtype   else profile.subtype
        execution = ExecutionMode(force_execution) if force_execution else profile.execution
    except ValueError as e:
        console.print(f"\n[red]Invalid override value:[/red] {e}\n")
        sys.exit(1)

    overrides = {k: v for k, v in [
        ("domain",    force_domain),
        ("subtype",   force_subtype),
        ("execution", force_execution),
    ] if v}

    override_str = "  ".join(f"{k}={v}" for k, v in overrides.items())
    console.print(f"[dim]User override applied — {override_str}[/dim]")
    logger.info("force_overrides_applied  %s", override_str)

    return ModelProfile(
        domain          = domain,
        subtype         = subtype,
        execution       = execution,
        confidence      = 1.0,
        confidence_tier = "graph",
        nms_inside      = profile.nms_inside,
        state_optional  = profile.state_optional,
    )


def _build_benchmark_caveats(profile: ModelProfile) -> List[str]:
    """
    Build the machine-readable benchmark limitation notes for a ModelProfile.
    These travel with every benchmark record so automated pipelines can check
    limitations without reading log output.
    """
    caveats: List[str] = []

    if profile.execution == ExecutionMode.KV_CACHE:
        caveats.append(
            "Single-pass KV-cache benchmark. Token-by-token generation latency "
            "not captured. Per-token latency will be lower; first-token latency "
            "will be higher."
        )

    elif profile.execution == ExecutionMode.PARTIALLY_STATEFUL:
        caveats.append(
            "Partially stateful model — state inputs are optional and were omitted. "
            "Benchmark reflects cold-start latency. Warm-start latency (with state) "
            "will differ."
        )

    elif profile.execution == ExecutionMode.STATEFUL:
        caveats.append(
            "Stateful model — running single-pass approximation with zero initial "
            "state. Benchmark reflects cold-start latency only. Real stateful "
            "latency will differ."
        )

    if profile.subtype == Subtype.TIMESERIES:
        caveats.append(
            "Time-series model — assumed fixed window input (T=96). Model may "
            "expect rolling context; benchmark reflects single-window latency only."
        )

    if profile.subtype == Subtype.NONE and profile.confidence < 0.5:
        caveats.append(
            "Model task could not be confidently classified — inputs generated "
            "from shape metadata only."
        )

    return caveats


def _profile_to_json(profile: ModelProfile) -> str:
    """Serialize ModelProfile to a stable JSON string."""
    return json.dumps({
        "domain":          profile.domain.value,
        "subtype":         profile.subtype.value,
        "execution":       profile.execution.value,
        "confidence":      round(profile.confidence, 4),
        "confidence_tier": profile.confidence_tier,
        "nms_inside":      profile.nms_inside,
        "state_optional":  profile.state_optional,
    }, sort_keys=True)


# ============================================================
# Shared helper: ModelInfo + DetectionResult + ModelProfile
# ============================================================

def _detect_task_from_onnx(
    onnx_model,
    forced_task:     Optional[str] = None,
    force_domain:    Optional[str] = None,
    force_subtype:   Optional[str] = None,
    force_execution: Optional[str] = None,
) -> _Detection:
    """
    Build ModelInfo from a loaded ONNX model, run three-tier task detection,
    build ModelProfile, and apply any --force-* overrides.

    Parameters
    ----------
    onnx_model      : loaded onnx.ModelProto
    forced_task     : legacy --task value (deprecated → maps to --force-domain)
    force_domain    : --force-domain override (e.g. "vision")
    force_subtype   : --force-subtype override (e.g. "detection")
    force_execution : --force-execution override (e.g. "kv_cache")

    Returns
    -------
    _Detection(info, result, profile)
    """
    import numpy as np

    graph = onnx_model.graph

    _ONNX_TO_NP = {
        1: np.float32, 2: np.uint8,   3: np.int8,
        5: np.int32,   6: np.int32,   7: np.int64,
        10: np.float16, 11: np.float64,
    }

    # ── Build TensorInfo for inputs ─────────────────────────────────────────
    inputs = []
    # Collect initializer names — inputs with a default initializer are optional
    initializer_names = {init.name for init in graph.initializer}

    for inp in graph.input:
        shape = []
        for dim in inp.type.tensor_type.shape.dim:
            shape.append(dim.dim_value if dim.dim_value > 0 else -1)

        elem_type = inp.type.tensor_type.elem_type
        dtype     = _ONNX_TO_NP.get(elem_type)

        # is_optional=True when the input has a default initializer value
        is_optional = inp.name in initializer_names

        inputs.append(TensorInfo(
            name        = inp.name,
            shape       = tuple(shape) if shape else None,
            dtype       = dtype,
            is_optional = is_optional,
        ))

    # ── Build TensorInfo for outputs ────────────────────────────────────────
    outputs = []
    for out in graph.output:
        shape = []
        for dim in out.type.tensor_type.shape.dim:
            shape.append(dim.dim_value if dim.dim_value > 0 else -1)
        dtype = _ONNX_TO_NP.get(out.type.tensor_type.elem_type)
        outputs.append(TensorInfo(
            name  = out.name,
            shape = tuple(shape) if shape else None,
            dtype = dtype,
        ))

    op_types = {node.op_type for node in graph.node}

    metadata = {}
    for prop in onnx_model.metadata_props:
        metadata[prop.key] = prop.value

    info = ModelInfo(
        format     = "onnx",
        inputs     = inputs,
        outputs    = outputs,
        op_types   = op_types,
        node_count = len(graph.node),
        metadata   = metadata,
    )

    # ── Handle --task deprecation ────────────────────────────────────────────
    # --task mapped to domain only (no subtype info). Emit deprecation warning.
    effective_force_domain = force_domain
    if forced_task and forced_task != "unknown" and not force_domain:
        console.print(
            f"[yellow]⚠  --task is deprecated. "
            f"Use --force-domain {forced_task} instead.[/yellow]"
        )
        logger.warning(
            "deprecated_flag_task  value=%s  "
            "use --force-domain %s instead",
            forced_task, forced_task,
        )
        effective_force_domain = forced_task

    # ── Detection ────────────────────────────────────────────────────────────
    # forced_task kept for detect_task() backward compat (GRAPH-tier override)
    forced = TaskType.from_str(forced_task) if forced_task and forced_task != "unknown" else None
    result  = detect_task(info, forced=forced.value if forced else None)
    profile = build_profile(info, result)

    # ── Migration consistency guard ──────────────────────────────────────────
    _assert_profile_consistency(result.primary, profile)

    # ── Apply --force-* overrides ────────────────────────────────────────────
    profile = _apply_force_overrides(
        profile,
        force_domain    = effective_force_domain,
        force_subtype   = force_subtype,
        force_execution = force_execution,
    )

    return _Detection(info=info, result=result, profile=profile)


# ============================================================
# CLI — benchmark sub-command  (unchanged)
# ============================================================

@click.command()
@click.argument("build_id")
@click.option("--runs",    default=100,  type=int)
@click.option("--warmup",  default=20,   type=int)
@click.option(
    "--compute-unit",
    type=click.Choice(["CPU_ONLY", "CPU_AND_GPU", "ALL"]),
    default="ALL",
)
@click.option("--json", "as_json", is_flag=True)
def benchmark(build_id, runs, warmup, compute_unit, as_json):
    """Benchmark a build on current device."""
    from ..commands.benchmark import benchmark as benchmark_cmd
    ctx = click.get_current_context()
    ctx.invoke(
        benchmark_cmd,
        build_id=build_id, runs=runs, warmup=warmup,
        compute_unit=compute_unit, as_json=as_json,
    )


# ============================================================
# CLI — build command  (Step 4 flag additions)
# ============================================================


@click.command()
@click.option("--model",   required=True, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--backend",
    type=click.Choice(["coreml", "tflite"]),
    default="coreml",
    help="""Backend to use for conversion. 
    NOTE: MLBuild only converts ONNX → CoreML or ONNX → TFLite.
    CoreML and TFLite models must be registered with 'mlbuild import',
    not converted. To convert between CoreML and TFLite, first export
    your model to ONNX, then build for the target backend.
""",
)
@click.option(
    "--target",
    required=True,
    type=click.Choice([
        # iPhone models (recommended)
        "iphone_16_pro_max", "iphone_16_pro", "iphone_16_plus", "iphone_16",
        "iphone_15_pro_max", "iphone_15_pro", "iphone_15_plus", "iphone_15",
        "iphone_14_pro_max", "iphone_14_pro", "iphone_14_plus", "iphone_14",
        "iphone_13_pro_max", "iphone_13_pro", "iphone_13_mini", "iphone_13",
        "iphone_12_pro_max", "iphone_12_pro", "iphone_12_mini", "iphone_12",
        # Mac
        "mac_m3", "mac_m2", "mac_m1",
        # Android (ABI-based)
        "android_arm64", "android_arm32",
        # Legacy chip targets (kept for CI compatibility)
        "apple_a18", "apple_a17", "apple_a16", "apple_a15", "apple_a14",
        "apple_m3", "apple_m2", "apple_m1",
        # Auto-detect
        "device-connected",
    ]),
)
@click.option("--name")
@click.option(
    "--quantize",
    type=click.Choice(["fp32", "fp16", "int8"]),
    default="fp32",
)
@click.option("--notes")

# ── Legacy flag (deprecated, kept for backward compat) ──────────────────────
@click.option(
    "--task",
    type=click.Choice(["vision", "nlp", "audio", "unknown"]),
    default=None,
    help="[Deprecated] Use --force-domain instead.",
)

# ── New override flags (Step 4) ──────────────────────────────────────────────
@click.option(
    "--force-domain", "force_domain",
    type=click.Choice(["vision", "nlp", "audio", "tabular"]),
    default=None,
    help="Override auto-detected domain.",
)
@click.option(
    "--force-subtype", "force_subtype",
    type=click.Choice([
        "detection", "segmentation", "timeseries",
        "recommendation", "generative", "multimodal", "none",
    ]),
    default=None,
    help="Override auto-detected behavioral subtype.",
)
@click.option(
    "--force-execution", "force_execution",
    type=click.Choice([
        "standard", "stateful", "partially_stateful", "kv_cache", "multi_input",
    ]),
    default=None,
    help="Override auto-detected execution mode. Useful for debugging backend differences.",
)
@click.option(
    "--dynamic-sweep", "dynamic_sweep",
    type=str,
    default=None,
    help="[Not yet implemented] Sweep dynamic dimensions. e.g. seq_len=32,64,128",
)

# ── Strict output flag (unchanged) ──────────────────────────────────────────
@click.option(
    "--strict-output", "strict_output",
    is_flag=True, default=False,
    help="Hard-fail on output validation warnings (overrides config.toml).",
)
def build(
    model:           Path,
    backend:         str,
    target:          str,
    name:            str,
    quantize:        str,
    notes:           str,
    task:            Optional[str],
    force_domain:    Optional[str],
    force_subtype:   Optional[str],
    force_execution: Optional[str],
    dynamic_sweep:   Optional[str],
    strict_output:   bool,
):
    # --dynamic-sweep: flag accepted, implementation deferred
    if dynamic_sweep is not None:
        console.print(
            "[dim]--dynamic-sweep flag received. "
            "Sweep mode is not yet implemented — running with single resolved shape.[/dim]\n"
        )
        logger.info("dynamic_sweep_flag_received  value=%s  not_yet_implemented", dynamic_sweep)

    strict_cfg = StrictOutputConfig.from_command(
        strict_flag   = strict_output,
        global_strict = _read_global_strict(),
    )

    # ── Resolve backend ──────────────────────────────────────────────────────
    from ...backends.registry import BackendRegistry

    try:
        backend_inst = BackendRegistry.get_backend(backend)
    except (ValueError, RuntimeError) as e:
        console.print(f"\n[red]Backend Error:[/red] {e}\n")
        sys.exit(1)

    # ============================================================
    # TFLite path
    # ============================================================
    if backend != "coreml":
        try:
            env_data        = collect_environment()
            env_fingerprint = hash_environment(env_data)
            source_hash     = compute_source_hash(model)

            config = {
                "target":       target,
                "quantization": {"type": quantize},
                "optimizer":    {},
            }
            config_hash = compute_config_hash(config)

            console.print(f"\n[bold]Building:[/bold] {Path(model).name}")
            console.print(f"Backend: {backend}")
            console.print(f"Target:  {target}")
            console.print(f"Quantization: {quantize}\n")

            # --- device-connected: resolve ABI from connected device ---
            original_target = target 
            device_abi  = None
            device_name = None

            if target == "device-connected":
                try:
                    from mlbuild.platforms.android.introspect import build_profile as _build_profile
                    console.print("[dim]Detecting connected Android device...[/dim]")
                    _profile = _build_profile()
                    device_abi  = _profile.primary_abi
                    device_name = f"{_profile.manufacturer} {_profile.model} (device-connected)"
                    # Remap to a concrete target the converter understands
                    _abi_to_target = {
                        "arm64-v8a":   "android_arm64",
                        "armeabi-v7a": "android_arm32",
                        "x86_64":      "android_arm64",  # closest supported
                    }
                    target = _abi_to_target.get(_profile.primary_abi, "android_arm64")
                    console.print(
                        f"  Device:  [bold]{device_name}[/bold]\n"
                        f"  ABI:     {device_abi}\n"
                        f"  Target:  {target}\n"
                    )
                except Exception as exc:
                    console.print(f"\n[red]No Android device detected.[/red] {exc}\n")
                    console.print("[dim]Connect a device via USB-C with USB debugging enabled.[/dim]\n")
                    sys.exit(1)

            representative_dataset = None
            if quantize == "int8":
                console.print("[dim]Generating INT8 calibration data...[/dim]")
                import onnx
                import numpy as np
                onnx_model   = onnx.load(str(model))
                input_shapes = backend_inst._get_input_shapes(onnx_model)
                def _make_representative_dataset(shapes):
                    def generator():
                        for _ in range(100):
                            yield [np.random.randn(*shape).astype(np.float32) for _, shape in shapes]
                    return generator
                representative_dataset = _make_representative_dataset(input_shapes)

            with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
                tp = p.add_task(f"Converting to TFLite ({quantize.upper()})...", total=None)
                tflite_path = backend_inst._export(
                    model_path            = Path(model),
                    quantize              = quantize,
                    name                  = name,
                    representative_dataset= representative_dataset,
                )
                p.update(tp, completed=True)

            artifact_hash = compute_source_hash(tflite_path)
            build_id      = _structured_build_id(
                source_hash, config_hash, artifact_hash,
                env_fingerprint, MLBUILD_VERSION,
            )

            artifacts_root = Path(".mlbuild/artifacts").resolve()
            final_path     = artifacts_root / f"{artifact_hash[:16]}_{name or tflite_path.stem}.tflite"
            artifacts_root.mkdir(parents=True, exist_ok=True)

            if not final_path.exists():
                shutil.copy2(str(tflite_path), str(final_path))
                graphs_root = Path(".mlbuild/graphs").resolve()
                graphs_root.mkdir(parents=True, exist_ok=True)
                graph_dest = graphs_root / f"{build_id}.onnx"
                if not graph_dest.exists():
                    shutil.copy2(str(model), str(graph_dest))
                graph_path = f"graphs/{build_id}.onnx"

            size_mb = Decimal(str(round(final_path.stat().st_size / (1024 * 1024), 6)))

            backend_versions = {"tflite": backend_inst._tf_version()}
            if "onnx2tf" in sys.modules:
                import onnx2tf
                backend_versions["onnx2tf"] = onnx2tf.__version__

            import onnx
            onnx_model = onnx.load(str(model))
            detection  = _detect_task_from_onnx(
                onnx_model,
                forced_task     = task,
                force_domain    = force_domain,
                force_subtype   = force_subtype,
                force_execution = force_execution,
            )

            warn = detection_warning(detection.result)
            if warn:
                console.print(f"\n[yellow]{warn}[/yellow]")

            # Build input roles and benchmark caveats
            _, input_roles = build_input_schemas_with_roles(
                detection.info, detection.result, detection.profile,
            )
            benchmark_caveats = _build_benchmark_caveats(detection.profile)
            model_profile_json = _profile_to_json(detection.profile)

            build_obj = Build(
                build_id         = build_id,
                artifact_hash    = artifact_hash,
                source_hash      = source_hash,
                config_hash      = config_hash,
                env_fingerprint  = env_fingerprint,
                name             = name,
                notes            = notes,
                created_at       = datetime.now(timezone.utc),
                source_path      = str(Path(model).resolve()),
                target_device    = original_target, 
                format           = backend,
                quantization     = config["quantization"],
                optimizer_config = config["optimizer"],
                backend_versions = backend_versions,
                environment_data = env_data,
                mlbuild_version  = MLBUILD_VERSION,
                python_version   = env_data["python"]["version"],
                platform         = env_data["hardware"]["cpu"]["system"],
                os_version       = env_data["hardware"]["cpu"]["release"],
                artifact_path    = str(final_path),
                size_mb          = size_mb,
                has_graph        = True,
                graph_format     = "onnx",
                graph_path       = graph_path,
                task_type        = detection.profile.domain.value,
                subtype          = detection.profile.subtype.value,
                execution_mode   = detection.profile.execution.value,
                nms_inside       = detection.profile.nms_inside,
                state_optional   = detection.profile.state_optional,
                device_abi       = device_abi,
                device_name      = device_name,
                model_profile_json = model_profile_json,
                benchmark_caveats  = benchmark_caveats,
                input_roles        = input_roles,
            )

            registry = LocalRegistry()
            try:
                registry.save_build(build_obj)
            except Exception:
                pass  # Duplicate build — artifact still valid

            _print_success(
                build_id, artifact_hash, source_hash, config_hash,
                env_fingerprint, size_mb, final_path,
                detection.profile, benchmark_caveats,
            )
            if device_name:
                console.print(f"Device:        {device_name}")
                console.print(f"Device ABI:    {device_abi}")
            return

        except MLBuildError as e:
            console.print("\n[bold red]Build failed[/bold red]\n")
            console.print(e.format())
            sys.exit(e.exit_code.value)
        except Exception as e:
            console.print(f"\n[bold red]Build failed[/bold red]: {e}\n")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # ============================================================
    # CoreML path
    # ============================================================
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = "0"
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)

    console.print("[dim]Deterministic mode enabled (PYTHONHASHSEED=0, seeds set)[/dim]\n")
    console.print(f"\n[bold]Building:[/bold] {os.path.basename(model)}")
    console.print(f"Target: {target}")
    console.print(f"Quantization: {quantize}\n")

    try:
        env_data        = collect_environment()
        env_fingerprint = hash_environment(env_data)

        is_reproducible, repro_warnings = validate_reproducibility()
        if repro_warnings:
            console.print("\n[yellow]⚠️  Reproducibility warnings:[/yellow]")
            for w in repro_warnings:
                console.print(f"    {w}")
            if any("[CRITICAL]" in w for w in repro_warnings):
                console.print("\n[bold red]CRITICAL: Build cannot proceed[/bold red]")
                sys.exit(1)
            console.print("\n[bold yellow]WARNING: Build may not be reproducible[/bold yellow]")
            console.print("Run: export PYTHONHASHSEED=0\n")

        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
            tp = p.add_task("Loading ONNX...", total=None)
            ir = load_model(str(model))
            p.update(tp, completed=True)

        source_hash = compute_source_hash(model)

        config = {
            "target":       target,
            "quantization": {"type": quantize},
            "optimizer":    {"compute_units": "ALL"},
        }

        try:
            import coremltools as ct
            coremltools_version = ct.__version__
        except ImportError:
            coremltools_version = "unknown"

        config_hash = compute_config_hash(config, coremltools_version=coremltools_version)

        # INT8 calibration
        calibration_data = None
        if quantize == "int8":
            console.print("[dim]Generating INT8 calibration data...[/dim]")
            from ...backends.coreml.exporter import ModelIngestion
            _, _, shape_tuples = ModelIngestion.extract_input_specs(ir)
            cal_config = CalibrationConfig(
                sample_count = 100,
                input_shape  = tuple(shape_tuples[0]),
                preprocessing = PreprocessingConfig(),
                seed         = 42,
            )
            cal_dataset         = CalibrationDataset(cal_config)
            calibration_samples = cal_dataset.generate_synthetic()
            cal_fingerprint     = cal_dataset.compute_fingerprint()
            console.print(f"  Calibration samples: {cal_fingerprint.sample_count}")
            console.print(f"  Calibration hash:    {cal_fingerprint.data_hash[:16]}...\n")
            calibration_data = list(calibration_samples)

        # Task detection (v2)
        import onnx as _onnx
        _onnx_model = _onnx.load(str(model))
        detection   = _detect_task_from_onnx(
            _onnx_model,
            forced_task     = task,
            force_domain    = force_domain,
            force_subtype   = force_subtype,
            force_execution = force_execution,
        )

        warn = detection_warning(detection.result)
        if warn:
            console.print(f"[yellow]{warn}[/yellow]\n")

        if target == "device-connected":
            # Detect connected device — must be iOS for CoreML
            try:
                from mlbuild.platforms.android.adb import devices as adb_devices
                android_devices = adb_devices()
                if android_devices:
                    console.print(
                        "\n[red]Device mismatch.[/red]\n"
                        "Connected device is Android but backend is CoreML.\n"
                        "CoreML only runs on Apple devices.\n"
                        "[dim]Use --backend tflite for Android, or connect an iPhone.[/dim]\n"
                    )
                    sys.exit(1)
            except Exception:
                pass

            # Detect connected iPhone chip via devicectl
            original_target = target
            try:
                from mlbuild.platforms.ios import idb as _idb
                from mlbuild.platforms.ios.chip_map import CHIP_MAP as _CHIP_MAP
                _idb.ensure_companion()
                _info = _idb.describe()
                _model = _info.get("model", "")
                _chip = _CHIP_MAP.get(_model, "")
                _chip_to_target = {
                    "A18 Pro": "apple_a18", "A18": "apple_a18",
                    "A17 Pro": "apple_a17",
                    "A16 Bionic": "apple_a16",
                    "A15 Bionic": "apple_a15",
                    "A14 Bionic": "apple_a14",
                }
                target = _chip_to_target.get(_chip, "apple_a17")
                console.print(f"  Detected: [bold]{_info.get('name', 'iPhone')}[/bold] ({_chip}) → target={target}\n")
            except Exception as exc:
                console.print(f"[yellow]⚠  Could not detect iPhone chip ({exc}) — defaulting to apple_a17[/yellow]\n")
                target = "apple_a17"

        # Convert
        # Resolve phone name to chip target
        from mlbuild.backends.coreml.exporter import resolve_target as _resolve_target
        target = _resolve_target(target)

        with tempfile.TemporaryDirectory() as tmp_root:
            tmp_root = Path(tmp_root)
            import io, contextlib

            with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
                tp = p.add_task("Converting to CoreML...", total=None)
                exporter = CoreMLExporter(target=target)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    mlpackage_path, _ = exporter.export(
                        ir,
                        output_dir     = tmp_root,
                        quantization   = quantize,
                        calibration_data = calibration_data,
                    )
                p.update(tp, completed=True)

            artifact_hash = compute_artifact_hash(mlpackage_path)
            build_id      = _structured_build_id(
                source_hash, config_hash, artifact_hash,
                env_fingerprint, MLBUILD_VERSION,
            )

            artifacts_root = Path(".mlbuild/artifacts").resolve()
            final_dir      = artifacts_root / artifact_hash
            artifacts_root.mkdir(parents=True, exist_ok=True)

            if final_dir.exists():
                existing_hash = compute_artifact_hash(final_dir)
                if existing_hash != artifact_hash:
                    raise RuntimeError("Artifact hash mismatch — possible corruption")
                console.print(f"[yellow]Reusing existing artifact {artifact_hash[:12]}[/yellow]")
            else:
                shutil.move(str(mlpackage_path), str(final_dir))
                verified_hash = compute_artifact_hash(final_dir)
                if verified_hash != artifact_hash:
                    shutil.rmtree(final_dir, ignore_errors=True)
                    raise RuntimeError("Artifact integrity verification failed")

            graphs_root = Path(".mlbuild/graphs").resolve()
            graphs_root.mkdir(parents=True, exist_ok=True)
            graph_dest = graphs_root / f"{build_id}.onnx"
            if not graph_dest.exists():
                shutil.copy2(str(model), str(graph_dest))
            graph_path = f"graphs/{build_id}.onnx"

            size_bytes = _directory_size_bytes(final_dir)
            size_mb    = Decimal(size_bytes) / Decimal(1024 * 1024)

            backend_versions: dict = {}
            for key, subkey in [("numpy", "numpy"), ("torch", "torch"),
                                 ("tensorflow", "tensorflow"), ("onnxruntime", "onnxruntime")]:
                if key in env_data and env_data[key].get("installed"):
                    backend_versions[subkey] = env_data[key]["version"]
            try:
                import coremltools as ct
                backend_versions["coremltools"] = ct.__version__
            except ImportError:
                backend_versions["coremltools"] = "unknown"
            backend_versions["onnx"] = ir.metadata.get("framework_version", "unknown")

            # Build input roles and benchmark caveats (v2)
            _, input_roles = build_input_schemas_with_roles(
                detection.info, detection.result, detection.profile,
            )
            benchmark_caveats  = _build_benchmark_caveats(detection.profile)
            model_profile_json = _profile_to_json(detection.profile)

            build_obj = Build(
                build_id         = build_id,
                artifact_hash    = artifact_hash,
                source_hash      = source_hash,
                config_hash      = config_hash,
                env_fingerprint  = env_fingerprint,
                name             = name,
                notes            = notes,
                created_at       = datetime.now(timezone.utc),
                source_path      = str(Path(model).resolve()),
                target_device    = original_target,
                format           = "coreml",
                quantization     = config["quantization"],
                optimizer_config = config["optimizer"],
                backend_versions = backend_versions,
                environment_data = env_data,
                mlbuild_version  = MLBUILD_VERSION,
                python_version   = env_data["python"]["version"],
                platform         = env_data["hardware"]["cpu"]["system"],
                os_version       = env_data["hardware"]["cpu"]["release"],
                artifact_path    = str(final_dir),
                size_mb          = size_mb,
                has_graph        = True,
                graph_format     = "onnx",
                graph_path       = graph_path,
                task_type        = detection.profile.domain.value,
                subtype          = detection.profile.subtype.value,
                execution_mode   = detection.profile.execution.value,
                nms_inside       = detection.profile.nms_inside,
                state_optional   = detection.profile.state_optional,
                model_profile_json = model_profile_json,
                benchmark_caveats  = benchmark_caveats,
                input_roles        = input_roles,
            )

            registry = LocalRegistry()
            try:
                registry.save_build(build_obj)
            except Exception:
                shutil.rmtree(final_dir, ignore_errors=True)
                raise

        _print_success(
            build_id, artifact_hash, source_hash, config_hash,
            env_fingerprint, size_mb, final_dir,
            detection.profile, benchmark_caveats,
        )

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


def _print_success(
    build_id:          str,
    artifact_hash:     str,
    source_hash:       str,
    config_hash:       str,
    env_fingerprint:   str,
    size_mb:           Decimal,
    artifact_path:     Path,
    profile:           ModelProfile,
    benchmark_caveats: List[str],
) -> None:
    """Unified success output for both CoreML and TFLite builds."""
    console.print(f"\n[bold green]✓ Build complete[/bold green]")
    console.print(f"Build ID:      {build_id[:16]}...")
    console.print(f"Artifact Hash: {artifact_hash[:16]}...")
    console.print(f"Source Hash:   {source_hash[:16]}...")
    console.print(f"Config Hash:   {config_hash[:16]}...")
    console.print(f"Env:           {env_fingerprint[:16]}...")
    console.print(f"Size:          {size_mb:.2f} MB")
    console.print(f"Artifact:      {artifact_path}", overflow="fold")
    console.print(f"Domain:        {profile.domain.value}")
    console.print(f"Subtype:       {profile.subtype.value}")
    console.print(f"Execution:     {profile.execution.value}")
    console.print(f"Confidence:    {profile.confidence:.2f} ({profile.confidence_tier})")
    if profile.nms_inside:
        console.print(f"NMS:           inside graph")
    if benchmark_caveats:
        console.print(f"\n[dim]Benchmark caveats:[/dim]")
        for caveat in benchmark_caveats:
            console.print(f"  [dim]⚠  {caveat}[/dim]")
    console.print()


# ============================================================
# Programmatic build API  (updated for Step 4)
# ============================================================

def run_build(
    model_path:      Path,
    target:          str,
    name:            str,
    quantization:    str             = "fp32",
    format:          str             = "coreml",
    registry:        Optional["LocalRegistry"] = None,
    notes:           Optional[str]   = None,
    task:            Optional[str]   = None,
    force_domain:    Optional[str]   = None,
    force_subtype:   Optional[str]   = None,
    force_execution: Optional[str]   = None,
) -> "Build":
    """
    Programmatic build API — no console output, no sys.exit.
    Returns the registered Build object.
    Used by: mlbuild explore, mlbuild optimize (future).
    """
    from ...registry.local import LocalRegistry as _LocalRegistry

    model_path = Path(model_path)
    _registry  = registry or _LocalRegistry()

    kwargs = dict(
        model_path      = model_path,
        target          = target,
        name            = name,
        quantize        = quantization,
        notes           = notes,
        task            = task,
        force_domain    = force_domain,
        force_subtype   = force_subtype,
        force_execution = force_execution,
        registry        = _registry,
    )

    if format == "coreml":
        return _run_coreml_build(**kwargs)
    elif format == "tflite":
        return _run_tflite_build(**kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _run_coreml_build(
    model_path:      Path,
    target:          str,
    name:            str,
    quantize:        str,
    notes:           Optional[str],
    task:            Optional[str],
    force_domain:    Optional[str],
    force_subtype:   Optional[str],
    force_execution: Optional[str],
    registry:        "LocalRegistry",
) -> "Build":
    import shutil, tempfile, io, contextlib

    env_data        = collect_environment()
    env_fingerprint = hash_environment(env_data)
    ir              = load_model(str(model_path))
    source_hash     = compute_source_hash(model_path)

    try:
        import coremltools as ct
        coremltools_version = ct.__version__
    except ImportError:
        coremltools_version = "unknown"

    config = {
        "target":       target,
        "quantization": {"type": quantize},
        "optimizer":    {"compute_units": "ALL"},
    }
    config_hash = compute_config_hash(config, coremltools_version=coremltools_version)

    import onnx as _onnx
    _onnx_model = _onnx.load(str(model_path))
    detection   = _detect_task_from_onnx(
        _onnx_model,
        forced_task     = task,
        force_domain    = force_domain,
        force_subtype   = force_subtype,
        force_execution = force_execution,
    )

    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_root = Path(tmp_root)
        exporter = CoreMLExporter(target=target)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            mlpackage_path, _ = exporter.export(
                ir, output_dir=tmp_root, quantization=quantize, calibration_data=None,
            )

        artifact_hash = compute_artifact_hash(mlpackage_path)
        build_id      = _structured_build_id(
            source_hash, config_hash, artifact_hash, env_fingerprint, MLBUILD_VERSION,
        )

        artifacts_root = Path(".mlbuild/artifacts").resolve()
        final_dir      = artifacts_root / artifact_hash
        artifacts_root.mkdir(parents=True, exist_ok=True)
        if not final_dir.exists():
            shutil.move(str(mlpackage_path), str(final_dir))

        graphs_root = Path(".mlbuild/graphs").resolve()
        graphs_root.mkdir(parents=True, exist_ok=True)
        graph_dest = graphs_root / f"{build_id}.onnx"
        if not graph_dest.exists():
            shutil.copy2(str(model_path), str(graph_dest))
        graph_path = f"graphs/{build_id}.onnx"

        size_mb = Decimal(_directory_size_bytes(final_dir)) / Decimal(1024 * 1024)

        backend_versions: dict = {}
        try:
            import coremltools as ct
            backend_versions["coremltools"] = ct.__version__
        except ImportError:
            backend_versions["coremltools"] = "unknown"
        backend_versions["onnx"] = ir.metadata.get("framework_version", "unknown")

        _, input_roles    = build_input_schemas_with_roles(detection.info, detection.result, detection.profile)
        benchmark_caveats = _build_benchmark_caveats(detection.profile)

        build_obj = Build(
            build_id          = build_id,
            artifact_hash     = artifact_hash,
            source_hash       = source_hash,
            config_hash       = config_hash,
            env_fingerprint   = env_fingerprint,
            name              = name,
            notes             = notes,
            created_at        = datetime.now(timezone.utc),
            source_path       = str(model_path.resolve()),
            target_device     = original_target,
            format            = "coreml",
            quantization      = config["quantization"],
            optimizer_config  = config["optimizer"],
            backend_versions  = backend_versions,
            environment_data  = env_data,
            mlbuild_version   = MLBUILD_VERSION,
            python_version    = env_data["python"]["version"],
            platform          = env_data["hardware"]["cpu"]["system"],
            os_version        = env_data["hardware"]["cpu"]["release"],
            artifact_path     = str(final_dir),
            size_mb           = size_mb,
            has_graph         = True,
            graph_format      = "onnx",
            graph_path        = graph_path,
            task_type         = detection.profile.domain.value,
            subtype           = detection.profile.subtype.value,
            execution_mode    = detection.profile.execution.value,
            nms_inside        = detection.profile.nms_inside,
            state_optional    = detection.profile.state_optional,
            model_profile_json = _profile_to_json(detection.profile),
            benchmark_caveats  = benchmark_caveats,
            input_roles        = input_roles,
        )

        try:
            registry.save_build(build_obj)
        except Exception:
            pass

    return build_obj


def _run_tflite_build(
    model_path:      Path,
    target:          str,
    name:            str,
    quantize:        str,
    notes:           Optional[str],
    task:            Optional[str],
    force_domain:    Optional[str],
    force_subtype:   Optional[str],
    force_execution: Optional[str],
    registry:        "LocalRegistry",
) -> "Build":
    import shutil
    from ...backends.registry import BackendRegistry

    env_data        = collect_environment()
    env_fingerprint = hash_environment(env_data)
    source_hash     = compute_source_hash(model_path)

    config = {
        "target":       target,
        "quantization": {"type": quantize},
        "optimizer":    {},
    }
    config_hash  = compute_config_hash(config)
    backend_inst = BackendRegistry.get_backend("tflite")
    tflite_path  = backend_inst._export(
        model_path=model_path, quantize=quantize, name=name, representative_dataset=None,
    )

    artifact_hash  = compute_source_hash(tflite_path)
    build_id       = _structured_build_id(
        source_hash, config_hash, artifact_hash, env_fingerprint, MLBUILD_VERSION,
    )

    artifacts_root = Path(".mlbuild/artifacts").resolve()
    final_path     = artifacts_root / f"{artifact_hash[:16]}_{name or tflite_path.stem}.tflite"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    if not final_path.exists():
        shutil.copy2(str(tflite_path), str(final_path))

    graphs_root = Path(".mlbuild/graphs").resolve()
    graphs_root.mkdir(parents=True, exist_ok=True)
    graph_dest  = graphs_root / f"{build_id}.onnx"
    if not graph_dest.exists():
        shutil.copy2(str(model_path), str(graph_dest))
    graph_path = f"graphs/{build_id}.onnx"

    size_mb          = Decimal(str(round(final_path.stat().st_size / (1024 * 1024), 6)))
    backend_versions = {"tflite": backend_inst._tf_version()}

    import onnx as _onnx
    _onnx_model = _onnx.load(str(model_path))
    detection   = _detect_task_from_onnx(
        _onnx_model,
        forced_task     = task,
        force_domain    = force_domain,
        force_subtype   = force_subtype,
        force_execution = force_execution,
    )

    _, input_roles    = build_input_schemas_with_roles(detection.info, detection.result, detection.profile)
    benchmark_caveats = _build_benchmark_caveats(detection.profile)

    build_obj = Build(
        build_id          = build_id,
        artifact_hash     = artifact_hash,
        source_hash       = source_hash,
        config_hash       = config_hash,
        env_fingerprint   = env_fingerprint,
        name              = name,
        notes             = notes,
        created_at        = datetime.now(timezone.utc),
        source_path       = str(model_path.resolve()),
        target_device     = target,
        format            = "tflite",
        quantization      = config["quantization"],
        optimizer_config  = config["optimizer"],
        backend_versions  = backend_versions,
        environment_data  = env_data,
        mlbuild_version   = MLBUILD_VERSION,
        python_version    = env_data["python"]["version"],
        platform          = env_data["hardware"]["cpu"]["system"],
        os_version        = env_data["hardware"]["cpu"]["release"],
        artifact_path     = str(final_path),
        size_mb           = size_mb,
        has_graph         = True,
        graph_format      = "onnx",
        graph_path        = graph_path,
        task_type         = detection.profile.domain.value,
        subtype           = detection.profile.subtype.value,
        execution_mode    = detection.profile.execution.value,
        nms_inside        = detection.profile.nms_inside,
        state_optional    = detection.profile.state_optional,
        model_profile_json = _profile_to_json(detection.profile),
        benchmark_caveats  = benchmark_caveats,
        input_roles        = input_roles,
    )

    try:
        registry.save_build(build_obj)
    except Exception:
        pass

    return build_obj