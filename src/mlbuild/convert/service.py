"""
Conversion service — orchestrates the full pipeline.

Responsibilities:
- Determinism seeding
- Path resolution via graph
- Dry run support
- Cache lookup per step
- Temp dir lifecycle management
- Per-step timeout + SIGINT cancellation
- Executor dispatch
- Output validation
- Registry registration
- ConvertResult assembly

The CLI calls run_convert() and gets back a ConvertResult.
Nothing else in this file is public API.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import List, Optional, Tuple

from mlbuild.convert.cache import build_cache_key, collect_env_versions, find_cached_build
from mlbuild.convert.detector import detect_format
from mlbuild.convert.graph import resolve_path
from mlbuild.convert.types import (
    CacheInfo,
    ConvertContext,
    ConvertOutput,
    ConvertParams,
    ConvertResult,
    ConvertStatus,
    RunMetadata,
    StepResult,
    ValidationResult,
)
from mlbuild.convert.validator import validate_output, Format

import mlbuild.convert.pytorch      # registers ("pytorch", "onnx")
import mlbuild.convert.coreml       # registers ("pytorch", "coreml"), ("onnx", "coreml")
import mlbuild.convert.tensorflow   # registers ("onnx", "tflite"), ("savedmodel", "tflite")
from mlbuild.core.errors import ConvertError, ConversionCancelled, ErrorCode

logger = logging.getLogger("mlbuild.convert.service")

# Temp dir root
MLBUILD_TMP_ROOT = Path.home() / ".mlbuild" / "tmp"


# ---------------------------------------------------------------------
# Determinism seeding
# ---------------------------------------------------------------------

def _seed_determinism() -> bool:
    """
    Best-effort deterministic seeding.
    Called once at the start of every conversion run.
    Returns True if seeding was applied.
    """
    import random
    random.seed(0)

    seeded = True
    try:
        import torch
        torch.manual_seed(0)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        import numpy as np
        np.random.seed(0)
    except ImportError:
        pass

    return seeded


# ---------------------------------------------------------------------
# Temp dir management
# ---------------------------------------------------------------------

def _make_run_tmp_dir(run_id: str) -> Path:
    tmp = MLBUILD_TMP_ROOT / run_id
    tmp.mkdir(parents=True, exist_ok=False)
    return tmp


def _make_step_dir(run_tmp: Path, step_index: int, src: str, dst: str) -> Path:
    name = f"step_{step_index + 1}_{src}_to_{dst}"
    step_dir = run_tmp / name
    step_dir.mkdir(parents=True, exist_ok=True)
    return step_dir


def _cleanup_tmp(run_tmp: Path) -> None:
    try:
        shutil.rmtree(run_tmp)
    except Exception as e:
        logger.warning(f"Failed to clean up temp dir {run_tmp}: {e}")


# ---------------------------------------------------------------------
# SIGINT cancellation
# ---------------------------------------------------------------------





# ---------------------------------------------------------------------
# Timeout runner
# ---------------------------------------------------------------------

def _run_with_timeout(fn, args: tuple, timeout: int):
    """
    Run fn(*args) in a thread with a hard timeout.
    Raises ConvertError on timeout.
    """
    import threading

    result = [None]
    error = [None]

    def target():
        try:
            result[0] = fn(*args)
        except Exception as e:
            error[0] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise ConvertError(
            f"Conversion timed out after {timeout}s.",
            stage="timeout",
            error_code=ErrorCode.CONVERSION_FAILED,
        )

    if error[0] is not None:
        raise error[0]

    return result[0]


# ---------------------------------------------------------------------
# Format → validator Format enum mapping
# ---------------------------------------------------------------------

_FORMAT_TO_VALIDATOR = {
    "onnx":    Format.ONNX,
    "coreml":  Format.COREML,
    "tflite":  Format.TFLITE,
}


# ---------------------------------------------------------------------
# Registry registration
# ---------------------------------------------------------------------

def _register_artifact(
    output: ConvertOutput,
    ctx: ConvertContext,
    src_format: str,
    dst_format: str,
    cache_key: str,
    run_id: str,
    parent_build_id: Optional[str],
) -> Optional[str]:
    """
    Register a conversion output artifact in the MLBuild registry.
    Returns build_id on success, None on failure (non-fatal).
    """
    try:
        from mlbuild import __version__ as MLBUILD_VERSION
        from mlbuild.core.hash import compute_source_hash, compute_config_hash
        from mlbuild.core.environment import collect_environment, hash_environment
        from mlbuild.core.types import Build
        from mlbuild.registry import LocalRegistry

        artifact_path = output.path
        artifact_hash = compute_source_hash(artifact_path)
        source_hash   = compute_source_hash(ctx.input_path)

        config = {
            "quantization": {
                "type": ctx.params.quantize.value
                        if hasattr(ctx.params.quantize, "value")
                        else ctx.params.quantize
            },
            "optimizer": {},
            "src_format": src_format,
            "dst_format": dst_format,
            "converter_version": output.converter_version,
            "cache_key": cache_key,
        }

        config_hash = compute_config_hash(config)
        env_data = collect_environment()
        env_fingerprint = hash_environment(env_data)

        # Deterministic build_id from all inputs
        build_id = hashlib.sha256(
            f"{source_hash}{config_hash}{artifact_hash}{env_fingerprint}".encode()
        ).hexdigest()

        size_mb = Decimal(str(
            artifact_path.stat().st_size / (1024 * 1024)
            if artifact_path.is_file()
            else sum(f.stat().st_size for f in artifact_path.rglob("*") if f.is_file()) / (1024 * 1024)
        ))

        target_device = (
            ctx.params.target.value
            if ctx.params.target and hasattr(ctx.params.target, "value")
            else ctx.params.target or "unknown"
        )

        build = Build(
            build_id=build_id,
            artifact_hash=artifact_hash,
            source_hash=source_hash,
            config_hash=config_hash,
            env_fingerprint=env_fingerprint,
            name=ctx.params.name or artifact_path.stem,
            notes=ctx.params.notes,
            created_at=datetime.now(timezone.utc),
            source_path=str(ctx.input_path.resolve()),
            target_device=target_device,
            format=dst_format,
            quantization=config["quantization"],
            optimizer_config={},
            backend_versions=ctx.run.env_versions or {},
            environment_data=env_data,
            mlbuild_version=MLBUILD_VERSION,
            python_version=env_data.get("python", {}).get("version", platform.python_version()),
            platform=env_data.get("hardware", {}).get("cpu", {}).get("system", platform.system()),
            os_version=env_data.get("hardware", {}).get("cpu", {}).get("release", platform.release()),
            artifact_path=str(artifact_path.resolve()),
            size_mb=size_mb,
            parent_build_id=parent_build_id,
        )

        registry = LocalRegistry()
        registry.save_build(build)
        logger.info(f"Registered artifact: {build_id[:16]}...")
        return build_id

    except Exception as e:
        logger.exception("Registry write failed (non-fatal).")
        return None


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def run_convert(
    model_path: Path,
    target_format: str,
    target: Optional[str],
    input_shape: tuple,
    quantize: str,
    load_mode: str,
    opset: Optional[int],
    keep_intermediate: bool,
    name: Optional[str],
    notes: Optional[str],
    no_register: bool,
    debug: bool,
    timeout: int,
    dry_run: bool = False,
) -> ConvertResult:
    """
    Full conversion pipeline.

    1. Seed determinism
    2. Detect input format
    3. Resolve conversion path (BFS)
    4. Dry run: print plan and return early
    5. For each step:
        a. Check cache
        b. Run executor with timeout
        c. Validate output
        d. Optionally register in MLBuild registry
        e. Record StepResult
    6. Clean up temp dir (unless failure/debug)
    7. Return ConvertResult
    """
    from mlbuild.convert.types import QuantizeMode, LoadMode, TargetDevice

    run_id = uuid.uuid4().hex[:12]
    start_total = time.time()

    # --- Validate input_shape ---
    if not isinstance(input_shape, tuple) or not input_shape:
        raise ConvertError(
            "input_shape must be a non-empty tuple",
            error_code=ErrorCode.CONVERSION_FAILED,
        )

    for dim in input_shape:
        if not isinstance(dim, int) or dim <= 0:
            raise ConvertError(
                f"Invalid input_shape dimension: {input_shape}",
                error_code=ErrorCode.CONVERSION_FAILED,
            )
        if dim > 10_000:
            raise ConvertError(
                f"input_shape dimension too large (OOM risk): {input_shape}",
                error_code=ErrorCode.CONVERSION_FAILED,
            )

    # --- Seed determinism ---
    deterministic = _seed_determinism()

    # --- Collect env versions once ---
    env_versions = collect_env_versions()

    run_metadata = RunMetadata(
        run_id=run_id,
        env_versions=env_versions,
        deterministic=deterministic,
        seed=0,
    )

    # --- Build typed params ---
    try:
        quantize_mode = QuantizeMode(quantize)
    except ValueError:
        raise ConvertError(
            f"Invalid quantize value '{quantize}'. Use fp32 or fp16.",
            error_code=ErrorCode.CONVERSION_FAILED,
        )

    try:
        load_mode_enum = LoadMode(load_mode)
    except ValueError:
        raise ConvertError(
            f"Invalid load_mode '{load_mode}'. Use auto, jit, or eager.",
            error_code=ErrorCode.CONVERSION_FAILED,
        )

    target_enum = None
    if target:
        try:
            target_enum = TargetDevice(target)
        except ValueError:
            raise ConvertError(
                f"Invalid target '{target}'.",
                error_code=ErrorCode.CONVERSION_FAILED,
            )

    params = ConvertParams(
        input_shape=input_shape,
        quantize=quantize_mode,
        load_mode=load_mode_enum,
        opset=opset,
        target=target_enum,
        name=name,
        notes=notes,
    )

    # --- Detect input format ---
    src_format = detect_format(model_path)

    # --- Resolve path ---
    steps = resolve_path(src_format, target_format)
    total_steps = len(steps)

    # --- Feature → OS compatibility validation ---
    # Only runs for CoreML targets — ONNX and TFLite don't use OS deployment targets
    if target_format == "coreml" and target:
        from mlbuild.convert.feature_compat import infer_features, validate_features_against_target
        features = infer_features(params)
        errors, warnings = validate_features_against_target(features, target)

        for w in warnings:
            logger.warning(w.format())

        if errors:
            msg = "\n\n".join(e.format() for e in errors)
            raise ConvertError(
                msg,
                stage="feature_compat",
                error_code=ErrorCode.CONVERSION_FAILED,
            )
        
    # --- Dry run ---
    if dry_run:
        _print_dry_run(steps, params, env_versions, model_path, src_format)
        return ConvertResult(
            status=ConvertStatus.SUCCESS,
            final_path=None,
            final_format=target_format,
            final_build_id=None,
            steps=(),
            intermediate_build_ids=(),
            cache_hits=(),
            total_duration_seconds=0.0,
            run=run_metadata,
        )

    # --- Setup temp dir ---
    run_tmp = _make_run_tmp_dir(run_id)

    # --- Setup cancellation ---


    step_results: List[StepResult] = []
    intermediate_build_ids: List[str] = []
    cache_hits: List[str] = []
    current_input = model_path
    final_build_id = None
    parent_build_id = None
    success = False

    try:
        for i, step in enumerate(steps):
            is_final = (i == total_steps - 1)
            step_dir = _make_step_dir(run_tmp, i, step.src, step.dst)

            ctx = ConvertContext(
                input_path=current_input,
                output_dir=step_dir,
                src_format=step.src,
                dst_format=step.dst,
                step_index=i,
                total_steps=total_steps,
                params=params,
                run=run_metadata,
            )

            # --- Cache lookup ---
            from mlbuild.registry import LocalRegistry
            registry = LocalRegistry()

            # Get converter version from executor module attr if available
            module = sys.modules.get(step.executor.__module__)
            converter_version = getattr(module, "CONVERTER_VERSION", "unknown")

            cache_key = build_cache_key(
                input_path=current_input,
                src_format=step.src,
                dst_format=step.dst,
                params=params,
                converter_version=converter_version,
                env_versions=env_versions,
                executor_name=step.executor.__name__,
            )

            cached = find_cached_build(cache_key, registry)

            if cached is not None:
                cached_path = Path(cached.artifact_path)

                if not cached_path.exists() or cached_path.stat().st_size == 0:
                    logger.warning("Cache artifact invalid — ignoring cache.")
                else:
                    logger.info(f"Cache hit for step {i+1}: {step.src} → {step.dst}")
                    cache_hits.append(cache_key)

                    # 🔧 lightweight validation instead of blind trust
                    validator_fmt = _FORMAT_TO_VALIDATOR.get(step.dst)
                    if validator_fmt:
                        validation = validate_output(cached_path, validator_fmt)
                        if not validation.passed:
                            logger.warning("Cached artifact failed validation — recomputing.")
                        else:
                            cache_info = CacheInfo(
                                hit=True,
                                cache_key=cache_key,
                                source_build_id=cached.build_id,
                            )

                            step_result = StepResult(
                                src_format=step.src,
                                dst_format=step.dst,
                                output_path=cached_path,
                                file_size_mb=float(cached.size_mb),
                                duration_seconds=0.0,
                                conversion=ConvertOutput(
                                    path=cached_path,
                                    converter_version=converter_version,
                                ),
                                validation=validation,
                                cache=cache_info,
                                build_id=cached.build_id,
                                parent_build_id=None,
                            )

                            step_results.append(step_result)
                            current_input = cached_path
                            if is_final:
                                final_build_id = cached.build_id
                            continue

            # --- Execute step ---
            step_start = time.time()
            try:
                output: ConvertOutput = _run_with_timeout(
                    step.executor, (ctx,), timeout=timeout
                )
            except ConversionCancelled:
                raise
            except ConvertError:
                raise
            except Exception as e:
                raise ConvertError(
                    f"Executor {step.executor.__name__} failed: {e}",
                    stage=f"{step.src}→{step.dst}",
                    error_code=ErrorCode.CONVERSION_FAILED,
                )

            step_duration = time.time() - step_start

            # --- Validate output ---
            validator_fmt = _FORMAT_TO_VALIDATOR.get(step.dst)
            if validator_fmt is not None:
                validation = validate_output(output.path, validator_fmt)
                if not validation.passed:
                    raise ConvertError(
                        f"Output validation failed for {step.dst}: "
                        f"{validation.warnings}",
                        stage=f"validate:{step.dst}",
                        error_code=ErrorCode.CONVERSION_FAILED,
                    )
            else:
                validation = ValidationResult(
                    passed=True,
                    format=step.dst,
                    warnings=["No validator available for this format"],
                )

            # --- Register artifact ---
            build_id = None
            if not no_register and (is_final or keep_intermediate):
                build_id = _register_artifact(
                    output=output,
                    ctx=ctx,
                    src_format=step.src,
                    dst_format=step.dst,
                    cache_key=cache_key,
                    run_id=run_id,
                    parent_build_id=parent_build_id,
                )
                if build_id:
                    if not is_final:
                        intermediate_build_ids.append(build_id)
                    else:
                        final_build_id = build_id
                    parent_build_id = build_id

            # --- Compute file size ---
            if output.path.is_file():
                size_mb = output.path.stat().st_size / (1024 * 1024)
            elif output.path.is_dir():
                size_mb = sum(
                    f.stat().st_size for f in output.path.rglob("*") if f.is_file()
                ) / (1024 * 1024)
            else:
                size_mb = 0.0

            cache_info = CacheInfo(
                hit=False,
                cache_key=cache_key,
                source_build_id=None,
            )

            step_result = StepResult(
                src_format=step.src,
                dst_format=step.dst,
                output_path=output.path,
                file_size_mb=size_mb,
                duration_seconds=step_duration,
                conversion=output,
                validation=validation,
                cache=cache_info,
                build_id=build_id,
                parent_build_id=parent_build_id if i > 0 else None,
            )
            step_results.append(step_result)
            current_input = output.path

        success = True


        # --- Cleanup on success ---
        if not debug:
            _cleanup_tmp(run_tmp)

        total_duration = time.time() - start_total
        final_step = step_results[-1] if step_results else None

        return ConvertResult(
            status=ConvertStatus.SUCCESS,
            final_path=final_step.output_path if final_step else None,
            final_format=target_format,
            final_build_id=final_build_id,
            steps=tuple(step_results),
            intermediate_build_ids=tuple(intermediate_build_ids),
            cache_hits=tuple(cache_hits),
            total_duration_seconds=total_duration,
            run=run_metadata,
        )

    except ConversionCancelled as e:
        logger.info(f"Conversion cancelled at step. Temp dir preserved: {run_tmp}")
        total_duration = time.time() - start_total
        return ConvertResult(
            status=ConvertStatus.CANCELLED,
            final_path=None,
            final_format=target_format,
            final_build_id=None,
            steps=tuple(step_results),
            intermediate_build_ids=tuple(intermediate_build_ids),
            cache_hits=tuple(cache_hits),
            total_duration_seconds=total_duration,
            run=run_metadata,
        )

    except Exception:

        # Always preserve temp on failure
        logger.info(f"Conversion failed. Temp dir preserved: {run_tmp}")
        if debug:
            logger.info("--debug: temp dir preserved on success too")
        raise


# ---------------------------------------------------------------------
# Dry run output
# ---------------------------------------------------------------------

def _print_dry_run(steps, params, env_versions, model_path, src_format):
    print()
    print("Plan:")
    for i, step in enumerate(steps):
        print(f"  Step {i+1}  {step.src} → {step.dst}    reason: {step.reason}")

    print()
    print("Cache:")
    print("  (cache lookup requires executing the pipeline)")

    print()
    print(f"Input:  {model_path}  [{src_format}]")
    print(f"Output: {steps[-1].dst}")
    print()
    print("No files will be written. Remove --dry-run to execute.")
    print()