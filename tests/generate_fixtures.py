"""
MLBuild test fixture generator — Step 1: Scaffold and shared utilities.

Usage
-----
    python generate_fixtures.py [--out-dir tests/fixtures] [--skip-coreml] [--skip-tflite]

What this script does
---------------------
Generates all 55 model fixture files (ONNX, CoreML, TFLite) used by the
MLBuild test suite.  Model weights are dummy (random, seeded) — architectures
are faithful so that input shapes, op sets, and execution-mode routing all
behave identically to real models.

Phases
------
Phase 1 (this file)  — Scaffold: seed, directories, shared helpers, manifest.
Phase 2              — Baseline ONNX: B01, B05, B09, B13.
Phase 3              — CoreML variants: B02/B03, B06/B07, B10/B11, B14/B15.
Phase 4              — TFLite variants: B04, B08, B12, B16.
Phase 5              — Detection baseline: D01–D04.
Phase 6              — Time-series baseline: T01–T04.
Phase 7              — Multimodal baseline: M01–M04.
Phase 8              — Recommendation baseline: R01–R02.
Phase 9              — Generative baseline: G01–G02.
Phase 10             — Stress models: S01–S14, SD1, ST1–ST2, SM1, SR1, SG1.
Phase 11             — Expected-failure ONNX sources: S09, S10, SD2, SM2, SG2.
Phase 12             — Manifest: write manifest.json with SHA256 per file.

Random seed
-----------
All models use seed=42 globally.  Same seed → identical bytes → deterministic
SHA256 hashes in the manifest.  Re-running the script on the same machine
produces byte-identical files.

CoreML availability
-------------------
CoreML conversion requires macOS + coremltools>=7.  On Linux the CoreML steps
are skipped gracefully and logged as warnings.  All ONNX and TFLite fixtures
are generated on any platform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Seed everything before importing torch / numpy ────────────────────────────
GLOBAL_SEED = 42

import numpy as np

np.random.seed(GLOBAL_SEED)

TORCH_AVAILABLE = False
try:
    import torch
    torch.manual_seed(GLOBAL_SEED)
    torch.cuda.manual_seed_all(GLOBAL_SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    TORCH_AVAILABLE = True
except ImportError:
    pass

# ── Platform guards ───────────────────────────────────────────────────────────
IS_MACOS        = platform.system() == "Darwin"
COREML_AVAILABLE = False

if IS_MACOS:
    try:
        import coremltools as ct  # noqa: F401
        COREML_AVAILABLE = True
    except ImportError:
        pass

TFLITE_AVAILABLE = False
try:
    import tensorflow  # noqa: F401
    TFLITE_AVAILABLE = True
except ImportError:
    pass

ONNX_AVAILABLE = False
try:
    import onnx  # noqa: F401
    ONNX_AVAILABLE = True
except ImportError:
    pass

ONNX2TF_AVAILABLE = False
try:
    import onnx2tf  # noqa: F401
    ONNX2TF_AVAILABLE = True
except ImportError:
    pass

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("mlbuild.fixtures")


# ============================================================
# Directory layout
# ============================================================

class FixtureDirs:
    """
    Canonical directory layout for the fixture tree.

    fixtures/
      onnx/        .onnx files (source for everything)
      coreml/      .mlmodel and .mlpackage files
      tflite/      .tflite files
      manifest.json
    """

    def __init__(self, root: Path):
        self.root    = root
        self.onnx    = root / "onnx"
        self.coreml  = root / "coreml"
        self.tflite  = root / "tflite"
        self.manifest = root / "manifest.json"

    def create(self) -> None:
        for d in (self.root, self.onnx, self.coreml, self.tflite):
            d.mkdir(parents=True, exist_ok=True)
        logger.info("fixture dirs created at %s", self.root)

    def onnx_path(self, model_id: str, stem: str) -> Path:
        return self.onnx / f"{model_id}_{stem}.onnx"

    def coreml_path(self, model_id: str, stem: str, fmt: str) -> Path:
        """fmt: 'mlmodel' | 'mlpackage'"""
        ext = "mlmodel" if fmt == "mlmodel" else "mlpackage"
        return self.coreml / f"{model_id}_{stem}.{ext}"

    def tflite_path(self, model_id: str, stem: str) -> Path:
        return self.tflite / f"{model_id}_{stem}.tflite"


# ============================================================
# Manifest entry
# ============================================================

@dataclass
class ManifestEntry:
    """
    One entry per model ID in manifest.json.

    All fields used by conftest.py and the unit/integration tests.
    """
    # Identity
    model_id:   str
    tier:       str   # "baseline" | "stress" | "stress/fail"
    modality:   str   # "vision" | "tabular" | "audio" | "nlp" | "detection"
                      # | "timeseries" | "multimodal" | "rec" | "generative"
    format:     str   # "onnx" | "mlmodel" | "mlpackage" | "tflite"
    size_class: str   # "tiny" | "medium" | "large"

    # File location (relative to fixture root)
    file: str

    # Expected ModelProfile fields
    expected_domain:    str   # "vision" | "nlp" | "audio" | "tabular"
    expected_subtype:   str   # "detection" | "timeseries" | ... | "none"
    expected_execution: str   # "standard" | "stateful" | "kv_cache" | ...
    expected_nms_inside: bool = False

    # Failure metadata
    expected_failure: bool         = False
    failure_target:   Optional[str] = None   # "coreml" | "tflite" | None
    failure_reason:   Optional[str] = None   # "dynamic_sequence_unsupported" | ...

    # Commands this model should exercise
    commands: List[str] = field(default_factory=list)

    # File integrity (populated after generation)
    sha256: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# Shared export helpers
# ============================================================

def _sha256(path: Path) -> str:
    """
    Compute SHA256 of a file, directory, or ONNX external-data pair.

    ONNX external data: torch 2.10+ writes large models as two files:
      model.onnx       — graph structure
      model.onnx.data  — weight tensors

    Both are hashed together so the SHA256 covers the full model.
    Directories (.mlpackage) hash all contained files sorted by name.
    """
    h = hashlib.sha256()
    if path.is_file():
        h.update(path.read_bytes())
        # Include companion .data file if present (ONNX external data format)
        data_file = path.with_suffix(path.suffix + ".data")
        if data_file.exists():
            h.update(data_file.name.encode())
            h.update(data_file.read_bytes())
    elif path.is_dir():
        # Directories (.mlpackage) — hash sorted file contents
        for f in sorted(path.rglob("*")):
            if f.is_file():
                h.update(f.name.encode())
                h.update(f.read_bytes())
    return h.hexdigest()


def _export_onnx(
    model:        "torch.nn.Module",
    dummy_inputs: tuple,
    path:         Path,
    input_names:  List[str],
    output_names: List[str],
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    opset:        int                                  = 18,
) -> Path:
    """
    Export a PyTorch model to ONNX.

    Parameters
    ----------
    model        : nn.Module in eval mode
    dummy_inputs : tuple of tensors matching the model's forward() signature
    path         : output .onnx file path
    input_names  : tensor names for ONNX graph inputs
    output_names : tensor names for ONNX graph outputs
    dynamic_axes : optional dynamic axis spec (e.g. {0: "batch"})
    opset        : ONNX opset version (default 18)

    Returns
    -------
    path — the written file path (for chaining)
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is required for ONNX export")

    import torch

    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_inputs,
            str(path),
            input_names        = input_names,
            output_names       = output_names,
            dynamic_axes       = dynamic_axes or {},
            opset_version      = opset,
            do_constant_folding = True,
        )

    # Validate the exported graph
    if ONNX_AVAILABLE:
        import onnx as _onnx
        _onnx.checker.check_model(str(path))

    logger.info("exported ONNX  %s  (%s)", path.name, _human_size(path))
    return path


def _to_coreml_nn(
    torch_model,
    example_inputs: tuple,
    input_names:    list,
    out_path:       Path,
    min_ios:        str = "14",
) -> Optional[Path]:
    """
    Convert a PyTorch model to CoreML NeuralNetwork format (.mlmodel).

    Parameters
    ----------
    torch_model    : nn.Module in eval mode
    example_inputs : tuple of example tensors (same as used for ONNX export)
    input_names    : list of input tensor names (for ct.TensorType)
    out_path       : destination .mlmodel file
    min_ios        : minimum iOS deployment target (default "14" → NeuralNetwork)

    Returns
    -------
    out_path on success, None if CoreML is unavailable (non-macOS).
    Logs a warning and returns None gracefully on any conversion error.

    Note: ONNX → CoreML via coremltools is not supported. CoreML conversion
    must go PyTorch → TorchScript → CoreML directly (source="pytorch").
    """
    if not COREML_AVAILABLE:
        logger.warning(
            "skipping CoreML .mlmodel for %s — coremltools not available",
            out_path.name,
        )
        return None

    import coremltools as ct
    import torch

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        torch_model.eval()
        with torch.no_grad():
            traced = torch.jit.trace(torch_model, example_inputs)

        ct_inputs = [
            ct.TensorType(name=name, shape=inp.shape)
            for name, inp in zip(input_names, example_inputs)
        ]

        model = ct.convert(
            traced,
            source                    = "pytorch",
            inputs                    = ct_inputs,
            minimum_deployment_target = ct.target.iOS14,
            convert_to                = "neuralnetwork",
        )
        model.save(str(out_path))
        logger.info("exported CoreML .mlmodel  %s  (%s)", out_path.name, _human_size(out_path))
        return out_path

    except Exception as e:
        logger.warning(
            "CoreML .mlmodel conversion failed for %s: %s", out_path.name, e
        )
        return None


def _to_coreml_mlprogram(
    torch_model,
    example_inputs: tuple,
    input_names:    list,
    out_path:       Path,
    min_ios:        str  = "15",
    fp16:           bool = False,
) -> Optional[Path]:
    """
    Convert a PyTorch model to CoreML MLProgram format (.mlpackage).

    Parameters
    ----------
    torch_model    : nn.Module in eval mode
    example_inputs : tuple of example tensors
    input_names    : list of input tensor names
    out_path       : destination .mlpackage directory
    min_ios        : minimum iOS deployment target (default "15" → MLProgram)
    fp16           : if True, apply linear weight quantization after conversion

    Returns
    -------
    out_path on success, None if CoreML is unavailable.

    Note: ONNX → CoreML via coremltools is not supported. CoreML conversion
    must go PyTorch → TorchScript → CoreML directly (source="pytorch").
    """
    if not COREML_AVAILABLE:
        logger.warning(
            "skipping CoreML .mlpackage for %s — coremltools not available",
            out_path.name,
        )
        return None

    import coremltools as ct
    import torch

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        torch_model.eval()
        with torch.no_grad():
            traced = torch.jit.trace(torch_model, example_inputs)

        ct_inputs = [
            ct.TensorType(name=name, shape=inp.shape)
            for name, inp in zip(input_names, example_inputs)
        ]

        model = ct.convert(
            traced,
            source                    = "pytorch",
            inputs                    = ct_inputs,
            minimum_deployment_target = ct.target.iOS15,
            convert_to                = "mlprogram",
        )

        if fp16:
            from coremltools.optimize.coreml import (
                OpLinearQuantizerConfig,
                OptimizationConfig,
                linear_quantize_weights,
            )
            op_config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
            config    = OptimizationConfig(global_config=op_config)
            model     = linear_quantize_weights(model, config)

        model.save(str(out_path))
        logger.info(
            "exported CoreML .mlpackage  %s  (%s)",
            out_path.name,
            _human_size(out_path),
        )
        return out_path

    except Exception as e:
        logger.warning(
            "CoreML .mlpackage conversion failed for %s: %s", out_path.name, e
        )
        return None


def _to_tflite_from_tf(
    tf_model,
    out_path:     Path,
    quantize_int8: bool = False,
) -> Optional[Path]:
    """
    Convert a TensorFlow/Keras model to TFLite.

    Parameters
    ----------
    tf_model      : tf.keras.Model
    out_path      : destination .tflite file
    quantize_int8 : if True, apply post-training integer quantization

    Returns
    -------
    out_path on success, None if TensorFlow is unavailable.
    """
    if not TFLITE_AVAILABLE:
        logger.warning(
            "skipping TFLite for %s — tensorflow not available", out_path.name
        )
        return None

    import tensorflow as tf

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)

        if quantize_int8:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            def _representative():
                for _ in range(100):
                    # Generic representative dataset — shape is inferred from
                    # the model's input spec inside each generator function
                    yield [
                        np.random.rand(*[
                            d if d is not None else 1
                            for d in tf_model.input_shape
                        ]).astype(np.float32)
                    ]

            converter.representative_dataset    = _representative
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type  = tf.uint8
            converter.inference_output_type = tf.uint8

        tflite_bytes = converter.convert()
        out_path.write_bytes(tflite_bytes)
        logger.info("exported TFLite  %s  (%s)", out_path.name, _human_size(out_path))
        return out_path

    except Exception as e:
        logger.warning("TFLite (from TF) conversion failed for %s: %s", out_path.name, e)
        return None


def _to_tflite_from_onnx(
    onnx_path: Path,
    out_path:  Path,
) -> Optional[Path]:
    """
    Convert an ONNX model to TFLite via onnx2tf.

    Parameters
    ----------
    onnx_path : source .onnx file
    out_path  : destination .tflite file

    Returns
    -------
    out_path on success, None if onnx2tf is unavailable.
    """
    if not ONNX2TF_AVAILABLE:
        logger.warning(
            "skipping TFLite (onnx2tf) for %s — onnx2tf not available",
            out_path.name,
        )
        return None

    import onnx2tf
    import tempfile
    import numpy as _np

    try:
        import onnx as _onnx_load
        import shutil
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="mlbuild_onnx2tf_") as tmp:
            tmp_dir = Path(tmp)

            # Inline external data so onnx2tf receives a single self-contained
            # .onnx file (no companion .onnx.data file to mishandle).
            proto = _onnx_load.load(str(onnx_path), load_external_data=True)
            inline_path = tmp_dir / "model_inline.onnx"
            _onnx_load.save_model(
                proto,
                str(inline_path),
                save_as_external_data=False,
            )

            # onnx2tf ships pre-computed numpy param files for certain op
            # patterns (hardswish, hardsigmoid in MobileNets) that were saved
            # with allow_pickle=True.  NumPy 1.24+ defaults to allow_pickle=False
            # and rejects them.  Patch np.load for the duration of the call only.
            _orig_np_load = _np.load
            _np.load = lambda *a, **kw: _orig_np_load(
                *a, **{**kw, "allow_pickle": True}
            )

            try:
                conv_dir = tmp_dir / "converted"
                conv_dir.mkdir()
                onnx2tf.convert(
                    input_onnx_file_path            = str(inline_path),
                    output_folder_path              = str(conv_dir),
                    non_verbose                     = True,
                    output_integer_quantized_tflite = False,
                )
            finally:
                _np.load = _orig_np_load  # always restore
            # onnx2tf writes the .tflite into the output folder
            tflite_files = list(conv_dir.glob("*.tflite"))
            if not tflite_files:
                raise FileNotFoundError("onnx2tf produced no .tflite file")

            shutil.copy2(str(tflite_files[0]), str(out_path))

        logger.info("exported TFLite (onnx2tf)  %s  (%s)", out_path.name, _human_size(out_path))
        return out_path

    except Exception as e:
        logger.warning(
            "TFLite (onnx2tf) conversion failed for %s: %s", out_path.name, e
        )
        return None


# ============================================================
# Shared utilities
# ============================================================

def _human_size(path: Path) -> str:
    """Human-readable file or directory size."""
    if path.is_file():
        size = path.stat().st_size
    elif path.is_dir():
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    else:
        return "?"
    if size < 1024:
        return f"{size} B"
    if size < 1024 ** 2:
        return f"{size / 1024:.1f} KB"
    if size < 1024 ** 3:
        return f"{size / 1024**2:.1f} MB"
    return f"{size / 1024**3:.2f} GB"


def _check_deps() -> None:
    """
    Log availability of all optional dependencies at startup.
    Does not exit — missing deps cause individual steps to skip gracefully.
    """
    checks = {
        "torch":        TORCH_AVAILABLE,
        "onnx":         ONNX_AVAILABLE,
        "coremltools":  COREML_AVAILABLE,
        "tensorflow":   TFLITE_AVAILABLE,
        "onnx2tf":      ONNX2TF_AVAILABLE,
    }
    for name, available in checks.items():
        status = "✓" if available else "✗ (skipped)"
        logger.info("dependency  %-16s %s", name, status)

    if not IS_MACOS:
        logger.warning("not macOS — all CoreML steps will be skipped")


def _section(title: str) -> None:
    """Log a visible section header."""
    bar = "─" * (60 - len(title))
    logger.info("")
    logger.info("── %s %s", title, bar)


# ============================================================
# Manifest writer
# ============================================================

class Manifest:
    """
    Accumulates ManifestEntry objects during generation and writes
    manifest.json at the end.

    Usage
    -----
    manifest = Manifest()
    manifest.add(entry)      # called by each generator function
    manifest.write(path)     # called once at the end
    manifest.load(path)      # used by conftest.py at test time
    """

    def __init__(self):
        self._entries: Dict[str, ManifestEntry] = {}

    def add(self, entry: ManifestEntry) -> None:
        if entry.model_id in self._entries:
            raise ValueError(f"Duplicate model_id in manifest: {entry.model_id}")
        self._entries[entry.model_id] = entry

    def stamp_sha256(self, model_id: str, path: Path) -> None:
        """Compute and store SHA256 for a generated file."""
        if model_id not in self._entries:
            raise KeyError(f"model_id not in manifest: {model_id}")
        if path and path.exists():
            self._entries[model_id].sha256 = _sha256(path)

    def write(self, path: Path) -> None:
        payload = {
            "version":     "1",
            "seed":        GLOBAL_SEED,
            "total":       len(self._entries),
            "generated_at": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "models":      {
                k: v.to_dict()
                for k, v in sorted(self._entries.items())
            },
        }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "manifest written  %s  (%d entries)", path.name, len(self._entries)
        )

    @staticmethod
    def load(path: Path) -> Dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self._entries)


# ============================================================
# Result tracker
# ============================================================

@dataclass
class GenerationResult:
    """
    Tracks generation outcomes for the final summary.
    """
    generated: List[str] = field(default_factory=list)
    skipped:   List[str] = field(default_factory=list)   # dep not available
    failed:    List[str] = field(default_factory=list)    # unexpected error

    def ok(self, model_id: str) -> None:
        self.generated.append(model_id)

    def skip(self, model_id: str, reason: str) -> None:
        logger.warning("SKIP  %s  — %s", model_id, reason)
        self.skipped.append(model_id)

    def fail(self, model_id: str, exc: Exception) -> None:
        logger.error("FAIL  %s  — %s", model_id, exc, exc_info=True)
        self.failed.append(model_id)

    def print_summary(self) -> None:
        total = len(self.generated) + len(self.skipped) + len(self.failed)
        logger.info("")
        logger.info("═" * 60)
        logger.info(
            "Generation complete  total=%d  generated=%d  skipped=%d  failed=%d",
            total, len(self.generated), len(self.skipped), len(self.failed),
        )
        if self.skipped:
            logger.info("  Skipped: %s", ", ".join(self.skipped))
        if self.failed:
            logger.error("  Failed:  %s", ", ".join(self.failed))
        logger.info("═" * 60)


# ============================================================
# CLI entry point
# ============================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate MLBuild test fixtures (dummy models, seed=42).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--out-dir",
        default="tests/fixtures",
        help="Root directory for generated fixtures (default: tests/fixtures)",
    )
    p.add_argument(
        "--skip-coreml",
        action="store_true",
        help="Skip all CoreML conversion steps (useful for Linux CI)",
    )
    p.add_argument(
        "--skip-tflite",
        action="store_true",
        help="Skip all TFLite conversion steps",
    )
    p.add_argument(
        "--only",
        nargs="*",
        metavar="MODEL_ID",
        help="Generate only these model IDs (e.g. --only B01 B13 D01)",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Set log level to DEBUG",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("MLBuild fixture generator — seed=%d", GLOBAL_SEED)
    logger.info("output directory: %s", args.out_dir)

    _check_deps()

    dirs     = FixtureDirs(Path(args.out_dir))
    manifest = Manifest()
    result   = GenerationResult()

    dirs.create()

    # Override platform guards if user passed skip flags
    global COREML_AVAILABLE, TFLITE_AVAILABLE
    if args.skip_coreml:
        COREML_AVAILABLE = False
        logger.info("--skip-coreml: CoreML steps disabled")
    if args.skip_tflite:
        TFLITE_AVAILABLE  = False
        ONNX2TF_AVAILABLE = False  # noqa: F841 — read by helpers
        logger.info("--skip-tflite: TFLite steps disabled")

    only_ids: Optional[set] = set(args.only) if args.only else None

    # ── Generator phases will be registered here in subsequent steps ──────────
    # Each phase adds generator functions to this list.
    # Format: list of (generator_fn, model_id) tuples.
    generators: List = []

    # ── Step 2: Baseline ONNX (B01, B05, B09, B13) ───────────────────────────
    try:
        from generate_fixtures_step2 import register_baseline_onnx
        register_baseline_onnx(generators)
    except ImportError as e:
        logger.warning("step2 not available: %s", e)

    # ── Step 3: CoreML baselines (B02/B03, B06/B07, B10/B11, B14/B15) ─────────
    try:
        from generate_fixtures_step3 import register_coreml_baselines
        register_coreml_baselines(generators)
    except ImportError as e:
        logger.warning("step3 not available: %s", e)

    # ── Step 4: TFLite baselines (B04, B08, B12, B16) ──────────────────────────
    try:
        from generate_fixtures_step4 import register_tflite_baselines
        register_tflite_baselines(generators)
    except ImportError as e:
        logger.warning("step4 not available: %s", e)

    # ── Step 5: Detection baselines (D01 ONNX, D02/D03 CoreML, D04 TFLite) ────
    try:
        from generate_fixtures_step5 import register_detection_baselines
        register_detection_baselines(generators)
    except ImportError as e:
        logger.warning("step5 not available: %s", e)

    # ── Step 6: Time-series baselines (T01 ONNX, T02/T03 CoreML, T04 TFLite) ──
    try:
        from generate_fixtures_step6 import register_timeseries_baselines
        register_timeseries_baselines(generators)
    except ImportError as e:
        logger.warning("step6 not available: %s", e)

    # ── Step 7: Multimodal baselines (M01 ONNX, M02/M03 CoreML, M04 TFLite) ───
    try:
        from generate_fixtures_step7 import register_multimodal_baselines
        register_multimodal_baselines(generators)
    except ImportError as e:
        logger.warning("step7 not available: %s", e)

    # ── Step 8: Recommendation baselines (R01 ONNX, R02 CoreML .mlpackage) ────
    try:
        from generate_fixtures_step8 import register_rec_baselines
        register_rec_baselines(generators)
    except ImportError as e:
        logger.warning("step8 not available: %s", e)

    # ── Step 9: Generative baselines (G01 ONNX, G02 CoreML .mlpackage FP16) ───
    try:
        from generate_fixtures_step9 import register_generative_baselines
        register_generative_baselines(generators)
    except ImportError as e:
        logger.warning("step9 not available: %s", e)

    # ── Step 10: Stress models (18 models) ──────────────────────────────────────
    try:
        from generate_fixtures_step10 import register_stress_models
        register_stress_models(generators)
    except ImportError as e:
        logger.warning("step10 not available: %s", e)

    # ── Step 11: Expected-failure ONNX sources (5 models) ────────────────────
    try:
        from generate_fixtures_step11 import register_failure_models
        register_failure_models(generators)
    except ImportError as e:
        logger.warning("step11 not available: %s", e)

    if not generators:
        logger.info(
            "no generators registered yet — scaffold only (Steps 2–11 pending)"
        )
    else:
        t0 = time.monotonic()
        for gen_fn, model_id in generators:
            if only_ids and model_id not in only_ids:
                continue
            try:
                gen_fn(dirs, manifest, result)
            except Exception as exc:
                result.fail(model_id, exc)

        logger.info("generation took %.1fs", time.monotonic() - t0)

    # Write manifest (even if empty — useful for scaffold verification)
    manifest.write(dirs.manifest)
    result.print_summary()

    return 1 if result.failed else 0


if __name__ == "__main__":
    sys.exit(main())