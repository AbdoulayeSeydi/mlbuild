"""
Step 4 — TFLite baseline variants.

Pipeline
--------
ONNX (from Step 2) → onnx2tf → .tflite

Source ONNX files must already exist on disk (run Step 2 first).
Skips gracefully if onnx2tf is unavailable or the source ONNX is missing.

Models
------
B04  vision    B01 ONNX → TFLite   (MobileNetV3-Small, [1,3,224,224]→[1,1000])
B08  tabular   B05 ONNX → TFLite   (MLP, [1,32]→[1,1])
B12  audio     B09 ONNX → TFLite   (CNN spectrogram, [1,1,64,101]→[1,35])
B16  nlp       B13 ONNX → TFLite   (BERT-style, input_ids+attention_mask→[1,2])
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("mlbuild.fixtures.step4")

# ONNX source stems — must match Step 2 output filenames exactly
_ONNX_VISION   = "B01_mobilenetv3small_vision.onnx"
_ONNX_TABULAR  = "B05_mlp_tabular.onnx"
_ONNX_AUDIO    = "B09_cnn_spectrogram_audio.onnx"
_ONNX_NLP      = "B13_bert_style_nlp.onnx"


def _src(dirs, filename: str) -> Path | None:
    """Resolve and verify a Step 2 ONNX source file."""
    path = dirs.onnx / filename
    if not path.exists():
        logger.warning("source ONNX not found: %s — run Step 2 first", filename)
        return None
    return path


# ============================================================
# B04 — Vision: TFLite
# ============================================================

def _gen_B04(dirs, manifest, result) -> None:
    """
    MobileNetV3-Small → TFLite via tf.keras (not onnx2tf).

    onnx2tf cannot handle B01's external-data ONNX file on this torch/numpy
    version: it reads the companion .onnx.data file with np.load, which fails
    regardless of allow_pickle setting. Bypass: use tf.keras.applications
    directly and convert with TFLiteConverter.

    Input:  pixel_values [1, 224, 224, 3] float32  (NHWC — TFLite standard)
    Output: logits       [1, 1000]         float32

    Detection target
    ----------------
    domain=VISION, subtype=NONE, execution=STANDARD
    """
    from generate_fixtures import _to_tflite_from_tf, ManifestEntry, TFLITE_AVAILABLE

    model_id = "B04"

    if not TFLITE_AVAILABLE:
        result.skip(model_id, "tensorflow not available")
        return

    try:
        import tensorflow as tf

        # MobileNetV3Small is in tf.keras.applications — no onnx2tf needed.
        # Input is NHWC [1, 224, 224, 3] (channels-last, standard for TFLite).
        tf_model = tf.keras.applications.MobileNetV3Small(
            input_shape = (224, 224, 3),
            weights     = None,
            classes     = 1000,
        )

        out = dirs.tflite_path(model_id, "mobilenetv3small_vision")
        converted = _to_tflite_from_tf(tf_model, out)
        if converted is None:
            raise RuntimeError(f"TFLite conversion returned None for {model_id}")

        manifest.add(ManifestEntry(
            model_id            = model_id,
            tier                = "baseline",
            modality            = "vision",
            format              = "tflite",
            size_class          = "small",
            file                = str(out.relative_to(dirs.root)),
            expected_domain     = "vision",
            expected_subtype    = "none",
            expected_execution  = "standard",
            expected_nms_inside = False,
            expected_failure    = False,
            commands            = ["inspect", "baseline", "status"],
        ))
        manifest.stamp_sha256(model_id, out)
        result.ok(model_id)

    except Exception as exc:
        result.fail(model_id, exc)


# ============================================================
# B08 — Tabular: TFLite
# ============================================================

def _gen_B08(dirs, manifest, result) -> None:
    """
    MLP tabular ONNX → TFLite.

    Source: B05_mlp_tabular.onnx
    Input:  features [1, 32]  float32
    Output: output   [1, 1]   float32

    Detection target
    ----------------
    domain=TABULAR, subtype=NONE, execution=STANDARD
    """
    from generate_fixtures import _to_tflite_from_onnx, ManifestEntry, ONNX2TF_AVAILABLE

    model_id = "B08"

    if not ONNX2TF_AVAILABLE:
        result.skip(model_id, "onnx2tf not available")
        return

    src = _src(dirs, _ONNX_TABULAR)
    if src is None:
        result.skip(model_id, "source ONNX B05 not found")
        return

    try:
        out = dirs.tflite_path(model_id, "mlp_tabular")
        converted = _to_tflite_from_onnx(src, out)
        if converted is None:
            raise RuntimeError(f"TFLite conversion returned None for {model_id}")

        manifest.add(ManifestEntry(
            model_id            = model_id,
            tier                = "baseline",
            modality            = "tabular",
            format              = "tflite",
            size_class          = "tiny",
            file                = str(out.relative_to(dirs.root)),
            expected_domain     = "tabular",
            expected_subtype    = "none",
            expected_execution  = "standard",
            expected_failure    = False,
            commands            = ["inspect", "baseline", "status"],
        ))
        manifest.stamp_sha256(model_id, out)
        result.ok(model_id)

    except Exception as exc:
        result.fail(model_id, exc)


# ============================================================
# B12 — Audio: TFLite
# ============================================================

def _gen_B12(dirs, manifest, result) -> None:
    """
    CNN spectrogram ONNX → TFLite.

    Source: B09_cnn_spectrogram_audio.onnx
    Input:  spectrogram [1, 1, 64, 101]  float32
    Output: logits      [1, 35]          float32

    Detection target
    ----------------
    domain=AUDIO, subtype=NONE, execution=STANDARD
    """
    from generate_fixtures import _to_tflite_from_onnx, ManifestEntry, ONNX2TF_AVAILABLE

    model_id = "B12"

    if not ONNX2TF_AVAILABLE:
        result.skip(model_id, "onnx2tf not available")
        return

    src = _src(dirs, _ONNX_AUDIO)
    if src is None:
        result.skip(model_id, "source ONNX B09 not found")
        return

    try:
        out = dirs.tflite_path(model_id, "cnn_spectrogram_audio")
        converted = _to_tflite_from_onnx(src, out)
        if converted is None:
            raise RuntimeError(f"TFLite conversion returned None for {model_id}")

        manifest.add(ManifestEntry(
            model_id            = model_id,
            tier                = "baseline",
            modality            = "audio",
            format              = "tflite",
            size_class          = "tiny",
            file                = str(out.relative_to(dirs.root)),
            expected_domain     = "audio",
            expected_subtype    = "none",
            expected_execution  = "standard",
            expected_failure    = False,
            commands            = ["inspect", "baseline", "budget"],
        ))
        manifest.stamp_sha256(model_id, out)
        result.ok(model_id)

    except Exception as exc:
        result.fail(model_id, exc)


# ============================================================
# B16 — NLP: TFLite
# ============================================================

def _gen_B16(dirs, manifest, result) -> None:
    """
    BERT-style 2-input classifier ONNX → TFLite.

    Source: B13_bert_style_nlp.onnx
    Inputs: input_ids [1, 128] int64  +  attention_mask [1, 128] int64
    Output: logits    [1, 2]   float32

    onnx2tf handles transformer ops (attention, layernorm) via its own
    TF op mappings. Unlike CoreML, TFLite does not require op decomposition.

    Detection target
    ----------------
    domain=NLP, subtype=NONE, execution=MULTI_INPUT
    """
    from generate_fixtures import _to_tflite_from_onnx, ManifestEntry, ONNX2TF_AVAILABLE

    model_id = "B16"

    if not ONNX2TF_AVAILABLE:
        result.skip(model_id, "onnx2tf not available")
        return

    src = _src(dirs, _ONNX_NLP)
    if src is None:
        result.skip(model_id, "source ONNX B13 not found")
        return

    try:
        out = dirs.tflite_path(model_id, "bert_style_nlp")
        converted = _to_tflite_from_onnx(src, out)
        if converted is None:
            raise RuntimeError(f"TFLite conversion returned None for {model_id}")

        manifest.add(ManifestEntry(
            model_id            = model_id,
            tier                = "baseline",
            modality            = "nlp",
            format              = "tflite",
            size_class          = "medium",
            file                = str(out.relative_to(dirs.root)),
            expected_domain     = "nlp",
            expected_subtype    = "none",
            expected_execution  = "multi_input",
            expected_failure    = False,
            commands            = ["inspect", "budget", "baseline"],
        ))
        manifest.stamp_sha256(model_id, out)
        result.ok(model_id)

    except Exception as exc:
        result.fail(model_id, exc)


# ============================================================
# Registration
# ============================================================

def register_tflite_baselines(generators: list) -> None:
    """
    Register Step 4 generators into the main generator list.

    Pipeline: ONNX (Step 2) → onnx2tf → .tflite
    All 4 generators depend on Step 2 ONNX files existing on disk.
    Skips gracefully if onnx2tf is unavailable (non-standard install).
    """
    generators.extend([
        (_gen_B04, "B04"),
        (_gen_B08, "B08"),
        (_gen_B12, "B12"),
        (_gen_B16, "B16"),
    ])
    logger.info(
        "step4: registered 4 TFLite baseline generators "
        "(B04 vision, B08 tabular, B12 audio, B16 nlp) "
        "— pipeline: ONNX → onnx2tf → TFLite"
    )