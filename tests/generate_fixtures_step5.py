"""
Step 5 — Detection baseline models.

Models
------
D01  detection  ONNX             SSD-lite style, [1,3,320,320] → boxes+scores, no NMS
D02  detection  CoreML .mlmodel  Same PyTorch model → NeuralNetwork
D03  detection  CoreML .mlpackage Same PyTorch model → MLProgram
D04  detection  TFLite           D01 ONNX → onnx2tf → .tflite

Architecture
------------
SSD-lite-style backbone: 4× stride-2 Conv blocks → dual detection heads.
  Input:  pixel_values  [1, 3, 320, 320]  float32
  Output: boxes         [1, 400, 4]       float32  (x1,y1,x2,y2 per anchor)
          scores         [1, 400, 80]      float32  (per-class scores, pre-softmax)

Output shape [N, 4] for boxes and named output "boxes" are the two
primary Tier 2 detection signals used by task_detection.py.

NMS is NOT baked into this graph. expected_nms_inside=False for all D0x.
(SD2 stress model is the NMS-baked variant — that's a later step.)

Detection target for all D0x
-----------------------------
domain=VISION, subtype=DETECTION, execution=STANDARD, nms_inside=False
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn

logger = logging.getLogger("mlbuild.fixtures.step5")


# ── Shared architecture ───────────────────────────────────────────────────────

class _ssd_lite_detector(nn.Module):
    """
    SSD-lite style detection backbone.

    4× stride-2 Conv blocks reduce 320×320 → 20×20 feature map.
    Dual heads produce boxes and class scores per anchor position.

    Outputs
    -------
    boxes   [B, 400, 4]   float32  — raw (x1,y1,x2,y2) per anchor
    scores  [B, 400, 80]  float32  — per-class logits, no softmax

    The named outputs ("boxes", "scores") and the [N, 4] box shape are
    the primary Tier 2 detection signals in task_detection.py.
    """

    def __init__(self, num_classes: int = 80):
        super().__init__()
        # Backbone: 3 → 32 → 64 → 128 → 256 (each stride=2, 320→160→80→40→20)
        self.backbone = nn.Sequential(
            nn.Conv2d(3,   32,  3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,  64,  3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,  128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        # Detection heads
        self.box_head   = nn.Conv2d(256, 4,           1)  # 4 coords per anchor
        self.score_head = nn.Conv2d(256, num_classes, 1)  # C scores per anchor

    def forward(self, pixel_values: torch.Tensor):
        # [B, 3, 320, 320] → [B, 256, 20, 20]
        feat = self.backbone(pixel_values)

        B, _, H, W = feat.shape               # H=W=20, H*W=400 anchors
        N = H * W

        # Box head: [B, 4, H, W] → [B, N, 4]
        boxes = self.box_head(feat)            # [B, 4, 20, 20]
        boxes = boxes.reshape(B, 4, N).permute(0, 2, 1)  # [B, 400, 4]

        # Score head: [B, C, H, W] → [B, N, C]
        scores = self.score_head(feat)         # [B, 80, 20, 20]
        scores = scores.reshape(B, -1, N).permute(0, 2, 1)  # [B, 400, 80]

        return boxes, scores


def _reset_seed() -> None:
    import numpy as np
    torch.manual_seed(42)
    np.random.seed(42)


def _build_model() -> _ssd_lite_detector:
    _reset_seed()
    model = _ssd_lite_detector(num_classes=80)
    model.eval()
    return model


# ── D01 — ONNX ───────────────────────────────────────────────────────────────

def _gen_D01(dirs, manifest, result) -> None:
    """
    SSD-lite detector → ONNX.

    Input:  pixel_values  [1, 3, 320, 320]  float32
    Outputs: boxes        [1, 400, 4]       float32
             scores       [1, 400, 80]      float32

    Named outputs "boxes" and "scores" trigger Tier 2 detection name heuristic.
    Output shape [N, 4] for boxes triggers the bbox dimension check.

    Also serves as the ONNX source for D04 (TFLite via onnx2tf).
    """
    from generate_fixtures import ManifestEntry

    model_id = "D01"

    try:
        model = _build_model()
        dummy = torch.zeros(1, 3, 320, 320)

        path = dirs.onnx_path(model_id, "ssdlite_detection")
        path.parent.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy,),
                str(path),
                input_names    = ["pixel_values"],
                output_names   = ["boxes", "scores"],
                dynamic_axes   = {},
                opset_version  = 18,
                do_constant_folding = True,
            )

        import onnx as _onnx
        _onnx.checker.check_model(str(path))
        logger.info("exported ONNX  %s", path.name)

        manifest.add(ManifestEntry(
            model_id            = model_id,
            tier                = "baseline",
            modality            = "detection",
            format              = "onnx",
            size_class          = "small",
            file                = str(path.relative_to(dirs.root)),
            expected_domain     = "vision",
            expected_subtype    = "detection",
            expected_execution  = "standard",
            expected_nms_inside = False,
            expected_failure    = False,
            commands            = ["inspect", "baseline", "budget"],
        ))
        manifest.stamp_sha256(model_id, path)
        result.ok(model_id)

    except Exception as exc:
        result.fail(model_id, exc)


# ── D02 — CoreML NeuralNetwork (.mlmodel) ────────────────────────────────────

def _gen_D02(dirs, manifest, result) -> None:
    """
    SSD-lite detector → CoreML NeuralNetwork.

    Same PyTorch model as D01, converted directly via TorchScript.
    Multi-output model: coremltools receives both boxes and scores outputs.
    """
    from generate_fixtures import _to_coreml_nn, ManifestEntry, COREML_AVAILABLE

    model_id = "D02"

    if not COREML_AVAILABLE:
        result.skip(model_id, "coremltools not available")
        return

    try:
        model = _build_model()
        dummy = torch.zeros(1, 3, 320, 320)
        out   = dirs.coreml_path(model_id, "ssdlite_detection_nn", "mlmodel")

        converted = _to_coreml_nn(model, (dummy,), ["pixel_values"], out)
        if converted is None:
            raise RuntimeError(f"CoreML NeuralNetwork conversion returned None for {model_id}")

        manifest.add(ManifestEntry(
            model_id            = model_id,
            tier                = "baseline",
            modality            = "detection",
            format              = "mlmodel",
            size_class          = "small",
            file                = str(out.relative_to(dirs.root)),
            expected_domain     = "vision",
            expected_subtype    = "detection",
            expected_execution  = "standard",
            expected_nms_inside = False,
            expected_failure    = False,
            commands            = ["inspect", "baseline"],
        ))
        manifest.stamp_sha256(model_id, out)
        result.ok(model_id)

    except Exception as exc:
        result.fail(model_id, exc)


# ── D03 — CoreML MLProgram (.mlpackage) ──────────────────────────────────────

def _gen_D03(dirs, manifest, result) -> None:
    """
    SSD-lite detector → CoreML MLProgram.

    Same PyTorch model as D01/D02, MLProgram format (iOS15+).
    """
    from generate_fixtures import _to_coreml_mlprogram, ManifestEntry, COREML_AVAILABLE

    model_id = "D03"

    if not COREML_AVAILABLE:
        result.skip(model_id, "coremltools not available")
        return

    try:
        model = _build_model()
        dummy = torch.zeros(1, 3, 320, 320)
        out   = dirs.coreml_path(model_id, "ssdlite_detection_mlprog", "mlpackage")

        converted = _to_coreml_mlprogram(model, (dummy,), ["pixel_values"], out)
        if converted is None:
            raise RuntimeError(f"CoreML MLProgram conversion returned None for {model_id}")

        manifest.add(ManifestEntry(
            model_id            = model_id,
            tier                = "baseline",
            modality            = "detection",
            format              = "mlpackage",
            size_class          = "small",
            file                = str(out.relative_to(dirs.root)),
            expected_domain     = "vision",
            expected_subtype    = "detection",
            expected_execution  = "standard",
            expected_nms_inside = False,
            expected_failure    = False,
            commands            = ["inspect", "baseline", "prune"],
        ))
        manifest.stamp_sha256(model_id, out)
        result.ok(model_id)

    except Exception as exc:
        result.fail(model_id, exc)


# ── D04 — TFLite ─────────────────────────────────────────────────────────────

def _gen_D04(dirs, manifest, result) -> None:
    """
    SSD-lite detector → TFLite.

    Primary: D01 ONNX → onnx2tf.
    Fallback: minimal TF Keras 3-block CNN (~200KB weights, not MobileNetV2).
    """
    from generate_fixtures import (
        _to_tflite_from_onnx, _to_tflite_from_tf,
        ManifestEntry, ONNX2TF_AVAILABLE, TFLITE_AVAILABLE,
    )

    model_id = "D04"

    if not TFLITE_AVAILABLE:
        result.skip(model_id, "tensorflow not available")
        return

    try:
        out = dirs.tflite_path(model_id, "ssdlite_detection")
        converted = None

        # Primary: D01 ONNX → onnx2tf
        d01_onnx = dirs.onnx / "D01_ssdlite_detection.onnx"
        if ONNX2TF_AVAILABLE and d01_onnx.exists():
            converted = _to_tflite_from_onnx(d01_onnx, out)

        # Fallback: minimal TF Keras detection model (tiny — no MobileNetV2)
        if converted is None:
            logger.warning("D04: onnx2tf failed — falling back to TF Keras model")
            import tensorflow as tf

            inp = tf.keras.Input(shape=(320, 320, 3), name="pixel_values")
            x = tf.keras.layers.Conv2D(16, 3, strides=2, padding="same", use_bias=False)(inp)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            pooled = tf.keras.layers.GlobalAveragePooling2D()(x)
            raw_boxes  = tf.keras.layers.Dense(400 * 4,  name="boxes_flat")(pooled)
            raw_scores = tf.keras.layers.Dense(400 * 80, name="scores_flat")(pooled)
            boxes  = tf.keras.layers.Reshape((400, 4),  name="boxes")(raw_boxes)
            scores = tf.keras.layers.Reshape((400, 80), name="scores")(raw_scores)
            tf_model  = tf.keras.Model(inputs=inp, outputs=[boxes, scores])
            converted = _to_tflite_from_tf(tf_model, out)

        if converted is None:
            raise RuntimeError(f"TFLite conversion returned None for {model_id}")

        manifest.add(ManifestEntry(
            model_id            = model_id,
            tier                = "baseline",
            modality            = "detection",
            format              = "tflite",
            size_class          = "small",
            file                = str(out.relative_to(dirs.root)),
            expected_domain     = "vision",
            expected_subtype    = "detection",
            expected_execution  = "standard",
            expected_nms_inside = False,
            expected_failure    = False,
            commands            = ["inspect", "baseline"],
        ))
        manifest.stamp_sha256(model_id, out)
        result.ok(model_id)

    except Exception as exc:
        result.fail(model_id, exc)


# ── Registration ──────────────────────────────────────────────────────────────

def register_detection_baselines(generators: list) -> None:
    """
    Register Step 5 generators.

    D01 must run before D04 — D04 uses D01's ONNX as its primary source.
    All four generators produce detection-domain fixtures:
      domain=VISION, subtype=DETECTION, nms_inside=False
    """
    generators.extend([
        (_gen_D01, "D01"),
        (_gen_D02, "D02"),
        (_gen_D03, "D03"),
        (_gen_D04, "D04"),
    ])
    logger.info(
        "step5: registered 4 detection baseline generators "
        "(D01 ONNX, D02 CoreML .mlmodel, D03 CoreML .mlpackage, D04 TFLite) "
        "— architecture: SSD-lite [1,3,320,320] → boxes[1,400,4] + scores[1,400,80]"
    )