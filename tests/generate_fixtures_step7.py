"""
Step 7 — Multimodal baseline models.

Models
------
M01  multimodal  ONNX             CLIP-style two-tower, pixel_values[1,3,224,224] + input_ids[1,77]
M02  multimodal  CoreML .mlmodel  Vision tower only → NeuralNetwork
M03  multimodal  CoreML .mlpackage Vision tower only → MLProgram
M04  multimodal  TFLite           MobileNet-style backbone [1,224,224,3] → TFLite

Architecture
------------
Two-tower CLIP-style (M01 ONNX only)
  Inputs:  pixel_values  [1, 3, 224, 224]  float32
           input_ids     [1, 77]           int64
  Output:  similarity    [1, 1]            float32  (dot product of L2-normed towers)

  Vision tower: 3× stride-2 Conv → GAP → Linear(256)
  Text tower:   Embedding(1000, 256) → mean pool → Linear(256)
  Output: dot(vision_embed, text_embed) — scalar similarity score

  Named inputs trigger MULTI_INPUT detection.
  Both image-shaped float input + int64 token input → MULTIMODAL subtype.

Vision tower only (M02/M03 CoreML)
  The full two-tower model traces correctly to TorchScript but converting
  mixed float+int inputs to CoreML requires separate input specs.
  M02/M03 use the vision tower only [1,3,224,224] → [1,256] embedding.
  This still exercises the multimodal architecture path (Conv + classification
  head with semantic embedding output).

TFLite (M04)
  MobileNet-style backbone with [1,224,224,3] NHWC input → [1,256] embedding.
  Built directly via TF Keras to avoid onnx2tf issues.

Detection target
----------------
M01: domain=VISION, subtype=MULTIMODAL, execution=MULTI_INPUT
M02/M03: domain=VISION, subtype=NONE, execution=STANDARD
M04: domain=VISION, subtype=NONE, execution=STANDARD
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("mlbuild.fixtures.step7")


# ── Shared architectures ──────────────────────────────────────────────────────

class _vision_tower(nn.Module):
    """
    Minimal vision tower: 3× stride-2 Conv → GAP → Linear projection.

    Input:  pixel_values  [B, 3, 224, 224]  float32
    Output: embedding     [B, 256]           float32  (L2-normalized)
    """
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3,  32,  3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64,  3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.backbone(pixel_values)
        x = self.proj(x)
        return F.normalize(x, dim=-1)


class _text_tower(nn.Module):
    """
    Minimal text tower: Embedding → mean pool → Linear projection.

    Input:  input_ids  [B, seq_len]  int64
    Output: embedding  [B, 256]      float32  (L2-normalized)
    """
    def __init__(self, vocab_size: int = 1000, embed_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.proj      = nn.Linear(embed_dim, embed_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)   # [B, seq, embed_dim]
        x = x.mean(dim=1)              # [B, embed_dim]
        x = self.proj(x)
        return F.normalize(x, dim=-1)


class _clip_style(nn.Module):
    """
    Two-tower CLIP-style model.

    Inputs:  pixel_values  [B, 3, 224, 224]  float32
             input_ids     [B, 77]            int64
    Output:  similarity    [B, 1]             float32

    The two named inputs (one float image + one int token) are the primary
    signal for MULTIMODAL subtype detection.
    """
    def __init__(self):
        super().__init__()
        self.vision = _vision_tower(embed_dim=256)
        self.text   = _text_tower(vocab_size=1000, embed_dim=256)

    def forward(
        self,
        pixel_values: torch.Tensor,  # [B, 3, 224, 224]
        input_ids:    torch.Tensor,  # [B, 77]
    ) -> torch.Tensor:               # [B, 1]
        v = self.vision(pixel_values)              # [B, 256]
        t = self.text(input_ids)                   # [B, 256]
        sim = (v * t).sum(dim=-1, keepdim=True)    # [B, 1]
        return sim


def _reset_seed() -> None:
    import numpy as np
    torch.manual_seed(42)
    np.random.seed(42)


# ── M01 — ONNX (two-tower) ───────────────────────────────────────────────────

def _gen_M01(dirs, manifest, result) -> None:
    """
    CLIP-style two-tower → ONNX.

    Two inputs: pixel_values [1,3,224,224] float32 + input_ids [1,77] int64.
    One output: similarity [1,1] float32.

    The mixed float+int input pair is the primary MULTIMODAL detection signal.
    """
    from generate_fixtures import ManifestEntry

    model_id = "M01"

    try:
        _reset_seed()
        model   = _clip_style().eval()
        pixels  = torch.zeros(1, 3, 224, 224)
        ids     = torch.zeros(1, 77, dtype=torch.int64)

        path = dirs.onnx_path(model_id, "clip_style_multimodal")
        path.parent.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            torch.onnx.export(
                model,
                (pixels, ids),
                str(path),
                input_names    = ["pixel_values", "input_ids"],
                output_names   = ["similarity"],
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
            modality            = "multimodal",
            format              = "onnx",
            size_class          = "small",
            file                = str(path.relative_to(dirs.root)),
            expected_domain     = "vision",
            expected_subtype    = "multimodal",
            expected_execution  = "multi_input",
            expected_nms_inside = False,
            expected_failure    = False,
            commands            = ["inspect", "baseline", "budget"],
        ))
        manifest.stamp_sha256(model_id, path)
        result.ok(model_id)

    except Exception as exc:
        result.fail(model_id, exc)


# ── M02 — CoreML NeuralNetwork (vision tower only) ───────────────────────────

def _gen_M02(dirs, manifest, result) -> None:
    """
    Vision tower → CoreML NeuralNetwork.

    Single float input [1,3,224,224] → embedding [1,256].
    CoreML mixed int64+float32 multi-input via TorchScript is fragile;
    using vision-only avoids that complexity for the baseline fixture.
    """
    from generate_fixtures import _to_coreml_nn, ManifestEntry, COREML_AVAILABLE

    model_id = "M02"

    if not COREML_AVAILABLE:
        result.skip(model_id, "coremltools not available")
        return

    try:
        _reset_seed()
        model = _vision_tower(embed_dim=256).eval()
        dummy = torch.zeros(1, 3, 224, 224)
        out   = dirs.coreml_path(model_id, "vision_tower_nn", "mlmodel")

        converted = _to_coreml_nn(model, (dummy,), ["pixel_values"], out)
        if converted is None:
            raise RuntimeError(f"CoreML NeuralNetwork conversion returned None for {model_id}")

        manifest.add(ManifestEntry(
            model_id            = model_id,
            tier                = "baseline",
            modality            = "multimodal",
            format              = "mlmodel",
            size_class          = "small",
            file                = str(out.relative_to(dirs.root)),
            expected_domain     = "vision",
            expected_subtype    = "none",
            expected_execution  = "standard",
            expected_nms_inside = False,
            expected_failure    = False,
            commands            = ["inspect", "baseline"],
        ))
        manifest.stamp_sha256(model_id, out)
        result.ok(model_id)

    except Exception as exc:
        result.fail(model_id, exc)


# ── M03 — CoreML MLProgram (vision tower only) ───────────────────────────────

def _gen_M03(dirs, manifest, result) -> None:
    """
    Vision tower → CoreML MLProgram.
    """
    from generate_fixtures import _to_coreml_mlprogram, ManifestEntry, COREML_AVAILABLE

    model_id = "M03"

    if not COREML_AVAILABLE:
        result.skip(model_id, "coremltools not available")
        return

    try:
        _reset_seed()
        model = _vision_tower(embed_dim=256).eval()
        dummy = torch.zeros(1, 3, 224, 224)
        out   = dirs.coreml_path(model_id, "vision_tower_mlprog", "mlpackage")

        converted = _to_coreml_mlprogram(model, (dummy,), ["pixel_values"], out)
        if converted is None:
            raise RuntimeError(f"CoreML MLProgram conversion returned None for {model_id}")

        manifest.add(ManifestEntry(
            model_id            = model_id,
            tier                = "baseline",
            modality            = "multimodal",
            format              = "mlpackage",
            size_class          = "small",
            file                = str(out.relative_to(dirs.root)),
            expected_domain     = "vision",
            expected_subtype    = "none",
            expected_execution  = "standard",
            expected_nms_inside = False,
            expected_failure    = False,
            commands            = ["inspect", "baseline", "prune"],
        ))
        manifest.stamp_sha256(model_id, out)
        result.ok(model_id)

    except Exception as exc:
        result.fail(model_id, exc)


# ── M04 — TFLite (MobileNet-style backbone) ──────────────────────────────────

def _gen_M04(dirs, manifest, result) -> None:
    """
    MobileNet-style backbone → TFLite via TF Keras.

    Input:  pixel_values  [1, 224, 224, 3]  float32  (NHWC)
    Output: embedding     [1, 256]           float32

    Built directly via TF Keras to avoid onnx2tf. Tiny custom backbone
    (3 depthwise-sep conv blocks) keeps the file small.
    """
    from generate_fixtures import _to_tflite_from_tf, ManifestEntry, TFLITE_AVAILABLE

    model_id = "M04"

    if not TFLITE_AVAILABLE:
        result.skip(model_id, "tensorflow not available")
        return

    try:
        import tensorflow as tf

        inp = tf.keras.Input(shape=(224, 224, 3), name="pixel_values")

        # Depthwise-sep block helper
        def _dsep(x, filters, stride=1):
            x = tf.keras.layers.DepthwiseConv2D(3, strides=stride, padding="same",
                                                 use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.Conv2D(filters, 1, use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            return tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(16, 3, strides=2, padding="same",
                                   use_bias=False)(inp)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = _dsep(x, 32, stride=2)   # 56×56
        x = _dsep(x, 64, stride=2)   # 28×28
        x = _dsep(x, 128, stride=2)  # 14×14

        x   = tf.keras.layers.GlobalAveragePooling2D()(x)
        out = tf.keras.layers.Dense(256, name="embedding")(x)

        tf_model = tf.keras.Model(inputs=inp, outputs=out)

        out_path  = dirs.tflite_path(model_id, "mobilenet_style_multimodal")
        converted = _to_tflite_from_tf(tf_model, out_path)
        if converted is None:
            raise RuntimeError(f"TFLite conversion returned None for {model_id}")

        manifest.add(ManifestEntry(
            model_id            = model_id,
            tier                = "baseline",
            modality            = "multimodal",
            format              = "tflite",
            size_class          = "small",
            file                = str(out_path.relative_to(dirs.root)),
            expected_domain     = "vision",
            expected_subtype    = "none",
            expected_execution  = "standard",
            expected_nms_inside = False,
            expected_failure    = False,
            commands            = ["inspect", "baseline"],
        ))
        manifest.stamp_sha256(model_id, out_path)
        result.ok(model_id)

    except Exception as exc:
        result.fail(model_id, exc)


# ── Registration ──────────────────────────────────────────────────────────────

def register_multimodal_baselines(generators: list) -> None:
    """
    Register Step 7 generators.

    M01: CLIP-style two-tower ONNX (MULTIMODAL subtype, MULTI_INPUT execution).
    M02/M03: Vision tower only CoreML (single-input, VISION domain).
    M04: TF Keras MobileNet-style TFLite (single-input, VISION domain).
    """
    generators.extend([
        (_gen_M01, "M01"),
        (_gen_M02, "M02"),
        (_gen_M03, "M03"),
        (_gen_M04, "M04"),
    ])
    logger.info(
        "step7: registered 4 multimodal baseline generators "
        "(M01 ONNX two-tower, M02 CoreML .mlmodel, M03 CoreML .mlpackage, M04 TFLite)"
    )