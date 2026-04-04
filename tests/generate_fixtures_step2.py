"""
Step 2 — Baseline ONNX generators.

Models
------
B01  vision    MobileNetV3-Small          1×3×224×224 → 1×1000
B05  tabular   MLP 512→256→128→1          1×32        → 1×1
B09  audio     CNN spectrogram            1×1×64×101  → 1×35
B13  nlp       BERT-style 2-input         input_ids[1×128] + attention_mask[1×128] → 1×2

All models use dummy (random) weights, seed=42.
Architectures are faithful — op types, input shapes, and output shapes match
the real models so that task_detection, task_inputs, and task_validation
all behave correctly.

Usage
-----
This module is imported by generate_fixtures.py.  Do not run it directly.

    from tests.generate_fixtures_step2 import register_baseline_onnx
    register_baseline_onnx(generators)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger("mlbuild.fixtures.step2")


# ============================================================
# B01 — Vision: MobileNetV3-Small
# ============================================================

def _gen_B01(dirs, manifest, result) -> None:
    """
    MobileNetV3-Small exported to ONNX.

    Architecture: torchvision.models.mobilenet_v3_small (dummy weights).
    Input:  pixel_values  [1, 3, 224, 224]  float32
    Output: logits        [1, 1000]          float32
    Ops:    Conv, BatchNormalization, HardSwish, GlobalAveragePool, Gemm

    Detection target
    ----------------
    domain=VISION, subtype=NONE, execution=STANDARD
    Tier 1 detection via Conv + GlobalAveragePool ops.
    """
    from generate_fixtures import (
        _export_onnx, FixtureDirs, Manifest, GenerationResult, ManifestEntry,
    )
    import torch
    import torch.nn as nn

    model_id = "B01"
    stem     = "mobilenetv3small_vision"

    try:
        # Use torchvision if available, otherwise build a structurally
        # equivalent dummy that has the same op profile.
        try:
            import torchvision.models as tvm
            model = tvm.mobilenet_v3_small(weights=None)
        except ImportError:
            # Fallback: minimal CNN with same input/output shape
            model = _minimal_cnn(in_channels=3, num_classes=1000)

        model.eval()
        dummy = torch.zeros(1, 3, 224, 224)   # zeros are deterministic with seed=42

        path = dirs.onnx_path(model_id, stem)
        _export_onnx(
            model        = model,
            dummy_inputs = (dummy,),
            path         = path,
            input_names  = ["pixel_values"],
            output_names = ["logits"],
            opset        = 18,
        )

        entry = ManifestEntry(
            model_id           = model_id,
            tier               = "baseline",
            modality           = "vision",
            format             = "onnx",
            size_class         = "tiny",
            file               = str(path.relative_to(dirs.root)),
            expected_domain    = "vision",
            expected_subtype   = "none",
            expected_execution = "standard",
            expected_nms_inside = False,
            expected_failure   = False,
            commands           = ["inspect", "baseline", "status", "prune"],
        )
        manifest.add(entry)
        manifest.stamp_sha256(model_id, path)
        result.ok(model_id)

    except Exception as exc:
        result.fail(model_id, exc)


# ============================================================
# B05 — Tabular: MLP regression
# ============================================================

def _gen_B05(dirs, manifest, result) -> None:
    """
    4-layer MLP: 32→512→256→128→1 (regression).

    Input:  features  [1, 32]  float32
    Output: output    [1, 1]   float32
    Ops:    Gemm, Relu

    Detection target
    ----------------
    domain=TABULAR, subtype=NONE, execution=STANDARD
    No Conv/Attention/LSTM ops — falls through to TABULAR via shape heuristic.
    Float rank-2 input with no integer inputs and no temporal structure.
    """
    from generate_fixtures import (
        _export_onnx, FixtureDirs, Manifest, GenerationResult, ManifestEntry,
    )
    import torch
    import torch.nn as nn

    model_id = "B05"
    stem     = "mlp_tabular"

    try:
        model = nn.Sequential(
            nn.Linear(32,  512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        model.eval()
        dummy = torch.zeros(1, 32)

        path = dirs.onnx_path(model_id, stem)
        _export_onnx(
            model        = model,
            dummy_inputs = (dummy,),
            path         = path,
            input_names  = ["features"],
            output_names = ["output"],
            opset        = 18,
        )

        entry = ManifestEntry(
            model_id           = model_id,
            tier               = "baseline",
            modality           = "tabular",
            format             = "onnx",
            size_class         = "tiny",
            file               = str(path.relative_to(dirs.root)),
            expected_domain    = "tabular",
            expected_subtype   = "none",
            expected_execution = "standard",
            expected_failure   = False,
            commands           = ["inspect", "baseline", "status", "rename"],
        )
        manifest.add(entry)
        manifest.stamp_sha256(model_id, path)
        result.ok(model_id)

    except Exception as exc:
        result.fail(model_id, exc)


# ============================================================
# B09 — Audio: CNN spectrogram classifier
# ============================================================

def _gen_B09(dirs, manifest, result) -> None:
    """
    Small CNN operating on a mel-spectrogram input.

    Input:  spectrogram  [1, 1, 64, 101]  float32  (channels, mel_bins, time)
    Output: logits       [1, 35]           float32  (35 speech command classes)
    Ops:    Conv, BatchNormalization, ReLU, MaxPool, AdaptiveAvgPool, Gemm

    Architecture mirrors the M5 / small CNN used for keyword spotting
    (Speech Commands dataset convention: 64 mel bins, ~1s at 16kHz → 101 frames).

    Detection target
    ----------------
    domain=AUDIO, subtype=NONE, execution=STANDARD
    Tier 1 detection: rank-4 float input with feature_dim < 256 → SPECTROGRAM role.
    Conv ops present but audio name pattern on input name wins at Tier 2.
    """
    from generate_fixtures import (
        _export_onnx, FixtureDirs, Manifest, GenerationResult, ManifestEntry,
    )
    import torch
    import torch.nn as nn

    model_id = "B09"
    stem     = "cnn_spectrogram_audio"

    try:
        model = _spectrogram_cnn(num_classes=35)
        model.eval()
        dummy = torch.zeros(1, 1, 64, 101)

        path = dirs.onnx_path(model_id, stem)
        _export_onnx(
            model        = model,
            dummy_inputs = (dummy,),
            path         = path,
            input_names  = ["spectrogram"],   # name triggers Tier 2 AUDIO signal
            output_names = ["logits"],
            opset        = 18,
        )

        entry = ManifestEntry(
            model_id           = model_id,
            tier               = "baseline",
            modality           = "audio",
            format             = "onnx",
            size_class         = "tiny",
            file               = str(path.relative_to(dirs.root)),
            expected_domain    = "audio",
            expected_subtype   = "none",
            expected_execution = "standard",
            expected_failure   = False,
            commands           = ["inspect", "baseline", "budget", "prune"],
        )
        manifest.add(entry)
        manifest.stamp_sha256(model_id, path)
        result.ok(model_id)

    except Exception as exc:
        result.fail(model_id, exc)


# ============================================================
# B13 — NLP: BERT-style 2-input classifier
# ============================================================

def _gen_B13(dirs, manifest, result) -> None:
    """
    Minimal BERT-style classifier with 2 integer inputs.

    Inputs:  input_ids       [1, 128]  int64
             attention_mask  [1, 128]  int64
    Output:  logits          [1, 2]    float32

    Architecture: Embedding → 2-layer Transformer encoder → mean pool → Linear.
    Uses torch.nn.TransformerEncoderLayer which exports Attention + LayerNorm + GELU
    — the exact ops that trigger Tier 1 NLP detection.

    Named inputs ensure Tier 2 name heuristics also fire:
      input_ids      → TOKEN_IDS role
      attention_mask → ATTENTION_MASK role

    Detection target
    ----------------
    domain=NLP, subtype=NONE, execution=MULTI_INPUT
    Tier 1: Attention + LayerNorm + integer inputs → NLP score=3.0
    Execution: MULTI_INPUT (2 inputs, same domain).
    """
    from generate_fixtures import (
        _export_onnx, FixtureDirs, Manifest, GenerationResult, ManifestEntry,
    )
    import torch
    import torch.nn as nn

    model_id = "B13"
    stem     = "bert_style_nlp"

    try:
        model = _bert_style_classifier(
            vocab_size   = 1000,   # small vocab — op types are identical, avoids 36MB embedding table
            hidden_size  = 256,    # small but realistic hidden dim
            num_heads    = 8,
            num_layers   = 2,
            seq_len      = 128,
            num_classes  = 2,
        )
        model.eval()

        # Two integer inputs — must be passed as separate tensors
        input_ids      = torch.zeros(1, 128, dtype=torch.int64)
        attention_mask = torch.ones(1, 128,  dtype=torch.int64)

        path = dirs.onnx_path(model_id, stem)

        # ONNX export with explicit input names for Tier 2 name detection
        with torch.no_grad():
            import torch.onnx
            torch.onnx.export(
                model,
                (input_ids, attention_mask),
                str(path),
                input_names        = ["input_ids", "attention_mask"],
                output_names       = ["logits"],
                dynamic_axes       = {},
                opset_version      = 14,
                do_constant_folding = True,
            )

        # Validate
        import onnx as _onnx
        _onnx.checker.check_model(str(path))
        logger.info("exported ONNX  %s", path.name)

        entry = ManifestEntry(
            model_id           = model_id,
            tier               = "baseline",
            modality           = "nlp",
            format             = "onnx",
            size_class         = "medium",
            file               = str(path.relative_to(dirs.root)),
            expected_domain    = "nlp",
            expected_subtype   = "none",
            expected_execution = "multi_input",
            expected_failure   = False,
            commands           = ["inspect", "budget", "baseline", "prune"],
        )
        manifest.add(entry)
        manifest.stamp_sha256(model_id, path)
        result.ok(model_id)

    except Exception as exc:
        result.fail(model_id, exc)


# ============================================================
# Architecture helpers
# ============================================================

def _minimal_cnn(in_channels: int = 3, num_classes: int = 1000):
    """
    Minimal CNN with same op profile as MobileNetV3-Small.
    Used as fallback when torchvision is not installed.
    Produces: Conv, BatchNorm, ReLU, AdaptiveAvgPool, Linear ops in ONNX graph.
    """
    import torch.nn as nn

    return nn.Sequential(
        # Stem
        nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        # Depthwise-sep block 1
        nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=16, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 24, kernel_size=1, bias=False),
        nn.BatchNorm2d(24),
        nn.ReLU(),
        # Depthwise-sep block 2
        nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1, groups=24, bias=False),
        nn.BatchNorm2d(24),
        nn.ReLU(),
        nn.Conv2d(24, 40, kernel_size=1, bias=False),
        nn.BatchNorm2d(40),
        nn.ReLU(),
        # Pool + classify
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(40, num_classes),
    )


def _spectrogram_cnn(num_classes: int = 35):
    """
    CNN for mel-spectrogram classification.

    Input shape: [B, 1, 64, 101]
    Ops: Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Linear

    Mirrors the M5 architecture used in the PyTorch keyword spotting tutorial
    adapted for 2D spectrogram input instead of 1D waveform.
    """
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),                         # → [B, 32, 32, 50]
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),                         # → [B, 64, 16, 25]
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),                 # → [B, 128, 1, 1]
        nn.Flatten(),                            # → [B, 128]
        nn.Linear(128, num_classes),
    )


class _bert_style_classifier(object.__class__):
    pass


import torch.nn as nn
import torch


class _bert_style_classifier(nn.Module):  # noqa: F811
    """
    Minimal BERT-style classifier.

    Exports Attention, LayerNormalization, and GELU ops into the ONNX graph
    — the exact Tier 1 NLP detection signals.

    The attention_mask input is accepted but not applied inside the transformer
    (torch's nn.TransformerEncoderLayer applies it via src_key_padding_mask).
    We pass it as a separate graph input so it appears in the ONNX graph and
    triggers the attention_mask name heuristic at Tier 2.
    """

    def __init__(
        self,
        vocab_size:  int = 30522,
        hidden_size: int = 256,
        num_heads:   int = 8,
        num_layers:  int = 2,
        seq_len:     int = 128,
        num_classes: int = 2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model          = hidden_size,
            nhead            = num_heads,
            dim_feedforward  = hidden_size * 4,
            dropout          = 0.0,   # no dropout — deterministic export
            activation       = "gelu",
            batch_first      = True,
            norm_first       = False,
        )
        self.encoder    = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        input_ids:      torch.Tensor,   # [B, seq_len] int64
        attention_mask: torch.Tensor,   # [B, seq_len] int64 — kept as graph input
    ) -> torch.Tensor:                  # [B, num_classes] float32
        # Embedding lookup
        x = self.embedding(input_ids)   # [B, seq_len, hidden_size]

        # Transformer encoder WITHOUT src_key_padding_mask.
        # src_key_padding_mask triggers _nested_tensor_from_mask which CoreML
        # cannot convert.  For fixture purposes the attention_mask tensor is
        # still accepted as a graph input (so the input signature is preserved
        # for ONNX detection tests), but the encoder runs dense with no masking.
        # This is semantically wrong for real inference but correct for testing
        # the task detection, input-role classification, and profiling pipeline.
        _ = attention_mask              # keep in graph signature via no-op use
        x = self.encoder(x)             # [B, seq_len, H] — no padding mask

        # Mean pool over sequence → classify
        x = x.mean(dim=1)              # [B, hidden_size]
        return self.classifier(x)       # [B, num_classes]


# ============================================================
# Registration
# ============================================================

def register_baseline_onnx(generators: list) -> None:
    """
    Register Step 2 generators into the main generator list.

    Called from generate_fixtures.py main() before the generation loop.

    Each entry is (generator_fn, model_id).  The model_id is used for
    --only filtering and for result tracking.
    """
    generators.extend([
        (_gen_B01, "B01"),
        (_gen_B05, "B05"),
        (_gen_B09, "B09"),
        (_gen_B13, "B13"),
    ])
    logger.info("step2: registered 4 baseline ONNX generators (B01, B05, B09, B13)")