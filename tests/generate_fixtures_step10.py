"""
Step 10 — Stress models (18 models).

S01   vision      ONNX              MobileNetV3 dynamic batch
S02   vision      ONNX              ResNet-style FP16
S03   vision      CoreML .mlpackage MobileNetV3 palettized (4-bit)
S04   vision      ONNX              Dual-output segmentation head
S05   vision      ONNX              ViT-style transformer (4L, 256H)
S06   vision      ONNX              CNN with baked Resize+Normalize
S07   nlp         ONNX              12-layer transformer dynamic seq
S08   nlp         CoreML .mlpackage DistilBERT-style FP16 MLProgram
S11   audio       ONNX              Wav2Vec2-style Conv1d+transformer
S12   tabular     ONNX              Oversized MLP (2048 hidden)
S13   vision      CoreML .mlmodel   EfficientNet-style NeuralNetwork
S14   nlp         ONNX              Dual-encoder (tokens + metadata_vec)
SD1   detection   ONNX              RT-DETR style transformer detector
ST1   timeseries  ONNX              PatchTST-style patched attention
ST2   timeseries  ONNX              Stateful LSTM with explicit h0/c0
SM1   multimodal  ONNX              Asymmetric image+token two-tower
SR1   rec         ONNX              Two-tower large Gather (vocab=100k)
SG1   generative  ONNX              GPT-2 + 12 KV-cache inputs
"""

from __future__ import annotations
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("mlbuild.fixtures.step10")


def _reset_seed():
    import numpy as np
    torch.manual_seed(42)
    np.random.seed(42)


# ─── S01 Vision: dynamic batch ───────────────────────────────────────────────
def _gen_S01(dirs, manifest, result):
    from generate_fixtures import ManifestEntry
    model_id = "S01"
    try:
        _reset_seed()
        try:
            import torchvision.models as tvm
            model = tvm.mobilenet_v3_small(weights=None).eval()
        except ImportError:
            model = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(32, 1000),
            ).eval()
        dummy = torch.zeros(1, 3, 224, 224)
        path = dirs.onnx_path(model_id, "mobilenetv3_dynamic_batch")
        path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            torch.onnx.export(model, (dummy,), str(path),
                input_names=["pixel_values"], output_names=["logits"],
                dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=18, do_constant_folding=True)
        import onnx as _o; _o.checker.check_model(str(path))
        logger.info("exported ONNX  %s", path.name)
        manifest.add(ManifestEntry(
            model_id="S01", tier="stress", modality="vision", format="onnx",
            size_class="small", file=str(path.relative_to(dirs.root)),
            expected_domain="vision", expected_subtype="none",
            expected_execution="standard", expected_failure=False,
            commands=["inspect"],
        ))
        manifest.stamp_sha256(model_id, path); result.ok(model_id)
    except Exception as exc: result.fail(model_id, exc)


# ─── S02 Vision: FP16 export ─────────────────────────────────────────────────
def _gen_S02(dirs, manifest, result):
    from generate_fixtures import ManifestEntry
    model_id = "S02"
    try:
        _reset_seed()
        model = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 1000),
        ).half().eval()   # FP16 weights
        dummy = torch.zeros(1, 3, 224, 224, dtype=torch.float16)
        path = dirs.onnx_path(model_id, "resnet_style_fp16")
        path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            torch.onnx.export(model, (dummy,), str(path),
                input_names=["pixel_values"], output_names=["logits"],
                dynamic_axes={}, opset_version=18, do_constant_folding=True)
        import onnx as _o; _o.checker.check_model(str(path))
        logger.info("exported ONNX  %s", path.name)
        manifest.add(ManifestEntry(
            model_id="S02", tier="stress", modality="vision", format="onnx",
            size_class="small", file=str(path.relative_to(dirs.root)),
            expected_domain="vision", expected_subtype="none",
            expected_execution="standard", expected_failure=False,
            commands=["inspect"],
        ))
        manifest.stamp_sha256(model_id, path); result.ok(model_id)
    except Exception as exc: result.fail(model_id, exc)


# ─── S03 Vision: CoreML palettized ───────────────────────────────────────────
def _gen_S03(dirs, manifest, result):
    from generate_fixtures import _to_coreml_mlprogram, ManifestEntry, COREML_AVAILABLE
    model_id = "S03"
    if not COREML_AVAILABLE:
        result.skip(model_id, "coremltools not available"); return
    try:
        _reset_seed()
        try:
            import torchvision.models as tvm
            model = tvm.mobilenet_v3_small(weights=None).eval()
        except ImportError:
            model = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 1000),
            ).eval()
        out = dirs.coreml_path(model_id, "mobilenetv3_palettized_mlprog", "mlpackage")

        import coremltools as ct

        dummy = torch.zeros(1, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            traced = torch.jit.trace(model, (dummy,))
        ct_inputs = [ct.TensorType(name="pixel_values", shape=dummy.shape)]
        mlmodel = ct.convert(traced, source="pytorch", inputs=ct_inputs,
                             minimum_deployment_target=ct.target.iOS15,
                             convert_to="mlprogram")

        # Palettize weights to 4-bit
        try:
            from coremltools.optimize.coreml import (
                palettize_weights, OptimizationConfig, OpPalettizerConfig,
            )
            op_config = OpPalettizerConfig(mode="kmeans", nbits=4)
            config    = OptimizationConfig(global_config=op_config)
            mlmodel   = palettize_weights(mlmodel, config)
        except Exception as pal_exc:
            logger.warning("S03: palettize_weights failed (%s) — saving unquantized", pal_exc)

        out.parent.mkdir(parents=True, exist_ok=True)
        mlmodel.save(str(out))
        from generate_fixtures import _human_size
        logger.info("exported CoreML .mlpackage  %s  (%s)", out.name, _human_size(out))

        manifest.add(ManifestEntry(
            model_id="S03", tier="stress", modality="vision", format="mlpackage",
            size_class="small", file=str(out.relative_to(dirs.root)),
            expected_domain="vision", expected_subtype="none",
            expected_execution="standard", expected_failure=False,
            commands=["inspect"],
        ))
        manifest.stamp_sha256(model_id, out); result.ok(model_id)
    except Exception as exc: result.fail(model_id, exc)


# ─── S04 Vision: dual-output segmentation head ───────────────────────────────
def _gen_S04(dirs, manifest, result):
    from generate_fixtures import ManifestEntry
    model_id = "S04"
    class _dual_out(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(),
            )
            self.cls_head = nn.Conv2d(64, 21, 1)   # 21 classes (VOC)
            self.seg_head = nn.Conv2d(64, 1, 1)    # binary mask
        def forward(self, x):
            f = self.backbone(x)
            return self.cls_head(f), self.seg_head(f)
    try:
        _reset_seed()
        model = _dual_out().eval()
        dummy = torch.zeros(1, 3, 224, 224)
        path = dirs.onnx_path(model_id, "deeplabv3_dual_output")
        path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            torch.onnx.export(model, (dummy,), str(path),
                input_names=["pixel_values"], output_names=["class_logits", "seg_mask"],
                dynamic_axes={}, opset_version=18, do_constant_folding=True)
        import onnx as _o; _o.checker.check_model(str(path))
        logger.info("exported ONNX  %s", path.name)
        manifest.add(ManifestEntry(
            model_id="S04", tier="stress", modality="vision", format="onnx",
            size_class="small", file=str(path.relative_to(dirs.root)),
            expected_domain="vision", expected_subtype="none",
            expected_execution="standard", expected_failure=False,
            commands=["inspect"],
        ))
        manifest.stamp_sha256(model_id, path); result.ok(model_id)
    except Exception as exc: result.fail(model_id, exc)


# ─── S05 Vision: ViT-style transformer ───────────────────────────────────────
def _gen_S05(dirs, manifest, result):
    from generate_fixtures import ManifestEntry
    from generate_fixtures_step9 import _gpt2_coreml  # reuse primitive attn

    # ViT-style: patch embed → transformer → class token → head
    class _vit_style(nn.Module):
        def __init__(self, patch=16, img=224, hidden=256, layers=4, heads=4, num_cls=1000):
            super().__init__()
            self.patch_embed = nn.Conv2d(3, hidden, patch, stride=patch)
            num_patches = (img // patch) ** 2
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden))
            self.pos_emb   = nn.Parameter(torch.zeros(1, num_patches + 1, hidden))
            self.layers = nn.ModuleList()
            for _ in range(layers):
                self.layers.append(nn.ModuleDict({
                    "q": nn.Linear(hidden, hidden), "k": nn.Linear(hidden, hidden),
                    "v": nn.Linear(hidden, hidden), "o": nn.Linear(hidden, hidden),
                    "ff1": nn.Linear(hidden, hidden*4), "ff2": nn.Linear(hidden*4, hidden),
                    "ln1": nn.LayerNorm(hidden), "ln2": nn.LayerNorm(hidden),
                }))
            self.head = nn.Linear(hidden, num_cls)
            self.num_heads = heads
            self.head_dim  = hidden // heads

        def forward(self, pixel_values):
            B = pixel_values.shape[0]
            x = self.patch_embed(pixel_values)                # [B, H, P, P]
            x = x.flatten(2).transpose(1, 2)                  # [B, N, H]
            cls = self.cls_token.expand(B, -1, -1)
            x   = torch.cat([cls, x], dim=1) + self.pos_emb  # [B, N+1, H]
            nh, dh = self.num_heads, self.head_dim
            for l in self.layers:
                T = x.shape[1]
                Q = l["q"](x).reshape(B,T,nh,dh).transpose(1,2)
                K = l["k"](x).reshape(B,T,nh,dh).transpose(1,2)
                V = l["v"](x).reshape(B,T,nh,dh).transpose(1,2)
                w = torch.softmax(torch.matmul(Q, K.transpose(-2,-1)) * (dh**-0.5), dim=-1)
                a = l["o"](torch.matmul(w,V).transpose(1,2).reshape(B,T,-1))
                x = l["ln1"](x + a)
                x = l["ln2"](x + l["ff2"](F.gelu(l["ff1"](x))))
            return self.head(x[:, 0])  # class token

    model_id = "S05"
    try:
        _reset_seed()
        model = _vit_style(patch=16, img=224, hidden=256, layers=4, heads=4, num_cls=1000).eval()
        dummy = torch.zeros(1, 3, 224, 224)
        path = dirs.onnx_path(model_id, "vit_style_vision")
        path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            torch.onnx.export(model, (dummy,), str(path),
                input_names=["pixel_values"], output_names=["logits"],
                dynamic_axes={}, opset_version=18, do_constant_folding=True)
        import onnx as _o; _o.checker.check_model(str(path))
        logger.info("exported ONNX  %s", path.name)
        manifest.add(ManifestEntry(
            model_id="S05", tier="stress", modality="vision", format="onnx",
            size_class="medium", file=str(path.relative_to(dirs.root)),
            expected_domain="vision", expected_subtype="none",
            expected_execution="standard", expected_failure=False,
            commands=["inspect"],
        ))
        manifest.stamp_sha256(model_id, path); result.ok(model_id)
    except Exception as exc: result.fail(model_id, exc)


# ─── S06 Vision: baked Resize+Normalize preprocessing ────────────────────────
def _gen_S06(dirs, manifest, result):
    from generate_fixtures import ManifestEntry
    class _preprocessed_cnn(nn.Module):
        def __init__(self):
            super().__init__()
            # Normalization parameters baked as buffers
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
            self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 1000),
            )
        def forward(self, x):
            # Baked Resize (bilinear interpolation) + Normalize
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            x = (x - self.mean) / self.std
            return self.backbone(x)

    model_id = "S06"
    try:
        _reset_seed()
        model = _preprocessed_cnn().eval()
        dummy = torch.zeros(1, 3, 256, 256)   # different input size — resize is baked in
        path = dirs.onnx_path(model_id, "resnet18_baked_preprocess")
        path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            torch.onnx.export(model, (dummy,), str(path),
                input_names=["raw_image"], output_names=["logits"],
                dynamic_axes={}, opset_version=18, do_constant_folding=True)
        import onnx as _o; _o.checker.check_model(str(path))
        logger.info("exported ONNX  %s", path.name)
        manifest.add(ManifestEntry(
            model_id="S06", tier="stress", modality="vision", format="onnx",
            size_class="small", file=str(path.relative_to(dirs.root)),
            expected_domain="vision", expected_subtype="none",
            expected_execution="standard", expected_failure=False,
            commands=["inspect"],
        ))
        manifest.stamp_sha256(model_id, path); result.ok(model_id)
    except Exception as exc: result.fail(model_id, exc)


# ─── S07 NLP: 12-layer transformer dynamic seq ───────────────────────────────
def _gen_S07(dirs, manifest, result):
    from generate_fixtures import ManifestEntry
    from generate_fixtures_step3 import _bert_coreml   # reuse primitive bert

    model_id = "S07"
    try:
        _reset_seed()
        # 12-layer, 256-hidden, dynamic sequence
        model = _bert_coreml(vocab_size=1000, hidden_size=256,
                             num_heads=8, num_layers=12, num_classes=2).eval()
        ids  = torch.zeros(1, 128, dtype=torch.int64)
        mask = torch.ones(1, 128, dtype=torch.int64)
        path = dirs.onnx_path(model_id, "transformer_12l_dynamic_seq")
        path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            torch.onnx.export(model, (ids, mask), str(path),
                input_names=["input_ids", "attention_mask"], output_names=["logits"],
                dynamic_axes={"input_ids": {1: "seq_len"}, "attention_mask": {1: "seq_len"}},
                opset_version=18, do_constant_folding=True)
        import onnx as _o; _o.checker.check_model(str(path))
        logger.info("exported ONNX  %s", path.name)
        manifest.add(ManifestEntry(
            model_id="S07", tier="stress", modality="nlp", format="onnx",
            size_class="medium", file=str(path.relative_to(dirs.root)),
            expected_domain="nlp", expected_subtype="none",
            expected_execution="multi_input", expected_failure=False,
            commands=["inspect"],
        ))
        manifest.stamp_sha256(model_id, path); result.ok(model_id)
    except Exception as exc: result.fail(model_id, exc)


# ─── S08 NLP: DistilBERT-style FP16 CoreML ───────────────────────────────────
def _gen_S08(dirs, manifest, result):
    from generate_fixtures import _to_coreml_mlprogram, ManifestEntry, COREML_AVAILABLE
    from generate_fixtures_step3 import _bert_coreml

    model_id = "S08"
    if not COREML_AVAILABLE:
        result.skip(model_id, "coremltools not available"); return
    try:
        _reset_seed()
        model = _bert_coreml(vocab_size=1000, hidden_size=256,
                             num_heads=8, num_layers=6, num_classes=2).eval()
        ids  = torch.zeros(1, 128, dtype=torch.int64)
        mask = torch.ones(1, 128, dtype=torch.int64)
        out  = dirs.coreml_path(model_id, "distilbert_style_fp16_mlprog", "mlpackage")
        converted = _to_coreml_mlprogram(model, (ids, mask),
                                         ["input_ids", "attention_mask"], out, fp16=True)
        if converted is None:
            raise RuntimeError(f"CoreML MLProgram conversion returned None for {model_id}")
        manifest.add(ManifestEntry(
            model_id="S08", tier="stress", modality="nlp", format="mlpackage",
            size_class="medium", file=str(out.relative_to(dirs.root)),
            expected_domain="nlp", expected_subtype="none",
            expected_execution="multi_input", expected_failure=False,
            commands=["inspect"],
        ))
        manifest.stamp_sha256(model_id, out); result.ok(model_id)
    except Exception as exc: result.fail(model_id, exc)


# ─── S11 Audio: Wav2Vec2-style Conv1d feature extractor ──────────────────────
def _gen_S11(dirs, manifest, result):
    from generate_fixtures import ManifestEntry

    class _wav2vec2_lite(nn.Module):
        """7-layer Conv1d feature extractor (Wav2Vec2-style) + 2-layer transformer."""
        def __init__(self):
            super().__init__()
            self.feature_extractor = nn.Sequential(
                nn.Conv1d(1, 32, 10, stride=5), nn.GELU(),
                nn.Conv1d(32, 64, 3, stride=2), nn.GELU(),
                nn.Conv1d(64, 64, 3, stride=2), nn.GELU(),
                nn.Conv1d(64, 64, 3, stride=2), nn.GELU(),
                nn.Conv1d(64, 64, 3, stride=2), nn.GELU(),
                nn.Conv1d(64, 64, 2, stride=2), nn.GELU(),
                nn.Conv1d(64, 64, 2, stride=2), nn.GELU(),
            )
            self.proj = nn.Linear(64, 128)
            self.ln   = nn.LayerNorm(128)
            self.head = nn.Linear(128, 1)

        def forward(self, waveform):
            x = self.feature_extractor(waveform)   # [B, 64, T']
            x = x.transpose(1, 2)                  # [B, T', 64]
            x = self.ln(self.proj(x))
            return self.head(x.mean(dim=1))         # [B, 1]

    model_id = "S11"
    try:
        _reset_seed()
        model = _wav2vec2_lite().eval()
        dummy = torch.zeros(1, 1, 16000)  # 1s @ 16kHz
        path  = dirs.onnx_path(model_id, "wav2vec2_style_audio")
        path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            torch.onnx.export(model, (dummy,), str(path),
                input_names=["waveform"], output_names=["logits"],
                dynamic_axes={}, opset_version=18, do_constant_folding=True)
        import onnx as _o; _o.checker.check_model(str(path))
        logger.info("exported ONNX  %s", path.name)
        manifest.add(ManifestEntry(
            model_id="S11", tier="stress", modality="audio", format="onnx",
            size_class="small", file=str(path.relative_to(dirs.root)),
            expected_domain="audio", expected_subtype="none",
            expected_execution="standard", expected_failure=False,
            commands=["inspect"],
        ))
        manifest.stamp_sha256(model_id, path); result.ok(model_id)
    except Exception as exc: result.fail(model_id, exc)


# ─── S12 Tabular: oversized MLP ──────────────────────────────────────────────
def _gen_S12(dirs, manifest, result):
    from generate_fixtures import ManifestEntry
    model_id = "S12"
    try:
        _reset_seed()
        model = nn.Sequential(
            nn.Linear(128, 2048), nn.ReLU(),
            nn.Linear(2048, 2048), nn.ReLU(),
            nn.Linear(2048, 1024), nn.ReLU(),
            nn.Linear(1024, 512),  nn.ReLU(),
            nn.Linear(512,  256),  nn.ReLU(),
            nn.Linear(256,  1),
        ).eval()
        dummy = torch.zeros(1, 128)
        path  = dirs.onnx_path(model_id, "oversized_mlp_tabular")
        path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            torch.onnx.export(model, (dummy,), str(path),
                input_names=["features"], output_names=["output"],
                dynamic_axes={}, opset_version=18, do_constant_folding=True)
        import onnx as _o; _o.checker.check_model(str(path))
        logger.info("exported ONNX  %s", path.name)
        manifest.add(ManifestEntry(
            model_id="S12", tier="stress", modality="tabular", format="onnx",
            size_class="medium", file=str(path.relative_to(dirs.root)),
            expected_domain="tabular", expected_subtype="none",
            expected_execution="standard", expected_failure=False,
            commands=["inspect"],
        ))
        manifest.stamp_sha256(model_id, path); result.ok(model_id)
    except Exception as exc: result.fail(model_id, exc)


# ─── S13 Vision: CoreML NeuralNetwork (.mlmodel) ─────────────────────────────
def _gen_S13(dirs, manifest, result):
    from generate_fixtures import _to_coreml_nn, ManifestEntry, COREML_AVAILABLE
    model_id = "S13"
    if not COREML_AVAILABLE:
        result.skip(model_id, "coremltools not available"); return
    try:
        _reset_seed()
        # EfficientNet-B0 style: compound-scaled CNN
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 1000),
        ).eval()
        dummy = torch.zeros(1, 3, 224, 224)
        out   = dirs.coreml_path(model_id, "efficientnet_style_nn", "mlmodel")
        converted = _to_coreml_nn(model, (dummy,), ["pixel_values"], out)
        if converted is None:
            raise RuntimeError(f"CoreML NeuralNetwork conversion returned None for {model_id}")
        manifest.add(ManifestEntry(
            model_id="S13", tier="stress", modality="vision", format="mlmodel",
            size_class="small", file=str(out.relative_to(dirs.root)),
            expected_domain="vision", expected_subtype="none",
            expected_execution="standard", expected_failure=False,
            commands=["inspect"],
        ))
        manifest.stamp_sha256(model_id, out); result.ok(model_id)
    except Exception as exc: result.fail(model_id, exc)


# ─── S14 NLP: dual-encoder (tokens + metadata_vec) ───────────────────────────
def _gen_S14(dirs, manifest, result):
    from generate_fixtures import ManifestEntry
    from generate_fixtures_step3 import _bert_coreml

    class _dual_encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_enc = _bert_coreml(vocab_size=1000, hidden_size=128,
                                          num_heads=4, num_layers=2, num_classes=64)
            self.meta_proj = nn.Linear(64, 64)
            self.head      = nn.Linear(128, 1)
        def forward(self, input_ids, attention_mask, metadata_vec):
            t = self.token_enc(input_ids, attention_mask)  # [B, 64]
            m = self.meta_proj(metadata_vec)               # [B, 64]
            return self.head(torch.cat([t, m], dim=-1))    # [B, 1]

    model_id = "S14"
    try:
        _reset_seed()
        model = _dual_encoder().eval()
        ids  = torch.zeros(1, 128, dtype=torch.int64)
        mask = torch.ones(1, 128, dtype=torch.int64)
        meta = torch.zeros(1, 64)
        path = dirs.onnx_path(model_id, "dual_encoder_nlp")
        path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            torch.onnx.export(model, (ids, mask, meta), str(path),
                input_names=["input_ids", "attention_mask", "metadata_vec"],
                output_names=["score"],
                dynamic_axes={}, opset_version=18, do_constant_folding=True)
        import onnx as _o; _o.checker.check_model(str(path))
        logger.info("exported ONNX  %s", path.name)
        manifest.add(ManifestEntry(
            model_id="S14", tier="stress", modality="nlp", format="onnx",
            size_class="small", file=str(path.relative_to(dirs.root)),
            expected_domain="nlp", expected_subtype="none",
            expected_execution="multi_input", expected_failure=False,
            commands=["inspect"],
        ))
        manifest.stamp_sha256(model_id, path); result.ok(model_id)
    except Exception as exc: result.fail(model_id, exc)


# ─── SD1 Detection: RT-DETR style transformer detector ───────────────────────
def _gen_SD1(dirs, manifest, result):
    from generate_fixtures import ManifestEntry
    from generate_fixtures_step3 import _bert_coreml  # reuse transformer block

    class _rt_detr_lite(nn.Module):
        """Simplified RT-DETR: CNN backbone + 2-layer cross-attention decoder."""
        def __init__(self, num_queries=100, num_classes=80):
            super().__init__()
            hidden = 128
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128), nn.ReLU(),
                nn.AdaptiveAvgPool2d(16),
            )
            self.queries = nn.Parameter(torch.zeros(1, num_queries, hidden))
            self.q = nn.Linear(hidden, hidden); self.k = nn.Linear(hidden, hidden)
            self.v = nn.Linear(hidden, hidden); self.o = nn.Linear(hidden, hidden)
            self.ln = nn.LayerNorm(hidden)
            self.box_head   = nn.Linear(hidden, 4)
            self.score_head = nn.Linear(hidden, num_classes)
            self.num_heads = 4; self.head_dim = hidden // 4

        def forward(self, pixel_values):
            B = pixel_values.shape[0]
            feat = self.backbone(pixel_values)         # [B, 128, 16, 16]
            mem  = feat.flatten(2).transpose(1,2)      # [B, 256, 128]
            q    = self.queries.expand(B, -1, -1)      # [B, 100, 128]
            T, S = q.shape[1], mem.shape[1]
            nh, dh = self.num_heads, self.head_dim
            Q = self.q(q).reshape(B,T,nh,dh).transpose(1,2)
            K = self.k(mem).reshape(B,S,nh,dh).transpose(1,2)
            V = self.v(mem).reshape(B,S,nh,dh).transpose(1,2)
            w = torch.softmax(torch.matmul(Q, K.transpose(-2,-1)) * (dh**-0.5), dim=-1)
            a = self.o(torch.matmul(w,V).transpose(1,2).reshape(B,T,-1))
            q = self.ln(q + a)
            return self.box_head(q), self.score_head(q)  # [B,100,4], [B,100,80]

    model_id = "SD1"
    try:
        _reset_seed()
        model = _rt_detr_lite().eval()
        dummy = torch.zeros(1, 3, 640, 640)
        path  = dirs.onnx_path(model_id, "rt_detr_style_detection")
        path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            torch.onnx.export(model, (dummy,), str(path),
                input_names=["pixel_values"], output_names=["boxes", "scores"],
                dynamic_axes={}, opset_version=18, do_constant_folding=True)
        import onnx as _o; _o.checker.check_model(str(path))
        logger.info("exported ONNX  %s", path.name)
        manifest.add(ManifestEntry(
            model_id="SD1", tier="stress", modality="detection", format="onnx",
            size_class="small", file=str(path.relative_to(dirs.root)),
            expected_domain="vision", expected_subtype="detection",
            expected_execution="standard", expected_nms_inside=False,
            expected_failure=False, commands=["inspect"],
        ))
        manifest.stamp_sha256(model_id, path); result.ok(model_id)
    except Exception as exc: result.fail(model_id, exc)


# ─── ST1 Timeseries: PatchTST-style dynamic seq ──────────────────────────────
def _gen_ST1(dirs, manifest, result):
    from generate_fixtures import ManifestEntry

    class _patch_tst(nn.Module):
        """Patch-based TS transformer: [B, 7, 96] → [B, 7, 24]."""
        def __init__(self, n_vars=7, patch_len=16, hidden=64, layers=2, pred_len=24):
            super().__init__()
            self.patch_len = patch_len
            self.patch_emb = nn.Linear(patch_len, hidden)
            self.attn = nn.ModuleList([nn.MultiheadAttention(hidden, 4, batch_first=True, dropout=0.0)
                                       for _ in range(layers)])
            self.lns  = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
            self.head = nn.Linear(hidden, pred_len)

        def forward(self, x):
            # x: [B, n_vars, seq_len]
            B, V, L = x.shape
            P = L // self.patch_len
            x = x.reshape(B * V, P, self.patch_len)    # [B*V, P, patch_len]
            x = self.patch_emb(x)                       # [B*V, P, hidden]
            for attn, ln in zip(self.attn, self.lns):
                a, _ = attn(x, x, x)
                x = ln(x + a)
            out = self.head(x.mean(dim=1))              # [B*V, pred_len]
            return out.reshape(B, V, -1)                # [B, n_vars, pred_len]

    model_id = "ST1"
    try:
        _reset_seed()
        model = _patch_tst(n_vars=7, patch_len=16, hidden=64, layers=2, pred_len=24).eval()
        dummy = torch.zeros(1, 7, 96)
        path  = dirs.onnx_path(model_id, "patch_tst_timeseries")
        path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            torch.onnx.export(model, (dummy,), str(path),
                input_names=["timeseries"], output_names=["forecast"],
                dynamic_axes={"timeseries": {2: "seq_len"}},
                opset_version=18, do_constant_folding=True)
        import onnx as _o; _o.checker.check_model(str(path))
        logger.info("exported ONNX  %s", path.name)
        manifest.add(ManifestEntry(
            model_id="ST1", tier="stress", modality="timeseries", format="onnx",
            size_class="tiny", file=str(path.relative_to(dirs.root)),
            expected_domain="audio", expected_subtype="timeseries",
            expected_execution="standard", expected_failure=False,
            commands=["inspect"],
        ))
        manifest.stamp_sha256(model_id, path); result.ok(model_id)
    except Exception as exc: result.fail(model_id, exc)


# ─── ST2 Timeseries: stateful LSTM with explicit h0/c0 ───────────────────────
def _gen_ST2(dirs, manifest, result):
    from generate_fixtures import ManifestEntry

    class _stateful_lstm(nn.Module):
        """LSTM that takes explicit hidden/cell state inputs and outputs them.
        6 tensors in: x[1,1,1], h0[2,1,64], c0[2,1,64]
        3 tensors out: y[1,1,64], h1[2,1,64], c1[2,1,64]
        """
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 64, num_layers=2, batch_first=True)

        def forward(self, x, h0, c0):
            y, (h1, c1) = self.lstm(x, (h0, c0))
            return y, h1, c1

    model_id = "ST2"
    try:
        _reset_seed()
        model = _stateful_lstm().eval()
        x  = torch.zeros(1, 1, 1)
        h0 = torch.zeros(2, 1, 64)
        c0 = torch.zeros(2, 1, 64)
        path = dirs.onnx_path(model_id, "stateful_lstm_timeseries")
        path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            torch.onnx.export(model, (x, h0, c0), str(path),
                input_names=["x", "h0", "c0"],
                output_names=["y", "h1", "c1"],
                dynamic_axes={}, opset_version=18, do_constant_folding=True)
        import onnx as _o; _o.checker.check_model(str(path))
        logger.info("exported ONNX  %s", path.name)
        manifest.add(ManifestEntry(
            model_id="ST2", tier="stress", modality="timeseries", format="onnx",
            size_class="tiny", file=str(path.relative_to(dirs.root)),
            expected_domain="audio", expected_subtype="timeseries",
            expected_execution="stateful", expected_failure=False,
            commands=["inspect"],
        ))
        manifest.stamp_sha256(model_id, path); result.ok(model_id)
    except Exception as exc: result.fail(model_id, exc)


# ─── SM1 Multimodal: asymmetric two-tower ────────────────────────────────────
def _gen_SM1(dirs, manifest, result):
    from generate_fixtures import ManifestEntry
    from generate_fixtures_step7 import _vision_tower, _text_tower

    class _asymmetric_clip(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = _vision_tower(embed_dim=256)
            self.text   = _text_tower(vocab_size=1000, embed_dim=256)
        def forward(self, pixel_values, input_ids):
            v = self.vision(pixel_values)
            t = self.text(input_ids[:, :32])   # shorter token seq (32 vs 77)
            return (v * t).sum(dim=-1, keepdim=True)

    model_id = "SM1"
    try:
        _reset_seed()
        model  = _asymmetric_clip().eval()
        pixels = torch.zeros(1, 3, 224, 224)
        ids    = torch.zeros(1, 32, dtype=torch.int64)
        path   = dirs.onnx_path(model_id, "asymmetric_twotower_multimodal")
        path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            torch.onnx.export(model, (pixels, ids), str(path),
                input_names=["pixel_values", "input_ids"],
                output_names=["similarity"],
                dynamic_axes={}, opset_version=18, do_constant_folding=True)
        import onnx as _o; _o.checker.check_model(str(path))
        logger.info("exported ONNX  %s", path.name)
        manifest.add(ManifestEntry(
            model_id="SM1", tier="stress", modality="multimodal", format="onnx",
            size_class="small", file=str(path.relative_to(dirs.root)),
            expected_domain="vision", expected_subtype="multimodal",
            expected_execution="multi_input", expected_failure=False,
            commands=["inspect"],
        ))
        manifest.stamp_sha256(model_id, path); result.ok(model_id)
    except Exception as exc: result.fail(model_id, exc)


# ─── SR1 Rec: large Gather (vocab=100k) ──────────────────────────────────────
def _gen_SR1(dirs, manifest, result):
    from generate_fixtures import ManifestEntry
    from generate_fixtures_step8 import _neural_mf

    model_id = "SR1"
    try:
        _reset_seed()
        # Large embedding tables dominate size: 100k×32 = ~13MB user, 500k×32 = ~64MB item
        # Keep item_vocab reasonable to avoid OOM: 50k
        model   = _neural_mf(user_vocab=100000, item_vocab=50000, embed_dim=32).eval()
        user_id = torch.zeros(1, dtype=torch.int64)
        item_id = torch.zeros(1, dtype=torch.int64)
        path    = dirs.onnx_path(model_id, "large_gather_rec")
        path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            torch.onnx.export(model, (user_id, item_id), str(path),
                input_names=["user_id", "item_id"], output_names=["score"],
                dynamic_axes={}, opset_version=18, do_constant_folding=True)
        import onnx as _o; _o.checker.check_model(str(path))
        logger.info("exported ONNX  %s", path.name)
        manifest.add(ManifestEntry(
            model_id="SR1", tier="stress", modality="rec", format="onnx",
            size_class="large", file=str(path.relative_to(dirs.root)),
            expected_domain="tabular", expected_subtype="recommendation",
            expected_execution="multi_input", expected_failure=False,
            commands=["inspect"],
        ))
        manifest.stamp_sha256(model_id, path); result.ok(model_id)
    except Exception as exc: result.fail(model_id, exc)


# ─── SG1 Generative: GPT-2 + KV-cache inputs ─────────────────────────────────
def _gen_SG1(dirs, manifest, result):
    from generate_fixtures import ManifestEntry

    class _gpt2_kvcache(nn.Module):
        """GPT-2 with explicit past_key/past_value inputs per layer.
        6 layers → 12 KV-cache tensors (6 past_key + 6 past_value).
        Inputs: input_ids [1,1] + past_key_{i} [1,4,N,32] + past_value_{i} [1,4,N,32]
        Outputs: logits [1,1,50257] + present_key_{i} + present_value_{i}
        """
        def __init__(self, vocab=50257, hidden=128, heads=4, layers=6, max_len=128):
            super().__init__()
            self.layers_n  = layers
            self.heads     = heads
            self.head_dim  = hidden // heads
            self.token_emb = nn.Embedding(vocab, hidden)
            self.pos_emb   = nn.Embedding(max_len, hidden)
            self.q  = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
            self.k  = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
            self.v  = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
            self.o  = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
            self.ff1 = nn.ModuleList([nn.Linear(hidden, hidden*4) for _ in range(layers)])
            self.ff2 = nn.ModuleList([nn.Linear(hidden*4, hidden)  for _ in range(layers)])
            self.ln1 = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
            self.ln2 = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
            self.ln_f    = nn.LayerNorm(hidden)
            self.lm_head = nn.Linear(hidden, vocab, bias=False)

        def forward(self, input_ids,
                    pk0, pv0, pk1, pv1, pk2, pv2,
                    pk3, pv3, pk4, pv4, pk5, pv5):
            past_keys   = [pk0, pk1, pk2, pk3, pk4, pk5]
            past_values = [pv0, pv1, pv2, pv3, pv4, pv5]
            B, T = input_ids.shape
            past_len = past_keys[0].shape[2]
            pos = torch.arange(past_len, past_len + T, device=input_ids.device).unsqueeze(0)
            x = self.token_emb(input_ids) + self.pos_emb(pos)
            nh, dh = self.heads, self.head_dim
            present_keys, present_values = [], []
            for i in range(self.layers_n):
                Q = self.q[i](x).reshape(B,T,nh,dh).transpose(1,2)
                K_cur = self.k[i](x).reshape(B,T,nh,dh).transpose(1,2)
                V_cur = self.v[i](x).reshape(B,T,nh,dh).transpose(1,2)
                K = torch.cat([past_keys[i], K_cur], dim=2)
                V = torch.cat([past_values[i], V_cur], dim=2)
                present_keys.append(K); present_values.append(V)
                w = torch.softmax(torch.matmul(Q, K.transpose(-2,-1)) * (dh**-0.5), dim=-1)
                a = self.o[i](torch.matmul(w,V).transpose(1,2).reshape(B,T,-1))
                x = self.ln1[i](x + a)
                x = self.ln2[i](x + self.ff2[i](F.gelu(self.ff1[i](x))))
            logits = self.lm_head(self.ln_f(x))
            return (logits, *present_keys, *present_values)

    model_id = "SG1"
    try:
        _reset_seed()
        model    = _gpt2_kvcache(vocab=50257, hidden=128, heads=4, layers=6).eval()
        ids      = torch.zeros(1, 1, dtype=torch.int64)
        past_len = 64
        past_kv  = [torch.zeros(1, 4, past_len, 32) for _ in range(12)]

        in_names  = ["input_ids"] + [f"pk{i}" for i in range(6)] + [f"pv{i}" for i in range(6)]
        out_names = ["logits"]    + [f"pk{i}_out" for i in range(6)] + [f"pv{i}_out" for i in range(6)]
        path = dirs.onnx_path(model_id, "gpt2_kvcache_generative")
        path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            try:
                # torch.export (new dynamo path) may conflict with static-shape
                # dims inside the KV-cat; fall back to legacy jit.trace exporter.
                torch.onnx.export(model, (ids, *past_kv), str(path),
                    input_names=in_names, output_names=out_names,
                    dynamic_axes={},
                    opset_version=18, do_constant_folding=True)
            except Exception:
                # Legacy TorchScript-based exporter — works for static KV shapes
                traced = torch.jit.trace(model, (ids, *past_kv))
                torch.onnx.export(traced, (ids, *past_kv), str(path),
                    input_names=in_names, output_names=out_names,
                    dynamic_axes={},
                    opset_version=17, do_constant_folding=True)
        import onnx as _o; _o.checker.check_model(str(path))
        logger.info("exported ONNX  %s", path.name)
        manifest.add(ManifestEntry(
            model_id="SG1", tier="stress", modality="generative", format="onnx",
            size_class="medium", file=str(path.relative_to(dirs.root)),
            expected_domain="nlp", expected_subtype="generative_stateful",
            expected_execution="kv_cache", expected_failure=False,
            commands=["inspect"],
        ))
        manifest.stamp_sha256(model_id, path); result.ok(model_id)
    except Exception as exc: result.fail(model_id, exc)


# ─── Registration ─────────────────────────────────────────────────────────────
def register_stress_models(generators: list) -> None:
    generators.extend([
        (_gen_S01, "S01"), (_gen_S02, "S02"), (_gen_S03, "S03"),
        (_gen_S04, "S04"), (_gen_S05, "S05"), (_gen_S06, "S06"),
        (_gen_S07, "S07"), (_gen_S08, "S08"), (_gen_S11, "S11"),
        (_gen_S12, "S12"), (_gen_S13, "S13"), (_gen_S14, "S14"),
        (_gen_SD1, "SD1"), (_gen_ST1, "ST1"), (_gen_ST2, "ST2"),
        (_gen_SM1, "SM1"), (_gen_SR1, "SR1"), (_gen_SG1, "SG1"),
    ])
    logger.info("step10: registered 18 stress model generators")