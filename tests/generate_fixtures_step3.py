"""
Step 3 — CoreML baseline variants (torch-native conversion).

Pipeline: PyTorch → TorchScript → CoreML (NeuralNetwork or MLProgram)
ONNX is NOT used. coremltools only accepts pytorch/tensorflow/milinternal.

B14/B15 use _bert_coreml, NOT _bert_style_classifier from step2.
_bert_style_classifier uses nn.TransformerEncoderLayer which CoreML
cannot lower (_transformer_encoder_layer_fwd has no MIL mapping).
_bert_coreml decomposes the transformer into primitive ops that CoreML
does support: Linear, MatMul, Softmax, LayerNorm, GELU.
"""

from __future__ import annotations
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
logger = logging.getLogger("mlbuild.fixtures.step3")

from generate_fixtures_step2 import _minimal_cnn, _spectrogram_cnn


# ── CoreML-compatible transformer (primitive ops only) ────────────────────────

class _bert_coreml(nn.Module):
    """
    BERT-style classifier using only CoreML-compatible primitive ops.

    nn.TransformerEncoderLayer is fused by TorchScript into
    _transformer_encoder_layer_fwd, which has no MIL mapping in coremltools.
    This class expresses the same computation via:
      Linear (Q/K/V/O projections), MatMul, Softmax, LayerNorm, GELU
    — all of which have direct MIL op mappings.

    Input:  input_ids [B, T] int64  +  attention_mask [B, T] int64
    Output: logits    [B, num_classes] float32
    """

    def __init__(
        self,
        vocab_size:  int = 1000,
        hidden_size: int = 256,
        num_heads:   int = 8,
        num_layers:  int = 2,
        num_classes: int = 2,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = hidden_size // num_heads

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

        self.q = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.k = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.v = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.o = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])

        self.ff1 = nn.ModuleList([nn.Linear(hidden_size, hidden_size * 4) for _ in range(num_layers)])
        self.ff2 = nn.ModuleList([nn.Linear(hidden_size * 4, hidden_size)  for _ in range(num_layers)])

        self.ln1 = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])

        self.classifier = nn.Linear(hidden_size, num_classes)

    def _attn(self, x: torch.Tensor, i: int) -> torch.Tensor:
        B, T, H = x.shape
        nh, dh  = self.num_heads, self.head_dim
        Q = self.q[i](x).reshape(B, T, nh, dh).transpose(1, 2)
        K = self.k[i](x).reshape(B, T, nh, dh).transpose(1, 2)
        V = self.v[i](x).reshape(B, T, nh, dh).transpose(1, 2)
        w = torch.matmul(Q, K.transpose(-2, -1)) * (float(dh) ** -0.5)
        w = torch.softmax(w, dim=-1)
        out = torch.matmul(w, V).transpose(1, 2).reshape(B, T, H)
        return self.o[i](out)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embedding(input_ids)
        _ = attention_mask              # graph input preserved, not used in encoder
        for i in range(len(self.q)):
            x = self.ln1[i](x + self._attn(x, i))
            x = self.ln2[i](x + self.ff2[i](F.gelu(self.ff1[i](x))))
        return self.classifier(x.mean(dim=1))


# ── Seed reset ────────────────────────────────────────────────────────────────

def _reset_seed():
    import numpy as np
    torch.manual_seed(42)
    np.random.seed(42)


# ── Generators ────────────────────────────────────────────────────────────────

def _gen_B02(dirs, manifest, result):
    from generate_fixtures import _to_coreml_nn, ManifestEntry, COREML_AVAILABLE
    model_id = "B02"
    if not COREML_AVAILABLE:
        result.skip(model_id, "coremltools not available"); return
    try:
        _reset_seed()
        try:
            import torchvision.models as tvm
            model = tvm.mobilenet_v3_small(weights=None)
        except ImportError:
            model = _minimal_cnn(in_channels=3, num_classes=1000)
        model.eval()
        dummy = torch.zeros(1, 3, 224, 224)
        out = dirs.coreml_path(model_id, "mobilenetv3small_vision_nn", "mlmodel")
        if _to_coreml_nn(model, (dummy,), ["pixel_values"], out) is None:
            raise RuntimeError(f"conversion returned None for {model_id}")
        manifest.add(ManifestEntry(
            model_id="B02", tier="baseline", modality="vision", format="mlmodel",
            size_class="small", file=str(out.relative_to(dirs.root)),
            expected_domain="vision", expected_subtype="none",
            expected_execution="standard", expected_nms_inside=False,
            expected_failure=False, commands=["inspect", "baseline", "status"],
        ))
        manifest.stamp_sha256(model_id, out); result.ok(model_id)
    except Exception as exc:
        result.fail(model_id, exc)


def _gen_B03(dirs, manifest, result):
    from generate_fixtures import _to_coreml_mlprogram, ManifestEntry, COREML_AVAILABLE
    model_id = "B03"
    if not COREML_AVAILABLE:
        result.skip(model_id, "coremltools not available"); return
    try:
        _reset_seed()
        try:
            import torchvision.models as tvm
            model = tvm.mobilenet_v3_small(weights=None)
        except ImportError:
            model = _minimal_cnn(in_channels=3, num_classes=1000)
        model.eval()
        dummy = torch.zeros(1, 3, 224, 224)
        out = dirs.coreml_path(model_id, "mobilenetv3small_vision_mlprog", "mlpackage")
        if _to_coreml_mlprogram(model, (dummy,), ["pixel_values"], out) is None:
            raise RuntimeError(f"conversion returned None for {model_id}")
        manifest.add(ManifestEntry(
            model_id="B03", tier="baseline", modality="vision", format="mlpackage",
            size_class="small", file=str(out.relative_to(dirs.root)),
            expected_domain="vision", expected_subtype="none",
            expected_execution="standard", expected_nms_inside=False,
            expected_failure=False, commands=["inspect", "baseline", "status", "prune"],
        ))
        manifest.stamp_sha256(model_id, out); result.ok(model_id)
    except Exception as exc:
        result.fail(model_id, exc)


def _gen_B06(dirs, manifest, result):
    from generate_fixtures import _to_coreml_nn, ManifestEntry, COREML_AVAILABLE
    model_id = "B06"
    if not COREML_AVAILABLE:
        result.skip(model_id, "coremltools not available"); return
    try:
        _reset_seed()
        model = nn.Sequential(
            nn.Linear(32, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1),
        ).eval()
        dummy = torch.zeros(1, 32)
        out = dirs.coreml_path(model_id, "mlp_tabular_nn", "mlmodel")
        if _to_coreml_nn(model, (dummy,), ["features"], out) is None:
            raise RuntimeError(f"conversion returned None for {model_id}")
        manifest.add(ManifestEntry(
            model_id="B06", tier="baseline", modality="tabular", format="mlmodel",
            size_class="tiny", file=str(out.relative_to(dirs.root)),
            expected_domain="tabular", expected_subtype="none",
            expected_execution="standard", expected_failure=False,
            commands=["inspect", "baseline", "status", "rename"],
        ))
        manifest.stamp_sha256(model_id, out); result.ok(model_id)
    except Exception as exc:
        result.fail(model_id, exc)


def _gen_B07(dirs, manifest, result):
    from generate_fixtures import _to_coreml_mlprogram, ManifestEntry, COREML_AVAILABLE
    model_id = "B07"
    if not COREML_AVAILABLE:
        result.skip(model_id, "coremltools not available"); return
    try:
        _reset_seed()
        model = nn.Sequential(
            nn.Linear(32, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1),
        ).eval()
        dummy = torch.zeros(1, 32)
        out = dirs.coreml_path(model_id, "mlp_tabular_mlprog", "mlpackage")
        if _to_coreml_mlprogram(model, (dummy,), ["features"], out) is None:
            raise RuntimeError(f"conversion returned None for {model_id}")
        manifest.add(ManifestEntry(
            model_id="B07", tier="baseline", modality="tabular", format="mlpackage",
            size_class="tiny", file=str(out.relative_to(dirs.root)),
            expected_domain="tabular", expected_subtype="none",
            expected_execution="standard", expected_failure=False,
            commands=["inspect", "baseline", "status"],
        ))
        manifest.stamp_sha256(model_id, out); result.ok(model_id)
    except Exception as exc:
        result.fail(model_id, exc)


def _gen_B10(dirs, manifest, result):
    from generate_fixtures import _to_coreml_nn, ManifestEntry, COREML_AVAILABLE
    model_id = "B10"
    if not COREML_AVAILABLE:
        result.skip(model_id, "coremltools not available"); return
    try:
        _reset_seed()
        model = _spectrogram_cnn(num_classes=35).eval()
        dummy = torch.zeros(1, 1, 64, 101)
        out = dirs.coreml_path(model_id, "cnn_spectrogram_audio_nn", "mlmodel")
        if _to_coreml_nn(model, (dummy,), ["spectrogram"], out) is None:
            raise RuntimeError(f"conversion returned None for {model_id}")
        manifest.add(ManifestEntry(
            model_id="B10", tier="baseline", modality="audio", format="mlmodel",
            size_class="tiny", file=str(out.relative_to(dirs.root)),
            expected_domain="audio", expected_subtype="none",
            expected_execution="standard", expected_failure=False,
            commands=["inspect", "baseline", "budget"],
        ))
        manifest.stamp_sha256(model_id, out); result.ok(model_id)
    except Exception as exc:
        result.fail(model_id, exc)


def _gen_B11(dirs, manifest, result):
    from generate_fixtures import _to_coreml_mlprogram, ManifestEntry, COREML_AVAILABLE
    model_id = "B11"
    if not COREML_AVAILABLE:
        result.skip(model_id, "coremltools not available"); return
    try:
        _reset_seed()
        model = _spectrogram_cnn(num_classes=35).eval()
        dummy = torch.zeros(1, 1, 64, 101)
        out = dirs.coreml_path(model_id, "cnn_spectrogram_audio_mlprog", "mlpackage")
        if _to_coreml_mlprogram(model, (dummy,), ["spectrogram"], out) is None:
            raise RuntimeError(f"conversion returned None for {model_id}")
        manifest.add(ManifestEntry(
            model_id="B11", tier="baseline", modality="audio", format="mlpackage",
            size_class="tiny", file=str(out.relative_to(dirs.root)),
            expected_domain="audio", expected_subtype="none",
            expected_execution="standard", expected_failure=False,
            commands=["inspect", "baseline", "budget", "prune"],
        ))
        manifest.stamp_sha256(model_id, out); result.ok(model_id)
    except Exception as exc:
        result.fail(model_id, exc)


def _gen_B14(dirs, manifest, result):
    from generate_fixtures import _to_coreml_nn, ManifestEntry, COREML_AVAILABLE
    model_id = "B14"
    if not COREML_AVAILABLE:
        result.skip(model_id, "coremltools not available"); return
    try:
        _reset_seed()
        model = _bert_coreml(vocab_size=1000, hidden_size=256, num_heads=8,
                             num_layers=2, num_classes=2).eval()
        ids  = torch.zeros(1, 128, dtype=torch.int64)
        mask = torch.ones(1,  128, dtype=torch.int64)
        out  = dirs.coreml_path(model_id, "bert_style_nlp_nn", "mlmodel")
        if _to_coreml_nn(model, (ids, mask), ["input_ids", "attention_mask"], out) is None:
            raise RuntimeError(f"conversion returned None for {model_id}")
        manifest.add(ManifestEntry(
            model_id="B14", tier="baseline", modality="nlp", format="mlmodel",
            size_class="medium", file=str(out.relative_to(dirs.root)),
            expected_domain="nlp", expected_subtype="none",
            expected_execution="multi_input", expected_failure=False,
            commands=["inspect", "budget", "baseline"],
        ))
        manifest.stamp_sha256(model_id, out); result.ok(model_id)
    except Exception as exc:
        result.fail(model_id, exc)


def _gen_B15(dirs, manifest, result):
    from generate_fixtures import _to_coreml_mlprogram, ManifestEntry, COREML_AVAILABLE
    model_id = "B15"
    if not COREML_AVAILABLE:
        result.skip(model_id, "coremltools not available"); return
    try:
        _reset_seed()
        model = _bert_coreml(vocab_size=1000, hidden_size=256, num_heads=8,
                             num_layers=2, num_classes=2).eval()
        ids  = torch.zeros(1, 128, dtype=torch.int64)
        mask = torch.ones(1,  128, dtype=torch.int64)
        out  = dirs.coreml_path(model_id, "bert_style_nlp_mlprog", "mlpackage")
        if _to_coreml_mlprogram(model, (ids, mask), ["input_ids", "attention_mask"], out) is None:
            raise RuntimeError(f"conversion returned None for {model_id}")
        manifest.add(ManifestEntry(
            model_id="B15", tier="baseline", modality="nlp", format="mlpackage",
            size_class="medium", file=str(out.relative_to(dirs.root)),
            expected_domain="nlp", expected_subtype="none",
            expected_execution="multi_input", expected_failure=False,
            commands=["inspect", "budget", "baseline", "prune"],
        ))
        manifest.stamp_sha256(model_id, out); result.ok(model_id)
    except Exception as exc:
        result.fail(model_id, exc)


def register_coreml_baselines(generators: list) -> None:
    generators.extend([
        (_gen_B02, "B02"), (_gen_B03, "B03"),
        (_gen_B06, "B06"), (_gen_B07, "B07"),
        (_gen_B10, "B10"), (_gen_B11, "B11"),
        (_gen_B14, "B14"), (_gen_B15, "B15"),
    ])
    logger.info(
        "step3: registered 8 CoreML baseline generators "
        "(B02/B03 vision, B06/B07 tabular, B10/B11 audio, B14/B15 nlp) "
        "— pipeline: PyTorch → TorchScript → CoreML"
    )