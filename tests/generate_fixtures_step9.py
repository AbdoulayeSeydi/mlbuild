"""
Step 9 — Generative baseline models.

Models
------
G01  generative  ONNX              GPT-2-style 6-layer decoder, [1,128] → [1,128,50257]
G02  generative  CoreML .mlpackage Same architecture → MLProgram FP16

Architecture
------------
GPT-2-style transformer decoder (single-pass, no KV-cache).
  Input:  input_ids  [1, 128]          int64   — token sequence
  Output: logits     [1, 128, 50257]   float32 — per-token vocabulary logits

  6 layers, hidden_size=128, num_heads=4, vocab_size=50257 (GPT-2 standard)
  Causal self-attention via explicit matmul (no nn.TransformerDecoder —
  that fuses to an op with no CoreML MIL mapping, same issue as BERT).

  vocab_size=50257 is kept to match GPT-2's real vocab for detection signal
  validity, but hidden_size=128 keeps the model ~30MB (not 548MB).

CoreML note
-----------
G02 uses _gpt2_coreml (primitive ops) not nn.TransformerDecoder.
Same design pattern as _bert_coreml in step3: Q/K/V as explicit Linear +
matmul attention + GELU FFN + LayerNorm — all ops with MIL mappings.
Causal masking is applied as an additive mask tensor (lower-triangular).

Detection target
----------------
domain=NLP, subtype=GENERATIVE_STATEFUL, execution=STANDARD
Large vocab output + decoder-only attention pattern → GENERATIVE subtype.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

logger = logging.getLogger("mlbuild.fixtures.step9")


# ── Shared architecture ───────────────────────────────────────────────────────

class _gpt2_coreml(nn.Module):
    """
    GPT-2-style decoder built from CoreML-compatible primitive ops.

    Avoids nn.TransformerDecoder which fuses to _transformer_decoder_layer_fwd
    — an op with no CoreML MIL mapping.

    Uses explicit primitive ops throughout:
      - nn.Embedding (gather)
      - nn.Linear for Q/K/V/O projections and FFN
      - torch.matmul for attention scores
      - torch.softmax
      - nn.LayerNorm
      - F.gelu

    Input:  input_ids  [B, T]          int64
    Output: logits     [B, T, vocab]   float32
    """

    def __init__(
        self,
        vocab_size:  int = 50257,
        hidden_size: int = 128,
        num_heads:   int = 4,
        num_layers:  int = 6,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads   = num_heads
        self.head_dim    = hidden_size // num_heads
        self.num_layers  = num_layers

        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb   = nn.Embedding(max_seq_len, hidden_size)

        # Per-layer attention projections
        self.q = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.k = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.v = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.o = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])

        # Per-layer FFN
        self.ff1 = nn.ModuleList([nn.Linear(hidden_size, hidden_size * 4) for _ in range(num_layers)])
        self.ff2 = nn.ModuleList([nn.Linear(hidden_size * 4, hidden_size)  for _ in range(num_layers)])

        # Per-layer LayerNorm (×2 per layer)
        self.ln1 = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])

        self.ln_f    = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def _causal_attn(self, x: torch.Tensor, i: int) -> torch.Tensor:
        B, T, H = x.shape
        nh, dh  = self.num_heads, self.head_dim

        Q = self.q[i](x).reshape(B, T, nh, dh).transpose(1, 2)   # [B,nh,T,dh]
        K = self.k[i](x).reshape(B, T, nh, dh).transpose(1, 2)
        V = self.v[i](x).reshape(B, T, nh, dh).transpose(1, 2)

        scale  = float(dh) ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale      # [B,nh,T,T]

        # Causal mask: upper-triangular filled with -inf
        mask   = torch.triu(torch.ones(T, T, dtype=x.dtype, device=x.device), diagonal=1) * -1e9
        scores = scores + mask

        attn = torch.softmax(scores, dim=-1)
        out  = torch.matmul(attn, V)                               # [B,nh,T,dh]
        out  = out.transpose(1, 2).reshape(B, T, H)
        return self.o[i](out)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device).unsqueeze(0)  # [1, T]

        x = self.token_emb(input_ids) + self.pos_emb(pos)  # [B, T, H]

        for i in range(self.num_layers):
            x = self.ln1[i](x + self._causal_attn(x, i))
            x = self.ln2[i](x + self.ff2[i](F.gelu(self.ff1[i](x))))

        x = self.ln_f(x)
        return self.lm_head(x)   # [B, T, vocab_size]


def _reset_seed() -> None:
    import numpy as np
    torch.manual_seed(42)
    np.random.seed(42)


# ── G01 — ONNX ───────────────────────────────────────────────────────────────

def _gen_G01(dirs, manifest, result) -> None:
    """
    GPT-2-style decoder → ONNX.

    Input:  input_ids  [1, 128]          int64
    Output: logits     [1, 128, 50257]   float32

    vocab_size=50257 matches GPT-2 — ensures the detection heuristic for
    large-vocab outputs fires correctly. hidden_size=128 with 6 layers keeps
    the model manageable (~28MB with external data).

    The output shape [B, T, 50257] is the primary GENERATIVE detection signal.
    """
    from generate_fixtures import ManifestEntry

    model_id = "G01"

    try:
        _reset_seed()
        model = _gpt2_coreml(
            vocab_size  = 50257,
            hidden_size = 128,
            num_heads   = 4,
            num_layers  = 6,
            max_seq_len = 128,
        ).eval()

        input_ids = torch.zeros(1, 128, dtype=torch.int64)
        path      = dirs.onnx_path(model_id, "gpt2_style_generative")
        path.parent.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            torch.onnx.export(
                model,
                (input_ids,),
                str(path),
                input_names    = ["input_ids"],
                output_names   = ["logits"],
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
            modality            = "generative",
            format              = "onnx",
            size_class          = "medium",
            file                = str(path.relative_to(dirs.root)),
            expected_domain     = "nlp",
            expected_subtype    = "generative_stateful",
            expected_execution  = "standard",
            expected_nms_inside = False,
            expected_failure    = False,
            commands            = ["inspect", "budget"],
        ))
        manifest.stamp_sha256(model_id, path)
        result.ok(model_id)

    except Exception as exc:
        result.fail(model_id, exc)


# ── G02 — CoreML MLProgram FP16 ──────────────────────────────────────────────

def _gen_G02(dirs, manifest, result) -> None:
    """
    GPT-2-style decoder → CoreML MLProgram with FP16 weights.

    Same architecture as G01 but converted to .mlpackage format with
    linear weight quantization (FP16). MLProgram format is required for
    FP16 weights — NeuralNetwork (.mlmodel) does not support it.

    The _gpt2_coreml architecture uses only primitive ops (no fused
    TransformerDecoder), so CoreML MIL conversion succeeds.
    """
    from generate_fixtures import _to_coreml_mlprogram, ManifestEntry, COREML_AVAILABLE

    model_id = "G02"

    if not COREML_AVAILABLE:
        result.skip(model_id, "coremltools not available")
        return

    try:
        _reset_seed()
        model = _gpt2_coreml(
            vocab_size  = 50257,
            hidden_size = 128,
            num_heads   = 4,
            num_layers  = 6,
            max_seq_len = 128,
        ).eval()

        input_ids = torch.zeros(1, 128, dtype=torch.int64)
        out       = dirs.coreml_path(model_id, "gpt2_style_generative_mlprog", "mlpackage")

        # fp16=True applies linear weight quantization after conversion
        converted = _to_coreml_mlprogram(
            model,
            (input_ids,),
            ["input_ids"],
            out,
            fp16 = True,
        )
        if converted is None:
            raise RuntimeError(f"CoreML MLProgram conversion returned None for {model_id}")

        manifest.add(ManifestEntry(
            model_id            = model_id,
            tier                = "baseline",
            modality            = "generative",
            format              = "mlpackage",
            size_class          = "medium",
            file                = str(out.relative_to(dirs.root)),
            expected_domain     = "nlp",
            expected_subtype    = "generative_stateful",
            expected_execution  = "standard",
            expected_nms_inside = False,
            expected_failure    = False,
            commands            = ["inspect", "budget", "prune"],
        ))
        manifest.stamp_sha256(model_id, out)
        result.ok(model_id)

    except Exception as exc:
        result.fail(model_id, exc)


# ── Registration ──────────────────────────────────────────────────────────────

def register_generative_baselines(generators: list) -> None:
    """
    Register Step 9 generators.

    G01: GPT-2-style ONNX (6 layers, hidden=128, vocab=50257, ctx=128).
    G02: Same → CoreML MLProgram with FP16 weight quantization.

    Both use _gpt2_coreml (primitive ops only) so CoreML conversion works.

    Expected profile:
      domain=NLP, subtype=GENERATIVE_STATEFUL, execution=STANDARD
    """
    generators.extend([
        (_gen_G01, "G01"),
        (_gen_G02, "G02"),
    ])
    logger.info(
        "step9: registered 2 generative baseline generators "
        "(G01 ONNX, G02 CoreML .mlpackage FP16) "
        "— GPT-2 style 6L hidden=128 vocab=50257 ctx=128"
    )