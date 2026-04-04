"""
Step 8 — Recommendation baseline models.

Models
------
R01  rec  ONNX              NeuralMF: user_id[1] + item_id[1] → score[1,1]
R02  rec  CoreML .mlpackage Same NeuralMF → MLProgram

Architecture
------------
Neural Matrix Factorization (NeuralMF)
  Inputs:  user_id   [1]     int64  — scalar user index
           item_id   [1]     int64  — scalar item index
  Output:  score     [1, 1]  float32

  Two Gather (embedding lookup) ops feed separate MLP towers that
  are concatenated and passed through a scoring head.

  Embedding sizes: user_vocab=1000, item_vocab=5000, embed_dim=32
  MLP: concat(64) → Linear(128) → ReLU → Linear(64) → ReLU → Linear(1)

Detection target
----------------
domain=TABULAR, subtype=RECOMMENDATION, execution=MULTI_INPUT
Two scalar int64 inputs + Gather ops → RECOMMENDATION subtype.
MULTI_INPUT execution: 2 inputs, same domain.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn

logger = logging.getLogger("mlbuild.fixtures.step8")


# ── Architecture ──────────────────────────────────────────────────────────────

class _neural_mf(nn.Module):
    """
    Neural Matrix Factorization for collaborative filtering.

    Two embedding lookups (Gather ops) feed an MLP scoring head.
    Scalar int64 inputs trigger the Tier 2 recommendation detection heuristic
    (two int64 scalar inputs → user/item ID pattern).

    Inputs:  user_id  [B]      int64
             item_id  [B]      int64
    Output:  score    [B, 1]   float32
    """

    def __init__(
        self,
        user_vocab: int = 1000,
        item_vocab: int = 5000,
        embed_dim:  int = 32,
    ):
        super().__init__()
        self.user_emb = nn.Embedding(user_vocab, embed_dim)
        self.item_emb = nn.Embedding(item_vocab, embed_dim)

        # MLP scoring head
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 128), nn.ReLU(),
            nn.Linear(128,          64),  nn.ReLU(),
            nn.Linear(64,           1),
        )

    def forward(
        self,
        user_id: torch.Tensor,  # [B] int64
        item_id: torch.Tensor,  # [B] int64
    ) -> torch.Tensor:          # [B, 1]
        u = self.user_emb(user_id)          # [B, embed_dim]
        v = self.item_emb(item_id)          # [B, embed_dim]
        x = torch.cat([u, v], dim=-1)       # [B, embed_dim * 2]
        return self.mlp(x)                  # [B, 1]


def _reset_seed() -> None:
    import numpy as np
    torch.manual_seed(42)
    np.random.seed(42)


# ── R01 — ONNX ───────────────────────────────────────────────────────────────

def _gen_R01(dirs, manifest, result) -> None:
    """
    NeuralMF → ONNX.

    Two scalar int64 inputs named "user_id" and "item_id".
    The Gather ops and named inputs trigger Tier 2 recommendation detection.
    """
    from generate_fixtures import ManifestEntry

    model_id = "R01"

    try:
        _reset_seed()
        model   = _neural_mf(user_vocab=1000, item_vocab=5000, embed_dim=32).eval()
        user_id = torch.zeros(1, dtype=torch.int64)
        item_id = torch.zeros(1, dtype=torch.int64)

        path = dirs.onnx_path(model_id, "neural_mf_rec")
        path.parent.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            torch.onnx.export(
                model,
                (user_id, item_id),
                str(path),
                input_names    = ["user_id", "item_id"],
                output_names   = ["score"],
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
            modality            = "rec",
            format              = "onnx",
            size_class          = "small",
            file                = str(path.relative_to(dirs.root)),
            expected_domain     = "tabular",
            expected_subtype    = "recommendation",
            expected_execution  = "multi_input",
            expected_nms_inside = False,
            expected_failure    = False,
            commands            = ["inspect", "baseline"],
        ))
        manifest.stamp_sha256(model_id, path)
        result.ok(model_id)

    except Exception as exc:
        result.fail(model_id, exc)


# ── R02 — CoreML MLProgram ────────────────────────────────────────────────────

def _gen_R02(dirs, manifest, result) -> None:
    """
    NeuralMF → CoreML MLProgram.

    Both user_id and item_id are int64 scalar inputs.
    coremltools handles int64 Gather inputs via ct.TensorType with dtype=int.
    """
    from generate_fixtures import _to_coreml_mlprogram, ManifestEntry, COREML_AVAILABLE
    import coremltools as ct

    model_id = "R02"

    if not COREML_AVAILABLE:
        result.skip(model_id, "coremltools not available")
        return

    try:
        _reset_seed()
        model   = _neural_mf(user_vocab=1000, item_vocab=5000, embed_dim=32).eval()
        user_id = torch.zeros(1, dtype=torch.int64)
        item_id = torch.zeros(1, dtype=torch.int64)
        out     = dirs.coreml_path(model_id, "neural_mf_rec_mlprog", "mlpackage")

        converted = _to_coreml_mlprogram(
            model,
            (user_id, item_id),
            ["user_id", "item_id"],
            out,
        )
        if converted is None:
            raise RuntimeError(f"CoreML MLProgram conversion returned None for {model_id}")

        manifest.add(ManifestEntry(
            model_id            = model_id,
            tier                = "baseline",
            modality            = "rec",
            format              = "mlpackage",
            size_class          = "small",
            file                = str(out.relative_to(dirs.root)),
            expected_domain     = "tabular",
            expected_subtype    = "recommendation",
            expected_execution  = "multi_input",
            expected_nms_inside = False,
            expected_failure    = False,
            commands            = ["inspect", "baseline", "prune"],
        ))
        manifest.stamp_sha256(model_id, out)
        result.ok(model_id)

    except Exception as exc:
        result.fail(model_id, exc)


# ── Registration ──────────────────────────────────────────────────────────────

def register_rec_baselines(generators: list) -> None:
    """
    Register Step 8 generators.

    R01: NeuralMF ONNX (user_id + item_id scalar int64 → score float32).
    R02: Same NeuralMF → CoreML MLProgram.

    Expected profile:
      domain=TABULAR, subtype=RECOMMENDATION, execution=MULTI_INPUT
    """
    generators.extend([
        (_gen_R01, "R01"),
        (_gen_R02, "R02"),
    ])
    logger.info(
        "step8: registered 2 recommendation baseline generators "
        "(R01 ONNX, R02 CoreML .mlpackage) "
        "— NeuralMF user_id+item_id Gather-based embedding"
    )