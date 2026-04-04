"""
Step 6 — Time-series baseline models.

Models
------
T01  timeseries  ONNX             2-layer LSTM forecaster, [1,96,1] → [1,24,1]
T02  timeseries  CoreML .mlmodel  Same LSTM → NeuralNetwork
T03  timeseries  CoreML .mlpackage Same LSTM → MLProgram
T04  timeseries  TFLite           TCN (dilated Conv1d), [1,1,96] → [1,1,24]

Architecture
------------
LSTM forecaster (T01–T03)
  Input:  timeseries  [1, 96, 1]  float32   — window of 96 timesteps, 1 feature
  Output: forecast    [1, 24, 1]  float32   — 24-step ahead prediction
  Ops:    LSTM, Linear

  nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
  Takes last 24 steps of encoder output as the forecast.

TCN forecaster (T04)
  Input:  timeseries  [1, 1, 96]  float32   — (batch, channels, time) — PyTorch Conv1d
  Output: forecast    [1, 1, 24]  float32
  Ops:    Conv1d (dilations 1, 2, 4), ReLU

Detection target for all T0x
-----------------------------
domain=AUDIO, subtype=TIMESERIES, execution=STANDARD
LSTM/GRU ops + temporal rank-3 float input triggers subtype=TIMESERIES.
Domain maps to AUDIO via LSTM heuristic (LSTM is primary audio signal).
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("mlbuild.fixtures.step6")


# ── Shared architectures ──────────────────────────────────────────────────────

class _lstm_forecaster(nn.Module):
    """
    2-layer LSTM sequence forecaster.

    Input:  timeseries  [B, 96, 1]  float32  (batch_first=True)
    Output: forecast    [B, 24, 1]  float32

    Uses the last pred_len steps of the encoder output as the forecast.
    Named input "timeseries" triggers the Tier 2 time-series name heuristic.
    LSTM op in the graph triggers Tier 1 LSTM/RNN scoring.
    """

    def __init__(
        self,
        input_size:  int = 1,
        hidden_size: int = 64,
        num_layers:  int = 2,
        pred_len:    int = 24,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = 0.0,   # no dropout — deterministic
        )
        self.head = nn.Linear(hidden_size, input_size)

    def forward(self, timeseries: torch.Tensor) -> torch.Tensor:
        # timeseries: [B, seq_len, input_size]
        enc, _ = self.lstm(timeseries)          # [B, seq_len, hidden_size]
        out     = enc[:, -self.pred_len:, :]    # [B, pred_len, hidden_size]
        return self.head(out)                    # [B, pred_len, input_size]


class _tcn_forecaster(nn.Module):
    """
    Temporal Convolutional Network (dilated) for time-series forecasting.

    Input:  timeseries  [B, 1, 96]  float32  (channels-first Conv1d format)
    Output: forecast    [B, 1, 24]  float32

    Three causal conv layers with doubling dilation (1, 2, 4) followed by
    a 1×1 projection head. Takes the last pred_len steps as the forecast.
    Dilated Conv1d ops serve as the Tier 2 TCN/time-series signal.
    """

    def __init__(self, channels: int = 32, pred_len: int = 24):
        super().__init__()
        self.pred_len = pred_len
        # Dilated causal convolutions — padding preserves length
        self.conv1 = nn.Conv1d(1,        channels, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, dilation=2, padding=2)
        self.conv3 = nn.Conv1d(channels, channels, kernel_size=3, dilation=4, padding=4)
        self.head  = nn.Conv1d(channels, 1,        kernel_size=1)

    def forward(self, timeseries: torch.Tensor) -> torch.Tensor:
        # timeseries: [B, 1, seq_len]
        x = F.relu(self.conv1(timeseries))   # [B, C, 96]
        x = F.relu(self.conv2(x))            # [B, C, 96]
        x = F.relu(self.conv3(x))            # [B, C, 96]
        x = self.head(x)                     # [B, 1, 96]
        return x[:, :, -self.pred_len:]      # [B, 1, 24]


def _reset_seed() -> None:
    import numpy as np
    torch.manual_seed(42)
    np.random.seed(42)


# ── T01 — ONNX (LSTM) ────────────────────────────────────────────────────────

def _gen_T01(dirs, manifest, result) -> None:
    """
    2-layer LSTM forecaster → ONNX.

    Input:  timeseries  [1, 96, 1]  float32
    Output: forecast    [1, 24, 1]  float32

    The named input "timeseries" and LSTM ops in the graph are the two primary
    detection signals for subtype=TIMESERIES scoring.

    Also serves as the ONNX source for T04 (TFLite via onnx2tf, if available).
    """
    from generate_fixtures import ManifestEntry

    model_id = "T01"

    try:
        _reset_seed()
        model = _lstm_forecaster(input_size=1, hidden_size=64,
                                 num_layers=2, pred_len=24).eval()
        dummy = torch.zeros(1, 96, 1)

        path = dirs.onnx_path(model_id, "lstm_timeseries")
        path.parent.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy,),
                str(path),
                input_names    = ["timeseries"],
                output_names   = ["forecast"],
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
            modality            = "timeseries",
            format              = "onnx",
            size_class          = "tiny",
            file                = str(path.relative_to(dirs.root)),
            expected_domain     = "audio",
            expected_subtype    = "timeseries",
            expected_execution  = "standard",
            expected_nms_inside = False,
            expected_failure    = False,
            commands            = ["inspect", "baseline", "budget"],
        ))
        manifest.stamp_sha256(model_id, path)
        result.ok(model_id)

    except Exception as exc:
        result.fail(model_id, exc)


# ── T02 — CoreML NeuralNetwork ────────────────────────────────────────────────

def _gen_T02(dirs, manifest, result) -> None:
    """
    LSTM forecaster → CoreML NeuralNetwork.

    CoreML handles LSTM via TorchScript → MIL LSTM op mapping.
    """
    from generate_fixtures import _to_coreml_nn, ManifestEntry, COREML_AVAILABLE

    model_id = "T02"

    if not COREML_AVAILABLE:
        result.skip(model_id, "coremltools not available")
        return

    try:
        _reset_seed()
        model = _lstm_forecaster(input_size=1, hidden_size=64,
                                 num_layers=2, pred_len=24).eval()
        dummy = torch.zeros(1, 96, 1)
        out   = dirs.coreml_path(model_id, "lstm_timeseries_nn", "mlmodel")

        converted = _to_coreml_nn(model, (dummy,), ["timeseries"], out)
        if converted is None:
            raise RuntimeError(f"CoreML NeuralNetwork conversion returned None for {model_id}")

        manifest.add(ManifestEntry(
            model_id            = model_id,
            tier                = "baseline",
            modality            = "timeseries",
            format              = "mlmodel",
            size_class          = "tiny",
            file                = str(out.relative_to(dirs.root)),
            expected_domain     = "audio",
            expected_subtype    = "timeseries",
            expected_execution  = "standard",
            expected_nms_inside = False,
            expected_failure    = False,
            commands            = ["inspect", "baseline"],
        ))
        manifest.stamp_sha256(model_id, out)
        result.ok(model_id)

    except Exception as exc:
        result.fail(model_id, exc)


# ── T03 — CoreML MLProgram ────────────────────────────────────────────────────

def _gen_T03(dirs, manifest, result) -> None:
    """
    LSTM forecaster → CoreML MLProgram.
    """
    from generate_fixtures import _to_coreml_mlprogram, ManifestEntry, COREML_AVAILABLE

    model_id = "T03"

    if not COREML_AVAILABLE:
        result.skip(model_id, "coremltools not available")
        return

    try:
        _reset_seed()
        model = _lstm_forecaster(input_size=1, hidden_size=64,
                                 num_layers=2, pred_len=24).eval()
        dummy = torch.zeros(1, 96, 1)
        out   = dirs.coreml_path(model_id, "lstm_timeseries_mlprog", "mlpackage")

        converted = _to_coreml_mlprogram(model, (dummy,), ["timeseries"], out)
        if converted is None:
            raise RuntimeError(f"CoreML MLProgram conversion returned None for {model_id}")

        manifest.add(ManifestEntry(
            model_id            = model_id,
            tier                = "baseline",
            modality            = "timeseries",
            format              = "mlpackage",
            size_class          = "tiny",
            file                = str(out.relative_to(dirs.root)),
            expected_domain     = "audio",
            expected_subtype    = "timeseries",
            expected_execution  = "standard",
            expected_nms_inside = False,
            expected_failure    = False,
            commands            = ["inspect", "baseline", "prune"],
        ))
        manifest.stamp_sha256(model_id, out)
        result.ok(model_id)

    except Exception as exc:
        result.fail(model_id, exc)


# ── T04 — TFLite (TCN) ───────────────────────────────────────────────────────

def _gen_T04(dirs, manifest, result) -> None:
    """
    TCN (dilated Conv1d) forecaster → TFLite.

    Primary: export TCN to ONNX, then onnx2tf → TFLite.
    Fallback: TF Keras Conv1D equivalent (avoids onnx2tf entirely).

    TCN is a different architecture from the LSTM (T01–T03) — dilated
    causal convolutions instead of recurrent layers. Provides coverage for
    the Conv1d-based timeseries detection path.
    """
    from generate_fixtures import (
        _to_tflite_from_onnx, _to_tflite_from_tf,
        ManifestEntry, ONNX2TF_AVAILABLE, TFLITE_AVAILABLE,
    )

    model_id = "T04"

    if not TFLITE_AVAILABLE:
        result.skip(model_id, "tensorflow not available")
        return

    try:
        out = dirs.tflite_path(model_id, "tcn_timeseries")
        converted = None

        # Primary: export TCN to ONNX first, then onnx2tf
        if ONNX2TF_AVAILABLE:
            import tempfile
            from pathlib import Path

            _reset_seed()
            tcn_model = _tcn_forecaster(channels=32, pred_len=24).eval()
            tcn_dummy = torch.zeros(1, 1, 96)

            with tempfile.TemporaryDirectory(prefix="mlbuild_t04_") as tmp:
                tcn_onnx = Path(tmp) / "T04_tcn_timeseries.onnx"
                with torch.no_grad():
                    torch.onnx.export(
                        tcn_model,
                        (tcn_dummy,),
                        str(tcn_onnx),
                        input_names   = ["timeseries"],
                        output_names  = ["forecast"],
                        dynamic_axes  = {},
                        opset_version = 18,
                        do_constant_folding = True,
                    )
                import onnx as _onnx
                _onnx.checker.check_model(str(tcn_onnx))
                converted = _to_tflite_from_onnx(tcn_onnx, out)

        # Fallback: TF Keras Conv1D equivalent (channels-last, [B, 96, 1] input)
        if converted is None:
            logger.warning("T04: onnx2tf path unavailable or failed — using TF Keras fallback")
            import tensorflow as tf

            inp = tf.keras.Input(shape=(96, 1), name="timeseries")
            x   = tf.keras.layers.Conv1D(32, 3, dilation_rate=1, padding="causal", activation="relu")(inp)
            x   = tf.keras.layers.Conv1D(32, 3, dilation_rate=2, padding="causal", activation="relu")(x)
            x   = tf.keras.layers.Conv1D(32, 3, dilation_rate=4, padding="causal", activation="relu")(x)
            out_seq = tf.keras.layers.Conv1D(1,  1, padding="same")(x)       # [B, 96, 1]
            forecast = out_seq[:, -24:, :]                                    # [B, 24, 1]
            tf_model  = tf.keras.Model(inputs=inp, outputs=forecast, name="forecast")
            converted = _to_tflite_from_tf(tf_model, out)

        if converted is None:
            raise RuntimeError(f"TFLite conversion returned None for {model_id}")

        manifest.add(ManifestEntry(
            model_id            = model_id,
            tier                = "baseline",
            modality            = "timeseries",
            format              = "tflite",
            size_class          = "tiny",
            file                = str(out.relative_to(dirs.root)),
            expected_domain     = "audio",
            expected_subtype    = "timeseries",
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

def register_timeseries_baselines(generators: list) -> None:
    """
    Register Step 6 generators.

    T01–T03: LSTM-based forecaster (ONNX + CoreML NeuralNetwork + MLProgram).
    T04: TCN-based forecaster (TFLite via onnx2tf or TF Keras fallback).

    Expected profile for all T0x:
      domain=AUDIO, subtype=TIMESERIES, execution=STANDARD
    """
    generators.extend([
        (_gen_T01, "T01"),
        (_gen_T02, "T02"),
        (_gen_T03, "T03"),
        (_gen_T04, "T04"),
    ])
    logger.info(
        "step6: registered 4 time-series baseline generators "
        "(T01 ONNX, T02 CoreML .mlmodel, T03 CoreML .mlpackage, T04 TFLite) "
        "— LSTM [1,96,1]→[1,24,1] + TCN dilated Conv1d"
    )