"""
Step 11 — Expected-failure ONNX sources (5 models).

These models generate ONNX files with deliberate issues.
No conversion is attempted at generation time — failures happen
at mlbuild build time during unit/integration tests.

S09   nlp         dynamic sequence → CoreML FAIL (dynamic_sequence_unsupported)
S10   vision      custom StochasticDepth op → TFLite FAIL (unsupported_op)
SD2   detection   baked NMS → TFLite FAIL (nms_op_unsupported)
SM2   multimodal  string-typed input → CoreML FAIL (string_input_unsupported)
SG2   generative  large model ONNX → CoreML FAIL (model_too_large)
"""

from __future__ import annotations
import logging
import torch
import torch.nn as nn

logger = logging.getLogger("mlbuild.fixtures.step11")


def _reset_seed():
    import numpy as np
    torch.manual_seed(42)
    np.random.seed(42)


# ─── S09 NLP: dynamic sequence → CoreML expected failure ─────────────────────
def _gen_S09(dirs, manifest, result):
    """
    BERT-style model with dynamic sequence axis.
    CoreML cannot convert models with dynamic sequence length.
    Expected failure: coremltools raises when tracing with dynamic input.
    """
    from generate_fixtures import ManifestEntry
    from generate_fixtures_step3 import _bert_coreml

    model_id = "S09"
    try:
        _reset_seed()
        model = _bert_coreml(vocab_size=1000, hidden_size=128,
                             num_heads=4, num_layers=2, num_classes=2).eval()
        ids  = torch.zeros(1, 64, dtype=torch.int64)
        mask = torch.ones(1, 64, dtype=torch.int64)
        path = dirs.onnx_path(model_id, "bert_dynamic_seq_fail")
        path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            torch.onnx.export(model, (ids, mask), str(path),
                input_names=["input_ids", "attention_mask"], output_names=["logits"],
                # Dynamic sequence axis — CoreML TorchScript tracing cannot
                # handle variable-length inputs without ct.RangeDim spec
                dynamic_axes={"input_ids": {1: "seq_len"},
                               "attention_mask": {1: "seq_len"}},
                opset_version=18, do_constant_folding=True)
        import onnx as _o; _o.checker.check_model(str(path))
        logger.info("exported ONNX  %s  [expected-failure source]", path.name)
        manifest.add(ManifestEntry(
            model_id="S09", tier="stress", modality="nlp", format="onnx",
            size_class="tiny", file=str(path.relative_to(dirs.root)),
            expected_domain="nlp", expected_subtype="none",
            expected_execution="multi_input",
            expected_failure=True, failure_target="coreml",
            failure_reason="dynamic_sequence_unsupported",
            commands=["inspect"],
        ))
        manifest.stamp_sha256(model_id, path); result.ok(model_id)
    except Exception as exc: result.fail(model_id, exc)


# ─── S10 Vision: custom StochasticDepth op → TFLite expected failure ─────────
def _gen_S10(dirs, manifest, result):
    """
    ResNet-18 style with a custom 'StochasticDepth' node injected via
    onnx.helper. TFLite conversion should fail on the unrecognized op.

    The custom op is inserted between two Conv layers so it's in the middle
    of the graph and clearly exercised by any op-checking conversion pass.
    """
    from generate_fixtures import ManifestEntry
    import onnx
    import onnx.helper as oh
    import numpy as np

    model_id = "S10"
    try:
        _reset_seed()

        # Build graph manually with onnx.helper to inject custom op
        X = oh.make_tensor_value_info("pixel_values", onnx.TensorProto.FLOAT, [1, 3, 224, 224])
        Y = oh.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, [1, 1000])

        # Conv weights
        w1 = np.random.randn(32, 3, 3, 3).astype(np.float32)
        w2 = np.random.randn(64, 32, 3, 3).astype(np.float32)
        w3 = np.random.randn(1000, 64).astype(np.float32)

        init_w1 = oh.make_tensor("w1", onnx.TensorProto.FLOAT, w1.shape, w1.flatten())
        init_w2 = oh.make_tensor("w2", onnx.TensorProto.FLOAT, w2.shape, w2.flatten())
        init_w3 = oh.make_tensor("w3", onnx.TensorProto.FLOAT, w3.shape, w3.flatten())

        nodes = [
            oh.make_node("Conv",  ["pixel_values", "w1"], ["conv1_out"],
                         strides=[2,2], pads=[1,1,1,1], kernel_shape=[3,3]),
            oh.make_node("Relu",  ["conv1_out"],          ["relu1_out"]),
            # ← custom op: TFLite has no mapping for this
            oh.make_node("StochasticDepth", ["relu1_out"], ["sd_out"],
                         domain="custom.mlbuild.test", survival_prob=0.8),
            oh.make_node("Conv",  ["sd_out", "w2"],       ["conv2_out"],
                         strides=[2,2], pads=[1,1,1,1], kernel_shape=[3,3]),
            oh.make_node("Relu",  ["conv2_out"],           ["relu2_out"]),
            oh.make_node("GlobalAveragePool", ["relu2_out"], ["gap_out"]),
            oh.make_node("Flatten", ["gap_out"],            ["flat_out"]),
            oh.make_node("Gemm",  ["flat_out", "w3"],       ["logits"]),
        ]

        graph = oh.make_graph(nodes, "s10_stochastic_depth",
                              [X], [Y], initializer=[init_w1, init_w2, init_w3])
        model = oh.make_model(graph, opset_imports=[
            oh.make_opsetid("", 18),
            oh.make_opsetid("custom.mlbuild.test", 1),
        ])
        model.ir_version = 8

        path = dirs.onnx_path(model_id, "resnet_custom_op_fail")
        path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(path))
        logger.info("exported ONNX  %s  [expected-failure source — custom StochasticDepth op]",
                    path.name)

        manifest.add(ManifestEntry(
            model_id="S10", tier="stress", modality="vision", format="onnx",
            size_class="tiny", file=str(path.relative_to(dirs.root)),
            expected_domain="vision", expected_subtype="none",
            expected_execution="standard",
            expected_failure=True, failure_target="tflite",
            failure_reason="unsupported_custom_op",
            commands=["inspect"],
        ))
        manifest.stamp_sha256(model_id, path); result.ok(model_id)
    except Exception as exc: result.fail(model_id, exc)


# ─── SD2 Detection: baked NMS → TFLite expected failure ──────────────────────
def _gen_SD2(dirs, manifest, result):
    """
    Detection model with NonMaxSuppression baked into the ONNX graph.
    TFLite conversion fails because TFLite's NMS op expects a specific
    SSD-style graph pattern, not a raw ONNX NonMaxSuppression node.
    expected_nms_inside=True since NMS is baked.
    """
    from generate_fixtures import ManifestEntry
    from generate_fixtures_step10 import _gen_S01   # import only for context

    import onnx
    import onnx.helper as oh
    import numpy as np

    model_id = "SD2"
    try:
        _reset_seed()

        # Minimal graph: boxes + scores → NMS → selected_indices
        boxes  = oh.make_tensor_value_info("boxes",  onnx.TensorProto.FLOAT, [1, 100, 4])
        scores = oh.make_tensor_value_info("scores", onnx.TensorProto.FLOAT, [1, 80, 100])
        out    = oh.make_tensor_value_info("selected_detections", onnx.TensorProto.INT64, [None, 3])

        max_out    = oh.make_tensor("max_out",    onnx.TensorProto.INT64, [], [10])
        iou_thresh = oh.make_tensor("iou_thresh", onnx.TensorProto.FLOAT, [], [0.45])
        score_thresh = oh.make_tensor("score_thresh", onnx.TensorProto.FLOAT, [], [0.25])

        nms_node = oh.make_node(
            "NonMaxSuppression",
            inputs=["boxes", "scores", "max_out", "iou_thresh", "score_thresh"],
            outputs=["selected_detections"],
            center_point_box=0,
        )

        graph = oh.make_graph(
            [nms_node], "sd2_baked_nms",
            [boxes, scores], [out],
            initializer=[max_out, iou_thresh, score_thresh],
        )
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 11)])
        model.ir_version = 7

        path = dirs.onnx_path(model_id, "yolov8_baked_nms_fail")
        path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(path))
        logger.info("exported ONNX  %s  [expected-failure source — baked NMS]", path.name)

        manifest.add(ManifestEntry(
            model_id="SD2", tier="stress", modality="detection", format="onnx",
            size_class="tiny", file=str(path.relative_to(dirs.root)),
            expected_domain="vision", expected_subtype="detection",
            expected_execution="standard", expected_nms_inside=True,
            expected_failure=True, failure_target="tflite",
            failure_reason="nms_op_unsupported",
            commands=["inspect"],
        ))
        manifest.stamp_sha256(model_id, path); result.ok(model_id)
    except Exception as exc: result.fail(model_id, exc)


# ─── SM2 Multimodal: string input → CoreML expected failure ──────────────────
def _gen_SM2(dirs, manifest, result):
    """
    CLIP-style model with a STRING-typed input for text.
    CoreML has no mapping for string tensor inputs — conversion fails with
    a type error rather than a missing-op error. This distinguishes it from
    S10 (custom op) and S09 (dynamic shape).
    """
    from generate_fixtures import ManifestEntry
    import onnx
    import onnx.helper as oh
    import numpy as np

    model_id = "SM2"
    try:
        _reset_seed()

        # Graph: float image input + STRING text input → float output
        pixel_values = oh.make_tensor_value_info("pixel_values",
                           onnx.TensorProto.FLOAT, [1, 3, 224, 224])
        text_input   = oh.make_tensor_value_info("text_tokens",
                           onnx.TensorProto.STRING, [1, 77])
        output       = oh.make_tensor_value_info("similarity",
                           onnx.TensorProto.FLOAT, [1, 1])

        # Pool the image, hash the text (dummy), dot them
        w_img  = np.random.randn(1, 256).astype(np.float32)
        w_text = np.random.randn(77, 256).astype(np.float32)
        w_out  = np.random.randn(256, 1).astype(np.float32)

        init_wi = oh.make_tensor("w_img",  onnx.TensorProto.FLOAT, w_img.shape,  w_img.flatten())
        init_wo = oh.make_tensor("w_out",  onnx.TensorProto.FLOAT, w_out.shape,  w_out.flatten())

        # The string input makes this unconvertible to CoreML
        # Use a Shape op to at least reference it in the graph
        nodes = [
            oh.make_node("GlobalAveragePool", ["pixel_values"],   ["gap"]),
            oh.make_node("Flatten",           ["gap"],            ["flat"]),
            oh.make_node("Gemm",              ["flat", "w_img"],  ["img_emb"]),
            oh.make_node("Shape",             ["text_tokens"],    ["text_shape"]),   # string ref
            oh.make_node("Gemm",              ["img_emb", "w_out"], ["similarity"]),
        ]

        graph = oh.make_graph(nodes, "sm2_string_input",
                              [pixel_values, text_input], [output],
                              initializer=[init_wi, init_wo])
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
        model.ir_version = 8

        path = dirs.onnx_path(model_id, "clip_string_input_fail")
        path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(path))
        logger.info("exported ONNX  %s  [expected-failure source — string input]", path.name)

        manifest.add(ManifestEntry(
            model_id="SM2", tier="stress", modality="multimodal", format="onnx",
            size_class="tiny", file=str(path.relative_to(dirs.root)),
            expected_domain="vision", expected_subtype="multimodal",
            expected_execution="multi_input",
            expected_failure=True, failure_target="coreml",
            failure_reason="string_input_unsupported",
            commands=["inspect"],
        ))
        manifest.stamp_sha256(model_id, path); result.ok(model_id)
    except Exception as exc: result.fail(model_id, exc)


# ─── SG2 Generative: large model → CoreML size guard expected failure ─────────
def _gen_SG2(dirs, manifest, result):
    """
    GPT-2-style model scaled up to stress the CoreML size guard.
    12 layers, hidden=512, vocab=50257 → ~200MB ONNX (with external data).
    At real TinyLlama scale (2.2GB) this would OOM; at this scale it tests
    MLBuild's size validation path without causing an actual OOM.

    No conversion is attempted — expected_failure is tested at build time
    by checking that mlbuild build exits non-zero with a size error.
    """
    from generate_fixtures import ManifestEntry
    from generate_fixtures_step9 import _gpt2_coreml

    model_id = "SG2"
    try:
        _reset_seed()
        # 12 layers, hidden=512 → ~200MB weights (external data)
        model = _gpt2_coreml(
            vocab_size  = 50257,
            hidden_size = 512,
            num_heads   = 8,
            num_layers  = 12,
            max_seq_len = 128,
        ).eval()
        input_ids = torch.zeros(1, 128, dtype=torch.int64)
        path = dirs.onnx_path(model_id, "large_gpt2_size_guard_fail")
        path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            torch.onnx.export(model, (input_ids,), str(path),
                input_names=["input_ids"], output_names=["logits"],
                dynamic_axes={}, opset_version=18, do_constant_folding=True)
        # Skip onnx checker for large models — checker itself can OOM
        logger.info("exported ONNX  %s  [expected-failure source — size guard]", path.name)

        manifest.add(ManifestEntry(
            model_id="SG2", tier="stress", modality="generative", format="onnx",
            size_class="large", file=str(path.relative_to(dirs.root)),
            expected_domain="nlp", expected_subtype="generative_stateful",
            expected_execution="standard",
            expected_failure=True, failure_target="coreml",
            failure_reason="model_size_constraint",
            commands=["inspect"],
        ))
        manifest.stamp_sha256(model_id, path); result.ok(model_id)
    except Exception as exc: result.fail(model_id, exc)


# ─── Registration ─────────────────────────────────────────────────────────────
def register_failure_models(generators: list) -> None:
    generators.extend([
        (_gen_S09, "S09"),
        (_gen_S10, "S10"),
        (_gen_SD2, "SD2"),
        (_gen_SM2, "SM2"),
        (_gen_SG2, "SG2"),
    ])
    logger.info(
        "step11: registered 5 expected-failure ONNX generators "
        "(S09 dynamic-seq CoreML, S10 custom-op TFLite, "
        "SD2 baked-NMS TFLite, SM2 string-input CoreML, SG2 size-guard CoreML)"
    )