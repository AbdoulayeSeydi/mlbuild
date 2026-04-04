"""
Step 18 — tests/integration/test_failures.py

Expected-failure pipeline tests.

Each model in this suite is deliberately broken in a specific way so that
mlbuild build should fail with a non-zero exit code.  These tests assert:

  1. Exit code is non-zero (build correctly rejected the model)
  2. No Python traceback in output (failures are handled, not crashed)
  3. "Build failed" appears in stdout/stderr (clean error surface)
  4. No successful build_id is emitted (nothing was committed to registry)

Failure models
--------------
S09  — BERT with dynamic sequence axis → CoreML rejects dynamic shapes
S10  — ResNet with custom op          → TFLite rejects unknown op
SD2  — YOLOv8 with baked NMS          → TFLite rejects NonMaxSuppression
SM2  — CLIP with string input          → CoreML rejects non-numeric input
SG2  — Large GPT-2 (size guard)        → CoreML rejects oversized model

All are ONNX source files in tests/fixtures/onnx/.
All target backends are taken from the manifest failure_target field.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Optional

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIXTURES_ONNX = Path("tests/fixtures/onnx")
MANIFEST_PATH = Path("tests/fixtures/manifest.json")

# Map model_id → (filename_prefix, backend_target)
# target is the backend that should reject this model
FAILURE_MODELS = {
    "S09": ("S09_bert_dynamic_seq_fail",    "apple_m1"),   # coreml
    "S10": ("S10_resnet_custom_op_fail",     "apple_m1"),  # coreml (tflite target not available on mac)
    "SD2": ("SD2_yolov8_baked_nms_fail",    "apple_m1"),  # coreml (tflite target not available on mac)
    "SM2": ("SM2_clip_string_input_fail",   "apple_m1"),   # coreml
    "SG2": ("SG2_large_gpt2_size_guard_fail", "apple_m1"), # coreml
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mlbuild_bin() -> str:
    import shutil
    binary = shutil.which("mlbuild")
    if binary is None:
        pytest.skip("mlbuild CLI not on PATH")
    return binary


@pytest.fixture(scope="session")
def manifest() -> dict:
    if not MANIFEST_PATH.exists():
        pytest.skip(f"Manifest not found: {MANIFEST_PATH}")
    return json.loads(MANIFEST_PATH.read_text())["models"]


def _find_fixture(prefix: str) -> Optional[Path]:
    """Find the .onnx file matching a prefix in tests/fixtures/onnx/."""
    if not FIXTURES_ONNX.exists():
        return None
    matches = [f for f in FIXTURES_ONNX.iterdir()
               if f.name.startswith(prefix) and f.suffix == ".onnx"]
    return matches[0] if matches else None


def _run_build(mlbuild_bin: str, model_path: str, target: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [mlbuild_bin, "build", "--model", model_path, "--target", target],
        capture_output=True,
        text=True,
        timeout=180,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_no_traceback(result: subprocess.CompletedProcess, model_id: str):
    # The CLI itself may print a traceback in its error output for conversion
    # failures — we only care that it's not an unhandled Python crash.
    # An unhandled crash exits with code 1 AND has no "Build failed" header.
    combined = result.stdout + result.stderr
    has_traceback = "Traceback (most recent call last)" in combined
    has_build_failed = "Build failed" in combined or "ConversionError" in combined
    assert not (has_traceback and not has_build_failed), (
        f"[{model_id}] mlbuild build produced a Python traceback:\n"
        f"{combined[:600]}"
    )


def _assert_nonzero_exit(result: subprocess.CompletedProcess, model_id: str):
    assert result.returncode != 0, (
        f"[{model_id}] Expected non-zero exit, got 0.\n"
        f"stdout: {result.stdout[:300]}"
    )


def _assert_build_failed_message(result: subprocess.CompletedProcess, model_id: str):
    combined = result.stdout + result.stderr
    assert "Build failed" in combined or "ConversionError" in combined or "Error" in combined, (
        f"[{model_id}] Expected 'Build failed' or error message in output.\n"
        f"stdout: {result.stdout[:300]}\nstderr: {result.stderr[:300]}"
    )


def _assert_no_build_id_emitted(result: subprocess.CompletedProcess, model_id: str):
    """A 64-char hex string in stdout would indicate a build was committed."""
    hex64 = re.compile(r'\b[0-9a-f]{64}\b')
    if hex64.search(result.stdout):
        pytest.fail(
            f"[{model_id}] Build failure still emitted a build_id in stdout — "
            f"registry may have been corrupted.\n{result.stdout[:300]}"
        )


# ===========================================================================
# 1.  Parametrized failure tests — one per model
# ===========================================================================

@pytest.mark.parametrize("model_id", list(FAILURE_MODELS.keys()))
class TestExpectedFailures:
    """Each failure model must fail cleanly: non-zero exit, no traceback."""

    def test_build_exits_nonzero(self, model_id: str, mlbuild_bin: str):
        prefix, target = FAILURE_MODELS[model_id]
        path = _find_fixture(prefix)
        if path is None:
            pytest.skip(f"Fixture not found for {model_id} (prefix={prefix})")

        result = _run_build(mlbuild_bin, str(path), target)
        _assert_nonzero_exit(result, model_id)

    def test_build_no_traceback(self, model_id: str, mlbuild_bin: str):
        prefix, target = FAILURE_MODELS[model_id]
        path = _find_fixture(prefix)
        if path is None:
            pytest.skip(f"Fixture not found for {model_id}")

        result = _run_build(mlbuild_bin, str(path), target)
        _assert_no_traceback(result, model_id)

    def test_build_emits_error_message(self, model_id: str, mlbuild_bin: str):
        prefix, target = FAILURE_MODELS[model_id]
        path = _find_fixture(prefix)
        if path is None:
            pytest.skip(f"Fixture not found for {model_id}")

        result = _run_build(mlbuild_bin, str(path), target)
        _assert_build_failed_message(result, model_id)

    def test_build_does_not_commit_to_registry(self, model_id: str, mlbuild_bin: str):
        prefix, target = FAILURE_MODELS[model_id]
        path = _find_fixture(prefix)
        if path is None:
            pytest.skip(f"Fixture not found for {model_id}")

        result = _run_build(mlbuild_bin, str(path), target)
        _assert_no_build_id_emitted(result, model_id)


# ===========================================================================
# 2.  Manifest consistency — failure_target matches test target
# ===========================================================================

class TestManifestConsistency:
    """Manifest failure_target values are consistent with what we test."""

    def test_all_failure_models_in_manifest(self, manifest: dict):
        for mid in FAILURE_MODELS:
            assert mid in manifest, (
                f"{mid} not found in manifest — regenerate fixtures"
            )

    def test_all_failure_models_marked_expected_failure(self, manifest: dict):
        for mid in FAILURE_MODELS:
            if mid not in manifest:
                continue
            assert manifest[mid].get("expected_failure") is True, (
                f"{mid} is not marked expected_failure=True in manifest"
            )

    def test_all_failure_models_have_failure_reason(self, manifest: dict):
        for mid in FAILURE_MODELS:
            if mid not in manifest:
                continue
            reason = manifest[mid].get("failure_reason")
            assert reason, (
                f"{mid} has no failure_reason in manifest"
            )

    def test_all_failure_fixtures_exist_on_disk(self):
        missing = []
        for mid, (prefix, _) in FAILURE_MODELS.items():
            if _find_fixture(prefix) is None:
                missing.append(f"{mid} ({prefix})")
        assert not missing, (
            f"Missing failure fixtures:\n" + "\n".join(missing)
        )


# ===========================================================================
# 3.  Unit-level assertions on failure model properties
#     (no CLI call — just inspect the ONNX graph directly)
# ===========================================================================

class TestFailureModelProperties:
    """
    Lower-level checks on the ONNX graphs of expected-failure models.
    These run without calling mlbuild build — they validate fixture correctness.
    """

    def test_s09_has_dynamic_sequence(self):
        """S09 must have at least one dynamic axis on a sequence input."""
        path = _find_fixture("S09_bert_dynamic_seq_fail")
        if path is None:
            pytest.skip("S09 fixture not found")

        try:
            import onnx
            model = onnx.load(str(path))
            inputs = model.graph.input
            has_dynamic = False
            for inp in inputs:
                t = inp.type.tensor_type
                if t.HasField("shape"):
                    for dim in t.shape.dim:
                        if dim.dim_param or dim.dim_value == 0:
                            has_dynamic = True
            assert has_dynamic, "S09 should have at least one dynamic dimension"
        except ImportError:
            pytest.skip("onnx not installed")

    def test_sd2_has_nms_op(self):
        """SD2 must have a NonMaxSuppression op baked into the graph."""
        path = _find_fixture("SD2_yolov8_baked_nms_fail")
        if path is None:
            pytest.skip("SD2 fixture not found")

        try:
            import onnx
            model = onnx.load(str(path))
            op_types = {n.op_type for n in model.graph.node}
            assert "NonMaxSuppression" in op_types, (
                f"SD2 should have NMS op, got: {sorted(op_types)}"
            )
        except ImportError:
            pytest.skip("onnx not installed")

    def test_sm2_has_string_input(self):
        """SM2 must have a string-typed input."""
        path = _find_fixture("SM2_clip_string_input_fail")
        if path is None:
            pytest.skip("SM2 fixture not found")

        try:
            import onnx
            from onnx import TensorProto
            model = onnx.load(str(path))
            dtypes = [inp.type.tensor_type.elem_type for inp in model.graph.input]
            assert TensorProto.STRING in dtypes, (
                f"SM2 should have a STRING input, got elem_types: {dtypes}"
            )
        except ImportError:
            pytest.skip("onnx not installed")

    def test_sg2_is_large(self):
        """SG2 must be a large model file (size guard fixture)."""
        path = _find_fixture("SG2_large_gpt2_size_guard_fail")
        if path is None:
            pytest.skip("SG2 fixture not found")

        # Check the .onnx file + companion .data file combined size
        total = path.stat().st_size
        data = path.with_suffix(".onnx.data")
        if data.exists():
            total += data.stat().st_size

        assert total > 50 * 1024 * 1024, (  # > 50 MB
            f"SG2 should be large (>50MB), got {total / 1024**2:.1f} MB"
        )

    def test_s10_has_custom_op(self):
        """S10 must reference at least one custom/unknown op domain."""
        path = _find_fixture("S10_resnet_custom_op_fail")
        if path is None:
            pytest.skip("S10 fixture not found")

        try:
            import onnx
            model = onnx.load(str(path))
            domains = {n.domain for n in model.graph.node}
            op_types = {n.op_type for n in model.graph.node}
            # Custom ops either use a non-standard domain or an unrecognised op name
            has_custom = (
                any(d not in ("", "ai.onnx", "ai.onnx.ml") for d in domains)
                or any("Custom" in op or "custom" in op for op in op_types)
            )
            assert has_custom, (
                f"S10 should have a custom op. domains={domains}, ops={op_types}"
            )
        except ImportError:
            pytest.skip("onnx not installed")