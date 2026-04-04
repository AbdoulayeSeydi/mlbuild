"""
tests/conftest.py — Shared pytest fixtures for MLBuild test suite.

Session-scoped fixtures load the fixture manifest once per pytest run and
provide model paths, expected profiles, and filtered model sets to tests.

Layout
------
simple_mlpackage    function-scoped  throwaway .mlpackage for unit tests
manifest            session-scoped   parsed manifest.json (all 55 entries)
fixture_path        session-scoped   resolves model_id → absolute Path
expected_profile    session-scoped   model_id → dict of expected_* fields
all_models          session-scoped   list of all 55 model_ids
baseline_models     session-scoped   32 baseline model_ids (tier="baseline")
stress_models       session-scoped   18 stress model_ids  (tier="stress")
failure_models      session-scoped   5 expected-failure model_ids
mlbuild_available   session-scoped   bool — True if `mlbuild` is on PATH
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable

import pytest

# ── Constants ─────────────────────────────────────────────────────────────────

# tests/fixtures/ relative to this conftest.py
TESTS_DIR    = Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
MANIFEST_FILE = FIXTURES_DIR / "manifest.json"


# ── Throwaway model fixture ───────────────────────────────────────────────────

@pytest.fixture
def simple_mlpackage(tmp_path: Path) -> Path:
    """
    Create a minimal .mlpackage for unit tests that need a real model file
    but don't care about its task type or architecture.

    Uses the legacy NeuralNetworkBuilder (identity activation) — fast and
    dependency-light. Not suitable for MLProgram-specific tests.

    Returns: path to the .mlpackage directory.
    """
    import coremltools as ct

    input_features  = [("input",  ct.models.datatypes.Array(1, 10))]
    output_features = [("output", ct.models.datatypes.Array(1, 10))]

    builder = ct.models.neural_network.NeuralNetworkBuilder(
        input_features,
        output_features,
    )
    builder.add_activation(
        name="identity",
        non_linearity="LINEAR",
        input_name="input",
        output_name="output",
    )

    mlpackage_path = tmp_path / "test_model.mlpackage"
    model = ct.models.MLModel(builder.spec)
    model.save(str(mlpackage_path))
    return mlpackage_path


# ── Manifest fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def manifest() -> dict[str, dict]:
    """
    Load tests/fixtures/manifest.json once per session.

    Returns a dict keyed by model_id:
        {
            "B01": {
                "model_id": "B01",
                "tier": "baseline",
                "format": "onnx",
                "file": "onnx/B01_mobilenetv3small_vision.onnx",
                "expected_domain": "vision",
                "expected_subtype": "none",
                "expected_execution": "standard",
                "expected_nms_inside": false,
                "expected_failure": false,
                ...
            },
            ...
        }

    Raises pytest.skip if the manifest file doesn't exist so individual tests
    are skipped rather than erroring out when fixtures haven't been generated.
    """
    if not MANIFEST_FILE.exists():
        pytest.skip(
            f"Fixture manifest not found: {MANIFEST_FILE}. "
            "Run `python tests/generate_fixtures.py --out-dir tests/fixtures` first."
        )

    raw = json.loads(MANIFEST_FILE.read_text())

    # Manifest may be a list or a dict depending on the generate_fixtures version
    if isinstance(raw, list):
        entries = {e["model_id"]: e for e in raw}
    elif isinstance(raw, dict):
        # Support both {"models": [...]} and {"B01": {...}, ...} shapes
        if "models" in raw:
            m = raw["models"]
            entries = m if isinstance(m, dict) else {e["model_id"]: e for e in m}
        else:
            entries = raw
    else:
        pytest.fail(f"Unexpected manifest format: {type(raw)}")

    assert len(entries) > 0, "Manifest is empty"
    return entries


@pytest.fixture(scope="session")
def fixture_path(manifest: dict[str, dict]) -> Callable[[str], Path]:
    """
    Returns a callable: fixture_path(model_id) → absolute Path.

    Asserts the file exists so failures surface as:
        AssertionError: fixture file missing for B01: tests/fixtures/onnx/B01_...onnx
    rather than a FileNotFoundError three frames deep.

    Usage in tests:
        def test_something(fixture_path):
            path = fixture_path("B01")
            assert path.suffix == ".onnx"
    """
    def _resolve(model_id: str) -> Path:
        assert model_id in manifest, (
            f"model_id '{model_id}' not found in manifest. "
            f"Known IDs: {sorted(manifest)[:10]}..."
        )
        entry = manifest[model_id]
        # manifest stores relative paths from FIXTURES_DIR
        rel = entry.get("file", "")
        path = FIXTURES_DIR / rel
        assert path.exists(), (
            f"Fixture file missing for {model_id}: {path}\n"
            "Re-run generate_fixtures.py to regenerate."
        )
        return path

    return _resolve


@pytest.fixture(scope="session")
def expected_profile(manifest: dict[str, dict]) -> Callable[[str], dict]:
    """
    Returns a callable: expected_profile(model_id) → dict of expected_* fields.

    The returned dict contains all manifest fields prefixed `expected_`:
        {
            "domain":      "vision",
            "subtype":     "none",
            "execution":   "standard",
            "nms_inside":  False,
            "failure":     False,
            "failure_target":  None,
            "failure_reason":  None,
        }

    Usage in tests:
        def test_detection(fixture_path, expected_profile):
            profile = detect(fixture_path("B01"))
            exp = expected_profile("B01")
            assert profile.domain == exp["domain"]
    """
    def _profile(model_id: str) -> dict:
        assert model_id in manifest, f"model_id '{model_id}' not in manifest"
        e = manifest[model_id]
        return {
            "domain":         e.get("expected_domain",     "unknown"),
            "subtype":        e.get("expected_subtype",    "none"),
            "execution":      e.get("expected_execution",  "standard"),
            "nms_inside":     e.get("expected_nms_inside", False),
            "failure":        e.get("expected_failure",    False),
            "failure_target": e.get("failure_target",      None),
            "failure_reason": e.get("failure_reason",      None),
        }

    return _profile


# ── Model-set fixtures ────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def all_models(manifest: dict[str, dict]) -> list[str]:
    """All 55 model IDs, sorted."""
    return sorted(manifest.keys())


@pytest.fixture(scope="session")
def baseline_models(manifest: dict[str, dict]) -> list[str]:
    """
    32 baseline model IDs (tier="baseline"), sorted.

    Used to parametrize detection/inputs/validation unit tests that only
    run against well-formed models with known-good expected profiles.
    """
    return sorted(
        mid for mid, e in manifest.items()
        if e.get("tier") == "baseline"
    )


@pytest.fixture(scope="session")
def stress_models(manifest: dict[str, dict]) -> list[str]:
    """18 stress model IDs (tier="stress", expected_failure=False), sorted."""
    return sorted(
        mid for mid, e in manifest.items()
        if e.get("tier") == "stress" and not e.get("expected_failure", False)
    )


@pytest.fixture(scope="session")
def failure_models(manifest: dict[str, dict]) -> list[str]:
    """
    5 expected-failure model IDs (expected_failure=True), sorted.

    These are S09, S10, SD2, SM2, SG2 — ONNX sources that should fail
    at conversion time in the integration tests (Step 18).
    """
    return sorted(
        mid for mid, e in manifest.items()
        if e.get("expected_failure", False)
    )


@pytest.fixture(scope="session")
def models_by_format(manifest: dict[str, dict]) -> dict[str, list[str]]:
    """
    Dict mapping format → sorted list of model_ids.

    Keys: "onnx", "mlmodel", "mlpackage", "tflite"

    Useful for format-specific tests:
        def test_coreml_only(models_by_format, fixture_path):
            for mid in models_by_format["mlpackage"]:
                ...
    """
    result: dict[str, list[str]] = {}
    for mid, e in manifest.items():
        fmt = e.get("format", "unknown")
        result.setdefault(fmt, []).append(mid)
    return {fmt: sorted(ids) for fmt, ids in result.items()}


@pytest.fixture(scope="session")
def models_by_domain(manifest: dict[str, dict]) -> dict[str, list[str]]:
    """Dict mapping expected_domain → sorted list of baseline model_ids."""
    result: dict[str, list[str]] = {}
    for mid, e in manifest.items():
        if e.get("tier") == "baseline":
            domain = e.get("expected_domain", "unknown")
            result.setdefault(domain, []).append(mid)
    return {d: sorted(ids) for d, ids in result.items()}


# ── CLI availability guard ────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def mlbuild_available() -> bool:
    """
    True if the `mlbuild` CLI is on PATH and returns exit code 0 for --version.

    Integration tests (Steps 17–18) use this to skip gracefully on systems
    where the package isn't installed.

    Usage:
        def test_build_smoke(mlbuild_available, fixture_path):
            if not mlbuild_available:
                pytest.skip("mlbuild CLI not available")
            ...
    """
    try:
        result = subprocess.run(
            ["mlbuild", "--version"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.fixture(scope="session")
def mlbuild_bin() -> str:
    """
    Path to the mlbuild binary (or just "mlbuild" if on PATH).

    Raises pytest.skip if not available — use this in integration tests
    instead of mlbuild_available so the skip happens automatically.
    """
    binary = shutil.which("mlbuild")
    if binary is None:
        pytest.skip("mlbuild CLI not on PATH — skipping integration test")
    return binary


# ── Parametrize helpers ───────────────────────────────────────────────────────

def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """
    Auto-parametrize tests that declare a `model_id` fixture parameter.

    If the test is marked with:
        @pytest.mark.parametrize_baseline
    it gets one invocation per baseline model ID.

    If marked with:
        @pytest.mark.parametrize_all
    it gets one invocation per all 55 model IDs.

    This keeps test files clean — no boilerplate parametrize calls needed.
    """
    # Only act if the manifest exists (otherwise tests skip via the fixture)
    if not MANIFEST_FILE.exists():
        return

    if "model_id" not in metafunc.fixturenames:
        return

    marker_baseline = metafunc.definition.get_closest_marker("parametrize_baseline")
    marker_all      = metafunc.definition.get_closest_marker("parametrize_all")

    if marker_baseline or marker_all:
        raw = json.loads(MANIFEST_FILE.read_text())
        if isinstance(raw, list):
            entries = raw
        elif "models" in raw:
            entries = raw["models"]
        else:
            entries = list(raw.values())

        if marker_baseline:
            ids = sorted(e["model_id"] for e in entries if e.get("tier") == "baseline")
        else:
            ids = sorted(e["model_id"] for e in entries)

        metafunc.parametrize("model_id", ids)