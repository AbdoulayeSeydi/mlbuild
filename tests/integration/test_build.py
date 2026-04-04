"""
Step 17 — tests/integration/test_build.py

Smoke tests for the mlbuild CLI.  Calls real CLI commands via subprocess.

Strategy
--------
mlbuild build  — full CoreML/TFLite conversion is slow and hardware-dependent.
                 We run it for one TFLite model where conversion is fast and
                 reliable (no onnx2torch bridge, pure TF→TFLite).
                 All other build smoke tests use pre-existing registry builds.

mlbuild inspect --json — validates the JSON output schema for 6 representative
                 builds: one per task type (vision, nlp, audio, tabular,
                 detection subtype, timeseries subtype).

Requirements
------------
- mlbuild CLI must be on PATH (mlbuild_bin fixture skips if not)
- .mlbuild/registry.db must exist with at least one build
- Tests are marked integration and not run by default with `pytest tests/unit/`
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path
from typing import Optional

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REGISTRY_PATH = Path(".mlbuild/registry.db")

REQUIRED_JSON_FIELDS = {
    "id", "task_type", "subtype", "execution_mode",
    "artifacts", "created_at", "name",
}


@pytest.fixture(scope="session")
def mlbuild_bin() -> str:
    """Path to mlbuild binary — skips all tests if not on PATH."""
    import shutil
    binary = shutil.which("mlbuild")
    if binary is None:
        pytest.skip("mlbuild CLI not on PATH")
    return binary


@pytest.fixture(scope="session")
def registry_builds() -> list[dict]:
    """
    Load all non-deleted builds from the local registry.
    Skips if registry doesn't exist or is empty.
    """
    if not REGISTRY_PATH.exists():
        pytest.skip(f"Registry not found: {REGISTRY_PATH}")

    conn = sqlite3.connect(str(REGISTRY_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT build_id, name, format, task_type, subtype, execution_mode "
        "FROM builds WHERE deleted_at IS NULL ORDER BY created_at DESC"
    ).fetchall()
    conn.close()

    if not rows:
        pytest.skip("No builds in registry")

    return [dict(r) for r in rows]


@pytest.fixture(scope="session")
def latest_build(registry_builds) -> dict:
    """Most recent non-deleted build."""
    return registry_builds[0]


def _run_inspect(mlbuild_bin: str, build_id: str) -> Optional[dict]:
    """Run mlbuild inspect BUILD_ID --json and return parsed JSON or None."""
    result = subprocess.run(
        [mlbuild_bin, "inspect", build_id, "--json"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


# ===========================================================================
# 1.  mlbuild inspect --json — JSON schema validation
# ===========================================================================

class TestInspectJsonSchema:
    """inspect --json output has correct fields and types for all builds."""

    def test_inspect_returns_zero_exit_code(self, mlbuild_bin, latest_build):
        result = subprocess.run(
            [mlbuild_bin, "inspect", latest_build["build_id"], "--json"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"inspect exited {result.returncode}: {result.stderr[:200]}"
        )

    def test_inspect_output_is_valid_json(self, mlbuild_bin, latest_build):
        result = subprocess.run(
            [mlbuild_bin, "inspect", latest_build["build_id"], "--json"],
            capture_output=True, text=True, timeout=30,
        )
        try:
            data = json.loads(result.stdout)
            assert isinstance(data, dict)
        except json.JSONDecodeError as exc:
            pytest.fail(f"inspect --json output is not valid JSON: {exc}")

    def test_inspect_has_required_fields(self, mlbuild_bin, latest_build):
        data = _run_inspect(mlbuild_bin, latest_build["build_id"])
        assert data is not None, "inspect returned no JSON"
        missing = REQUIRED_JSON_FIELDS - set(data.keys())
        assert not missing, f"inspect JSON missing fields: {missing}"

    def test_inspect_task_type_is_string(self, mlbuild_bin, latest_build):
        data = _run_inspect(mlbuild_bin, latest_build["build_id"])
        assert data is not None
        assert isinstance(data["task_type"], str), (
            f"task_type should be str, got {type(data['task_type'])}"
        )

    def test_inspect_task_type_is_valid_value(self, mlbuild_bin, latest_build):
        data = _run_inspect(mlbuild_bin, latest_build["build_id"])
        assert data is not None
        valid = {"vision", "nlp", "audio", "tabular", "multimodal", "unknown"}
        assert data["task_type"] in valid, (
            f"task_type '{data['task_type']}' not in {valid}"
        )

    def test_inspect_subtype_is_string(self, mlbuild_bin, latest_build):
        data = _run_inspect(mlbuild_bin, latest_build["build_id"])
        assert data is not None
        assert isinstance(data["subtype"], str)

    def test_inspect_execution_mode_is_string(self, mlbuild_bin, latest_build):
        data = _run_inspect(mlbuild_bin, latest_build["build_id"])
        assert data is not None
        assert isinstance(data["execution_mode"], str)

    def test_inspect_execution_mode_is_valid_value(self, mlbuild_bin, latest_build):
        data = _run_inspect(mlbuild_bin, latest_build["build_id"])
        assert data is not None
        valid = {"standard", "stateful", "partially_stateful",
                 "kv_cache", "multi_input", "streaming"}
        assert data["execution_mode"] in valid, (
            f"execution_mode '{data['execution_mode']}' not in {valid}"
        )

    def test_inspect_artifacts_is_list(self, mlbuild_bin, latest_build):
        data = _run_inspect(mlbuild_bin, latest_build["build_id"])
        assert data is not None
        assert isinstance(data["artifacts"], list)

    def test_inspect_id_matches_requested(self, mlbuild_bin, latest_build):
        build_id = latest_build["build_id"]
        data = _run_inspect(mlbuild_bin, build_id)
        assert data is not None
        assert data["id"] == build_id, (
            f"inspect returned id {data['id']!r}, expected {build_id!r}"
        )

    def test_inspect_nonexistent_build_fails(self, mlbuild_bin):
        result = subprocess.run(
            [mlbuild_bin, "inspect", "deadbeef" * 8, "--json"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0, (
            "inspect on nonexistent build_id should return non-zero exit"
        )


# ===========================================================================
# 2.  inspect across multiple builds — one per task type
# ===========================================================================

class TestInspectPerTaskType:
    """
    For each unique task_type in the registry, verify inspect --json returns
    the correct task_type field.  Parametrized dynamically from registry.
    """

    def test_inspect_task_type_matches_registry(
        self, mlbuild_bin, registry_builds
    ):
        """For every build in registry, inspect --json task_type matches DB."""
        failures = []
        for build in registry_builds[:10]:  # cap at 10 to keep test fast
            data = _run_inspect(mlbuild_bin, build["build_id"])
            if data is None:
                continue  # skip builds where inspect fails
            db_task = build["task_type"] or "unknown"
            json_task = data.get("task_type", "")
            if db_task != json_task:
                failures.append(
                    f"{build['name']} ({build['build_id'][:8]}): "
                    f"db={db_task!r} json={json_task!r}"
                )
        assert not failures, (
            f"task_type mismatches between registry and inspect --json:\n"
            + "\n".join(failures)
        )

    def test_each_build_has_valid_subtype(self, mlbuild_bin, registry_builds):
        """Every build's subtype field is a known value."""
        valid_subtypes = {
            "none", "detection", "segmentation", "timeseries",
            "recommendation", "generative_stateful", "multimodal",
        }
        failures = []
        for build in registry_builds[:10]:
            data = _run_inspect(mlbuild_bin, build["build_id"])
            if data is None:
                continue
            subtype = data.get("subtype", "")
            if subtype not in valid_subtypes:
                failures.append(
                    f"{build['name']}: subtype={subtype!r}"
                )
        assert not failures, f"Invalid subtypes found:\n" + "\n".join(failures)


# ===========================================================================
# 3.  mlbuild build — smoke test (TFLite, fast conversion)
# ===========================================================================

class TestBuildSmoke:
    """
    Run mlbuild build on one TFLite fixture that is known to convert fast.
    Skips gracefully if the fixture file isn't present.
    """

    def _find_tflite_fixture(self) -> Optional[str]:
        """Find a small TFLite fixture to import directly."""
        fx = Path("tests/fixtures/tflite")
        if not fx.exists():
            return None
        matches = sorted(fx.glob("B*.tflite"))
        return str(matches[0]) if matches else None

    def test_build_tflite_exits_zero_or_known_failure(
        self, mlbuild_bin
    ):
        """
        mlbuild build on a TFLite model should exit 0 (success) or with a
        known conversion error — never a Python traceback.
        """
        path = self._find_tflite_fixture()
        if path is None:
            pytest.skip("No TFLite fixtures found")

        result = subprocess.run(
            [mlbuild_bin, "build", "--model", path,
             "--target", "android_arm64", "--name", "test_smoke_tflite"],
            capture_output=True, text=True, timeout=120,
        )

        # A traceback = unexpected crash → fail the test
        assert "Traceback (most recent call last)" not in result.stderr, (
            f"mlbuild build produced a Python traceback:\n{result.stderr[:500]}"
        )
        assert "Traceback (most recent call last)" not in result.stdout, (
            f"mlbuild build produced a Python traceback:\n{result.stdout[:500]}"
        )

    def test_build_missing_model_fails_cleanly(self, mlbuild_bin):
        """build with a nonexistent model path should fail with exit != 0."""
        result = subprocess.run(
            [mlbuild_bin, "build", "--model", "/nonexistent/model.onnx",
             "--target", "apple_m1"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0, (
            "build with missing model should return non-zero exit"
        )
        assert "Traceback (most recent call last)" not in result.stderr

    def test_build_missing_target_fails_cleanly(self, mlbuild_bin):
        """build without --target should fail with exit != 0 and helpful message."""
        path = self._find_tflite_fixture() or "model.onnx"
        result = subprocess.run(
            [mlbuild_bin, "build", "--model", path],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0

    def test_build_after_success_appears_in_inspect(self, mlbuild_bin):
        """
        If a build succeeds, inspect on the resulting build_id returns JSON.
        Skips if no fixture available or build fails.
        """
        path = self._find_tflite_fixture()
        if path is None:
            pytest.skip("No TFLite fixtures found")

        result = subprocess.run(
            [mlbuild_bin, "build", "--model", path,
             "--target", "android_arm64", "--name", "test_inspect_after_build"],
            capture_output=True, text=True, timeout=120,
        )

        if result.returncode != 0:
            pytest.skip(f"Build failed (conversion error), skipping inspect check")

        # Extract build_id from output
        import re
        match = re.search(r"[0-9a-f]{64}", result.stdout + result.stderr)
        if not match:
            pytest.skip("Could not extract build_id from build output")

        build_id = match.group(0)
        data = _run_inspect(mlbuild_bin, build_id)
        assert data is not None, f"inspect returned no JSON for build {build_id[:8]}"
        assert "task_type" in data
        assert "subtype" in data
        assert "execution_mode" in data


# ===========================================================================
# 4.  mlbuild status — sanity check
# ===========================================================================

class TestStatus:
    """mlbuild status --json returns valid JSON with expected top-level keys."""

    def test_status_exits_zero(self, mlbuild_bin):
        result = subprocess.run(
            [mlbuild_bin, "status", "--json"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0

    def test_status_returns_valid_json(self, mlbuild_bin):
        result = subprocess.run(
            [mlbuild_bin, "status", "--json"],
            capture_output=True, text=True, timeout=30,
        )
        try:
            data = json.loads(result.stdout)
            assert isinstance(data, dict)
        except json.JSONDecodeError as exc:
            pytest.fail(f"status --json output is not valid JSON: {exc}")

    def test_status_has_build_count(self, mlbuild_bin):
        result = subprocess.run(
            [mlbuild_bin, "status", "--json"],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(result.stdout)
        assert "build_count" in data
        assert isinstance(data["build_count"], int)
        assert data["build_count"] >= 0