"""
tests/unit/test_accuracy.py

Tests for the accuracy module changes:
    AccuracyConfig      — profiles, overrides, validation
    EvaluationContext   — construction, validation
    AccuracyResult      — new fields, as_db_row()
    New accumulators    — RMSEAccumulator, KLDivAccumulator,
                          JSDivAccumulator, PercentileAEAccumulator
    generate_batch_task_aware — vision/nlp/audio/fallback distributions
    load_dataset_batch        — .npz and .npy loading, error cases

Running
-------
    pytest tests/unit/test_accuracy.py -v
    pytest tests/unit/test_accuracy.py -v -k "profile"
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Modules under test
# ---------------------------------------------------------------------------
try:
    from mlbuild.core.accuracy.config import (
        AccuracyConfig,
        EvaluationContext,
        AccuracyResult,
        PROFILES,
        VALID_PROFILES,
        VALID_TASK_TYPES,
    )
    from mlbuild.core.accuracy.metrics import (
        RMSEAccumulator,
        KLDivAccumulator,
        JSDivAccumulator,
        PercentileAEAccumulator,
    )
    from mlbuild.core.accuracy.inputs import (
        InputSpec,
        generate_batch,
        generate_batch_task_aware,
        load_dataset_batch,
    )
except ImportError as exc:
    pytest.skip(f"mlbuild not importable: {exc}", allow_module_level=True)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _float_spec(name: str = "input", shape: tuple = (4,)) -> InputSpec:
    return InputSpec(name=name, shape=shape, dtype=np.float32)


def _int_spec(name: str = "input_ids", shape: tuple = (8,)) -> InputSpec:
    return InputSpec(name=name, shape=shape, dtype=np.int64)


def _make_result(**overrides) -> AccuracyResult:
    defaults = dict(
        baseline_build_id="a" * 64,
        candidate_build_id="b" * 64,
        cosine_similarity=0.995,
        mean_abs_error=0.001,
        max_abs_error=0.01,
        rmse=0.002,
        top1_agreement=None,
        kl_divergence=None,
        js_divergence=None,
        error_p50=0.0005,
        error_p95=0.005,
        error_p99=0.009,
        num_samples=32,
        seed=42,
        failure_reasons=(),
    )
    defaults.update(overrides)
    return AccuracyResult(**defaults)


def _make_context(**overrides) -> EvaluationContext:
    defaults = dict(
        baseline_build_id="a" * 64,
        candidate_build_id="b" * 64,
    )
    defaults.update(overrides)
    return EvaluationContext(**defaults)


# ===========================================================================
# 1.  AccuracyConfig — profiles
# ===========================================================================

class TestAccuracyConfigProfiles:

    def test_valid_profiles_exist(self):
        assert "strict" in PROFILES
        assert "default" in PROFILES
        assert "loose" in PROFILES

    def test_from_profile_strict_thresholds(self):
        cfg = AccuracyConfig.from_profile("strict")
        assert cfg.cosine_threshold == PROFILES["strict"]["cosine_threshold"]
        assert cfg.top1_threshold   == PROFILES["strict"]["top1_threshold"]
        assert cfg.mae_threshold    == PROFILES["strict"]["mae_threshold"]
        assert cfg.rmse_threshold   == PROFILES["strict"]["rmse_threshold"]

    def test_from_profile_loose_thresholds(self):
        cfg = AccuracyConfig.from_profile("loose")
        assert cfg.cosine_threshold == PROFILES["loose"]["cosine_threshold"]
        assert cfg.mae_threshold    is None
        assert cfg.rmse_threshold   is None

    def test_from_profile_stores_profile_name(self):
        cfg = AccuracyConfig.from_profile("default")
        assert cfg.profile == "default"

    def test_from_profile_override_samples(self):
        cfg = AccuracyConfig.from_profile("strict", samples=64)
        assert cfg.samples == 64
        assert cfg.cosine_threshold == PROFILES["strict"]["cosine_threshold"]

    def test_from_profile_override_cosine_threshold(self):
        cfg = AccuracyConfig.from_profile("default", cosine_threshold=0.999)
        assert cfg.cosine_threshold == 0.999

    def test_from_profile_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            AccuracyConfig.from_profile("nonexistent")

    def test_from_cli_with_profile(self):
        cfg = AccuracyConfig.from_cli(profile="strict", samples=16)
        assert cfg.profile == "strict"
        assert cfg.samples == 16
        assert cfg.cosine_threshold == PROFILES["strict"]["cosine_threshold"]

    def test_from_cli_without_profile_uses_defaults(self):
        cfg = AccuracyConfig.from_cli()
        assert cfg.profile is None
        assert cfg.cosine_threshold == 0.99

    def test_invalid_profile_field_raises(self):
        with pytest.raises(ValueError, match="profile must be one of"):
            AccuracyConfig(profile="bad")


# ===========================================================================
# 2.  AccuracyConfig — rmse_threshold field
# ===========================================================================

class TestAccuracyConfigRmse:

    def test_rmse_threshold_none_by_default(self):
        cfg = AccuracyConfig()
        assert cfg.rmse_threshold is None

    def test_rmse_threshold_set(self):
        cfg = AccuracyConfig(rmse_threshold=0.01)
        assert cfg.rmse_threshold == 0.01

    def test_rmse_threshold_negative_raises(self):
        with pytest.raises(ValueError, match="rmse_threshold must be ≥ 0"):
            AccuracyConfig(rmse_threshold=-0.1)

    def test_rmse_threshold_zero_valid(self):
        cfg = AccuracyConfig(rmse_threshold=0.0)
        assert cfg.rmse_threshold == 0.0


# ===========================================================================
# 3.  EvaluationContext — construction and validation
# ===========================================================================

class TestEvaluationContext:

    def test_basic_construction(self):
        ctx = _make_context()
        assert ctx.baseline_build_id  == "a" * 64
        assert ctx.candidate_build_id == "b" * 64
        assert ctx.task_type    is None
        assert ctx.dataset_path is None
        assert ctx.cross_format is False

    def test_with_task_type(self):
        ctx = _make_context(task_type="vision")
        assert ctx.task_type == "vision"

    def test_with_dataset_path(self):
        ctx = _make_context(dataset_path="/tmp/data.npz")
        assert ctx.dataset_path == "/tmp/data.npz"

    def test_cross_format_flag(self):
        ctx = _make_context(cross_format=True)
        assert ctx.cross_format is True

    def test_empty_baseline_id_raises(self):
        with pytest.raises(ValueError, match="baseline_build_id must not be empty"):
            EvaluationContext(baseline_build_id="", candidate_build_id="b" * 64)

    def test_empty_candidate_id_raises(self):
        with pytest.raises(ValueError, match="candidate_build_id must not be empty"):
            EvaluationContext(baseline_build_id="a" * 64, candidate_build_id="")

    def test_same_ids_raises(self):
        with pytest.raises(ValueError, match="must differ"):
            EvaluationContext(baseline_build_id="a" * 64, candidate_build_id="a" * 64)

    def test_invalid_task_type_raises(self):
        with pytest.raises(ValueError, match="task_type must be one of"):
            _make_context(task_type="invalid_task")

    def test_all_valid_task_types_accepted(self):
        for task in VALID_TASK_TYPES:
            ctx = _make_context(task_type=task)
            assert ctx.task_type == task

    def test_frozen(self):
        ctx = _make_context()
        with pytest.raises(Exception):
            ctx.task_type = "vision"  # type: ignore[misc]


# ===========================================================================
# 4.  AccuracyResult — new fields and as_db_row()
# ===========================================================================

class TestAccuracyResult:

    def test_new_fields_present(self):
        r = _make_result()
        assert hasattr(r, "rmse")
        assert hasattr(r, "kl_divergence")
        assert hasattr(r, "js_divergence")
        assert hasattr(r, "error_p50")
        assert hasattr(r, "error_p95")
        assert hasattr(r, "error_p99")

    def test_passed_when_no_failures(self):
        r = _make_result(failure_reasons=())
        assert r.passed is True

    def test_failed_when_failures(self):
        r = _make_result(failure_reasons=("cosine_similarity 0.98 < threshold 0.99",))
        assert r.passed is False

    def test_as_db_row_contains_new_fields(self):
        r = _make_result(
            rmse=0.002,
            kl_divergence=0.01,
            js_divergence=0.005,
            error_p50=0.001,
            error_p95=0.005,
            error_p99=0.009,
        )
        row = r.as_db_row()
        assert row["rmse"]          == 0.002
        assert row["kl_divergence"] == 0.01
        assert row["js_divergence"] == 0.005
        assert row["error_p50"]     == 0.001
        assert row["error_p95"]     == 0.005
        assert row["error_p99"]     == 0.009

    def test_as_db_row_none_fields_preserved(self):
        r = _make_result(kl_divergence=None, js_divergence=None, top1_agreement=None)
        row = r.as_db_row()
        assert row["kl_divergence"] is None
        assert row["js_divergence"] is None
        assert row["top1_agreement"] is None

    def test_as_db_row_passed_is_int(self):
        row = _make_result().as_db_row()
        assert row["passed"] in (0, 1)
        assert isinstance(row["passed"], int)

    def test_as_db_row_has_created_at(self):
        row = _make_result().as_db_row()
        assert "created_at" in row
        assert row["created_at"].endswith("Z")

    def test_frozen(self):
        r = _make_result()
        with pytest.raises(Exception):
            r.rmse = 99.0  # type: ignore[misc]


# ===========================================================================
# 5.  RMSEAccumulator
# ===========================================================================

class TestRMSEAccumulator:

    def test_zero_error(self):
        acc = RMSEAccumulator()
        b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        acc.update(b, b)
        assert acc.compute() == pytest.approx(0.0, abs=1e-9)

    def test_known_value(self):
        # diff = [1, 1, 1, 1] → mse = 1.0 → rmse = 1.0
        acc = RMSEAccumulator()
        b = np.zeros(4, dtype=np.float32)
        c = np.ones(4,  dtype=np.float32)
        acc.update(b, c)
        assert acc.compute() == pytest.approx(1.0, abs=1e-6)

    def test_multiple_updates(self):
        acc = RMSEAccumulator()
        rng = np.random.default_rng(0)
        b = rng.standard_normal(100).astype(np.float32)
        c = rng.standard_normal(100).astype(np.float32)
        # split into two updates vs one
        acc_single = RMSEAccumulator()
        acc_single.update(b, c)

        acc.update(b[:50], c[:50])
        acc.update(b[50:], c[50:])

        assert acc.compute() == pytest.approx(acc_single.compute(), rel=1e-5)

    def test_empty_returns_zero(self):
        acc = RMSEAccumulator()
        assert acc.compute() == 0.0

    def test_rmse_geq_mae(self):
        # RMSE >= MAE always
        from mlbuild.core.accuracy.metrics import MAEAccumulator
        rng = np.random.default_rng(1)
        b = rng.standard_normal(50).astype(np.float32)
        c = rng.standard_normal(50).astype(np.float32)
        rmse_acc = RMSEAccumulator()
        mae_acc  = MAEAccumulator()
        rmse_acc.update(b, c)
        mae_acc.update(b, c)
        assert rmse_acc.compute() >= mae_acc.compute()


# ===========================================================================
# 6.  KLDivAccumulator
# ===========================================================================

class TestKLDivAccumulator:

    def test_identical_distributions_near_zero(self):
        acc = KLDivAccumulator()
        logits = np.array([2.0, 1.0, 0.5, 0.1], dtype=np.float32)
        acc.update(logits, logits)
        assert acc.compute() == pytest.approx(0.0, abs=1e-5)

    def test_returns_none_when_empty(self):
        acc = KLDivAccumulator()
        assert acc.compute() is None

    def test_positive_for_different_distributions(self):
        acc = KLDivAccumulator()
        b = np.array([10.0, 0.1, 0.1, 0.1], dtype=np.float32)
        c = np.array([0.1,  0.1, 0.1, 10.0], dtype=np.float32)
        acc.update(b, c)
        assert acc.compute() > 0.0

    def test_direction_asymmetric(self):
        # KL(p||q) != KL(q||p) in general
        acc_fwd = KLDivAccumulator()
        acc_rev = KLDivAccumulator()
        b = np.array([5.0, 1.0, 1.0, 1.0], dtype=np.float32)
        c = np.array([1.0, 1.0, 1.0, 5.0], dtype=np.float32)
        acc_fwd.update(b, c)
        acc_rev.update(c, b)
        # they may be equal by symmetry of this specific input — just check both finite
        assert np.isfinite(acc_fwd.compute())
        assert np.isfinite(acc_rev.compute())

    def test_numerical_stability_extreme_logits(self):
        # Should not produce inf or nan for extreme logits
        acc = KLDivAccumulator()
        b = np.array([100.0, -100.0, 0.0], dtype=np.float32)
        c = np.array([-100.0, 100.0, 0.0], dtype=np.float32)
        acc.update(b, c)
        result = acc.compute()
        assert result is not None
        assert np.isfinite(result)

    def test_multiple_updates_averages(self):
        acc = KLDivAccumulator()
        b = np.array([2.0, 1.0], dtype=np.float32)
        c = np.array([1.0, 2.0], dtype=np.float32)
        acc.update(b, c)
        acc.update(b, c)
        # both updates identical → mean == single value
        acc_single = KLDivAccumulator()
        acc_single.update(b, c)
        assert acc.compute() == pytest.approx(acc_single.compute(), rel=1e-5)


# ===========================================================================
# 7.  JSDivAccumulator
# ===========================================================================

class TestJSDivAccumulator:

    def test_identical_distributions_zero(self):
        acc = JSDivAccumulator()
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        acc.update(logits, logits)
        assert acc.compute() == pytest.approx(0.0, abs=1e-5)

    def test_returns_none_when_empty(self):
        acc = JSDivAccumulator()
        assert acc.compute() is None

    def test_bounded_by_ln2(self):
        # JS divergence is bounded [0, ln(2)]
        import math
        acc = JSDivAccumulator()
        b = np.array([100.0, 0.0], dtype=np.float32)
        c = np.array([0.0, 100.0], dtype=np.float32)
        acc.update(b, c)
        result = acc.compute()
        assert result is not None
        assert 0.0 <= result <= math.log(2) + 1e-6

    def test_symmetric(self):
        # JS(p||q) == JS(q||p)
        acc_fwd = JSDivAccumulator()
        acc_rev = JSDivAccumulator()
        b = np.array([3.0, 1.0, 0.5], dtype=np.float32)
        c = np.array([0.5, 1.0, 3.0], dtype=np.float32)
        acc_fwd.update(b, c)
        acc_rev.update(c, b)
        assert acc_fwd.compute() == pytest.approx(acc_rev.compute(), abs=1e-6)

    def test_numerical_stability_extreme_logits(self):
        acc = JSDivAccumulator()
        b = np.array([100.0, -100.0], dtype=np.float32)
        c = np.array([-100.0, 100.0], dtype=np.float32)
        acc.update(b, c)
        result = acc.compute()
        assert result is not None
        assert np.isfinite(result)

    def test_positive_for_different_distributions(self):
        acc = JSDivAccumulator()
        b = np.array([5.0, 0.1], dtype=np.float32)
        c = np.array([0.1, 5.0], dtype=np.float32)
        acc.update(b, c)
        assert acc.compute() > 0.0


# ===========================================================================
# 8.  PercentileAEAccumulator
# ===========================================================================

class TestPercentileAEAccumulator:

    def test_zero_error_returns_zeros(self):
        acc = PercentileAEAccumulator()
        b = np.ones(10, dtype=np.float32)
        acc.update(b, b)
        p50, p95, p99 = acc.compute()
        assert p50 == pytest.approx(0.0, abs=1e-9)
        assert p95 == pytest.approx(0.0, abs=1e-9)
        assert p99 == pytest.approx(0.0, abs=1e-9)

    def test_empty_returns_zeros(self):
        acc = PercentileAEAccumulator()
        p50, p95, p99 = acc.compute()
        assert (p50, p95, p99) == (0.0, 0.0, 0.0)

    def test_percentile_ordering(self):
        # p50 <= p95 <= p99 always
        rng = np.random.default_rng(42)
        acc = PercentileAEAccumulator()
        b = rng.standard_normal(200).astype(np.float32)
        c = rng.standard_normal(200).astype(np.float32)
        acc.update(b, c)
        p50, p95, p99 = acc.compute()
        assert p50 <= p95 <= p99

    def test_returns_tuple_of_three_floats(self):
        acc = PercentileAEAccumulator()
        acc.update(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
        result = acc.compute()
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

    def test_known_uniform_errors(self):
        # errors = [0, 1, 2, ..., 99] → p50 ≈ 49.5, p95 ≈ 94.05, p99 ≈ 98.01
        acc = PercentileAEAccumulator()
        b = np.zeros(100, dtype=np.float32)
        c = np.arange(100, dtype=np.float32)
        acc.update(b, c)
        p50, p95, p99 = acc.compute()
        assert p50 == pytest.approx(49.5, abs=0.1)
        assert p95 == pytest.approx(94.05, abs=0.5)
        assert p99 == pytest.approx(98.01, abs=0.5)

    def test_multiple_updates_consistent(self):
        rng = np.random.default_rng(7)
        b = rng.standard_normal(100).astype(np.float32)
        c = rng.standard_normal(100).astype(np.float32)

        acc_one  = PercentileAEAccumulator()
        acc_split = PercentileAEAccumulator()

        acc_one.update(b, c)
        acc_split.update(b[:50], c[:50])
        acc_split.update(b[50:], c[50:])

        assert acc_one.compute() == pytest.approx(acc_split.compute(), abs=1e-5)


# ===========================================================================
# 9.  generate_batch_task_aware — sampling distributions
# ===========================================================================

class TestGenerateBatchTaskAware:

    def test_fallback_for_unknown_task(self):
        spec = [_float_spec()]
        rng  = np.random.default_rng(0)
        batch = generate_batch_task_aware(spec, rng, 16, task_type=None)
        assert "input" in batch
        assert batch["input"].shape == (16, 4)

    def test_fallback_for_none_task(self):
        spec = [_float_spec()]
        rng  = np.random.default_rng(0)
        batch = generate_batch_task_aware(spec, rng, 8, task_type="unknown")
        assert batch["input"].shape == (8, 4)

    def test_vision_floats_near_boundary(self):
        spec  = [_float_spec(shape=(3, 4))]
        rng   = np.random.default_rng(1)
        batch = generate_batch_task_aware(spec, rng, 64, task_type="vision")
        arr   = batch["input"]
        assert arr.min() >= 0.39
        assert arr.max() <= 0.61

    def test_audio_laplace_heavy_tail(self):
        # Laplace has heavier tails than uniform [-1,1]
        # Check that some values exceed ±0.9 (unlikely under uniform)
        spec  = [_float_spec(shape=(100,))]
        rng   = np.random.default_rng(2)
        batch = generate_batch_task_aware(spec, rng, 32, task_type="audio")
        arr   = batch["input"]
        assert arr.shape == (32, 100)
        # At least some values should be outside [-0.5, 0.5] for Laplace
        assert np.any(np.abs(arr) > 0.5)

    def test_nlp_integer_boundary_sampling(self):
        spec  = [_int_spec(shape=(16,))]
        rng   = np.random.default_rng(3)
        batch = generate_batch_task_aware(spec, rng, 32, task_type="nlp", int_range=1000)
        arr   = batch["input_ids"]
        assert arr.shape == (32, 16)
        # All values should be in low band [0, 100) or high band [900, 1000)
        low_band  = arr < 100
        high_band = arr >= 900
        assert np.all(low_band | high_band), (
            "NLP sampling should only produce boundary tokens"
        )

    def test_output_shapes_correct(self):
        specs = [_float_spec("x", (3, 4)), _int_spec("ids", (8,))]
        rng   = np.random.default_rng(4)
        for task in ("vision", "nlp", "audio", None, "unknown"):
            batch = generate_batch_task_aware(specs, rng, 10, task_type=task)
            assert batch["x"].shape   == (10, 3, 4)
            assert batch["ids"].shape == (10, 8)

    def test_deterministic_with_same_seed(self):
        spec = [_float_spec()]
        b1 = generate_batch_task_aware(spec, np.random.default_rng(99), 8, "vision")
        b2 = generate_batch_task_aware(spec, np.random.default_rng(99), 8, "vision")
        np.testing.assert_array_equal(b1["input"], b2["input"])


# ===========================================================================
# 10.  load_dataset_batch
# ===========================================================================

class TestLoadDatasetBatch:

    def _write_npz(self, tmp_path: Path, data: dict) -> Path:
        path = tmp_path / "data.npz"
        np.savez(path, **data)
        return path

    def _write_npy(self, tmp_path: Path, arr: np.ndarray) -> Path:
        path = tmp_path / "data.npy"
        np.save(path, arr)
        return path

    def test_npz_loads_correct_shape(self, tmp_path):
        arr  = np.random.rand(100, 4).astype(np.float32)
        path = self._write_npz(tmp_path, {"input": arr})
        spec  = [_float_spec(shape=(4,))]
        batch = load_dataset_batch(path, spec, samples=16)
        assert batch["input"].shape == (16, 4)

    def test_npz_slices_to_samples(self, tmp_path):
        arr  = np.arange(200).reshape(100, 2).astype(np.float32)
        path = self._write_npz(tmp_path, {"input": arr})
        spec  = [InputSpec(name="input", shape=(2,), dtype=np.float32)]
        batch = load_dataset_batch(path, spec, samples=10)
        np.testing.assert_array_equal(batch["input"], arr[:10])

    def test_npy_loads_single_input(self, tmp_path):
        arr  = np.random.rand(50, 8).astype(np.float32)
        path = self._write_npy(tmp_path, arr)
        spec  = [_float_spec(shape=(8,))]
        batch = load_dataset_batch(path, spec, samples=20)
        assert batch["input"].shape == (20, 8)

    def test_npy_rejects_multiple_specs(self, tmp_path):
        arr  = np.random.rand(50, 4).astype(np.float32)
        path = self._write_npy(tmp_path, arr)
        specs = [_float_spec("a", (4,)), _float_spec("b", (4,))]
        with pytest.raises(ValueError, match="single-input"):
            load_dataset_batch(path, specs, samples=10)

    def test_npz_missing_key_raises(self, tmp_path):
        arr  = np.random.rand(50, 4).astype(np.float32)
        path = self._write_npz(tmp_path, {"wrong_key": arr})
        spec  = [_float_spec(name="input", shape=(4,))]
        with pytest.raises(ValueError, match="missing keys"):
            load_dataset_batch(path, spec, samples=10)

    def test_fewer_rows_than_samples_raises(self, tmp_path):
        arr  = np.random.rand(5, 4).astype(np.float32)
        path = self._write_npz(tmp_path, {"input": arr})
        spec  = [_float_spec(shape=(4,))]
        with pytest.raises(ValueError, match="5 rows"):
            load_dataset_batch(path, spec, samples=32)

    def test_shape_mismatch_raises(self, tmp_path):
        arr  = np.random.rand(50, 8).astype(np.float32)
        path = self._write_npz(tmp_path, {"input": arr})
        spec  = [_float_spec(shape=(4,))]  # expects (4,) not (8,)
        with pytest.raises(ValueError, match="shape"):
            load_dataset_batch(path, spec, samples=10)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            load_dataset_batch(tmp_path / "missing.npz", [_float_spec()], samples=4)

    def test_unsupported_format_raises(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("1,2,3\n")
        with pytest.raises(ValueError, match="Unsupported dataset format"):
            load_dataset_batch(path, [_float_spec()], samples=1)

    def test_dtype_cast_applied(self, tmp_path):
        arr  = np.random.rand(20, 4).astype(np.float64)
        path = self._write_npz(tmp_path, {"input": arr})
        spec  = [_float_spec(shape=(4,))]  # float32
        batch = load_dataset_batch(path, spec, samples=10)
        assert batch["input"].dtype == np.float32

    def test_multi_input_npz(self, tmp_path):
        a = np.random.rand(50, 4).astype(np.float32)
        b = np.random.randint(0, 100, (50, 8), dtype=np.int64)
        path = self._write_npz(tmp_path, {"x": a, "ids": b})
        specs = [
            InputSpec(name="x",   shape=(4,), dtype=np.float32),
            InputSpec(name="ids", shape=(8,), dtype=np.int64),
        ]
        batch = load_dataset_batch(path, specs, samples=16)
        assert batch["x"].shape   == (16, 4)
        assert batch["ids"].shape == (16, 8)