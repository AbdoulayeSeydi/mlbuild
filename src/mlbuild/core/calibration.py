"""
Production-grade calibration dataset management for INT8 quantization.

Design goals:
- Strict determinism (no global RNG mutation)
- Stable, versioned fingerprinting
- Schema-validated preprocessing
- Observable dataset statistics
- Future-proof hashing guarantees
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np


# ============================================================
# Preprocessing (Structured + Versioned)
# ============================================================

@dataclass(frozen=True)
class Normalize:
    mean: float
    std: float

@dataclass(frozen=True)
class Scale:
    factor: float

@dataclass(frozen=True)
class Clip:
    min: float
    max: float

@dataclass(frozen=True)
class PreprocessingConfig:
    normalize: Optional[Normalize] = None
    scale: Optional[Scale] = None
    clip: Optional[Clip] = None

    def to_dict(self):
        return asdict(self)


# ============================================================
# Calibration Config
# ============================================================

@dataclass(frozen=True)
class CalibrationConfig:
    sample_count: int
    input_shape: Tuple[int, ...]
    preprocessing: PreprocessingConfig
    seed: int = 42

    def to_dict(self):
        return {
            "sample_count": self.sample_count,
            "input_shape": list(self.input_shape),
            "preprocessing": self.preprocessing.to_dict(),
            "seed": self.seed,
        }


# ============================================================
# Dataset Summary (Observability)
# ============================================================

@dataclass(frozen=True)
class DatasetSummary:
    min: float
    max: float
    mean: float
    std: float

    @staticmethod
    def from_samples(samples: List[np.ndarray]) -> "DatasetSummary":
        stacked = np.concatenate([s.reshape(-1) for s in samples])
        return DatasetSummary(
            min=float(stacked.min()),
            max=float(stacked.max()),
            mean=float(stacked.mean()),
            std=float(stacked.std()),
        )


# ============================================================
# Fingerprint (Versioned + Strict)
# ============================================================

@dataclass(frozen=True)
class CalibrationFingerprint:
    fingerprint_version: int
    config_hash: str
    data_hash: str
    sample_count: int
    input_shape: Tuple[int, ...]
    summary: DatasetSummary

    def to_dict(self):
        return {
            "fingerprint_version": self.fingerprint_version,
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "sample_count": self.sample_count,
            "input_shape": list(self.input_shape),
            "summary": asdict(self.summary),
        }


# ============================================================
# Calibration Dataset
# ============================================================

class CalibrationDataset:

    FINGERPRINT_VERSION = 1

    def __init__(self, config: CalibrationConfig):
        self.config = config
        self._data: Optional[List[np.ndarray]] = None

    # --------------------------
    # Deterministic Synthetic Data
    # --------------------------
    def generate_synthetic(self) -> List[np.ndarray]:
        """
        Generate synthetic calibration data.

        WARNING:
        Synthetic calibration is NOT statistically meaningful
        and should NOT be used for production quantization.
        """
        rng = np.random.default_rng(self.config.seed)

        samples: List[np.ndarray] = []

        for _ in range(self.config.sample_count):
            sample = rng.random(self.config.input_shape, dtype=np.float32)
            sample = self._apply_preprocessing(sample)
            self._validate_sample(sample)
            samples.append(sample)

        self._data = samples
        return samples

    # --------------------------
    # Load from Directory (Flexible + Strict)
    # --------------------------
    def load_from_directory(self, data_dir: Path) -> List[np.ndarray]:
        data_dir = Path(data_dir)

        files = sorted(data_dir.glob("*.npy"))

        if len(files) < self.config.sample_count:
            raise ValueError(
                f"Expected at least {self.config.sample_count} samples, "
                f"found {len(files)}"
            )

        samples: List[np.ndarray] = []

        for path in files[: self.config.sample_count]:
            sample = np.load(path)

            if sample.shape != self.config.input_shape:
                raise ValueError(
                    f"{path.name} has shape {sample.shape}, "
                    f"expected {self.config.input_shape}"
                )

            sample = sample.astype(np.float32, copy=False)
            sample = self._apply_preprocessing(sample)
            self._validate_sample(sample)
            samples.append(sample)

        self._data = samples
        return samples

    # --------------------------
    # Preprocessing
    # --------------------------
    def _apply_preprocessing(self, sample: np.ndarray) -> np.ndarray:
        p = self.config.preprocessing

        if p.normalize:
            sample = (sample - p.normalize.mean) / p.normalize.std

        if p.scale:
            sample = sample * p.scale.factor

        if p.clip:
            sample = np.clip(sample, p.clip.min, p.clip.max)

        return sample

    # --------------------------
    # Validation
    # --------------------------
    def _validate_sample(self, sample: np.ndarray):
        if sample.dtype != np.float32:
            raise ValueError("Calibration sample must be float32")

        if not np.isfinite(sample).all():
            raise ValueError("Calibration sample contains NaN or Inf")

    # --------------------------
    # Fingerprinting
    # --------------------------
    def compute_fingerprint(self) -> CalibrationFingerprint:
        if self._data is None:
            raise ValueError("No calibration data loaded")

        # Config hash (sorted, deterministic)
        config_json = json.dumps(
            self.config.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        config_hash = hashlib.sha256(config_json.encode()).hexdigest()

        # Strict data hash
        data_hasher = hashlib.sha256()

        for index, sample in enumerate(self._data):
            data_hasher.update(str(index).encode())
            data_hasher.update(sample.dtype.str.encode())
            data_hasher.update(str(sample.shape).encode())
            data_hasher.update(sample.tobytes(order="C"))

        data_hash = data_hasher.hexdigest()

        summary = DatasetSummary.from_samples(self._data)

        return CalibrationFingerprint(
            fingerprint_version=self.FINGERPRINT_VERSION,
            config_hash=config_hash,
            data_hash=data_hash,
            sample_count=len(self._data),
            input_shape=self.config.input_shape,
            summary=summary,
        )

    # --------------------------
    # Accessor (Immutable Exposure)
    # --------------------------
    @property
    def data(self) -> Tuple[np.ndarray, ...]:
        if self._data is None:
            raise ValueError("Calibration data not initialized")
        return tuple(self._data)
