"""
Calibration data loader for static INT8 quantization.

Supported sources
-----------------
• Directory of images
• Directory of .npy files
• Single .npz file

Output
------
Iterator[dict[str, np.ndarray]]

Each sample is returned as:

    {input_name: np.ndarray(batch=1, ...)}

This format is directly compatible with CoreML PostTrainingQuantizer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class CalibrationError(ValueError):
    pass


class CalibrationLoader:
    """
    Calibration loader.

    Parameters
    ----------
    source:
        Directory of images, directory of .npy files, or .npz file.
    input_name:
        Model input name.
    input_shape:
        Full model input shape including batch dimension.
    layout:
        "nchw" or "nhwc" (required for image inputs).
    max_samples:
        Maximum samples to load.
    preprocess:
        Optional preprocessing function applied to image arrays.
    npz_key:
        Key to read from npz file (required if multiple arrays exist).
    """

    def __init__(
        self,
        source: Path,
        input_name: str,
        input_shape: tuple[int, ...],
        layout: Optional[str] = None,
        max_samples: int = 200,
        preprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        npz_key: Optional[str] = None,
    ):

        if max_samples <= 0:
            raise CalibrationError("max_samples must be > 0")

        self.source = Path(source)
        self.input_name = input_name
        self.input_shape = input_shape
        self.layout = layout
        self.max_samples = max_samples
        self.preprocess = preprocess
        self.npz_key = npz_key

        self._kind = self._detect_kind()

        # metadata caches
        self._npz_size: Optional[int] = None
        self._npz_key: Optional[str] = None

        logger.info(
            "calibration loader initialized "
            "source=%s kind=%s shape=%s max_samples=%d",
            self.source,
            self._kind,
            self.input_shape,
            self.max_samples,
        )

    # --------------------------------------------------
    # Source detection
    # --------------------------------------------------

    def _detect_kind(self) -> str:

        if self.source.is_file() and self.source.suffix.lower() == ".npz":
            return "npz"

        if self.source.is_dir():

            for f in self.source.iterdir():

                suffix = f.suffix.lower()

                if suffix in IMAGE_EXTENSIONS:
                    return "images"

                if suffix == ".npy":
                    return "npy"

            raise CalibrationError(
                f"Directory {self.source} contains no images or .npy files"
            )

        raise CalibrationError(
            f"Calibration source must be a directory or .npz file: {self.source}"
        )

    # --------------------------------------------------
    # Metadata
    # --------------------------------------------------

    def __len__(self) -> int:

        if self._kind == "npz":

            if self._npz_size is None:

                with np.load(self.source) as data:

                    if self.npz_key is None:

                        if len(data.keys()) != 1:
                            raise CalibrationError(
                                "npz contains multiple arrays — specify npz_key"
                            )

                        self._npz_key = next(iter(data.keys()))

                    else:

                        if self.npz_key not in data:
                            raise CalibrationError(
                                f"npz key '{self.npz_key}' not found"
                            )

                        self._npz_key = self.npz_key

                    self._npz_size = data[self._npz_key].shape[0]

            return min(self._npz_size, self.max_samples)

        if self._kind == "images":

            count = 0
            for f in self.source.iterdir():
                if f.suffix.lower() in IMAGE_EXTENSIONS:
                    count += 1
                    if count >= self.max_samples:
                        break

            return count

        if self._kind == "npy":

            count = 0
            for _ in self.source.glob("*.npy"):
                count += 1
                if count >= self.max_samples:
                    break

            return count

        return 0

    # --------------------------------------------------
    # Sample iterator
    # --------------------------------------------------

    def iter_samples(self) -> Iterator[dict[str, np.ndarray]]:

        logger.info(
            "loading calibration samples "
            "source=%s kind=%s max_samples=%d",
            self.source,
            self._kind,
            self.max_samples,
        )

        if self._kind == "npz":
            yield from self._load_npz()

        elif self._kind == "images":
            yield from self._load_images()

        elif self._kind == "npy":
            yield from self._load_npy()

    # --------------------------------------------------
    # NPZ loader
    # --------------------------------------------------

    def _load_npz(self) -> Iterator[dict[str, np.ndarray]]:

        with np.load(self.source) as data:

            key = self._npz_key or self.npz_key

            if key is None:
                if len(data.keys()) != 1:
                    raise CalibrationError(
                        "npz contains multiple arrays — specify npz_key"
                    )
                key = next(iter(data.keys()))

            arr = data[key]

            logger.info(
                "npz calibration key=%s shape=%s",
                key,
                arr.shape,
            )

            for i in range(min(arr.shape[0], self.max_samples)):

                sample = arr[i]

                sample = self._ensure_batch(sample)

                yield {self.input_name: sample}

    # --------------------------------------------------
    # NPY loader
    # --------------------------------------------------

    def _load_npy(self) -> Iterator[dict[str, np.ndarray]]:

        count = 0

        for f in sorted(self.source.glob("*.npy")):

            arr = np.load(f)

            arr = self._ensure_batch(arr)

            yield {self.input_name: arr}

            count += 1

            if count >= self.max_samples:
                break

    # --------------------------------------------------
    # Image loader
    # --------------------------------------------------

    def _load_images(self) -> Iterator[dict[str, np.ndarray]]:

        if self.layout not in {"nchw", "nhwc"}:
            raise CalibrationError(
                "layout must be specified for image inputs: 'nchw' or 'nhwc'"
            )

        try:
            from PIL import Image
        except ImportError as exc:
            raise CalibrationError(
                "Pillow is required for image calibration"
            ) from exc

        shape = self.input_shape

        if len(shape) != 4:
            raise CalibrationError(
                f"Image inputs must be 4D but got shape {shape}"
            )

        if self.layout == "nchw":
            _, c, h, w = shape
        else:
            _, h, w, c = shape

        count = 0

        for f in sorted(self.source.iterdir()):

            if f.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            with Image.open(f) as img:

                img = img.convert("RGB" if c == 3 else "L")

                img = img.resize((w, h), Image.BILINEAR)

                arr = np.asarray(img, dtype=np.float32)

            if self.preprocess is not None:
                arr = self.preprocess(arr)

            if self.layout == "nchw":
                arr = arr.transpose(2, 0, 1)

            arr = arr[np.newaxis, ...]

            arr = self._validate_shape(arr)

            yield {self.input_name: arr}

            count += 1

            if count >= self.max_samples:
                break

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _ensure_batch(self, arr: np.ndarray) -> np.ndarray:

        expected_ndim = len(self.input_shape)

        if arr.ndim == expected_ndim - 1:
            arr = arr[np.newaxis, ...]

        if arr.ndim != expected_ndim:
            raise CalibrationError(
                f"Invalid ndim {arr.ndim}, expected {expected_ndim}"
            )

        return self._validate_shape(arr)

    def _validate_shape(self, arr: np.ndarray) -> np.ndarray:

        if len(arr.shape) != len(self.input_shape):
            raise CalibrationError(
                f"Shape mismatch {arr.shape} vs {self.input_shape}"
            )

        for i, (actual, expected) in enumerate(
            zip(arr.shape, self.input_shape)
        ):
            if expected is not None and actual != expected:
                raise CalibrationError(
                    f"Shape mismatch at dim {i}: {actual} vs {expected}"
                )

        return arr

    # --------------------------------------------------
    # Materialization (explicit)
    # --------------------------------------------------

    def as_list(
        self,
        max_samples: Optional[int] = None,
    ) -> list[dict[str, np.ndarray]]:

        limit = max_samples or self.max_samples

        samples = []

        for i, sample in enumerate(self.iter_samples()):

            samples.append(sample)

            if i + 1 >= limit:
                break

        logger.info(
            "calibration loaded samples=%d source=%s",
            len(samples),
            self.source,
        )

        return samples