"""Readers for the vector file formats used by the benchmark datasets."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _read_vecs(path: str | Path, dtype: np.dtype, count: int | None = None) -> np.ndarray:
    """Read the TEXMEX *vecs format: each row is an int32 dim followed by dim values."""
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    dim = int(np.frombuffer(raw[:4].tobytes(), dtype=np.int32)[0])
    item = np.dtype(dtype).itemsize
    row_bytes = 4 + dim * item
    n = raw.size // row_bytes
    if count is not None:
        n = min(n, count)
    rows = raw[: n * row_bytes].reshape(n, row_bytes)
    return np.ascontiguousarray(rows[:, 4:]).view(dtype).reshape(n, dim)


def read_fvecs(path, count=None) -> np.ndarray:
    return _read_vecs(path, np.float32, count)


def read_ivecs(path, count=None) -> np.ndarray:
    return _read_vecs(path, np.int32, count)


def read_bvecs(path, count=None) -> np.ndarray:
    return _read_vecs(path, np.uint8, count)


def write_fvecs(path, x: np.ndarray) -> None:
    x = np.ascontiguousarray(x, dtype=np.float32)
    n, d = x.shape
    out = np.empty((n, d + 1), dtype=np.float32)
    out[:, 0] = np.array([d], dtype=np.int32).view(np.float32)[0]
    out[:, 1:] = x
    out.tofile(path)


def read_hdf5(path, key: str, count=None) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as f:
        ds = f[key]
        n = ds.shape[0] if count is None else min(count, ds.shape[0])
        return np.asarray(ds[:n])
