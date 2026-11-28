"""Dataset registry, download and loading.

Every dataset is turned into the same on-disk layout under datasets/<name>/:
    base.npy      float32 (N, d) database vectors
    queries.npy   float32 (Q, d) query vectors
    meta.json     metric, sizes, source
Angular datasets are L2-normalised so every index can use L2 distance.
"""
from __future__ import annotations

import json
import subprocess
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import io

DATA_ROOT = Path(__file__).resolve().parents[2] / "datasets"


@dataclass
class DatasetSpec:
    name: str
    url: str
    metric: str  # "l2" or "angular"
    loader: Callable[[Path, int | None], tuple[np.ndarray, np.ndarray]]
    default_n: int | None = None  # base subset size (None = all)
    notes: str = ""
    archive: str = field(init=False)

    def __post_init__(self):
        self.archive = self.url.rsplit("/", 1)[-1]


def _texmex_loader(prefix: str, reader=io.read_fvecs):
    def load(raw_dir: Path, n: int | None):
        d = raw_dir / prefix
        base = reader(d / f"{prefix}_base.{_ext(reader)}", n).astype(np.float32)
        queries = reader(d / f"{prefix}_query.{_ext(reader)}").astype(np.float32)
        return base, queries

    return load


def _ext(reader) -> str:
    return {io.read_fvecs: "fvecs", io.read_bvecs: "bvecs"}[reader]


def _hdf5_loader(raw_dir: Path, n: int | None, archive: str):
    path = raw_dir / archive
    base = io.read_hdf5(path, "train", n).astype(np.float32)
    queries = io.read_hdf5(path, "test").astype(np.float32)
    return base, queries


REGISTRY: dict[str, DatasetSpec] = {
    "sift": DatasetSpec(
        "sift", "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz", "l2",
        _texmex_loader("sift"), notes="1M x 128, 10k queries",
    ),
    "gist": DatasetSpec(
        "gist", "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz", "l2",
        _texmex_loader("gist"), default_n=500_000, notes="1M x 960, 1k queries; subset for memory",
    ),
    "glove": DatasetSpec(
        "glove", "http://ann-benchmarks.com/glove-100-angular.hdf5", "angular",
        lambda r, n: _hdf5_loader(r, n, "glove-100-angular.hdf5"),
        default_n=1_000_000, notes="1.18M x 100, 10k queries",
    ),
    "deep": DatasetSpec(
        "deep", "http://ann-benchmarks.com/deep-image-96-angular.hdf5", "angular",
        lambda r, n: _hdf5_loader(r, n, "deep-image-96-angular.hdf5"),
        default_n=1_000_000, notes="10M x 96, 10k queries; first 1M used",
    ),
}


NATIVE_LABELS = {"yfcc"}  # datasets with their own (multi-label) attributes; see vdb/data/yfcc.py


def schemes(name: str) -> tuple[str, ...]:
    """Attribute schemes evaluated for a dataset."""
    return ("native",) if name in NATIVE_LABELS else ("independent", "correlated")


def raw_dir(name: str) -> Path:
    return DATA_ROOT / name / "raw"


def download(name: str) -> Path:
    spec = REGISTRY[name]
    rd = raw_dir(name)
    rd.mkdir(parents=True, exist_ok=True)
    target = rd / spec.archive
    if not target.exists():
        tmp = target.with_suffix(target.suffix + ".part")
        subprocess.run(["curl", "-L", "--fail", "-C", "-", "-o", str(tmp), spec.url], check=True)
        tmp.rename(target)
    if spec.archive.endswith(".tar.gz") and not (rd / spec.name).exists():
        with tarfile.open(target) as tf:
            tf.extractall(rd, filter="data")
    return rd


def prepare(name: str, n: int | None = None) -> Path:
    """Download (if needed) and write base.npy / queries.npy / meta.json."""
    spec = REGISTRY[name]
    n = n if n is not None else spec.default_n
    out = DATA_ROOT / name
    rd = download(name)
    base, queries = spec.loader(rd, n)
    if spec.metric == "angular":
        base = _normalize(base)
        queries = _normalize(queries)
    np.save(out / "base.npy", np.ascontiguousarray(base, dtype=np.float32))
    np.save(out / "queries.npy", np.ascontiguousarray(queries, dtype=np.float32))
    meta = {
        "name": name, "metric": spec.metric, "distance": "l2",
        "n_base": int(base.shape[0]), "n_queries": int(queries.shape[0]),
        "dim": int(base.shape[1]), "source": spec.url,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def load(name: str, mmap: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    d = DATA_ROOT / name
    mode = "r" if mmap else None
    base = np.load(d / "base.npy", mmap_mode=mode)
    queries = np.load(d / "queries.npy")
    meta = json.loads((d / "meta.json").read_text())
    return base, queries, meta
