"""Everything a plan needs for one (dataset, label scheme): vectors, labels,
an attribute index (label -> ids), and the vector indexes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data import datasets
from ..index import HNSWIndex, IVFIndex, PartitionedIVF

ROOT = Path(__file__).resolve().parents[2]


class AttributeIndex:
    """Inverted index on the label column: ids per label, sorted ascending."""

    def __init__(self, labels: np.ndarray):
        self.n = labels.size
        self.order = np.argsort(labels, kind="stable").astype(np.int64)
        self.offsets = np.zeros(int(labels.max()) + 2, np.int64)
        self.offsets[1:] = np.cumsum(np.bincount(labels))

    def eligible(self, label: int) -> np.ndarray:
        if label < 0 or label + 1 >= self.offsets.size:
            return self.order[:0]
        return self.order[self.offsets[label]:self.offsets[label + 1]]

    def count(self, label: int) -> int:
        return int(self.eligible(label).size)

    def selectivity(self, label: int) -> float:
        return self.count(label) / self.n


@dataclass
class FilterContext:
    base: np.ndarray
    labels: np.ndarray
    attr: AttributeIndex
    ivf: IVFIndex | None = None
    ivf_list_labels: np.ndarray | None = None
    pivf: PartitionedIVF | None = None
    hnsw: HNSWIndex | None = None

    @property
    def n(self) -> int:
        return self.base.shape[0]

    @classmethod
    def from_parts(cls, base, labels, ivf=None, hnsw=None, partitioned=True) -> "FilterContext":
        labels = np.ascontiguousarray(labels, np.int32)
        return cls(
            base=base, labels=labels, attr=AttributeIndex(labels), ivf=ivf, hnsw=hnsw,
            ivf_list_labels=ivf.list_labels(labels) if ivf is not None else None,
            pivf=PartitionedIVF.from_ivf(ivf, labels) if (ivf is not None and partitioned) else None,
        )

    @classmethod
    def load(cls, dataset: str, scheme: str, ivf_file: str | None = None, hnsw_file: str | None = None,
             partitioned: bool = True) -> "FilterContext":
        base, _, _ = datasets.load(dataset, mmap=False)
        labels = np.load(datasets.DATA_ROOT / dataset / scheme / "labels.npy")
        idx = ROOT / "indexes" / dataset
        ivf_path = idx / ivf_file if ivf_file else next(idx.glob("ivf_*.npz"), None)
        hnsw_path = idx / hnsw_file if hnsw_file else next(idx.glob("hnsw_*.npz"), None)
        ivf = IVFIndex.load(ivf_path, base) if ivf_path else None
        hnsw = HNSWIndex.load(hnsw_path, base) if hnsw_path else None
        return cls.from_parts(base, labels, ivf, hnsw, partitioned)
