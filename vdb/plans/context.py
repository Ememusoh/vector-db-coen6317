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

    @classmethod
    def from_postings(cls, order: np.ndarray, offsets: np.ndarray, n: int) -> "AttributeIndex":
        """Multi-label attributes: ids with predicate p are order[offsets[p]:offsets[p+1]]."""
        self = cls.__new__(cls)
        self.n, self.order, self.offsets = n, np.asarray(order, np.int64), np.asarray(offsets, np.int64)
        return self

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

    def prepare(self, label: int):
        """Called before each (untimed) query; single-label attributes need nothing."""

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
        idx = ROOT / "indexes" / dataset
        ivf_path = idx / ivf_file if ivf_file else next(idx.glob("ivf_*.npz"), None)
        hnsw_path = idx / hnsw_file if hnsw_file else next(idx.glob("hnsw_*.npz"), None)
        ivf = IVFIndex.load(ivf_path, base) if ivf_path else None
        hnsw = HNSWIndex.load(hnsw_path, base) if hnsw_path else None
        d = datasets.DATA_ROOT / dataset / scheme
        if (d / "postings.npz").exists():
            z = np.load(d / "postings.npz")
            return MultiLabelContext.from_postings(base, z["order"], z["offsets"], ivf, hnsw)
        return cls.from_parts(base, np.load(d / "labels.npy"), ivf, hnsw, partitioned)


class MultiLabelContext(FilterContext):
    """Multi-label attributes (a vector can carry many tags, e.g. YFCC).

    Plans test "labels[id] == label". Here a vector matches predicate p when p
    is among its tags, so before each query `prepare(p)` writes p into a
    per-vector buffer at the ids in p's posting list (and clears the previous
    predicate's ids): a bitmap of the predicate, as a filtered vector database
    would build before searching. This runs outside the timed region for every
    plan alike. Partitioning lists by label is not defined for overlapping
    labels, so there is no partitioned IVF.
    """

    @classmethod
    def from_postings(cls, base, order, offsets, ivf=None, hnsw=None) -> "MultiLabelContext":
        n = base.shape[0]
        self = cls(base=base, labels=np.full(n, -1, np.int32), attr=AttributeIndex.from_postings(order, offsets, n),
                   ivf=ivf, hnsw=hnsw, pivf=None,
                   ivf_list_labels=np.full(n, -1, np.int32) if ivf is not None else None)
        self._pos = np.empty(n, np.int64)  # position of each id in IVF list order
        if ivf is not None:
            self._pos[ivf.ids] = np.arange(ivf.ids.size)
        self._current = -1
        return self

    def prepare(self, label: int):
        if label == self._current:
            return
        if self._current >= 0:
            old = self.attr.eligible(self._current)
            self.labels[old] = -1
            if self.ivf_list_labels is not None:
                self.ivf_list_labels[self._pos[old]] = -1
        ids = self.attr.eligible(label)
        self.labels[ids] = label
        if self.ivf_list_labels is not None:
            self.ivf_list_labels[self._pos[ids]] = label
        self._current = label
