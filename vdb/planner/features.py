"""Query features for the planner.

Global selectivity comes from the attribute index. Local selectivity comes from
a coarse histogram: the space is cut into a few hundred k-means cells and we
store how many vectors of each label fall in each cell. For a query we look at
its nearest cells and measure how common the label is there. If the label is
rare near the query but common overall (correlated labels, query outside the
label's region), index-based plans will struggle; the ratio exposes that.
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = ("log_sel", "log_nelig", "log_local4", "log_local16", "log_ratio4", "log_ratio16")


class CoarseStats:
    def __init__(self, centroids: np.ndarray, counts: np.ndarray):
        self.centroids = np.ascontiguousarray(centroids, np.float32)  # (C, d)
        self.cnorm = (self.centroids.astype(np.float64) ** 2).sum(1)
        self.counts = np.ascontiguousarray(counts, np.float64)        # (C, L)
        self.sizes = self.counts.sum(1)
        self.label_totals = self.counts.sum(0)
        self.n = float(self.sizes.sum())

    @staticmethod
    def build_centroids(base: np.ndarray, n_cells: int = 256, sample: int = 100_000, seed: int = 0):
        import faiss

        rng = np.random.default_rng(seed)
        x = base[np.sort(rng.choice(base.shape[0], min(sample, base.shape[0]), replace=False))]
        km = faiss.Kmeans(base.shape[1], n_cells, niter=20, seed=seed, verbose=False)
        km.train(np.ascontiguousarray(x, np.float32))
        return km.centroids

    @classmethod
    def build(cls, base: np.ndarray, labels: np.ndarray, centroids: np.ndarray | None = None,
              n_cells: int = 256, seed: int = 0) -> "CoarseStats":
        import faiss

        if centroids is None:
            centroids = cls.build_centroids(base, n_cells, seed=seed)
        index = faiss.IndexFlatL2(centroids.shape[1])
        index.add(np.ascontiguousarray(centroids, np.float32))
        cell = np.empty(base.shape[0], np.int64)
        for s in range(0, base.shape[0], 200_000):
            _, a = index.search(np.ascontiguousarray(base[s:s + 200_000], np.float32), 1)
            cell[s:s + 200_000] = a[:, 0]
        L = int(labels.max()) + 1
        counts = np.bincount(cell * L + labels, minlength=centroids.shape[0] * L).reshape(-1, L)
        return cls(centroids, counts)

    def _row(self, d2: np.ndarray, label: int) -> np.ndarray:
        top16 = np.argpartition(d2, 16)[:16]
        top16 = top16[np.argsort(d2[top16])]
        top4 = top16[:4]
        tot = self.label_totals[label] if label < self.label_totals.size else 0.0
        sel = max(tot, 0.5) / self.n
        lc = self.counts[:, label] if label < self.counts.shape[1] else np.zeros(self.counts.shape[0])
        l4 = (lc[top4].sum() + 0.5) / (self.sizes[top4].sum() + 1.0)
        l16 = (lc[top16].sum() + 0.5) / (self.sizes[top16].sum() + 1.0)
        return np.array([np.log10(sel), np.log10(max(tot, 0.5)), np.log10(l4), np.log10(l16),
                         np.log10(l4 / sel), np.log10(l16 / sel)])

    def features(self, q: np.ndarray, label: int) -> np.ndarray:
        d2 = self.cnorm - 2.0 * (self.centroids @ q)
        return self._row(d2, label)

    def features_batch(self, Q: np.ndarray, labels: np.ndarray) -> np.ndarray:
        d2 = self.cnorm[None, :] - 2.0 * (Q.astype(np.float32) @ self.centroids.T)
        return np.stack([self._row(d2[i], int(labels[i])) for i in range(Q.shape[0])])
