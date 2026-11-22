"""Filtered query workloads.

A workload is a list of (query_id, label) pairs. For synthetic attributes we
pair every query with each target label, so each query is evaluated at every
selectivity level. Queries are split into validation (planner fitting) and
test (held-out evaluation) by query id, so the same query vector never
appears in both.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Workload:
    query_ids: np.ndarray    # (W,) int32 index into queries
    labels: np.ndarray       # (W,) int32 filter value: attribute == label
    selectivity: np.ndarray  # (W,) float64 measured fraction of base passing
    split: np.ndarray        # (W,) uint8 0 = validation, 1 = test

    def __len__(self):
        return self.query_ids.size

    def subset(self, mask) -> "Workload":
        return Workload(self.query_ids[mask], self.labels[mask], self.selectivity[mask], self.split[mask])

    def save(self, path):
        np.savez(path, query_ids=self.query_ids, labels=self.labels,
                 selectivity=self.selectivity, split=self.split)

    @classmethod
    def load(cls, path) -> "Workload":
        z = np.load(path)
        return cls(z["query_ids"], z["labels"], z["selectivity"], z["split"])


def label_selectivity(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels)
    return counts / labels.size


def build_workload(n_queries: int, labels: np.ndarray, target_labels, val_fraction: float = 0.3,
                   max_queries: int | None = 1000, seed: int = 0) -> Workload:
    rng = np.random.default_rng(seed)
    qids = np.arange(n_queries)
    if max_queries is not None and n_queries > max_queries:
        qids = np.sort(rng.choice(n_queries, max_queries, replace=False))
    is_test = np.ones(qids.size, dtype=np.uint8)
    is_test[rng.permutation(qids.size)[: round(val_fraction * qids.size)]] = 0

    sel = label_selectivity(labels)
    target_labels = np.asarray(target_labels, dtype=np.int32)
    q = np.repeat(qids, target_labels.size).astype(np.int32)
    lab = np.tile(target_labels, qids.size)
    return Workload(q, lab, sel[lab], np.repeat(is_test, target_labels.size))
