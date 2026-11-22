"""Download a dataset and build labels, workloads and ground truth.

Usage:
    python scripts/prepare_dataset.py sift
    python scripts/prepare_dataset.py glove --n 1000000 --schemes independent correlated

Output (datasets/<name>/):
    base.npy, queries.npy, meta.json
    <scheme>/labels.npy     int32 (N,)  attribute value per base vector
    <scheme>/workload.npz   query_ids, labels, selectivity, split
    <scheme>/gt.npz         ids (W, k), dists (W, k); -1 / inf = fewer than k eligible
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vdb.data import attributes, datasets, groundtruth, workload  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=sorted(datasets.REGISTRY))
    ap.add_argument("--n", type=int, default=None, help="base subset size")
    ap.add_argument("--schemes", nargs="+", default=list(attributes.SCHEMES))
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--max-queries", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t = time.time()
    out = datasets.prepare(args.name, args.n)
    base, queries, meta = datasets.load(args.name)
    print(f"[{args.name}] base {base.shape}, queries {queries.shape} ({time.time() - t:.0f}s)")

    for scheme in args.schemes:
        sd = out / scheme
        sd.mkdir(exist_ok=True)
        t = time.time()
        labels = attributes.make_labels(scheme, base, seed=args.seed)
        np.save(sd / "labels.npy", labels)
        _, targets = attributes.label_sizes(base.shape[0])
        wl = workload.build_workload(queries.shape[0], labels, targets,
                                     max_queries=args.max_queries, seed=args.seed)
        wl.save(sd / "workload.npz")
        ids, dists = groundtruth.filtered_knn(base, queries, labels, wl.query_ids, wl.labels, k=args.k)
        np.savez(sd / "gt.npz", ids=ids, dists=dists)
        sel = workload.label_selectivity(labels)[targets]
        info = {"scheme": scheme, "seed": args.seed, "k": args.k, "n_workload": len(wl),
                "target_labels": targets.tolist(), "selectivity": sel.tolist(),
                "n_labels": int(labels.max() + 1),
                "short_entries": int((ids[:, 9] < 0).sum())}
        (sd / "info.json").write_text(json.dumps(info, indent=2))
        print(f"  {scheme}: {len(wl)} filtered queries, selectivities "
              f"{[f'{s:.4f}' for s in sel]} ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
