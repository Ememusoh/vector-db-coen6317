Fastest single plan reaching recall@10 ≥ 0.95, per selectivity (from the budget sweeps).

| Dataset | Labels | Selectivity | Fastest plan at recall ≥ 0.95 | Budget | p50 ms | Pre-filter p50 ms |
|---|---|---|---|---|---|---|
| Deep1M | independent | 50% | HNSW in-search | 32 | 0.126 | 6.783 |
| Deep1M | independent | 10% | Partitioned IVF in-search | 64 | 0.082 | 5.869 |
| Deep1M | independent | 1% | Partitioned IVF in-search | 256 | 0.092 | 0.131 |
| Deep1M | independent | 0.1% | Pre-filter (exact) | – | 0.014 | 0.014 |
| Deep1M | correlated | 50% | IVF post-filter | 32 | 0.324 | 6.818 |
| Deep1M | correlated | 10% | Partitioned IVF in-search | 1024 | 0.358 | 5.847 |
| Deep1M | correlated | 1% | Pre-filter (exact) | – | 0.131 | 0.131 |
| Deep1M | correlated | 0.1% | Pre-filter (exact) | – | 0.014 | 0.014 |
| GIST-500K | independent | 50% | HNSW in-search | 128 | 1.980 | 31.786 |
| GIST-500K | independent | 10% | Partitioned IVF in-search | 128 | 0.526 | 14.115 |
| GIST-500K | independent | 1% | Partitioned IVF in-search | 256 | 0.258 | 0.603 |
| GIST-500K | independent | 0.1% | Pre-filter (exact) | – | 0.037 | 0.037 |
| GIST-500K | correlated | 50% | HNSW in-search | 128 | 2.653 | 31.019 |
| GIST-500K | correlated | 10% | Partitioned IVF in-search | 512 | 2.198 | 13.823 |
| GIST-500K | correlated | 1% | Partitioned IVF in-search | 512 | 0.388 | 0.594 |
| GIST-500K | correlated | 0.1% | Pre-filter (exact) | – | 0.037 | 0.037 |
| GloVe-100 | independent | 50% | IVF post-filter | 256 | 0.790 | 7.014 |
| GloVe-100 | independent | 10% | Partitioned IVF in-search | 1024 | 0.545 | 6.328 |
| GloVe-100 | independent | 1% | Pre-filter (exact) | – | 0.143 | 0.143 |
| GloVe-100 | independent | 0.1% | Pre-filter (exact) | – | 0.016 | 0.016 |
| GloVe-100 | correlated | 50% | HNSW post-filter | 256 | 1.382 | 7.731 |
| GloVe-100 | correlated | 10% | Partitioned IVF in-search | 1024 | 0.650 | 6.222 |
| GloVe-100 | correlated | 1% | Pre-filter (exact) | – | 0.134 | 0.134 |
| GloVe-100 | correlated | 0.1% | Pre-filter (exact) | – | 0.015 | 0.015 |
| SIFT1M | independent | 50% | HNSW in-search | 32 | 0.153 | 14.749 |
| SIFT1M | independent | 10% | Partitioned IVF in-search | 128 | 0.144 | 7.856 |
| SIFT1M | independent | 1% | Partitioned IVF in-search | 256 | 0.103 | 0.173 |
| SIFT1M | independent | 0.1% | Pre-filter (exact) | – | 0.017 | 0.017 |
| SIFT1M | correlated | 50% | HNSW in-search | 64 | 0.206 | 12.437 |
| SIFT1M | correlated | 10% | Partitioned IVF in-search | 1024 | 0.404 | 7.049 |
| SIFT1M | correlated | 1% | Pre-filter (exact) | – | 0.160 | 0.160 |
| SIFT1M | correlated | 0.1% | Pre-filter (exact) | – | 0.017 | 0.017 |
| YFCC | native | 3.7% | IVF in-search | 64 | 0.209 | 5.596 |
| YFCC | native | 0.42% | IVF in-search | 256 | 0.189 | 0.436 |
| YFCC | native | 0.045% | Pre-filter (exact) | – | 0.050 | 0.050 |
| YFCC | native | 0.0016% | Pre-filter (exact) | – | 0.004 | 0.004 |
