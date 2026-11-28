Index build time and memory (IVF nlist ≈ 4√N, HNSW M=32, efConstruction=200). IVF stores its own copy of the vectors in list order; the HNSW graph is stored in addition to the raw vectors.

| Dataset | Vectors | Raw vectors MiB | IVF build s | IVF MiB | Part. IVF extra MiB | HNSW build s | HNSW graph MiB |
|---|---|---|---|---|---|---|---|
| SIFT1M | 1M × 128 | 488 | 19.3 | 498 | 1.38 | 55.8 | 260 |
| GIST-500K | 0.5M × 960 | 1831 | 11.7 | 1842 | 0.69 | 150.6 | 130 |
| GloVe-100 | 1M × 100 | 381 | 19.0 | 391 | 1.38 | 83.6 | 260 |
| Deep1M | 1M × 96 | 366 | 18.6 | 375 | 1.38 | 46.7 | 260 |
