Planner vs baselines on held-out test queries (end-to-end latency, planning included). Speed-up is relative to the tuned threshold rule.

| Dataset | Labels | Method | Recall@10 | Mean ms | p50 ms | p95 ms | Speed-up |
|---|---|---|---|---|---|---|---|
| SIFT1M | independent | Cost-model planner | 0.999 | 0.451 | 0.196 | 1.716 | 12.5 |
| SIFT1M | independent | Threshold rule | 1.000 | 5.643 | 4.259 | 14.604 | 1.0 |
| SIFT1M | independent | Best fixed config | 1.000 | 5.613 | 4.173 | 14.527 | 1.0 |
| SIFT1M | independent | Oracle (hindsight) | 1.000 | 0.087 | 0.076 | 0.213 | 64.8 |
| SIFT1M | correlated | Cost-model planner | 0.914 | 0.562 | 0.170 | 0.835 | 8.8 |
| SIFT1M | correlated | Threshold rule | 1.000 | 4.933 | 3.870 | 12.284 | 1.0 |
| SIFT1M | correlated | Best fixed config | 1.000 | 4.979 | 3.899 | 12.662 | 1.0 |
| SIFT1M | correlated | Oracle (hindsight) | 0.902 | 0.208 | 0.123 | 0.250 | 23.7 |
| GloVe-100 | independent | Cost-model planner | 0.959 | 0.284 | 0.264 | 0.784 | 1.1 |
| GloVe-100 | independent | Threshold rule | 0.970 | 0.308 | 0.257 | 0.812 | 1.0 |
| GloVe-100 | independent | Best fixed config | 1.000 | 3.220 | 3.280 | 6.737 | 0.1 |
| GloVe-100 | independent | Oracle (hindsight) | 0.958 | 0.117 | 0.092 | 0.336 | 2.6 |
| GloVe-100 | correlated | Cost-model planner | 0.959 | 0.298 | 0.250 | 0.767 | 1.0 |
| GloVe-100 | correlated | Threshold rule | 0.955 | 0.285 | 0.218 | 0.797 | 1.0 |
| GloVe-100 | correlated | Best fixed config | 1.000 | 3.216 | 3.210 | 6.746 | 0.1 |
| GloVe-100 | correlated | Oracle (hindsight) | 0.943 | 0.135 | 0.126 | 0.345 | 2.1 |
| Deep1M | independent | Cost-model planner | 0.980 | 0.260 | 0.123 | 0.967 | 0.8 |
| Deep1M | independent | Threshold rule | 0.998 | 0.210 | 0.152 | 0.542 | 1.0 |
| Deep1M | independent | Best fixed config | 1.000 | 3.186 | 3.204 | 6.737 | 0.1 |
| Deep1M | independent | Oracle (hindsight) | 0.971 | 0.052 | 0.050 | 0.100 | 4.1 |
| Deep1M | correlated | Cost-model planner | 0.926 | 0.154 | 0.126 | 0.429 | 1.3 |
| Deep1M | correlated | Threshold rule | 0.934 | 0.198 | 0.119 | 0.728 | 1.0 |
| Deep1M | correlated | Best fixed config | 1.000 | 3.173 | 3.219 | 6.698 | 0.1 |
| Deep1M | correlated | Oracle (hindsight) | 0.929 | 0.077 | 0.077 | 0.145 | 2.6 |
