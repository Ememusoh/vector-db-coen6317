Methods tuned for recall@10 = 0.95 on validation, scored on identical test executions. 'Worst selectivity' is the lowest per-selectivity mean recall.

| Dataset | Labels | Method | Recall@10 | Worst selectivity | Mean ms | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| SIFT1M | both | Cost-model planner | 0.951 | 0.821 | 0.308 | 0.186 | 0.581 |
| SIFT1M | indep. | Cost-model planner | 0.999 | 0.997 | 0.249 | 0.197 | 0.595 |
| SIFT1M | corr. | Cost-model planner | 0.902 | 0.642 | 0.366 | 0.175 | 0.383 |
| SIFT1M | both | Planner (global sel. only) | 0.950 | 0.897 | 6.395 | 0.143 | 54.404 |
| SIFT1M | indep. | Planner (global sel. only) | 0.980 | 0.922 | 0.142 | 0.163 | 0.272 |
| SIFT1M | corr. | Planner (global sel. only) | 0.919 | 0.805 | 12.647 | 0.133 | 122.224 |
| SIFT1M | both | Earlier planner (cutoff) | 0.967 | 0.943 | 0.900 | 0.118 | 12.294 |
| SIFT1M | indep. | Earlier planner (cutoff) | 0.975 | 0.949 | 0.108 | 0.109 | 0.206 |
| SIFT1M | corr. | Earlier planner (cutoff) | 0.959 | 0.923 | 1.693 | 0.141 | 12.452 |
| SIFT1M | both | Threshold rule | 1.000 | 1.000 | 5.549 | 3.916 | 14.836 |
| SIFT1M | indep. | Threshold rule | 1.000 | 1.000 | 6.152 | 4.252 | 15.106 |
| SIFT1M | corr. | Threshold rule | 1.000 | 1.000 | 4.945 | 3.916 | 12.632 |
| SIFT1M | both | Best fixed config | 1.000 | 1.000 | 5.547 | 3.915 | 14.835 |
| SIFT1M | indep. | Best fixed config | 1.000 | 1.000 | 6.151 | 4.250 | 15.105 |
| SIFT1M | corr. | Best fixed config | 1.000 | 1.000 | 4.943 | 3.915 | 12.630 |
| SIFT1M | both | Oracle (hindsight) | 0.951 | 0.820 | 0.148 | 0.090 | 0.240 |
| SIFT1M | indep. | Oracle (hindsight) | 1.000 | 1.000 | 0.087 | 0.076 | 0.213 |
| SIFT1M | corr. | Oracle (hindsight) | 0.902 | 0.639 | 0.208 | 0.123 | 0.250 |
| GIST-500K | both | Cost-model planner | 0.953 | 0.898 | 0.587 | 0.478 | 1.613 |
| GIST-500K | indep. | Cost-model planner | 0.966 | 0.916 | 0.524 | 0.440 | 1.321 |
| GIST-500K | corr. | Cost-model planner | 0.939 | 0.881 | 0.649 | 0.523 | 1.977 |
| GIST-500K | both | Planner (global sel. only) | 0.959 | 0.862 | 0.762 | 0.648 | 1.833 |
| GIST-500K | indep. | Planner (global sel. only) | 0.979 | 0.916 | 0.702 | 0.760 | 1.325 |
| GIST-500K | corr. | Planner (global sel. only) | 0.939 | 0.808 | 0.822 | 0.619 | 2.273 |
| GIST-500K | both | Earlier planner (cutoff) | 0.964 | 0.943 | 0.839 | 0.472 | 2.393 |
| GIST-500K | indep. | Earlier planner (cutoff) | 0.979 | 0.967 | 0.744 | 0.440 | 2.292 |
| GIST-500K | corr. | Earlier planner (cutoff) | 0.949 | 0.908 | 0.933 | 0.525 | 2.575 |
| GIST-500K | both | Threshold rule | 0.969 | 0.928 | 1.196 | 0.343 | 4.663 |
| GIST-500K | indep. | Threshold rule | 0.994 | 0.982 | 1.146 | 0.584 | 3.824 |
| GIST-500K | corr. | Threshold rule | 0.945 | 0.875 | 1.245 | 0.334 | 5.440 |
| GIST-500K | both | Best fixed config | 0.974 | 0.922 | 2.261 | 0.683 | 8.584 |
| GIST-500K | indep. | Best fixed config | 0.997 | 0.991 | 2.051 | 0.954 | 6.679 |
| GIST-500K | corr. | Best fixed config | 0.950 | 0.853 | 2.471 | 0.621 | 9.995 |
| GIST-500K | both | Oracle (hindsight) | 0.950 | 0.857 | 0.333 | 0.298 | 0.846 |
| GIST-500K | indep. | Oracle (hindsight) | 0.962 | 0.876 | 0.311 | 0.260 | 0.789 |
| GIST-500K | corr. | Oracle (hindsight) | 0.938 | 0.838 | 0.356 | 0.326 | 0.917 |
| GloVe-100 | both | Cost-model planner | 0.954 | 0.905 | 0.250 | 0.230 | 0.560 |
| GloVe-100 | indep. | Cost-model planner | 0.959 | 0.912 | 0.246 | 0.284 | 0.508 |
| GloVe-100 | corr. | Cost-model planner | 0.949 | 0.886 | 0.253 | 0.215 | 0.604 |
| GloVe-100 | both | Planner (global sel. only) | 0.950 | 0.897 | 0.238 | 0.243 | 0.509 |
| GloVe-100 | indep. | Planner (global sel. only) | 0.959 | 0.912 | 0.246 | 0.284 | 0.508 |
| GloVe-100 | corr. | Planner (global sel. only) | 0.941 | 0.880 | 0.229 | 0.237 | 0.509 |
| GloVe-100 | both | Earlier planner (cutoff) | 0.973 | 0.958 | 0.499 | 0.309 | 1.142 |
| GloVe-100 | indep. | Earlier planner (cutoff) | 0.983 | 0.966 | 0.424 | 0.472 | 1.046 |
| GloVe-100 | corr. | Earlier planner (cutoff) | 0.963 | 0.950 | 0.575 | 0.231 | 1.383 |
| GloVe-100 | both | Threshold rule | 0.962 | 0.897 | 0.327 | 0.236 | 0.977 |
| GloVe-100 | indep. | Threshold rule | 0.970 | 0.912 | 0.358 | 0.277 | 1.039 |
| GloVe-100 | corr. | Threshold rule | 0.955 | 0.882 | 0.297 | 0.230 | 0.831 |
| GloVe-100 | both | Best fixed config | 1.000 | 1.000 | 3.553 | 3.420 | 8.371 |
| GloVe-100 | indep. | Best fixed config | 1.000 | 1.000 | 3.491 | 3.420 | 8.072 |
| GloVe-100 | corr. | Best fixed config | 1.000 | 1.000 | 3.614 | 3.341 | 8.604 |
| GloVe-100 | both | Oracle (hindsight) | 0.950 | 0.896 | 0.126 | 0.109 | 0.343 |
| GloVe-100 | indep. | Oracle (hindsight) | 0.958 | 0.897 | 0.117 | 0.092 | 0.336 |
| GloVe-100 | corr. | Oracle (hindsight) | 0.943 | 0.880 | 0.135 | 0.126 | 0.345 |
| Deep1M | both | Cost-model planner | 0.949 | 0.912 | 0.104 | 0.108 | 0.196 |
| Deep1M | indep. | Cost-model planner | 0.980 | 0.961 | 0.106 | 0.111 | 0.195 |
| Deep1M | corr. | Cost-model planner | 0.918 | 0.863 | 0.102 | 0.107 | 0.198 |
| Deep1M | both | Planner (global sel. only) | 0.938 | 0.855 | 0.369 | 0.107 | 0.350 |
| Deep1M | indep. | Planner (global sel. only) | 0.972 | 0.907 | 0.094 | 0.112 | 0.145 |
| Deep1M | corr. | Planner (global sel. only) | 0.905 | 0.729 | 0.644 | 0.097 | 3.980 |
| Deep1M | both | Earlier planner (cutoff) | 0.964 | 0.939 | 0.128 | 0.115 | 0.276 |
| Deep1M | indep. | Earlier planner (cutoff) | 0.980 | 0.961 | 0.107 | 0.114 | 0.195 |
| Deep1M | corr. | Earlier planner (cutoff) | 0.949 | 0.917 | 0.149 | 0.122 | 0.393 |
| Deep1M | both | Threshold rule | 0.966 | 0.917 | 0.181 | 0.135 | 0.486 |
| Deep1M | indep. | Threshold rule | 0.998 | 0.994 | 0.191 | 0.155 | 0.463 |
| Deep1M | corr. | Threshold rule | 0.934 | 0.841 | 0.171 | 0.130 | 0.643 |
| Deep1M | both | Best fixed config | 1.000 | 1.000 | 3.232 | 3.211 | 7.016 |
| Deep1M | indep. | Best fixed config | 1.000 | 1.000 | 3.244 | 3.200 | 7.072 |
| Deep1M | corr. | Best fixed config | 1.000 | 1.000 | 3.220 | 3.221 | 6.944 |
| Deep1M | both | Oracle (hindsight) | 0.950 | 0.897 | 0.064 | 0.060 | 0.137 |
| Deep1M | indep. | Oracle (hindsight) | 0.971 | 0.945 | 0.052 | 0.050 | 0.100 |
| Deep1M | corr. | Oracle (hindsight) | 0.929 | 0.849 | 0.077 | 0.077 | 0.145 |
| YFCC | both | Cost-model planner | 0.965 | 0.908 | 0.086 | 0.098 | 0.176 |
| YFCC | both | Planner (global sel. only) | 0.966 | 0.898 | 0.093 | 0.098 | 0.212 |
| YFCC | both | Earlier planner (cutoff) | 0.971 | 0.925 | 0.104 | 0.097 | 0.238 |
| YFCC | both | Threshold rule | 0.959 | 0.860 | 0.093 | 0.085 | 0.271 |
| YFCC | both | Best fixed config | 1.000 | 1.000 | 1.732 | 0.116 | 10.551 |
| YFCC | both | Oracle (hindsight) | 0.951 | 0.897 | 0.060 | 0.066 | 0.125 |
