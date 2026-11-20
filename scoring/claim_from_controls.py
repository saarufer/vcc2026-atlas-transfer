"""#348 预算的 claim 只由 400 细胞组的 jackknife 采样噪声决定 -> 可仅凭对照细胞估计。
对每个 context 对照 h5ad: 抽 R 组随机 400 细胞, jackknife_correction(TS=5e4) 得 C, claim ≈ n_pert × mean(C) / (G-1)。
Run: evalgpu python harness/line97_claim_from_ctrl.py ctx1.h5ad [ctx2.h5ad ...]
"""
import sys
import numpy as np, anndata as ad
from scipy import sparse
from cell_eval2.moments import jackknife_correction

import os
TS, NC, R, NP = 5e4, int(os.environ.get("NC", 400)), 20, int(os.environ.get("NP", 300))
for f in sys.argv[1:]:
    a = ad.read_h5ad(f, backed="r"); G = a.shape[1]
    X = sparse.csr_matrix(a.X[:], dtype=np.float64)
    depth = np.asarray(X.sum(1)).ravel()
    rng = np.random.default_rng(0); Cs = []
    for _ in range(R):
        rows = np.sort(rng.choice(X.shape[0], NC, replace=False))
        Cs.append(jackknife_correction(X[rows], np.zeros(NC, np.intp), 1, TS)[0])
    Cs = np.array(Cs)
    print(f"{f.split('/')[-1]}: 细胞 {X.shape[0]} UMI中位 {np.median(depth):.0f} | jk C {Cs.mean():.3f}±{Cs.std():.3f} -> claim≈{NP * Cs.mean() / (G - 1):.4f}", flush=True)
