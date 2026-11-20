"""#348 MSE 修正预算诊断: 对一个官方(或折)预测 h5ad, 逐 context 计算
budget = 跨扰动中心化平方和(行权 1/(G-1), 去自身靶基因), claim_up = Σ_p w_p C_pred,p (jackknife, bulk_lognorm TS=5e4)。
r = min(1, budget/claim), claim = Σ w min(C_pred, C_real) <= claim_up, 故 budget/claim_up 是 r 的下界。
Run: evalgpu python harness/line47_mse_budget.py PRED_H5AD [control_label]
"""
import sys
import numpy as np, anndata as ad
from scipy import sparse
from cell_eval2.moments import jackknife_correction
from cell_eval2.metrics.delta import _across_pert_budget

TS = 5e4
a = ad.read_h5ad(sys.argv[1], backed="r")
ctrl = sys.argv[2] if len(sys.argv) > 2 else "non-targeting"
obs = a.obs.copy()
if "context" not in obs:
    obs["context"] = "X"
genes = np.asarray(a.var_names).astype(str); gi = {g: i for i, g in enumerate(genes)}; G = len(genes)
for ctx in obs["context"].unique():
    idx = np.flatnonzero((obs["context"] == ctx).values)
    tg = obs["target_gene"].values[idx].astype(str)
    perts, bulks, C = [], [], []
    for t in sorted(set(tg) - {ctrl}):
        rows = np.sort(idx[tg == t])
        X = a.X[rows[0]:rows[-1] + 1] if rows[-1] - rows[0] + 1 == len(rows) else a[rows].X
        X = sparse.csr_matrix(X) if not sparse.issparse(X) else X.tocsr()
        P = np.asarray(X.sum(0)).ravel(); S = P.sum()
        bulks.append(np.log1p(TS * P / S)); perts.append(t)
        C.append(jackknife_correction(X, np.zeros(X.shape[0], np.intp), 1, TS)[0])
    B = np.stack(bulks); C = np.array(C)
    excl = {k: np.array([gi[t]]) for k, t in enumerate(perts) if t in gi}
    w = np.array([1.0 / (G - (1 if k in excl else 0)) for k in range(len(perts))])
    bud = _across_pert_budget(B, excl, w); cl = float((C * w).sum())
    print(f"[{ctx}] perts {len(perts)} budget {bud:.5g} claim_up(ΣwC_pred) {cl:.5g} 比 {bud/cl:.3f} | C_pred 中位 {np.median(C):.4g}", flush=True)
