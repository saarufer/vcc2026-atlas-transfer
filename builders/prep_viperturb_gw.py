"""VIPerturb K562-Flex 全基因组(binA/B/C 中已转换者) -> data/VIPflexGW_full_statistics.npz + _var.npz(含 sqmean, 供 z/共识头/可信度权重)。
h5ad X = 原始计数 CSR; 靶 = obs.target_gene, 对照 = non-targeting; batch = obs.sample。靶集合 = 源库 437 靶。流式。"""
import os, resource, sys
resource.setrlimit(resource.RLIMIT_AS, (min(40 * 2**30, resource.getrlimit(resource.RLIMIT_AS)[1]), resource.getrlimit(resource.RLIMIT_AS)[1]))
import numpy as np, h5py, anndata as ad, pandas as pd
from scipy import sparse
import os; D = os.environ.get("KP_DATA", "vcc_data/kp_atlasshift/data/"); S = "vcc_data/state_train/viperturb/"
bins = [b for b in ("binA", "binB", "binC") if os.path.exists(S + b + ".h5ad")]
targets = list(np.load(D + "K562_GWPS_CPM_full_statistics.npz", allow_pickle=False)["targets"].astype(str)); tl = {t: i for i, t in enumerate(targets)}; nT = len(targets)
genes = None; batches = []; acc = None
for b in bins:
    a = ad.read_h5ad(S + b + ".h5ad", backed="r")
    g0 = a.var_names.astype(str).values
    if genes is None: genes = g0; G = len(genes)
    assert (g0 == genes).all(), "基因轴不一致"
    lab = a.obs["target_gene"].astype(str).values; smp = a.obs["sample"].astype(str).values
    for s_ in pd.unique(smp):
        if s_ not in batches: batches.append(s_)
    bl = {s_: i for i, s_ in enumerate(batches)}
    if acc is None or acc["nB"] < len(batches):
        nB = len(batches)
        if acc is None:
            acc = dict(nB=nB, cs=np.zeros((nT + 400, G)), cp=np.zeros((nT + 400, G)), cq=np.zeros((nT + 400, G)), gn=np.zeros(nT + 400, np.int64), tbn=np.zeros((nT, 400), np.int64), tbl=np.zeros((nT, 400)))
        acc["nB"] = nB
    bc = np.array([bl[x] for x in smp]); rg = np.array([tl.get(t, -1) for t in lab], np.int64); ctrl = lab == "non-targeting"; rg[ctrl] = nT + bc[ctrl]
    with h5py.File(S + b + ".h5ad") as h:
        X_ = h["X"]; ip = X_["indptr"]; N = a.shape[0]
        for s in range(0, N, 5000):
            e = min(N, s + 5000); sel = np.arange(s, e); keep = rg[sel] >= 0
            if not keep.any(): continue
            p0, p1 = ip[s], ip[e]
            X = sparse.csr_matrix((X_["data"][p0:p1].astype(np.float64), X_["indices"][p0:p1], ip[s:e + 1] - p0), shape=(e - s, G))[keep]
            g = rg[sel][keep]; d = np.asarray(X.sum(1)).ravel(); ok = d > 0; X, g, d = X[ok], g[ok], d[ok]; bb = bc[sel][keep][ok]
            A = sparse.csr_matrix((np.ones(len(g)), (g, np.arange(len(g)))), shape=(nT + 400, len(g)))
            Xc = X.multiply((1e6 / d)[:, None]).tocsr()
            acc["cs"] += (A @ X).toarray(); acc["cp"] += (A @ Xc).toarray(); acc["cq"] += (A @ Xc.multiply(Xc)).toarray()
            acc["gn"] += np.bincount(g, minlength=nT + 400)
            pm = g < nT; np.add.at(acc["tbn"], (g[pm], bb[pm]), 1); np.add.at(acc["tbl"], (g[pm], bb[pm]), d[pm])
        print(b, "完成", flush=True)
nB = len(batches); cs, cp, cq, gn = acc["cs"], acc["cp"], acc["cq"], acc["gn"]; tbn, tbl = acc["tbn"][:, :nB], acc["tbl"][:, :nB]
cN = gn[nT:nT + nB]; okb = cN > 0
cprob = np.zeros((nB, G)); cprob[okb] = cs[nT:nT + nB][okb] / cs[nT:nT + nB][okb].sum(1, keepdims=True)
ccpm = np.zeros((nB, G)); ccpm[okb] = cp[nT:nT + nB][okb] / cN[okb, None]; csq = np.zeros((nB, G)); csq[okb] = cq[nT:nT + nB][okb] / cN[okb, None]
n = gn[:nT]; means = np.divide(cp[:nT], n[:, None], out=np.zeros((nT, G)), where=n[:, None] > 0); sq = np.divide(cq[:nT], n[:, None], out=np.zeros((nT, G)), where=n[:, None] > 0)
cw = np.divide(tbn, n[:, None], out=np.zeros_like(tbl), where=n[:, None] > 0); lib = tbl.sum(1, keepdims=True); lw = np.divide(tbl, lib, out=np.zeros_like(tbl), where=lib > 0)
common = dict(source=np.asarray("VIPflexGW"), targets=np.asarray(targets), genes=np.asarray(genes, dtype=str), n_cells=n,
    target_count_sums=cs[:nT].astype(np.float32), target_mean_cpm=means.astype(np.float32),
    matched_control_probability=(lw @ cprob).astype(np.float32), matched_control_mean_cpm=(cw @ ccpm).astype(np.float32),
    global_control_probability=(cs[nT:nT + nB].sum(0) / cs[nT:nT + nB].sum()).astype(np.float32), global_control_mean_cpm=(cp[nT:nT + nB].sum(0) / cN.sum()).astype(np.float32),
    measured_genes=np.ones(G, bool))
np.savez_compressed(D + "VIPflexGW_full_statistics.npz", **common)
np.savez_compressed(D + "VIPflexGW_full_statistics_var.npz", **common, target_cpm_sqmean=sq.astype(np.float32), matched_control_cpm_sqmean=(cw @ csq).astype(np.float32),
    matched_control_n=(cw @ cN.astype(float)).astype(np.float32))
val = set(pd.read_csv(D + "pert_counts.csv").target_gene.astype(str))
print("完成 bins", bins, "| val 靶覆盖(>=20)", sum(1 for t, k in zip(targets, n) if t in val and k >= 20), "库内靶", int((n >= 20).sum()), "对照", int(cN.sum()), "批次", nB)
