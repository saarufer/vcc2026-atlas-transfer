"""GSE295214(2026 H1 全基因组 CRISPRi Perturb-seq, 85 个 10x h5) -> kaipengm2 格式源统计 data/H1g_full_statistics.npz。
每个 h5 = 一个 batch; guide 判定: 最大 guide UMI >=5 且占该细胞 guide UMI >=70%; 靶 = guide 名 '_' 前缀; non-targeting -> 对照。
靶集合 = 现有源库 437 靶(val 300 + H1 2025) ∪ ess_targets.txt(必需文库共同靶)。流式, 低内存。
"""
import glob, resource
resource.setrlimit(resource.RLIMIT_AS, (min(56 * 2**30, resource.getrlimit(resource.RLIMIT_AS)[1]), resource.getrlimit(resource.RLIMIT_AS)[1]))
import numpy as np, h5py
from scipy import sparse
import os; D = os.environ.get("KP_DATA", "vcc_data/kp_atlasshift/data/")
fs = sorted(glob.glob("vcc_data/raw/gse295214/h5/*.h5"))
targets = list(np.load(D + "K562_GWPS_CPM_full_statistics.npz", allow_pickle=False)["targets"].astype(str))
targets += [t for t in open(D + "ess_targets.txt").read().split() if t not in set(targets)]
tl = {t: i for i, t in enumerate(targets)}; nT, nB = len(targets), len(fs)
genes = None
for bi, f in enumerate(fs):
    with h5py.File(f) as h:
        m = h["matrix"]; ft = m["features"]; typ = ft["feature_type"][:].astype(str); nm = ft["name"][:].astype(str)
        shp = m["shape"][:]
        X = sparse.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(shp)).T.tocsr().astype(np.float64)
    gex = np.flatnonzero(typ == "Gene Expression"); gd = np.flatnonzero(typ == "CRISPR Guide Capture")
    if genes is None:
        genes = np.asarray(list(dict.fromkeys(nm[gex])), dtype=str); gl = {g: i for i, g in enumerate(genes)}
        cs = np.zeros((nT + nB, len(genes))); cp = np.zeros_like(cs); gn = np.zeros(nT + nB, np.int64)
        tbn = np.zeros((nT, nB), np.int64); tbl = np.zeros((nT, nB))
    proj = sparse.csr_matrix((np.ones(len(gex)), (np.arange(len(gex)), [gl.get(g, 0) for g in nm[gex]])), shape=(len(gex), len(genes)))
    G = X[:, gd]; E = X[:, gex] @ proj
    gt = np.asarray(G.sum(1)).ravel(); gm = np.asarray(G.max(1).todense()).ravel(); ga = np.asarray(G.argmax(1)).ravel()
    ok = (gm >= 5) & (gm >= 0.7 * np.maximum(gt, 1))
    lab = np.array([nm[gd][j].split("_")[0] for j in ga])
    ctrl = ok & (lab == "non-targeting")
    grp = np.array([tl.get(l, -1) for l in lab]); grp[~ok] = -1; grp[ctrl] = nT + bi
    sel = np.flatnonzero(grp >= 0); d = np.asarray(E.sum(1)).ravel()
    sel = sel[d[sel] > 0]
    Es = E[sel]; g = grp[sel]; dd = d[sel]
    A = sparse.csr_matrix((np.ones(len(sel)), (g, np.arange(len(sel)))), shape=(nT + nB, len(sel)))
    cs += (A @ Es).toarray(); cp += (A @ Es.multiply((1e6 / dd)[:, None])).toarray(); gn += np.bincount(g, minlength=nT + nB)
    p = g < nT; np.add.at(tbn, (g[p], bi), 1); np.add.at(tbl, (g[p], bi), dd[p])
    print(f"{bi+1}/{nB} {f.split('/')[-1][:30]} 细胞 {X.shape[0]} 判定 {ok.mean():.2f} 对照 {ctrl.sum()} 入库靶细胞 {p.sum()}", flush=True)
cN = gn[nT:]; okb = cN > 0
cprob = np.zeros((nB, len(genes))); cprob[okb] = cs[nT:][okb] / cs[nT:][okb].sum(1, keepdims=True)
ccpm = np.zeros((nB, len(genes))); ccpm[okb] = cp[nT:][okb] / cN[okb, None]
n = gn[:nT]; means = np.divide(cp[:nT], n[:, None], out=np.zeros((nT, len(genes))), where=n[:, None] > 0)
cw = np.divide(tbn, n[:, None], out=np.zeros_like(tbl), where=n[:, None] > 0); lib = tbl.sum(1, keepdims=True)
lw = np.divide(tbl, lib, out=np.zeros_like(tbl), where=lib > 0)
np.savez_compressed(D + "H1g_full_statistics.npz", source=np.asarray("H1g_GSE295214"), targets=np.asarray(targets), genes=genes, n_cells=n,
    target_count_sums=cs[:nT].astype(np.float32), target_mean_cpm=means.astype(np.float32),
    matched_control_probability=(lw @ cprob).astype(np.float32), matched_control_mean_cpm=(cw @ ccpm).astype(np.float32),
    global_control_probability=(cs[nT:].sum(0) / cs[nT:].sum()).astype(np.float32), global_control_mean_cpm=(cp[nT:].sum(0) / cN.sum()).astype(np.float32),
    measured_genes=np.ones(len(genes), bool))
import pandas as pd
val = set(pd.read_csv(D + "pert_counts.csv").target_gene.astype(str))
print("完成: 靶", nT, "有细胞(>=20)", int((n >= 20).sum()), "val 靶覆盖(>=20)", sum(1 for t, k in zip(targets, n) if t in val and k >= 20), "对照细胞", int(cN.sum()), "靶细胞中位", np.median(n[n > 0]))
