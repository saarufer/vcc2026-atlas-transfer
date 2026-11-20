"""GSE337988(Genentech DLD-1 CRISPRi 5000 基因 Perturb-seq)低 MOI 通道 -> kaipengm2 格式源统计 data/DLD1_full_statistics.npz。
每个 GEX h5 + crispr.umi_correct.h5 一组(=一个 batch); guide 判定: 最大 UMI>=10 且 >= 5×次大(环境污染重); 靶 = guide 名 '_' 前缀; NTC -> 对照。
靶集合 = 源库 437 靶。流式, 低内存。
"""
import glob, os, resource
resource.setrlimit(resource.RLIMIT_AS, (min(32 * 2**30, resource.getrlimit(resource.RLIMIT_AS)[1]), resource.getrlimit(resource.RLIMIT_AS)[1]))
import numpy as np, h5py, pandas as pd
from scipy import sparse
import os; D = os.environ.get("KP_DATA", "vcc_data/kp_atlasshift/data/"); H = "vcc_data/raw/gse337988_dld1/h5/"
gex = sorted(glob.glob(H + "*.filtered_feature_bc_matrix.h5"))
targets = list(np.load(D + "K562_GWPS_CPM_full_statistics.npz", allow_pickle=False)["targets"].astype(str))
tl = {t: i for i, t in enumerate(targets)}; nT, nB = len(targets), len(gex); genes = None


def load10x(p, csc_rows_are_features=True):
    with h5py.File(p) as h:
        m = h["matrix"]; shp = m["shape"][:]
        X = sparse.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(shp)).T.tocsr()
        bc = np.array([b.decode().split("-")[0] for b in m["barcodes"][:]]); nm = m["features/name"][:].astype(str)
    return X, bc, nm


for bi, g in enumerate(gex):
    gsm = os.path.basename(g).split("_")[0]; cr = glob.glob(H + gsm + "_*.crispr.umi_correct.h5")
    if not cr: print("缺 guide 文件", gsm); continue
    X, bc, nm = load10x(g); G, gbc, gnm = load10x(cr[0])
    if genes is None:
        genes = np.asarray(list(dict.fromkeys(nm)), dtype=str); gl = {x: i for i, x in enumerate(genes)}
        cs = np.zeros((nT + nB, len(genes))); cp = np.zeros_like(cs); gn = np.zeros(nT + nB, np.int64)
        tbn = np.zeros((nT, nB), np.int64); tbl = np.zeros((nT, nB))
    proj = sparse.csr_matrix((np.ones(len(nm)), (np.arange(len(nm)), [gl[x] for x in nm])), shape=(len(nm), len(genes)))
    gi = {b: i for i, b in enumerate(gbc)}; rows = np.array([gi.get(b, -1) for b in bc]); has = rows >= 0
    Gm = sparse.csr_matrix((X.shape[0], G.shape[1])); Gm = G[np.where(has, rows, 0)].multiply(has[:, None]).tocsr()
    # 环境 guide 污染重(每细胞 ~百种 guide 低计数): 判定用 最大>=10 且 >= 5×次大
    Gm.sort_indices(); nnz = np.diff(Gm.indptr); rws = np.repeat(np.arange(Gm.shape[0]), nnz)
    order = np.lexsort((-Gm.data, rws)); dsort = Gm.data[order]; isort = Gm.indices[order]; st = Gm.indptr[:-1]
    gm = np.where(nnz > 0, dsort[np.minimum(st, len(dsort) - 1)], 0); g2 = np.where(nnz > 1, dsort[np.minimum(st + 1, len(dsort) - 1)], 0)
    ga = np.where(nnz > 0, isort[np.minimum(st, len(isort) - 1)], 0)
    ok = has & (gm >= 10) & (gm >= 5 * g2); lab = np.array([gnm[j].split("_")[0] for j in ga])
    grp = np.array([tl.get(l, -1) for l in lab]); grp[~ok] = -1; ctrl = ok & (lab == "NTC"); grp[ctrl] = nT + bi
    E = X.astype(np.float64) @ proj; d = np.asarray(E.sum(1)).ravel(); sel = np.flatnonzero((grp >= 0) & (d > 0))
    Es = E[sel]; gg = grp[sel]; dd = d[sel]
    A = sparse.csr_matrix((np.ones(len(sel)), (gg, np.arange(len(sel)))), shape=(nT + nB, len(sel)))
    cs += (A @ Es).toarray(); cp += (A @ Es.multiply((1e6 / dd)[:, None])).toarray(); gn += np.bincount(gg, minlength=nT + nB)
    p = gg < nT; np.add.at(tbn, (gg[p], bi), 1); np.add.at(tbl, (gg[p], bi), dd[p])
    print(f"{bi+1}/{nB} {gsm}: 细胞 {X.shape[0]}, 有 guide {has.mean():.2f}, 判定单 guide {ok.mean():.2f}, NTC {ctrl.sum()}, 入库靶细胞 {p.sum()}, UMI 中位 {np.median(d):.0f}", flush=True)
cN = gn[nT:]; okb = cN > 0
cprob = np.zeros((nB, len(genes))); cprob[okb] = cs[nT:][okb] / cs[nT:][okb].sum(1, keepdims=True)
ccpm = np.zeros((nB, len(genes))); ccpm[okb] = cp[nT:][okb] / cN[okb, None]
n = gn[:nT]; means = np.divide(cp[:nT], n[:, None], out=np.zeros((nT, len(genes))), where=n[:, None] > 0)
cw = np.divide(tbn, n[:, None], out=np.zeros_like(tbl), where=n[:, None] > 0); lib = tbl.sum(1, keepdims=True)
lw = np.divide(tbl, lib, out=np.zeros_like(tbl), where=lib > 0)
np.savez_compressed(D + "DLD1_full_statistics.npz", source=np.asarray("DLD1_GSE337988"), targets=np.asarray(targets), genes=genes, n_cells=n,
    target_count_sums=cs[:nT].astype(np.float32), target_mean_cpm=means.astype(np.float32),
    matched_control_probability=(lw @ cprob).astype(np.float32), matched_control_mean_cpm=(cw @ ccpm).astype(np.float32),
    global_control_probability=(cs[nT:].sum(0) / cs[nT:].sum()).astype(np.float32), global_control_mean_cpm=(cp[nT:].sum(0) / cN.sum()).astype(np.float32),
    measured_genes=np.ones(len(genes), bool))
val = set(pd.read_csv(D + "pert_counts.csv").target_gene.astype(str))
print("完成: val 靶覆盖(>=20 细胞)", sum(1 for t, k in zip(targets, n) if t in val and k >= 20), "(>=10)", sum(1 for t, k in zip(targets, n) if t in val and k >= 10), "对照细胞", int(cN.sum()), "靶细胞中位", np.median(n[n > 0]))
