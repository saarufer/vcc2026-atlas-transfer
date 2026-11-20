"""KOLF2.1J Pan-Genome CRISPRi(figshare 27261219, 266 万细胞, CSC 存储) -> data/KOLFpan_full_statistics(+_var).npz。
按基因列流式: 每列 (行索引, 计数) 按细胞分组(437 靶 + 3 批次 NTC 对照)累加 计数和 / CPM 和 / CPM 平方和。只处理官方基因轴上的基因。"""
import resource
resource.setrlimit(resource.RLIMIT_AS, (min(48 * 2**30, resource.getrlimit(resource.RLIMIT_AS)[1]), resource.getrlimit(resource.RLIMIT_AS)[1]))
import numpy as np, h5py, pandas as pd, time
P = "vcc_data/raw/kolf/KOLF_Pan_Genome_QC_Filtered.h5ad"; import os; D = os.environ.get("KP_DATA", "vcc_data/kp_atlasshift/data/")
h = h5py.File(P)
def cat(grp):
    g = h["obs"][grp]; return g["categories"][:].astype(str)[g["codes"][:]] if isinstance(g, h5py.Group) else g[:].astype(str)
lab = cat("gene_target"); bat = cat("batch"); bcodes, batches = pd.factorize(bat); nB = len(batches)
vidx = h["var"]["_index"][:].astype(str)
official = pd.read_csv(D + "gene_names.csv").iloc[:, 0].astype(str).values
keep_g = [j for j, g in enumerate(vidx) if g in set(official)]; genes = vidx[keep_g]; G = len(genes)
targets = list(np.load(D + "K562_GWPS_CPM_full_statistics.npz", allow_pickle=False)["targets"].astype(str)); tl = {t: i for i, t in enumerate(targets)}; nT = len(targets)
rg = np.array([tl.get(t, -1) for t in lab], np.int64); ctrl = lab == "NTC"; rg[ctrl] = nT + bcodes[ctrl]
d = h["obs"]["total_counts"][:].astype(np.float64); NG = nT + nB
print(f"细胞 {len(lab)}, 入组 {(rg >= 0).sum()}, 基因 {G}/{len(vidx)}, 批次 {list(batches)}", flush=True)
X = h["X"]; ip = X["indptr"][:]; data = X["data"]; ind = X["indices"]
cs = np.zeros((NG, G)); cp = np.zeros((NG, G)); cq = np.zeros((NG, G)); t0 = time.time()
inv = np.where(d > 0, 1e6 / np.maximum(d, 1), 0)
for k, j in enumerate(keep_g):
    a, b = ip[j], ip[j + 1]
    if b > a:
        r = ind[a:b]; v = data[a:b].astype(np.float64); g = rg[r]; m = g >= 0
        if m.any():
            g = g[m]; v = v[m]; c = v * inv[r[m]]
            cs[:, k] = np.bincount(g, weights=v, minlength=NG); cp[:, k] = np.bincount(g, weights=c, minlength=NG); cq[:, k] = np.bincount(g, weights=c * c, minlength=NG)
    if k % 2000 == 0: print(f"{k}/{G} {time.time() - t0:.0f}s", flush=True)
gn = np.bincount(rg[rg >= 0], minlength=NG); n = gn[:nT]; cN = gn[nT:]
tbn = np.zeros((nT, nB), np.int64); tbl = np.zeros((nT, nB)); pm = (rg >= 0) & (rg < nT)
np.add.at(tbn, (rg[pm], bcodes[pm]), 1); np.add.at(tbl, (rg[pm], bcodes[pm]), d[pm])
okb = cN > 0; cprob = np.zeros((nB, G)); cprob[okb] = cs[nT:][okb] / cs[nT:][okb].sum(1, keepdims=True)
ccpm = np.zeros((nB, G)); ccpm[okb] = cp[nT:][okb] / cN[okb, None]; csq = np.zeros((nB, G)); csq[okb] = cq[nT:][okb] / cN[okb, None]
means = np.divide(cp[:nT], n[:, None], out=np.zeros((nT, G)), where=n[:, None] > 0); sq = np.divide(cq[:nT], n[:, None], out=np.zeros((nT, G)), where=n[:, None] > 0)
cw = np.divide(tbn, n[:, None], out=np.zeros_like(tbl), where=n[:, None] > 0); lib = tbl.sum(1, keepdims=True); lw = np.divide(tbl, lib, out=np.zeros_like(tbl), where=lib > 0)
common = dict(source=np.asarray("KOLFpan"), targets=np.asarray(targets, dtype=str), genes=np.asarray(genes, dtype=str), n_cells=n,
    target_count_sums=cs[:nT].astype(np.float32), target_mean_cpm=means.astype(np.float32),
    matched_control_probability=(lw @ cprob).astype(np.float32), matched_control_mean_cpm=(cw @ ccpm).astype(np.float32),
    global_control_probability=(cs[nT:].sum(0) / cs[nT:].sum()).astype(np.float32), global_control_mean_cpm=(cp[nT:].sum(0) / cN.sum()).astype(np.float32),
    measured_genes=np.ones(G, bool))
np.savez_compressed(D + "KOLFpan_full_statistics.npz", **common)
np.savez_compressed(D + "KOLFpan_full_statistics_var.npz", **common, target_cpm_sqmean=sq.astype(np.float32), matched_control_cpm_sqmean=(cw @ csq).astype(np.float32),
    matched_control_n=(cw @ cN.astype(float)).astype(np.float32))
val = set(pd.read_csv(D + "pert_counts.csv").target_gene.astype(str))
print("完成 | val 靶覆盖(>=20)", sum(1 for t, k in zip(targets, n) if t in val and k >= 20), "库内靶", int((n >= 20).sum()), "对照", int(cN.sum()), "靶细胞中位", np.median(n[n > 0]), flush=True)
