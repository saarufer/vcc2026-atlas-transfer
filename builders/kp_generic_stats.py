"""Generic per-cell Perturb-seq h5ad -> AtlasShift *_full_statistics.npz (same fields as prepare.py / kp_xatlas_local.py).
Per (match-key, label) group: count sums + sum of per-cell CPM over the panel genes (official 18533 ∪ H1 genes);
library size = each cell's total counts over ALL genes. Targets = official 300 ∪ H1 targets. Controls matched by
--match cols (e.g. sample/batch), weighted by each match group's share of the target's counts / cells.
Optional --split col writes one npz per value (e.g. cell line). Raw counts: layers['counts'] if present else .X.
Run: python kp_generic_stats.py --path X.h5ad --target-col gene --control NT --match Batch_info --split cell_type \
        --prefix Jiang24_ --data-dir DATA --raw-dir RAW [--filter col=val ...]
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True); ap.add_argument("--target-col", required=True)
    ap.add_argument("--control", required=True); ap.add_argument("--match", nargs="*", default=[])
    ap.add_argument("--split", default=None); ap.add_argument("--filter", action="append", default=[])
    ap.add_argument("--prefix", required=True); ap.add_argument("--data-dir", required=True); ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--chunk", type=int, default=40000)
    a = ap.parse_args()
    data, raw = Path(a.data_dir), Path(a.raw_dir)
    official_t = pd.read_csv(data / "pert_counts.csv")["target_gene"].astype(str).tolist()
    official_g = pd.read_csv(data / "gene_names.csv")["gene_name"].astype(str).tolist()
    h1_t = pd.read_csv(raw / "pert_counts_Training.csv")["target_gene"].astype(str).tolist()
    h1g = pd.read_csv(raw / "gene_names.csv"); gcol = "gene_name" if "gene_name" in h1g else h1g.columns[0]
    genes = list(dict.fromkeys(official_g + h1g[gcol].astype(str).tolist()))
    targets = list(dict.fromkeys(official_t + [t for t in h1_t if t != "non-targeting"]))
    gl = {g: i for i, g in enumerate(genes)}
    d = ad.read_h5ad(a.path, backed="r")
    layer = "counts" if "counts" in d.layers else None
    var = [str(v) for v in d.var_names]
    vmap = np.array([gl.get(v, -1) for v in var]); vok = vmap >= 0
    measured = np.zeros(len(genes), bool); measured[vmap[vok]] = True
    obs = d.obs
    keep = np.ones(d.n_obs, bool)
    for f in a.filter:
        c, v = f.split("=", 1); keep &= obs[c].astype(str).values == v
    lab = obs[a.target_col].astype(str).values
    keep &= np.isin(lab, targets + [a.control])
    split = obs[a.split].astype(str).values if a.split else np.array(["all"] * d.n_obs)
    mkey = (obs[a.match].astype(str).agg("|".join, axis=1).values if a.match else np.array(["all"] * d.n_obs))
    print(f"{a.path}: {d.shape}, 矩阵 {'layers[counts]' if layer else '.X'}, 保留细胞 {keep.sum()}, 面板基因测到 {vok.sum()}", flush=True)
    acc = {}                                    # (split, mkey, label) -> [count_sum, cpm_sum, n]
    for i0 in range(0, d.n_obs, a.chunk):
        i1 = min(i0 + a.chunk, d.n_obs); idx = np.nonzero(keep[i0:i1])[0]
        if not len(idx):
            continue
        X = d[i0:i1].to_memory().layers[layer] if layer else d.X[i0:i1]   # backed 层切片会错形, 先载入内存
        X = (X.tocsr() if sp.issparse(X) else sp.csr_matrix(X))[idx].astype(np.float64)
        lib = np.asarray(X.sum(1)).ravel(); lib[lib == 0] = 1
        Xp = X[:, np.nonzero(vok)[0]]
        Xc = sp.diags(1e6 / lib) @ Xp
        keys = list(zip(split[i0:i1][idx], mkey[i0:i1][idx], lab[i0:i1][idx]))
        kdf = pd.Series(range(len(keys)), index=pd.MultiIndex.from_tuples(keys))
        for k, r in kdf.groupby(level=[0, 1, 2]).indices.items():   # 位置索引(避免重复标签的 n² 放大)
            cs = np.zeros(len(genes)); cc = np.zeros(len(genes)); c2 = np.zeros(len(genes))
            cs[vmap[vok]] = np.asarray(Xp[r].sum(0)).ravel(); cc[vmap[vok]] = np.asarray(Xc[r].sum(0)).ravel(); c2[vmap[vok]] = np.asarray(Xc[r].multiply(Xc[r]).sum(0)).ravel()
            if k in acc:
                acc[k][0] += cs; acc[k][1] += cc; acc[k][2] += len(r); acc[k][3] += c2
            else:
                acc[k] = [cs, cc, len(r), c2]
        print(f"  {i1}/{d.n_obs} 组 {len(acc)}", flush=True)
    for sv in sorted(set(k[0] for k in acc)):
        sub = {k[1:]: v for k, v in acc.items() if k[0] == sv}
        ctrl = {m: v for (m, l), v in sub.items() if l == a.control}
        if not ctrl:
            print(f"  [{sv}] 无对照, 跳过"); continue
        gcnt = sum(v[0] for v in ctrl.values()); gprob = gcnt / gcnt.sum()
        gmean = sum(v[1] for v in ctrl.values()) / sum(v[2] for v in ctrl.values())
        T, G = len(targets), len(genes)
        tc = np.zeros((T, G)); tcpm = np.zeros((T, G)); mp = np.zeros((T, G)); mc = np.zeros((T, G)); tn = np.zeros(T, np.int32)
        tsq = np.zeros((T, G)); csq = np.zeros((T, G)); cn = np.zeros(T)
        gsq = sum(v[3] for v in ctrl.values()) / sum(v[2] for v in ctrl.values()); gn = sum(v[2] for v in ctrl.values())
        for ti, t in enumerate(targets):
            rows = [(m, v) for (m, l), v in sub.items() if l == t]
            if not rows:
                continue
            tn[ti] = sum(v[2] for _, v in rows); tc[ti] = sum(v[0] for _, v in rows); tcpm[ti] = sum(v[1] for _, v in rows) / tn[ti]; tsq[ti] = sum(v[3] for _, v in rows) / tn[ti]
            lib = tc[ti].sum()
            for m, v in rows:
                c = ctrl.get(m)
                p, mn, sqm, nn = (gprob, gmean, gsq, gn) if c is None else (c[0] / max(c[0].sum(), 1), c[1] / c[2], c[3] / c[2], c[2])
                mp[ti] += v[0].sum() / max(lib, 1) * p; mc[ti] += v[2] / tn[ti] * mn; csq[ti] += v[2] / tn[ti] * sqm; cn[ti] += v[2] / tn[ti] * nn
        name = f"{a.prefix}{sv}" if a.split else a.prefix.rstrip("_")
        np.savez_compressed(data / f"{name}_full_statistics_var.npz", source=np.asarray(name), targets=np.asarray(targets),
                            genes=np.asarray(genes), n_cells=tn, target_count_sums=tc.astype(np.float32),
                            target_mean_cpm=tcpm.astype(np.float32), matched_control_probability=mp.astype(np.float32),
                            matched_control_mean_cpm=mc.astype(np.float32), global_control_probability=gprob.astype(np.float32),
                            global_control_mean_cpm=gmean.astype(np.float32), measured_genes=measured,
                            target_cpm_sqmean=tsq.astype(np.float32), matched_control_cpm_sqmean=csq.astype(np.float32), matched_control_n=cn.astype(np.float32))
        print(f"Saved {name}: 靶有细胞 {(tn >= 20).sum()}/{T}(≥20), 官方靶 {(tn[:300] >= 20).sum()}, 对照细胞 {sum(v[2] for v in ctrl.values())}", flush=True)


if __name__ == "__main__":
    main()
