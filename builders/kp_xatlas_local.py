"""Local drop-in for AtlasShift prepare_xatlas (kaipengm2, MIT): build HCT116/HEK293T_full_statistics.npz from the
local X-Atlas/Orion parquet batches instead of the remote slaf/lance scan. Same output fields and semantics:
per-target count sums + mean per-cell CPM; controls (Non-Targeting) capped at 250 per sample; per-sample matched
control probability / mean CPM weighted by each sample's share of the target's counts / cells.
Run: python kp_xatlas_local.py HCT116|HEK293T DATA_DIR RAW_DIR
"""
import os
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd, polars as pl

CONTROL = "Non-Targeting"
XA = Path("vcc_data/raw/xatlas")


def main(source, data, raw):
    data, raw = Path(data), Path(raw)
    official_targets = pd.read_csv(data / "pert_counts.csv")["target_gene"].astype(str).tolist()
    official_genes = pd.read_csv(data / "gene_names.csv")["gene_name"].astype(str).tolist()
    h1_targets = pd.read_csv(raw / "pert_counts_Training.csv")["target_gene"].astype(str).tolist()
    h1_genes = pd.read_csv(raw / "gene_names.csv"); gcol = "gene_name" if "gene_name" in h1_genes else h1_genes.columns[0]
    genes = list(dict.fromkeys(official_genes + h1_genes[gcol].astype(str).tolist()))
    targets = list(dict.fromkeys(official_targets + [t for t in h1_targets if t != "non-targeting"]))
    if os.environ.get("EXTRA_TARGETS"):   # 额外靶(如必需靶清单), 用于上下文映射训练
        targets = list(dict.fromkeys(targets + [l.strip() for l in open(os.environ["EXTRA_TARGETS"]) if l.strip()]))
    tset = set(targets) | {CONTROL}
    gl = {g: i for i, g in enumerate(genes)}
    gm = pl.read_parquet(XA / "metadata_gene.parquet")
    tok2col = {int(t): gl.get(str(s), -1) for t, s in zip(gm["gene_token_id"].to_list(), gm["gene_name"].to_list())}
    maxtok = max(tok2col); tokmap = np.full(maxtok + 1, -1, np.int64)
    for t, c in tok2col.items():
        tokmap[t] = c
    batches = sorted(glob.glob(str(XA / "data" / f"{source}_*.parquet")))
    keys, kidx, n_of = [], {}, []
    count_sums, cpm_sums, cpm_sq = [], [], []
    ctrl_seen = {}
    G = len(genes)
    for bi, b in enumerate(batches):
        df = pl.read_parquet(b, columns=["sample", "gene_target", "total_counts", "gene_token_id", "gene_expression"])
        df = df.filter(pl.col("gene_target").is_in(list(tset)))
        for sample, tgt, tot, tok, val in zip(df["sample"].to_list(), df["gene_target"].to_list(), df["total_counts"].to_list(),
                                              df["gene_token_id"].to_list(), df["gene_expression"].to_list()):
            if tgt == CONTROL:
                c = ctrl_seen.get(sample, 0)
                if c >= 250:
                    continue
                ctrl_seen[sample] = c + 1
            k = (sample, tgt)
            if k not in kidx:
                kidx[k] = len(keys); keys.append(k); n_of.append(0)
                count_sums.append(np.zeros(G, np.float32)); cpm_sums.append(np.zeros(G, np.float32)); cpm_sq.append(np.zeros(G, np.float32))
            i = kidx[k]; n_of[i] += 1
            tok = np.asarray(tok, np.int64); val = np.asarray(val, np.float64)
            ok = tok <= maxtok; cols = tokmap[tok[ok]]; v = val[ok]; m = cols >= 0
            np.add.at(count_sums[i], cols[m], v[m])
            _c = 1e6 * v[m] / max(tot, 1)
            np.add.at(cpm_sums[i], cols[m], _c); np.add.at(cpm_sq[i], cols[m], _c * _c)
        if bi % 10 == 0:
            print(f"  {source} batch {bi+1}/{len(batches)} groups {len(keys)}", flush=True)
    def _stack(lst):   # 逐行搬入预分配数组并释放列表元素, 峰值≈单份
        A = np.empty((len(lst), G), np.float32)
        for _i in range(len(lst)): A[_i] = lst[_i]; lst[_i] = None
        return A
    count_sums = _stack(count_sums); cpm_sums = _stack(cpm_sums); cpm_sq = _stack(cpm_sq); group_n = np.array(n_of)
    ctrl_by = {s: i for i, (s, l) in enumerate(keys) if l == CONTROL}
    crow = np.array(list(ctrl_by.values()))
    gcounts = count_sums[crow].sum(0); gprob = gcounts / gcounts.sum()
    gmean = cpm_sums[crow].sum(0) / group_n[crow].sum()
    shape = (len(targets), G)
    tc = np.zeros(shape); tcpm = np.zeros(shape); mp = np.zeros(shape); mc = np.zeros(shape); tn = np.zeros(len(targets), np.int32)
    tsq = np.zeros(shape); csq = np.zeros(shape); cn = np.zeros(len(targets))
    by_t = {}
    for i, (s, l) in enumerate(keys):
        by_t.setdefault(l, []).append(i)
    for ti, t in enumerate(targets):
        rows = by_t.get(t, [])
        if not rows:
            continue
        tn[ti] = group_n[rows].sum(); tc[ti] = count_sums[rows].sum(0); tcpm[ti] = cpm_sums[rows].sum(0) / tn[ti]; tsq[ti] = cpm_sq[rows].sum(0) / tn[ti]
        lib = tc[ti].sum()
        for r in rows:
            ci = ctrl_by.get(keys[r][0])
            if ci is None:
                p, mn, sqm, nn = gprob, gmean, cpm_sq[crow].sum(0) / group_n[crow].sum(), group_n[crow].sum()
            else:
                p = count_sums[ci] / max(count_sums[ci].sum(), 1); mn = cpm_sums[ci] / group_n[ci]; sqm = cpm_sq[ci] / group_n[ci]; nn = group_n[ci]
            mp[ti] += count_sums[r].sum() / max(lib, 1) * p
            mc[ti] += group_n[r] / tn[ti] * mn; csq[ti] += group_n[r] / tn[ti] * sqm; cn[ti] += group_n[r] / tn[ti] * nn
    measured = np.zeros(G, bool); measured[tokmap[tokmap >= 0]] = True
    np.savez_compressed(data / f"{source}_full_statistics_var.npz", source=np.asarray(source), targets=np.asarray(targets),
                        genes=np.asarray(genes), n_cells=tn, target_count_sums=tc.astype(np.float32),
                        target_mean_cpm=tcpm.astype(np.float32), matched_control_probability=mp.astype(np.float32),
                        matched_control_mean_cpm=mc.astype(np.float32), global_control_probability=gprob.astype(np.float32),
                        global_control_mean_cpm=gmean.astype(np.float32), measured_genes=measured,
                        target_cpm_sqmean=tsq.astype(np.float32), matched_control_cpm_sqmean=csq.astype(np.float32), matched_control_n=cn.astype(np.float32))
    print(f"Saved {source}: targets with cells {(tn > 0).sum()}/{len(targets)}, controls {group_n[crow].sum()}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
