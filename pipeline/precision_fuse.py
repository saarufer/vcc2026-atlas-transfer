"""Precision-weighted multi-source fusion : per (source, target, gene) delta-method variance of the
effect, weights w = cap_study * 1/(v + tau2), per-study share capped at max(cap, 1/n_studies_present), pooled
effect shrunk by its pooled variance. Effects come from AtlasShift's own aligned_effect (same centring / prior).
Needs *_full_statistics_var.npz (fields target_cpm_sqmean, matched_control_cpm_sqmean, matched_control_n).
"""
import os
import numpy as np
from model import load_xatlas, aligned_effect

LN2 = np.log(2.0)


def _var_log2(d, targets, genes, pseudocount):
    """Delta-method variance of log2((m_t+pc)/(m_c+pc)) aligned to (targets, genes); inf where unavailable."""
    rows = {str(t): i for i, t in enumerate(d["targets"].astype(str))}
    cols = {str(g): i for i, g in enumerate(d["genes"].astype(str))}
    ri = np.array([rows.get(str(t), -1) for t in targets]); ci = np.array([cols.get(str(g), -1) for g in genes])
    out = np.full((len(targets), len(genes)), np.inf)
    okr, okc = ri >= 0, ci >= 0
    R, C = ri[okr], ci[okc]
    n = d["n_cells"].astype(np.float64)[R][:, None]
    mt = d["target_mean_cpm"].astype(np.float64)[np.ix_(R, C)]
    st = d["target_cpm_sqmean"].astype(np.float64)[np.ix_(R, C)]
    mc = d["matched_control_mean_cpm"].astype(np.float64)
    sc = d["matched_control_cpm_sqmean"].astype(np.float64)
    mc = mc[np.ix_(R, C)] if mc.ndim == 2 else mc[C][None, :]
    sc = sc[np.ix_(R, C)] if sc.ndim == 2 else sc[C][None, :]
    nc = d["matched_control_n"].astype(np.float64)[R][:, None]
    vt = np.maximum(st - mt ** 2, 0) * n / np.maximum(n - 1, 1)
    vc = np.maximum(sc - mc ** 2, 0) * nc / np.maximum(nc - 1, 1)
    v = (vt / np.maximum(n, 1) / (mt + pseudocount) ** 2 + vc / np.maximum(nc, 1) / (mc + pseudocount) ** 2) / LN2 ** 2
    v = np.where(n >= 10, v, np.inf)
    out[np.ix_(np.flatnonzero(okr), np.flatnonzero(okc))] = v
    return out


def precision_fuse(root, names, studies, targets, genes, *, space, pseudocount=1.0, tau2=0.05, cap=0.125,
                   prior_counts=100000, common_subtract=1.0):
    """names: list of stats basenames (without _full_statistics_var.npz); studies: list of study labels (same length)."""
    T, G = len(targets), len(genes)
    E, V, S = [], [], []
    for nm, st in zip(names, studies):
        path = root / f"{nm}_full_statistics_var.npz"
        src = load_xatlas(path, prior_counts=prior_counts)
        full, mask = aligned_effect(src, src.targets, genes, space=space, pseudocount_cpm=pseudocount,
                                    common_subtract=common_subtract)
        look = {str(t): i for i, t in enumerate(src.targets)}
        found = np.array([look.get(str(t), -1) for t in targets]); rows = np.flatnonzero(found >= 0)
        e = np.zeros((T, G)); m = np.zeros((T, G), bool)
        e[rows] = full[found[rows]]; m[rows] = mask[found[rows]]
        with np.load(path, allow_pickle=False) as d:
            v = _var_log2(d, targets, genes, pseudocount)
        if space == "bulk_delta":
            v = v * LN2 ** 2
        v = np.where(m, v, np.inf)
        E.append(e); V.append(v); S.append(st)
    E = np.stack(E); V = np.stack(V)
    t2 = tau2 if space == "log2fc" else tau2 * LN2 ** 2
    W = np.where(np.isfinite(V), 1.0 / (V + t2), 0.0)
    ustud = sorted(set(S)); sidx = {s: [i for i, x in enumerate(S) if x == s] for s in ustud}
    for _ in range(4):                                         # per-study share cap
        Wk = np.stack([W[sidx[s]].sum(0) for s in ustud]); Wt = Wk.sum(0)
        npres = (Wk > 0).sum(0)
        lim = np.maximum(cap, 1.0 / np.maximum(npres, 1))
        share = np.divide(Wk, Wt, out=np.zeros_like(Wk), where=Wt > 0)
        f = np.where(share > lim, lim / np.maximum(share, 1e-12), 1.0)
        for k, s in enumerate(ustud):
            W[sidx[s]] *= f[k]
    Wt = W.sum(0)
    pooled = np.divide((W * E).sum(0), Wt, out=np.zeros((T, G)), where=Wt > 0)
    Vp = np.divide((W ** 2 * np.where(np.isfinite(V), V, 0)).sum(0) + 0, Wt ** 2, out=np.full((T, G), np.inf), where=Wt > 0)
    if float(os.environ.get("PF_SHRINK", 1)) > 0:
        pooled = pooled * pooled ** 2 / (pooled ** 2 + float(os.environ.get("PF_SHRINK", 1)) * Vp + 1e-12)
    return pooled.astype(np.float32), (Wt > 0).astype(np.float32)
