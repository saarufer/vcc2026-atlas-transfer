"""Rank shaping within depth bins: redistribute a gene's counts among cells of similar library size, so both the
per-gene count total and the depth-normalised mean (what the scorer's lfc / nmae use) stay ~fixed, while the
within-group ranks against controls shift in the predicted direction.
  up   : within bin, new = (1-rho)*x + rho*bin_mean
  down : within bin, keep (1-rho) of each cell, give removed mass to the bin's top-k cells
"""
import numpy as np


def _round_cols(F, totals):
    I = np.floor(F).astype(np.int64)
    rem = np.rint(totals - I.sum(0)).astype(np.int64)
    frac = F - I
    for j in np.flatnonzero(rem > 0):
        k = int(rem[j]); idx = np.argpartition(-frac[:, j], k - 1)[:k]; I[idx, j] += 1
    return I


def _shape_block(C, sel_up, sel_dn, rho, k):
    out = C.copy()
    r = np.broadcast_to(np.asarray(rho, dtype=np.float64), (C.shape[1],))
    if sel_up.any():
        U = C[:, sel_up]; ru = r[sel_up][None, :]; out[:, sel_up] = (1 - ru) * U + ru * U.mean(0, keepdims=True)
    if sel_dn.any():
        D = C[:, sel_dn]; rd = r[sel_dn]; F = (1 - rd)[None, :] * D; removed = rd * D.sum(0)
        top = np.argsort(-D, axis=0)[:k]
        cols = np.arange(D.shape[1])
        for r in range(k):
            F[top[r], cols] += removed / k
        out[:, sel_dn] = F
    return out


def rank_shape(counts, sign, rho, n_bins=None, topk_per_bin=2, min_total=5):
    import os
    n_bins = int(os.environ.get("RS_BINS", 10)) if n_bins is None else n_bins
    C = counts.astype(np.float64)
    totals = C.sum(0)
    sel_up = (sign > 0) & (totals >= min_total)
    sel_dn = (sign < 0) & (totals >= min_total)
    if not (sel_up.any() or sel_dn.any()):
        return counts
    depth = C.sum(1)
    order = np.argsort(depth)
    F = C.copy()
    for b in np.array_split(order, n_bins):
        F[b] = _shape_block(C[b], sel_up, sel_dn, rho, min(topk_per_bin, len(b)))
    sel = sel_up | sel_dn
    out = counts.copy()
    out[:, sel] = _round_cols(F[:, sel], totals[sel])
    return out
