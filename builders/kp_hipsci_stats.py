"""HipSci (Feng 2026, 34 iPSC lines; only a per-target lfc table in our delta bank) -> synthetic AtlasShift npz.
Control profile = KOLF (iPSC) global control probability / mean CPM (proxy). Target probability = control ∝ fc,
fc decoded from the bank (constant-M encoding: fc = (M + Δ)/M). Pseudo counts = N_PSEUDO cells × KOLF mean depth.
Run: python kp_hipsci_stats.py DATA RAW  (needs DATA/KOLFstrong_full_statistics.npz)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad

N_PSEUDO = 100


def main(data, raw):
    data, raw = Path(data), Path(raw)
    k = np.load(data / "KOLFstrong_full_statistics.npz", allow_pickle=False)
    genes = k["genes"].astype(str); targets = k["targets"].astype(str)
    ctrl_p = k["global_control_probability"].astype(np.float64); ctrl_m = k["global_control_mean_cpm"].astype(np.float64)
    depth = float(np.median(k["target_count_sums"].sum(1)[k["n_cells"] > 0] / np.maximum(k["n_cells"][k["n_cells"] > 0], 1)))
    b = ad.read_h5ad("vcc_data/delta_bank/HipSci_GW.h5ad")
    M = np.asarray(b.uns["mu_ctrl"], np.float64).ravel()
    gi = pd.Series(np.arange(b.n_vars), index=[str(v) for v in b.var_names]); gi = gi[~gi.index.duplicated()]
    col = gi.reindex(genes).values; ok = ~np.isnan(col); ci = col[ok].astype(int)
    t2r = {str(t): i for i, t in enumerate(b.obs_names)}
    X = np.asarray(b.X, np.float64)
    T, G = len(targets), len(genes)
    tc = np.zeros((T, G)); tcpm = np.zeros((T, G)); tn = np.zeros(T, np.int32)
    for ti, t in enumerate(targets):
        r = t2r.get(t)
        if r is None:
            continue
        fc = np.ones(G); fc[ok] = np.clip((M[ci] + X[r, ci]) / M[ci], 0, 20)
        p = ctrl_p * fc; p /= p.sum()
        tc[ti] = p * depth * N_PSEUDO; tcpm[ti] = ctrl_m * fc; tn[ti] = N_PSEUDO
    measured = np.zeros(G, bool); measured[ok] = True
    mp = np.tile(ctrl_p, (T, 1)); mc = np.tile(ctrl_m, (T, 1))
    np.savez_compressed(data / "HipSci_full_statistics.npz", source=np.asarray("HipSci"), targets=targets, genes=genes,
                        n_cells=tn, target_count_sums=tc.astype(np.float32), target_mean_cpm=tcpm.astype(np.float32),
                        matched_control_probability=mp.astype(np.float32), matched_control_mean_cpm=mc.astype(np.float32),
                        global_control_probability=ctrl_p.astype(np.float32), global_control_mean_cpm=ctrl_m.astype(np.float32),
                        measured_genes=measured)
    print(f"Saved HipSci: 靶 {(tn > 0).sum()}/{T}, 官方靶 {(tn[:300] > 0).sum()}, 基因 {ok.sum()}, 伪深度 {depth:.0f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
