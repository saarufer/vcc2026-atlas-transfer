"""训练堆叠值模型(HGB 回归, 中心化用法): 特征 = line86 FN(P/vote 按折标准化) 去掉尺度依赖的 Pnorm_t; 目标 Ỹ/折尺度。
argv[1] = 留出折(可空) -> data/stackval_hgb{_noX}.joblib"""
import os, sys, numpy as np, joblib
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from line62_pds_fast import C as CD
FN = ["P", "vote", "zmean", "zmax", "npos", "nneg", "nsrc", "sgnmean", "logcpm", "kappa_t", "nsrc_t", "vote_abs_rank", "P_abs_rank"]
EX = sys.argv[1] if len(sys.argv) > 1 else ""
Xs, ys = [], []; rng = np.random.default_rng(0)
for f in [x for x in ["H1t", "HCT", "Jurkat", "HepG2", "K562e"] if x != EX]:
    d = dict(np.load(CD + f"stack_{f}.npz", allow_pickle=True)); m = (d["gw"] > 0)[None, :] & np.ones(d["P"].shape, bool)
    for k, o in enumerate(d["own"]):
        if o >= 0: m[k, int(o)] = False
    X = np.stack([d[n][m] for n in FN], 1).astype(np.float64); X[:, 0] /= np.std(X[:, 0]) + 1e-9; X[:, 1] /= np.std(X[:, 1]) + 1e-9
    y = d["Yt"][m] / np.std(d["Yt"][m]); idx = rng.choice(len(y), min(len(y), 1_500_000), replace=False); Xs.append(X[idx]); ys.append(y[idx])
mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500, l2_regularization=1.0, random_state=0)
mdl.fit(np.concatenate(Xs), np.concatenate(ys))
joblib.dump({"model": mdl, "features": FN}, f"vcc_data/kp_atlasshift/data/stackval_hgb{'_no' + EX if EX else ''}.joblib"); print("saved", EX)
