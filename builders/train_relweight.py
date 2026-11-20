"""训练最终可信度分类器(全部 5 折): P(全源投票方向 = 真值方向), 按真值幅度加权; 只用尺度无关特征。
输出 vcc_data/kp_atlasshift/data/relweight_hgb.joblib(+ 特征名)。"""
import os, sys, numpy as np, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from line62_pds_fast import C as CD
FW = ["zmean", "zmax", "npos", "nneg", "nsrc", "sgnmean", "logcpm", "kappa_t", "nsrc_t", "vote_abs_rank", "P_abs_rank", "vote_abs_z"]
Xs, ys, ws = [], [], []; rng = np.random.default_rng(0)
EXCL = sys.argv[1] if len(sys.argv) > 1 else ""
for f in [x for x in ["H1t", "HCT", "Jurkat", "HepG2", "K562e"] if x != EXCL]:
    d = dict(np.load(CD + f"stack_{f}.npz", allow_pickle=True)); m = (d["gw"] > 0)[None, :] & np.ones(d["P"].shape, bool)
    for k, o in enumerate(d["own"]):
        if o >= 0: m[k, int(o)] = False
    m &= (np.abs(d["vote"]) > 0) & (d["Yt"] != 0)
    v = d["vote"]; d["vote_abs_z"] = np.abs(v) / (np.std(v[m]) + 1e-9)
    X = np.stack([d[n][m] for n in FW], 1); y = (np.sign(v[m]) == np.sign(d["Yt"][m])).astype(int)
    sc = np.std(d["Yt"][m]); w = np.abs(d["Yt"][m]) / sc
    idx = rng.choice(len(y), min(len(y), 1_500_000), replace=False); Xs.append(X[idx]); ys.append(y[idx]); ws.append(w[idx])
    print(f, len(y), "方向对比例", y.mean().round(3), flush=True)
clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=1000, l2_regularization=1.0, random_state=0)
clf.fit(np.concatenate(Xs), np.concatenate(ys), sample_weight=np.concatenate(ws))
joblib.dump({"model": clf, "features": FW}, f"vcc_data/kp_atlasshift/data/relweight_hgb{'_no' + EXCL if EXCL else ''}.joblib"); print("saved")
