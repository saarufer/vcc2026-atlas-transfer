"""单基因方向特征(细胞系专属 + 迁移): 供方向概率模型训练/应用。
feats(context_h5ad, targets, genes, raw_cell, sagree, pairs_csv) -> dict of [T, G] float32 特征矩阵(均以“上调”为正向):
  rc      迁移单细胞 log2FC
  sa      带符号多源一致度
  gen     跨靶通用响应(rc 的靶均值, 按基因)
  proj    rc 投影到 context 对照 top-50 主轴(按靶尺度对齐)
  coexp   −corr(靶基因, 基因) 于 context 对照细胞(敲低靶 -> 正相关基因下调)
  nbr     启动子邻居(−1, 邻居随靶下调)
 以及按基因的 lexpr(对照 log 均值), det(检出率)。内存: 3000 细胞 × G 稠密 float32。
"""
import numpy as np, anndata as ad, pandas as pd
from scipy import sparse


def context_stats(context_h5ad, genes, n=3000, k=50, seed=0):
    c = ad.read_h5ad(context_h5ad, backed="r"); cg = {g: i for i, g in enumerate(c.var_names.astype(str))}
    sel = np.sort(np.random.default_rng(seed).choice(c.shape[0], min(n, c.shape[0]), replace=False))
    X = sparse.csr_matrix(c.X[sel], dtype=np.float32)
    col = np.array([cg.get(g, -1) for g in genes]); ok = col >= 0
    X = X[:, np.where(ok, col, 0)].toarray(); X[:, ~ok] = 0
    lib = X.sum(1, keepdims=True); lib[lib == 0] = 1
    L = np.log1p(X / lib * 1e4).astype(np.float32); del X
    lexpr = L.mean(0); det = (L > 0).mean(0)
    Lc = L - lexpr
    _, _, Vt = np.linalg.svd(Lc, full_matrices=False); P = Vt[:k].astype(np.float32)
    sd = Lc.std(0); sd[sd == 0] = 1
    Z = (Lc / sd).astype(np.float32); del Lc, L
    return dict(P=P, Z=Z, lexpr=lexpr.astype(np.float32), det=det.astype(np.float32))


def feats(cs, targets, genes, raw_cell, sagree, pairs_csv=None):
    gi = {g: i for i, g in enumerate(genes)}
    rc = raw_cell.astype(np.float32); T, G = rc.shape
    gen = np.broadcast_to(rc.mean(0, keepdims=True), (T, G)).astype(np.float32)
    pr = (rc @ cs["P"].T) @ cs["P"]; pr *= (np.linalg.norm(rc, axis=1) / (np.linalg.norm(pr, axis=1) + 1e-12))[:, None]
    coexp = np.zeros((T, G), np.float32); n = cs["Z"].shape[0]
    for i, t in enumerate(targets):
        j = gi.get(str(t))
        if j is not None:
            coexp[i] = -(cs["Z"][:, j] @ cs["Z"]) / n
    nbr = np.zeros((T, G), np.float32)
    if pairs_csv is not None:
        pp = pd.read_csv(pairs_csv); ti = {str(t): i for i, t in enumerate(targets)}
        for t, g in zip(pp.target.astype(str), pp.neighbor.astype(str)):
            if t in ti and g in gi:
                nbr[ti[t], gi[g]] = -1
    return dict(rc=rc, sa=sagree.astype(np.float32), gen=gen, proj=pr.astype(np.float32), coexp=coexp, nbr=nbr,
                lexpr=np.broadcast_to(cs["lexpr"], (T, G)), det=np.broadcast_to(cs["det"], (T, G)))


FEATS = ["rc", "sa", "gen", "proj", "coexp", "nbr", "lexpr", "det"]


def design(F, a, b):
    """行 = (靶 a, 基因 b) 对; 列: 有符号特征 + 幅度特征 + 交互。"""
    cols = [F[k][a, b] for k in FEATS]
    rc = F["rc"][a, b]
    cols += [np.abs(rc), np.sign(rc) * np.abs(F["sa"][a, b]), np.sign(rc) * np.sign(F["proj"][a, b])]
    return np.stack(cols, 1).astype(np.float32)
