"""逐细胞乘性发射(EMIT_MODE=fc): 每个发射细胞 = 其模板对照细胞谱 × 跨细胞共享的逐基因倍数 f, 行归一后按自身深度整数化。
f 用迭代比例拟合使深度加权伪批量 = 目标 bulk_probability(伪批量决定 pds/mse); 细胞平均 CPM 随之近似。
与 dual_moment_counts 的区别: 后者按深度倾斜重加权模板细胞, 本函数让每个细胞的分布整体移动(更像真实敲低)。"""
import numpy as np


def fc_emit(template, probability, bulk_probability, *, depths, seed, iterations=30, fit="bulk"):
    x = np.asarray(template, dtype=np.float64)
    x = x / np.maximum(x.sum(axis=1, keepdims=True), 1e-30)
    d = np.asarray(depths, dtype=np.float64).ravel()
    target = np.asarray(bulk_probability if fit == "bulk" else probability, dtype=np.float64)
    target = target / target.sum()
    z = d / d.sum()                                   # 伪批量 = Σ_i d_i q_i / Σ d
    base = z @ x if fit == "bulk" else x.mean(0)
    f = np.divide(target, base, out=np.zeros_like(target), where=base > 0)
    missing = (base == 0) & (target > 0)              # 模板全零的基因: 按目标概率均匀补到每个细胞
    for _ in range(iterations):
        q = x * f[None, :]
        q[:, missing] += target[missing][None, :]
        q /= np.maximum(q.sum(axis=1, keepdims=True), 1e-30)
        cur = z @ q if fit == "bulk" else q.mean(0)
        ratio = np.divide(target, cur, out=np.ones_like(target), where=cur > 0)
        if np.abs(cur - target).sum() < 1e-6:
            break
        f *= ratio
    q = x * f[None, :]; q[:, missing] += target[missing][None, :]
    q /= np.maximum(q.sum(axis=1, keepdims=True), 1e-30)
    expected = q * d[:, None]
    integer = np.floor(expected).astype(np.int32); fractions = expected - integer
    rng = np.random.default_rng(seed)
    for i in range(len(q)):                           # 与 dual_moment_counts 相同的系统残差分配: 精确保留每细胞深度
        residual = int(d[i]) - int(integer[i].sum())
        if residual > 0:
            cumulative = np.cumsum(fractions[i]); cumulative *= residual / cumulative[-1]
            locations = np.searchsorted(cumulative, np.arange(residual) + rng.random(), side="right")
            np.add.at(integer[i], np.minimum(locations, len(cumulative) - 1), 1)
    return integer


def hybrid_emit(template, probability, bulk_probability, *, depths, seed, strength=1.0):
    """混合发射(EMIT_MODE=hybrid): 先把模板细胞按单细胞通道目标做逐基因乘性预偏移(每个细胞分布整体移动),
    再用 dual_moment_counts 同时精确匹配两通道(保留单细胞/伪批量解耦)。strength=预偏移指数 s (f^s)。"""
    from model import dual_moment_counts
    x = np.asarray(template, dtype=np.float64)
    x = x / np.maximum(x.sum(axis=1, keepdims=True), 1e-30)
    p = np.asarray(probability, dtype=np.float64)
    m = x.mean(0)
    f = np.ones_like(m); ok = (m > 0) & (p > 0)
    f[ok] = (p[ok] / m[ok]) ** strength
    x2 = x * f[None, :]
    x2 /= np.maximum(x2.sum(axis=1, keepdims=True), 1e-30)
    return dual_moment_counts(x2, probability, bulk_probability, depths=depths, seed=seed)
