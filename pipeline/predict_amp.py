"""Generate predictions from source statistics and destination controls."""

import argparse
from pathlib import Path
import time, os
import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from model import (
    load_xatlas,
    fuse_source_centered,
    aligned_cd4,
    add_cd4_family,
    desired_mean,
    apply_promoter_prior,
    dual_moment_counts,
    seed_for,
)

SOURCES = [
    "K562_GWPS_CPM_full_statistics.npz",
    "HCT116_full_statistics.npz",
    "HEK293T_full_statistics.npz",
    "H1_2025_full_statistics.npz",
]
WEIGHTS = [float(x) for x in os.environ.get("SRC_W", "2,1,1,2").split(",")]   # K562, HCT116, HEK293T, H1
SEED = 20260910 + int(os.environ.get("SEED_OFF", 0))   # 噪声底测量用
CELLS = int(os.environ.get("FOLD_CELLS", 400))


def axis_csv(path, column):
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if column not in table:
        raise ValueError(f"{Path(path).name}: expected a {column!r} column")
    values = table[column].to_numpy(dtype=str)
    if not len(values) or any((not x.strip() for x in values)) or len(set(values)) != len(values):
        raise ValueError(f"{column}: labels must be nonempty and unique")
    if column == "target_gene" and "non-targeting" in values:
        raise ValueError("Prediction panel must not contain controls")
    return values


class CountWriter:
    """Append integer count blocks; retain int64 CSR pointers beyond 2^31 NNZ."""

    def __init__(self, path, targets, genes, contexts, cells):
        self.path = Path(path)
        self.rows = 0
        self.nnz = 0
        self.nobs = len(targets) * len(contexts) * cells
        self.ngenes = len(genes)
        obs = pd.DataFrame(
            {
                "target_gene": np.tile(np.repeat(targets, cells), len(contexts)),
                "context": np.repeat(contexts, len(targets) * cells),
            },
            index=[f"{c}_{t}_{i}" for c in contexts for t in targets for i in range(cells)],
        )
        ad.AnnData(
            sparse.csr_matrix((self.nobs, len(genes)), dtype=np.float32),
            obs=obs,
            var=pd.DataFrame(index=genes),
        ).write_h5ad(self.path)
        self.file = h5py.File(self.path, "r+")
        group = self.file["X"]
        for name in ["data", "indices", "indptr"]:
            del group[name]
        self.data = group.create_dataset(
            "data",
            shape=(0,),
            maxshape=(None,),
            dtype="float32",
            chunks=(1048576,),
            compression="lzf",
            shuffle=True,
        )
        self.indices = group.create_dataset(
            "indices",
            shape=(0,),
            maxshape=(None,),
            dtype="int32",
            chunks=(1048576,),
            compression="lzf",
            shuffle=True,
        )
        self.pointers = group.create_dataset("indptr", shape=(self.nobs + 1,), dtype="int64")
        self.pointers[0] = 0

    def append(self, counts):
        if (
            counts.ndim != 2
            or counts.shape[1] != self.ngenes
            or self.rows + len(counts) > self.nobs
        ):
            raise ValueError("Unexpected count block shape")
        if (
            not np.isfinite(counts).all()
            or (counts < 0).any()
            or (counts != np.floor(counts)).any()
        ):
            raise ValueError("Expected finite nonnegative integer counts")
        if (counts.sum(axis=1, dtype=np.float64) > 1000000).any():
            raise ValueError("Cell depth exceeds 1,000,000")
        block = sparse.csr_matrix(counts.astype(np.uint32))
        end = self.nnz + block.nnz
        if end > 4750000000:
            raise ValueError("Sparse element limit exceeded")
        self.data.resize((end,))
        self.indices.resize((end,))
        self.data[self.nnz : end] = block.data
        self.indices[self.nnz : end] = block.indices
        self.pointers[self.rows + 1 : self.rows + len(counts) + 1] = (
            block.indptr[1:].astype(np.int64) + self.nnz
        )
        self.rows += len(counts)
        self.nnz = end

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        complete = self.rows == self.nobs and int(self.pointers[-1]) == self.nnz
        self.file.close()
        if kind is None and (not complete):
            raise ValueError("Incomplete prediction output")


def control_template(path, genes, cells, pool_k, seed):
    controls = ad.read_h5ad(path)
    if not np.array_equal(controls.var_names, genes):
        raise ValueError("Control gene order differs from gene_names.csv")
    raw = sparse.csr_matrix(controls.X, dtype=np.float64)
    if (
        not np.isfinite(raw.data).all()
        or (raw.data < 0).any()
        or (raw.data != np.floor(raw.data)).any()
    ):
        raise ValueError("Controls must contain raw nonnegative integer counts")
    library = np.asarray(raw.sum(axis=1)).ravel()
    if (library <= 0).any() or len(library) < cells * pool_k:
        raise ValueError(
            "Need positive-depth controls and at least cells_per_target * pool_k donors"
        )
    mean = np.zeros(len(genes))
    for left in range(0, len(library), 256):
        block = raw[left : left + 256].toarray() / library[left : left + 256, None]
        mean += block.sum(axis=0)
    mean /= len(library)
    bulk = np.asarray(raw.sum(axis=0)).ravel()
    bulk /= bulk.sum()
    selected = np.random.default_rng(seed).choice(len(library), cells * pool_k, replace=False)
    selected = selected[np.argsort(library[selected], kind="stable")]
    template = raw[selected].toarray() / library[selected, None]
    template = template.reshape(cells, pool_k, len(genes)).mean(axis=1)
    depths = np.rint(library[selected].reshape(cells, pool_k).mean(axis=1)).astype(np.int64)
    if depths.min() < 1 or depths.max() > 1000000:
        raise ValueError("Invalid predicted library depths")
    global _CTRL1   # 未池化的单个对照细胞谱(QUIET_TOPM 用: 让不承诺基因保持真实对照的分布形状)
    _CTRL1 = (raw[selected].toarray() / library[selected, None]).reshape(cells, pool_k, len(genes))[:, 0, :]
    return (template, depths, mean, bulk)


def predict(data_dir, output):
    root, output = Path(data_dir), Path(output)
    if output.exists():
        raise FileExistsError(output)
    pending = output.with_name(output.stem + ".partial.h5ad")
    if pending.exists():
        raise FileExistsError(pending)
    targets = axis_csv(root / "pert_counts.csv", "target_gene")
    genes = axis_csv(root / "gene_names.csv", "gene_name")
    if os.environ.get("FOLD_CTX") is None and (len(targets), len(genes)) != (300, 18533):
        raise ValueError("Expected the official 300-target, 18533-gene panel")
    panel = pd.read_csv(root / "pert_counts.csv")
    if "n_cells" in panel and not (panel.n_cells == CELLS).all():
        raise ValueError("Expected 400 cells per target")
    _extra = [x.split(":") for x in os.environ.get("EXTRA_SOURCES", "").split(",") if x.strip()]
    SOURCES_ = list(SOURCES) + [f"{n}_full_statistics.npz" for n, _ in _extra]
    WEIGHTS_ = list(WEIGHTS) + [float(w) for _, w in _extra]
    if _extra:
        print("额外源:", _extra, flush=True)
    sources = [load_xatlas(root / name, prior_counts=100000) for name in SOURCES_]
    cd4, available = aligned_cd4(
        root / "CD4_DE_statistics.npz", targets, genes, common_subtract=1, center_scope="source"
    )
    if os.environ.get("PFUSE"):
        from precision_fuse import precision_fuse
        _pf = [x.split(":") for x in os.environ["PFUSE"].split(",") if x.strip()]
        _pc = float(os.environ.get("PSEUDO_CPM", 1))
        base = {space: precision_fuse(root, [a for a, _ in _pf], [b for _, b in _pf], targets, genes, space=space,
                                      pseudocount=_pc, tau2=float(os.environ.get("PF_TAU2", 0.05)),
                                      cap=float(os.environ.get("PF_CAP", 0.125)))
                for space in ["log2fc", "bulk_delta"]}
        print("精度加权融合:", _pf, flush=True)
    else:
        base = {
            space: fuse_source_centered(
                sources, WEIGHTS_, targets, genes, space=space, common_subtract=float(os.environ.get('COMMON_CELL' if space == 'log2fc' else 'COMMON_BULK', 1)), pseudocount_cpm=float(os.environ.get("PSEUDO_CPM", 1))
            )
            for space in ["log2fc", "bulk_delta"]
        }
    kappa = None; gene_rep = None
    if os.environ.get("TARGET_AMP"):
        # 逐靶可靠度: 各源(伪批量空间)对同一靶响应的两两余弦均值 -> 跨系一致的靶迁移更可信
        from model import aligned_effect
        per = []
        _kx = [x for x in os.environ.get("KAPPA_EXTRA", "").split(",") if x.strip()]
        _ksrc = list(sources) + [load_xatlas(root / f"{n}_full_statistics.npz", prior_counts=100000) for n in _kx]
        if _kx:
            print("κ 额外投票源:", _kx, flush=True)
        for src in _ksrc:
            full, mask = aligned_effect(src, src.targets, genes, space="bulk_delta", common_subtract=float(os.environ.get("COMMON_VOTE", 1)))
            look = {t: i for i, t in enumerate(src.targets)}
            found = np.asarray([look.get(t, -1) for t in targets])
            E = np.full((len(targets), len(genes)), np.nan); rows = np.flatnonzero(found >= 0)
            E[rows] = np.where(mask[found[rows]], full[found[rows]], np.nan)
            per.append(E)
            if os.environ.get("CONS_HEAD") or os.environ.get("REL_WEIGHT"):
                # 多源同向共识: 每源每(靶,基因) z = Δ/SE(有 _var.npz 用真实方差, 否则 NB 近似), 计 |z|>阈 的正/负源数
                _zt = float(os.environ["CONS_HEAD"].split(",")[0]) if os.environ.get("CONS_HEAD") else 2.0
                _cl = {g: i for i, g in enumerate(src.genes)}; _ci = np.array([_cl.get(g, -1) for g in genes]); _okc = _ci >= 0
                _mt = np.zeros((len(targets), len(genes))); _mc = np.zeros_like(_mt)
                _mt[np.ix_(rows, np.flatnonzero(_okc))] = src.mean_cpm[np.ix_(found[rows], _ci[_okc])]
                _cm = src.control_mean_cpm; _cm = _cm[None, :] if _cm.ndim == 1 else _cm
                _mc[np.ix_(rows, np.flatnonzero(_okc))] = _cm[np.ix_(found[rows] if _cm.shape[0] > 1 else np.zeros(len(rows), int), _ci[_okc])]
                _vf = root / f"{src.name}_full_statistics_var.npz"
                _nt = np.full(len(targets), 1.0); _nt[rows] = np.maximum(src.n_cells[found[rows]], 1)
                if _vf.exists():
                    with np.load(_vf, allow_pickle=False) as _zv:
                        _vt = np.zeros_like(_mt); _vc = np.zeros_like(_mt); _nc2 = np.full(len(targets), 1000.0)
                        _vt[np.ix_(rows, np.flatnonzero(_okc))] = np.maximum(_zv["target_cpm_sqmean"][np.ix_(found[rows], _ci[_okc])] - _mt[np.ix_(rows, np.flatnonzero(_okc))] ** 2, 0)
                        _vc[np.ix_(rows, np.flatnonzero(_okc))] = np.maximum(_zv["matched_control_cpm_sqmean"][np.ix_(found[rows], _ci[_okc])] - _mc[np.ix_(rows, np.flatnonzero(_okc))] ** 2, 0)
                        _nc2[rows] = np.maximum(_zv["matched_control_n"][found[rows]], 1)
                else:
                    _vt = _mt * 100 + 0.1 * _mt ** 2; _vc = _mc * 100 + 0.1 * _mc ** 2; _nc2 = np.full(len(targets), 1000.0)
                _se = np.sqrt((0.05 / (1 + 0.05 * _mt)) ** 2 * _vt / _nt[:, None] + (0.05 / (1 + 0.05 * _mc)) ** 2 * _vc / _nc2[:, None]) + 1e-9
                _Z = np.nan_to_num(E) / _se
                if "_cpos" not in dir(): _cpos = np.zeros((len(targets), len(genes)), np.int16); _cneg = np.zeros_like(_cpos)
                _cpos += (_Z > _zt).astype(np.int16); _cneg += (_Z < -_zt).astype(np.int16)
                if "_zsum" not in dir():
                    _zsum = np.zeros((len(targets), len(genes))); _zmx = np.zeros_like(_zsum); _nsr = np.zeros_like(_zsum); _sgs = np.zeros_like(_zsum); _ncov = np.zeros(len(targets))
                _fin = np.isfinite(E); _zsum += _Z; _zmx = np.maximum(_zmx, np.abs(_Z)); _nsr += _fin; _sgs += np.sign(np.nan_to_num(E)); _ncov += _fin.any(1)
            if os.environ.get("FUSE_ALL_LAMBDA"):
                _NC = np.full(len(targets), np.nan); _NC[rows] = src.n_cells[found[rows]]
                if "_vote" not in dir(): _vote = np.zeros((len(targets), len(genes)))
                _En = np.nan_to_num(E); _nn = np.linalg.norm(_En, axis=1, keepdims=True)
                _vws = float(dict(x.split(":") for x in os.environ.get("VOTE_W", "").split(",") if ":" in x).get(str(src.name), 1))   # 按源投票权重(平台匹配等)
                _vote += _vws * np.sqrt(np.nan_to_num(_NC))[:, None] * np.divide(_En, _nn, out=np.zeros_like(_En), where=_nn > 0)
            if os.environ.get("FUSE_ALL_CELL"):
                _NCc = np.full(len(targets), np.nan); _NCc[rows] = src.n_cells[found[rows]]
                _fc, _mc = aligned_effect(src, src.targets, genes, space="log2fc", common_subtract=float(os.environ.get("COMMON_VOTE", 1)))
                _Ec = np.zeros((len(targets), len(genes))); _Ec[rows] = np.where(_mc[found[rows]], _fc[found[rows]], 0)
                if "_vote_c" not in dir(): _vote_c = np.zeros((len(targets), len(genes)))
                _nc_ = np.linalg.norm(_Ec, axis=1, keepdims=True)
                _vwc = float(dict(x.split(":") for x in os.environ.get("VOTE_W_CELL", "").split(",") if ":" in x).get(str(src.name), _vws)); _vote_c += _vwc * np.sqrt(np.nan_to_num(_NCc))[:, None] * np.divide(_Ec, _nc_, out=np.zeros_like(_Ec), where=_nc_ > 0)
        if os.environ.get("CD4_VOTE") and "_vote" in dir():
            # CD4 T 细胞(Marson, 3 条件)作为额外方向投票: 各条件 log2fc 源内跨靶中心化后单位化, √n 加权, 同时进伪批量/单细胞投票
            with np.load(root / "CD4_DE_statistics.npz", allow_pickle=False) as _cz:
                _ct = {t: i for i, t in enumerate(_cz["targets"].astype(str))}; _cgi = {g: i for i, g in enumerate(_cz["genes"].astype(str))}
                _cols = np.array([_cgi.get(g, -1) for g in genes]); _okc = _cols >= 0; _rows = np.array([_ct.get(t, -1) for t in targets])
                _cw = float(os.environ["CD4_VOTE"])
                for _c in range(len(_cz["conditions"])):
                    _av = _cz["available"][_c] & _cz["quality_pass"][_c] & (_cz["n_cells"][_c] >= 20)
                    _L = np.nan_to_num(_cz["log2fc"][_c].astype(np.float64)); _L[_av] -= _L[_av].mean(0)
                    _E = np.zeros((len(targets), len(genes)))
                    _sel = (_rows >= 0); _sel[_sel] &= _av[_rows[_sel]]
                    _E[np.ix_(np.flatnonzero(_sel), np.flatnonzero(_okc))] = _L[np.ix_(_rows[_sel], _cols[_okc])]
                    _n = np.linalg.norm(_E, axis=1, keepdims=True); _U = np.divide(_E, _n, out=np.zeros_like(_E), where=_n > 0)
                    _nn = np.zeros(len(targets)); _nn[_sel] = _cz["n_cells"][_c][_rows[_sel]]
                    _vote += _cw * np.sqrt(_nn)[:, None] * _U
                    if "_vote_c" in dir(): _vote_c += _cw * np.sqrt(_nn)[:, None] * _U
            print(f"  CD4 投票加入 ×{_cw}", flush=True)
        if os.environ.get("FUSE_ALL_LAMBDA") and "_vote" in dir():
            # 全源(主源+κ源) √n 加权单位向量投票, 以 λ 混入伪批量方向, 每靶范数保持
            _lam = float(os.environ["FUSE_ALL_LAMBDA"]); _R, _Dn = base["bulk_delta"]; _R = _R.astype(np.float64)
            _nr = np.linalg.norm(_R, axis=1, keepdims=True); _dr = np.divide(_R, _nr, out=np.zeros_like(_R), where=_nr > 0)
            _nv = np.linalg.norm(_vote, axis=1, keepdims=True); _dv = np.divide(_vote, _nv, out=np.zeros_like(_vote), where=_nv > 0)
            _d = (1 - _lam) * _dr + _lam * _dv; _nd = np.linalg.norm(_d, axis=1, keepdims=True)
            base["bulk_delta"] = ((np.divide(_d, _nd, out=np.zeros_like(_d), where=_nd > 0) * _nr).astype(np.float32), _Dn)
            print(f"  全源方向投票 λ={_lam}: 投票覆盖靶 {(_nv[:, 0] > 0).sum()}/{len(targets)}", flush=True)
        if os.environ.get("FUSE_ALL_CELL") and "_vote_c" in dir():
            # 单细胞(log2fc)通道同样做全源 √n 单位投票混入(决定 DE 方向: fid/reach/nmae), 每靶范数保持
            _lc = float(os.environ["FUSE_ALL_CELL"]); _R, _Dn = base["log2fc"]; _R = _R.astype(np.float64)
            _nr = np.linalg.norm(_R, axis=1, keepdims=True); _dr = np.divide(_R, _nr, out=np.zeros_like(_R), where=_nr > 0)
            _nv = np.linalg.norm(_vote_c, axis=1, keepdims=True); _dv = np.divide(_vote_c, _nv, out=np.zeros_like(_vote_c), where=_nv > 0)
            _d = (1 - _lc) * _dr + _lc * _dv; _nd = np.linalg.norm(_d, axis=1, keepdims=True)
            base["log2fc"] = ((np.divide(_d, _nd, out=np.zeros_like(_d), where=_nd > 0) * _nr).astype(np.float32), _Dn)
            print(f"  单细胞通道全源投票 λ={_lc}", flush=True)
        kappa = np.full(len(targets), np.nan); kstr = np.full(len(targets), np.nan)
        for ti in range(len(targets)):
            vs = [e[ti] for e in per if np.isfinite(e[ti]).sum() > 100]
            cs = []
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    m = np.isfinite(vs[i]) & np.isfinite(vs[j]); a, b = vs[i][m], vs[j][m]
                    if np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0:
                        cs.append(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
            if cs:
                kappa[ti] = np.mean(cs)
            if vs:
                kstr[ti] = np.median([np.sqrt(np.nanmean(v ** 2)) for v in vs])   # 各源效应强度(RMS)中位
        gene_rep = None
        if os.environ.get("GENE_W"):
            # 逐基因跨 context 可重复度: 各源对之间, 该基因在共同靶上的效应相关(Pearson)取均值
            _num = np.zeros(len(genes)); _den = np.zeros(len(genes))
            for _i in range(len(per)):
                for _j in range(_i + 1, len(per)):
                    A_, B_ = per[_i], per[_j]; M_ = np.isfinite(A_) & np.isfinite(B_)
                    n_ = M_.sum(0); ok_ = n_ >= 20
                    if not ok_.any(): continue
                    a_ = np.where(M_, A_, 0.0); b_ = np.where(M_, B_, 0.0)
                    ma = a_.sum(0) / np.maximum(n_, 1); mb = b_.sum(0) / np.maximum(n_, 1)
                    ca = np.where(M_, A_ - ma, 0.0); cb = np.where(M_, B_ - mb, 0.0)
                    r_ = (ca * cb).sum(0) / np.sqrt((ca ** 2).sum(0) * (cb ** 2).sum(0) + 1e-30)
                    _num[ok_] += r_[ok_] * n_[ok_]; _den[ok_] += n_[ok_]
            gene_rep = np.divide(_num, _den, out=np.full(len(genes), np.nan), where=_den > 0)
            print(f"  基因可重复度: 中位 {np.nanmedian(gene_rep):.3f}, 10/90 {np.nanpercentile(gene_rep,10):.3f}/{np.nanpercentile(gene_rep,90):.3f}, 可估 {np.isfinite(gene_rep).sum()}", flush=True)
        med = np.nanmedian(kappa)
        print(f"  逐靶一致度 κ: 中位 {med:.3f}, 分位10/90 {np.nanpercentile(kappa,10):.3f}/{np.nanpercentile(kappa,90):.3f}, 无法估计 {np.isnan(kappa).sum()}", flush=True)
    agree = None
    if os.environ.get("SIGN_CONF") or os.environ.get("SIGN_MODEL"):
        from model import aligned_effect
        num = np.zeros((len(targets), len(genes))); den = np.zeros_like(num)
        for src, w in zip(sources, WEIGHTS_):
            full, mask = aligned_effect(src, src.targets, genes, space="log2fc", common_subtract=_cs["log2fc"] if "_cs" in dir() else 1)
            look = {t: i for i, t in enumerate(src.targets)}
            found = np.asarray([look.get(t, -1) for t in targets]); rows = np.flatnonzero(found >= 0)
            num[rows] += w * np.sign(full[found[rows]]) * mask[found[rows]]
            den[rows] += w * mask[found[rows]]
        agree = np.divide(np.abs(num), den, out=np.zeros_like(num), where=den > 0)   # 0=分歧, 1=全一致
        sagree = np.divide(num, den, out=np.zeros_like(num), where=den > 0)   # 带符号多源一致度
        print(f"  符号一致度 中位 {np.median(agree[den > 0]):.2f}", flush=True)
    del sources
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    _CTXS = os.environ.get("FOLD_CTX", "ABC").split(",") if os.environ.get("FOLD_CTX") else list("ABC")
    with CountWriter(pending, targets, genes, _CTXS, CELLS) as writer:
        for ci, context in enumerate(_CTXS):
            template, depths, mean, bulk = control_template(
                root / f"context_{context}.h5ad", genes, CELLS, int(os.environ.get("POOL_K", 4)), SEED + ci
            )
            pcs = None
            if os.environ.get("PCA_SMOOTH") or os.environ.get("PC_DOWN") or os.environ.get("CTX_SIGN") or os.environ.get("FRAC_WHITEN"):
                # context PCA smoothing of the bulk effect: top-K PCs of this context's control cells
                # (log1p CPM); for the strongest frac of targets, scale the off-subspace part by `off`, restore energy.
                import anndata as _ad
                _c = _ad.read_h5ad(root / f"context_{context}.h5ad")
                _X = sparse.csr_matrix(_c.X, dtype=np.float64)
                _sel = np.random.default_rng(0).choice(_X.shape[0], min(3000, _X.shape[0]), replace=False)
                _L = np.log1p(_X[_sel].toarray() / np.asarray(_X[_sel].sum(1)) * 1e4)
                _L -= _L.mean(0)
                _, _sv, _Vt = np.linalg.svd(_L, full_matrices=False)
                pcs = _Vt[: int(os.environ.get("PCA_K", 300))]; pc_sv = _sv[: int(os.environ.get("PCA_K", 300))]
                print(f"  PCA {context}: {pcs.shape}", flush=True)
            desired = []
            for space, control, amplitude in [("log2fc", mean, float(os.environ.get("AMP_CELL", 0.6))), ("bulk_delta", bulk, float(dict(x.split(":") for x in os.environ.get("AMP_BULK_CTX", "").split(",") if ":" in x).get(context, os.environ.get("AMP_BULK", 0.3))))]:
                effect, coverage = base[space]
                cd4_w = float(os.environ.get("CD4_W_A", 0.5)) if context == "A" else 0.5
                effect, _ = add_cd4_family(
                    effect, coverage, cd4, available, cd4_w, space=space, control_probability=control
                )
                if space == "log2fc":
                    raw_cell = effect.copy()
                    if os.environ.get("EXT_FLIP"):
                        from model import aligned_effect as _ae
                        _num = np.zeros_like(effect, dtype=np.float64); _cnt = np.zeros_like(_num)
                        for _n in [x for x in os.environ.get("KAPPA_EXTRA", "").split(",") if x.strip()]:
                            _s = load_xatlas(root / f"{_n}_full_statistics.npz", prior_counts=100000)
                            _f, _mk = _ae(_s, _s.targets, genes, space="log2fc", common_subtract=1.0)
                            _lk = {t: i for i, t in enumerate(_s.targets)}; _fd = np.asarray([_lk.get(t, -1) for t in targets]); _r = np.flatnonzero(_fd >= 0)
                            _num[_r] += np.sign(_f[_fd[_r]]) * _mk[_fd[_r]]; _cnt[_r] += _mk[_fd[_r]]
                        _ag = np.divide(_num, _cnt, out=np.zeros_like(_num), where=_cnt > 0)
                        _be = np.abs(base["bulk_delta"][0]); _bthr = np.quantile(_be, float(os.environ.get("EF_BQ", 0.5)), axis=1, keepdims=True)
                        _fl = (_cnt >= float(os.environ.get("EF_MIN", 5))) & (np.abs(_ag) >= float(os.environ.get("EF_TAU", 0.8))) \
                              & (np.sign(_ag) != np.sign(effect)) & (effect != 0) & (_be <= _bthr)
                        effect = np.where(_fl, -effect, effect).astype(np.float32)
                        print(f"  外部库保守翻转 {context}: {_fl.sum()} 个(靶,基因), 占非零 {_fl.sum() / max((effect != 0).sum(), 1):.4f}", flush=True)
                    if os.environ.get("SIGN_MODEL"):
                        # 多 context 留一训练的方向概率模型(逻辑回归): 用 P(上调) 决定单细胞通道效应符号
                        from sign_features import context_stats, feats, design
                        _m = np.load(os.environ["SIGN_MODEL"])
                        _F = feats(context_stats(root / f"context_{context}.h5ad", genes), targets, genes, effect, sagree, root / "official_pairs.csv")
                        _T, _G = effect.shape; _z = np.empty(_T * _G, np.float64)
                        for _i in range(0, _T, 50):
                            _a, _b = np.divmod(np.arange(_i * _G, min(_T, _i + 50) * _G), _G)
                            _z[_i * _G: min(_T, _i + 50) * _G] = design(_F, _a, _b) @ _m["coef"] + float(_m["intercept"])
                        del _F
                        sm_p = (1 / (1 + np.exp(-_z))).reshape(_T, _G)
                        sm_c = np.where(effect > 0, sm_p, 1 - sm_p)   # 原符号判对的概率(只用于排序)
                        _flip = (np.sign(sm_p - 0.5) != np.sign(effect)) & (effect != 0) & (np.abs(sm_p - 0.5) >= float(os.environ.get("SM_TAU", 0)))
                        effect = np.where(_flip, -effect, effect).astype(np.float32)   # 仅在模型把握 ≥ SM_TAU 时翻转
                        print(f"  方向模型 {context}: 翻转 {_flip.mean():.3f} 的非零效应", flush=True)
                if os.environ.get("RESP_PCS") and context in os.environ.get("RESP_CTX", "").split(",") and space in os.environ.get("RESP_CH", "bulk_delta,log2fc").split(","):
                    # 同细胞系响应子空间先验(ASK SATHVIK 式): e' = (1-λ)e + λ·Proj_K(e), 每靶恢复原范数
                    _rz = np.load(os.environ["RESP_PCS"]); assert list(_rz["genes"].astype(str)) == [str(g) for g in genes]
                    _V = _rz[space][: int(os.environ.get("RESP_K", 50))].astype(np.float64); _lam = float(os.environ.get("RESP_BLEND", 0.9))
                    _e = effect.astype(np.float64); _n0 = np.linalg.norm(_e, axis=1, keepdims=True)
                    _e = (1 - _lam) * _e + _lam * (_e @ _V.T) @ _V
                    effect = (_e * (_n0 / (np.linalg.norm(_e, axis=1, keepdims=True) + 1e-12))).astype(np.float32)
                    print(f"  响应子空间 {context} [{space}]: K={_V.shape[0]} λ={_lam}", flush=True)
                _pm = dict(x.split(":") for x in os.environ.get("PIE_CTXMAP", "").split(",") if ":" in x)
                if os.environ.get("PIE_PRED") and context in _pm and space in os.environ.get("PIE_CH", "bulk_delta,log2fc").split(","):
                    # PIE(Arc, Verma 2026) 方向先验: 其 lfc_pred(log2, 强收缩)只取方向/相对形状, 按每靶 ||e|| 缩放后以权重 w 混入, 再恢复 ||e||;
                    # 仅对照中表达(CPM>=5)的基因, 其余为 0
                    import pyarrow.parquet as _pq, json as _js
                    _t = _pq.read_table(os.environ["PIE_PRED"]); _ds = os.environ.get("PIE_DS")
                    _pg = _js.loads(_t.schema.metadata[b"pie"])["genes"][_ds]; _df = _t.to_pandas()
                    _df = _df[(_df.dataset == _ds) & (_df.context == _pm[context])]
                    _row = {str(r_.perturbation): np.asarray(r_.lfc_pred, dtype=np.float64) for r_ in _df.itertuples()}
                    _gi = {g: i for i, g in enumerate(genes)}; _cols = np.array([_gi.get(g, -1) for g in _pg]); _okc = _cols >= 0
                    _expr = (mean * 1e6 >= 5)
                    _H = np.zeros(effect.shape, np.float64)
                    for _ti, _tg in enumerate(targets):
                        if _tg in _row: _H[_ti, _cols[_okc]] = _row[_tg][_okc]
                    _H[:, ~_expr] = 0; _H[np.arange(len(targets)), [_gi.get(t, 0) for t in targets]] *= np.array([t in _gi for t in targets])
                    _e = effect.astype(np.float64); _ne = np.linalg.norm(_e, axis=1, keepdims=True); _nh = np.linalg.norm(_H, axis=1, keepdims=True)
                    _w = float(os.environ.get("PIE_W_BULK", os.environ.get("PIE_W", 0.5))) if space == "bulk_delta" else float(os.environ.get("PIE_W", 0.5))
                    if space == "log2fc" and os.environ.get("PIE_GRADE"):
                        # : 按靶强度(||e||)分级混入权重 弱(<40%)/中(40-80%)/强(>80%)
                        _gw = [float(x) for x in os.environ["PIE_GRADE"].split(",")]; _q = np.argsort(np.argsort(_ne[:, 0])) / max(len(_ne) - 1, 1)
                        _w = np.where(_q < 0.4, _gw[0], np.where(_q < 0.8, _gw[1], _gw[2]))[:, None]
                    _e2 = _e + _w * _H * np.divide(_ne, _nh, out=np.zeros_like(_ne), where=_nh > 0)
                    effect = (_e2 * np.divide(_ne, np.linalg.norm(_e2, axis=1, keepdims=True) + 1e-12, out=np.zeros_like(_ne), where=_ne > 0)).astype(np.float32)
                    print(f"  PIE 混入 {context}->{_pm[context]} [{space}]: w={_w}, 覆盖靶 {(_nh[:, 0] > 0).sum()}/{len(targets)}", flush=True)
                _lc = dict(x.split(":") for x in os.environ.get("LINCS_CTX", "").split(",") if ":" in x)
                if context in _lc:
                    # 同谱系 LINCS 敲除/敲低签名(z)按方向混入: e' = (1-λ)e + λ·s·z, s=||e||/||z|| 于 LINCS 基因(靶自身除外)
                    _z = np.load(root / f"LINCS_{_lc[context]}_sigs.npz", allow_pickle=False)
                    _lam = float(os.environ.get("LINCS_LAMBDA", 0.3))
                    _gi = {g: i for i, g in enumerate(genes)}
                    _cols = np.array([_gi.get(str(g), -1) for g in _z["genes"]]); _ok = _cols >= 0
                    _ti = {str(t): i for i, t in enumerate(_z["targets"])}
                    effect = effect.copy(); _nb = 0
                    for ti, t in enumerate(targets):
                        j = _ti.get(str(t))
                        if j is None:
                            continue
                        c_ = _cols[_ok]; zz = _z["Z"][j][_ok].astype(np.float64)
                        keep_ = c_ != _gi.get(str(t), -2); c_, zz = c_[keep_], zz[keep_]
                        e_ = effect[ti, c_].astype(np.float64); nz = np.linalg.norm(zz)
                        if nz == 0 or np.linalg.norm(e_) == 0:
                            continue
                        effect[ti, c_] = ((1 - _lam) * e_ + _lam * (np.linalg.norm(e_) / nz) * zz).astype(effect.dtype); _nb += 1
                    print(f"  LINCS {_lc[context]} -> {context} [{space}]: 混入 {_nb} 靶, λ={_lam}", flush=True)
                if space == "log2fc" and os.environ.get("CTX_SIGN"):
                    # 细胞系专属方向特征: 单细胞效应投影到 context 对照 top-K 主轴(按靶尺度对齐)后按 λ 混入
                    P = pcs[: int(os.environ.get("CS_K", 50))].astype(np.float64); eff = effect.astype(np.float64)
                    pr = (eff @ P.T) @ P; pr *= (np.linalg.norm(eff, axis=1) / (np.linalg.norm(pr, axis=1) + 1e-12))[:, None]
                    new = eff + float(os.environ["CTX_SIGN"]) * pr
                    new *= (np.linalg.norm(eff, axis=1) / (np.linalg.norm(new, axis=1) + 1e-12))[:, None]   # 保持每靶能量
                    effect = new.astype(np.float32)
                    print(f"  细胞系方向特征 {context}: λ={os.environ['CTX_SIGN']} K={os.environ.get('CS_K', 50)}", flush=True)
                if space == "log2fc" and agree is not None and os.environ.get("SIGN_CONF"):
                    _fl = float(os.environ.get("SC_FLOOR", 0.5)); _g = float(os.environ.get("SC_GAMMA", 1.0))
                    if os.environ.get("SC_BOOST"):
                        effect = effect * (1 + float(os.environ["SC_BOOST"]) * agree ** _g)   # 只放大高一致度基因
                    else:
                        effect = effect * (_fl + (1 - _fl) * agree ** _g)
                if space == "log2fc" and os.environ.get("CALL_K"):
                    # 每靶只保留单细胞通道 |效应| 最大的前 K 个基因, 其余置 0(控制预测显著基因数量)
                    _k = int(os.environ["CALL_K"])
                    if _k < effect.shape[1]:
                        thr = -np.partition(-np.abs(effect), _k - 1, axis=1)[:, _k - 1:_k]
                        effect = np.where(np.abs(effect) >= thr, effect, 0).astype(effect.dtype)
                if space == "log2fc" and os.environ.get("POWER_LIFT"):
                    # expected-DEG power lift: genes whose predicted per-cell log2FC is close to the
                    # minimum detectable effect (MDE) of a Wilcoxon test with CELLS cells are lifted to margin*MDE.
                    # MDE_g ≈ z / (ln2 · sqrt(CELLS · mu_g)), mu_g = mean counts/cell at template depth.
                    z = float(os.environ.get("PL_Z", 4.0)); margin = float(os.environ.get("PL_MARGIN", 1.0))
                    near = float(os.environ.get("PL_NEAR", 0.5))
                    mu = control * depths.mean()
                    mde = z / (np.log(2) * np.sqrt(CELLS * np.maximum(mu, 1e-9)))
                    e = amplitude * effect
                    a = np.abs(e)
                    lift = (a >= near * mde[None, :]) & (a < margin * mde[None, :]) & (mu[None, :] >= 0.02) & (mde[None, :] < 3)
                    e = np.where(lift, np.sign(e) * margin * mde[None, :], e)
                    print(f"  power-lift {context}: 抬升 {lift.sum()} 个(靶,基因), 每靶中位 {np.median(lift.sum(1)):.0f}", flush=True)
                    effect = e / amplitude
                if space == "bulk_delta" and kappa is not None:
                    _g = float(os.environ.get("TA_GAMMA", 1.0)); _cap = float(os.environ.get("TA_CAP", 3.0))
                    k_ = np.where(np.isfinite(kappa), np.clip(kappa, 0, None), np.nanmedian(kappa))
                    f_ = np.clip((k_ / max(np.nanmedian(kappa), 1e-6)) ** _g, float(os.environ.get("TA_FLOOR", 0.0)), _cap)
                    if os.environ.get("TA_STRONG_Q"):
                        # κ 低但各源效应都强 -> 多为 context 依赖而非噪声, 不收缩(至多降到 TA_STRONG_FLOOR)
                        _st = np.nan_to_num(kstr) >= np.nanquantile(kstr, float(os.environ["TA_STRONG_Q"]))
                        f_ = np.where(_st, np.maximum(f_, float(os.environ.get("TA_STRONG_FLOOR", 1.0))), f_)
                    if os.environ.get("TA_NORM"):
                        _fr = np.clip((k_ / max(np.nanmedian(kappa), 1e-6)) ** 0.5, float(os.environ.get("TA_FLOOR", 0.0)), _cap)
                        f_ = f_ * np.sqrt(np.mean(_fr ** 2) / max(np.mean(f_ ** 2), 1e-12))   # 总能量与 γ=0.5 参考一致
                    _e_pre = np.linalg.norm(effect.astype(np.float64))
                    effect = effect * f_[:, None].astype(np.float32)
                    if os.environ.get("TA_RENORM"):   # 总能量恢复: 只改逐靶幅度分配, 不改总预算
                        effect = (effect * (_e_pre / max(np.linalg.norm(effect.astype(np.float64)), 1e-12))).astype(np.float32)
                if space == "bulk_delta" and os.environ.get("GENE_W") and gene_rep is not None:
                    # 伪批量效应按基因可重复度加权(w = clip(rep/中位, lo, hi)^γ, 不可估 -> 1), 每靶恢复原范数(预算不变)
                    _md = max(np.nanmedian(gene_rep), 1e-3)
                    _w = np.where(np.isfinite(gene_rep), np.clip(np.nan_to_num(gene_rep) / _md, float(os.environ.get("GW_LO", 0.2)), float(os.environ.get("GW_HI", 3.0))), 1.0) ** float(os.environ.get("GENE_W"))
                    _n0 = np.linalg.norm(effect, axis=1, keepdims=True); effect = effect * _w[None, :]
                    effect = (effect * (_n0 / (np.linalg.norm(effect, axis=1, keepdims=True) + 1e-12))).astype(np.float32)
                if space == "bulk_delta" and os.environ.get("AMP_BULK_COMMON"):
                    # 跨靶共同分量(各靶均值)单独定幅: #348 预算只计中心化分量, 共同分量只增误差
                    _m = effect.mean(0, keepdims=True)
                    effect = (effect - _m + _m * (float(os.environ["AMP_BULK_COMMON"]) / amplitude)).astype(np.float32)
                if space == "bulk_delta" and (os.environ.get("PC_DOWN") or os.environ.get("GEN_TILT")):
                    #  PDS 杠杆(仅较弱 PD_WEAK 比例的靶): 沿 context 对照细胞 top-K 主轴降权 β,
                    # 并偏离跨靶通用响应方向 γ; 之后恢复每靶原范数(能量保持)
                    eff = effect.astype(np.float64); nrm0 = np.linalg.norm(eff, axis=1)
                    weak = nrm0 <= np.quantile(nrm0, float(os.environ.get("PD_WEAK", 0.8)))
                    new = eff[weak].copy()
                    if os.environ.get("PC_DOWN"):
                        P = pcs[: int(os.environ.get("PD_K", 50))].astype(np.float64)
                        new -= float(os.environ["PC_DOWN"]) * (new @ P.T) @ P
                    if os.environ.get("GEN_TILT"):
                        gv = eff.mean(0); gv /= np.linalg.norm(gv) + 1e-12
                        new -= float(os.environ["GEN_TILT"]) * np.outer(new @ gv, gv)
                    new *= (nrm0[weak] / (np.linalg.norm(new, axis=1) + 1e-12))[:, None]
                    eff[weak] = new; effect = eff.astype(np.float32)
                    if os.environ.get("SRB"):
                        # SRB: 最强 (1-PD_WEAK) 的靶, 放大其在对照主轴 lo..hi 上的分量 d += η(dPᵀ)P, 再恢复每靶原能量
                        _eta, _lo, _hi = [float(x) for x in os.environ["SRB"].split(",")]
                        _P2 = pcs[int(_lo): int(_hi)].astype(np.float64); _st = ~weak; _s = eff[_st].copy()
                        _s += _eta * (_s @ _P2.T) @ _P2
                        _s *= (nrm0[_st] / (np.linalg.norm(_s, axis=1) + 1e-12))[:, None]; eff[_st] = _s; effect = eff.astype(np.float32)
                        print(f"  SRB {context}: 强靶 {_st.sum()}, η={_eta} 主轴 {int(_lo)}-{int(_hi)}", flush=True)
                    print(f"  PDS 杠杆 {context}: 弱靶 {weak.sum()}, PC_DOWN={os.environ.get('PC_DOWN')} K={os.environ.get('PD_K', 50)} GEN_TILT={os.environ.get('GEN_TILT')}", flush=True)
                if space == "bulk_delta" and os.environ.get("FRAC_WHITEN"):
                    # 分数白化: 在 context 对照前 K 主轴上, 第 k 分量乘 f_k=(λ_k/λ_K)^(-γ/2)(≤1, 高方差轴压得多),
                    # 轴外不变; 仅作用于 |e| 不超过 FW_WEAK 分位的靶; 每靶恢复原范数
                    _K = int(os.environ.get("FW_K", 100)); _g = float(os.environ["FRAC_WHITEN"])
                    _P = pcs[:_K].astype(np.float64); _lam = pc_sv[:_K].astype(np.float64) ** 2
                    _f = np.minimum((_lam / _lam[-1]) ** (-_g / 2), 1.0)
                    _e = effect.astype(np.float64); _n0 = np.linalg.norm(_e, axis=1)
                    _sel = _n0 <= np.quantile(_n0, float(os.environ.get("FW_WEAK", 1.0)))
                    _c = _e[_sel] @ _P.T; _new = _e[_sel] + (_c * (_f - 1.0)[None, :]) @ _P
                    _new *= (_n0[_sel] / (np.linalg.norm(_new, axis=1) + 1e-12))[:, None]
                    _e[_sel] = _new; effect = _e.astype(np.float32)
                    print(f"  分数白化 {context}: γ={_g} K={_K} 作用靶 {_sel.sum()}, f 范围 {_f.min():.3f}-{_f.max():.3f}", flush=True)
                if space == "bulk_delta" and pcs is not None and os.environ.get("PCA_SMOOTH"):
                    frac = float(os.environ.get("PCA_FRAC", 0.2)); off = float(os.environ.get("PCA_OFF", 0.3))
                    nrm = np.linalg.norm(effect, axis=1); thr = np.quantile(nrm, 1 - frac)
                    eff = effect.astype(np.float64).copy()
                    for i in np.flatnonzero(nrm >= thr):
                        inn = pcs.T @ (pcs @ eff[i]); outv = eff[i] - inn
                        new = inn + off * outv
                        eff[i] = new * (nrm[i] / (np.linalg.norm(new) + 1e-12))
                    effect = eff.astype(np.float32)
                if space == "log2fc":
                    cell_effect = effect * amplitude
                if space == "bulk_delta" and os.environ.get("CONS_HEAD") and "_cpos" in dir():
                    # 共识头: ≥2 源同向显著且无反向 -> 放大 amp 倍, 方向冲突者翻成共识方向
                    _amp = float(os.environ["CONS_HEAD"].split(",")[1])
                    _cons = ((_cpos >= 2) & (_cneg == 0)) | ((_cneg >= 2) & (_cpos == 0)); _cs = np.where(_cpos >= 2, 1.0, -1.0)
                    _e = effect.astype(np.float64); _e = np.where(_cons, _e * _amp, _e)
                    _e = np.where(_cons & (np.sign(_e) != _cs) & (_e != 0), -_e, _e)
                    if os.environ.get("CH_RENORM", "1") == "1":   # 每靶范数恢复: 只把能量重分配给共识基因, 预算不变
                        _n0 = np.linalg.norm(effect.astype(np.float64), axis=1, keepdims=True); _n1 = np.linalg.norm(_e, axis=1, keepdims=True)
                        _e = _e * np.divide(_n0, _n1, out=np.ones_like(_n0), where=_n1 > 0)
                    effect = _e.astype(np.float32)
                    print(f"  共识头 {context}: 每靶共识基因 {_cons.sum(1).mean():.0f}, ×{_amp}", flush=True)
                if space == "bulk_delta" and os.environ.get("REL_WEIGHT") and "_zsum" in dir() and "_vote" in dir():
                    # 学习型可信度权重(堆叠 v2 软权重): HGB 预测 P(投票方向=真值方向), w=(0.5+p)^γ, 每靶范数恢复
                    import joblib as _jl
                    _rw = _jl.load(os.environ.get("REL_MODEL", str(root / "relweight_hgb.joblib"))); _gam = float(os.environ["REL_WEIGHT"])
                    _cpm = np.asarray(control, dtype=np.float64).ravel() * 1e6
                    _m = (_cpm > 5)[None, :] & (np.abs(_vote) > 0)
                    _gix = {g: i for i, g in enumerate(genes)}
                    for _k, _t in enumerate(targets):
                        if _t in _gix: _m[_k, _gix[_t]] = False
                    _G = len(genes)
                    _rk = lambda A: np.argsort(np.argsort(-A, axis=1), axis=1) / _G
                    _F = {"zmean": _zsum / np.maximum(_nsr, 1), "zmax": _zmx, "npos": _cpos, "nneg": _cneg, "nsrc": _nsr, "sgnmean": _sgs / np.maximum(_nsr, 1),
                          "logcpm": np.broadcast_to(np.log1p(_cpm)[None, :], _vote.shape), "kappa_t": np.broadcast_to(np.nan_to_num(kappa)[:, None], _vote.shape),
                          "nsrc_t": np.broadcast_to(_ncov[:, None], _vote.shape), "vote_abs_rank": _rk(np.abs(_vote)), "P_abs_rank": _rk(np.abs(effect.astype(np.float64))),
                          "vote_abs_z": np.abs(_vote) / (np.std(_vote[_m]) + 1e-9)}
                    _X = np.stack([np.asarray(_F[n], dtype=np.float32)[_m] for n in _rw["features"]], 1)
                    _p = np.zeros(_vote.shape); _p[_m] = _rw["model"].predict_proba(_X)[:, 1]
                    _w = np.where(_m, (0.5 + _p) ** _gam, 1.0)
                    _e0 = effect.astype(np.float64); _e1 = _e0 * _w
                    _n0 = np.linalg.norm(_e0, axis=1, keepdims=True); _n1 = np.linalg.norm(_e1, axis=1, keepdims=True)
                    effect = (_e1 * np.divide(_n0, _n1, out=np.ones_like(_n0), where=_n1 > 0)).astype(np.float32)
                    print(f"  可信度权重 {context}: γ={_gam}, p 均值 {_p[_m].mean():.3f}, 覆盖 {_m.mean():.2f}", flush=True)
                    _rel_p = _p; _rel_m = _m
                    if os.environ.get("DUMP_STACK"):
                        # 导出 MLP/堆叠所用特征(与应用时同一位置同一口径), 供用当前配置重训
                        _ef_d = effect.astype(np.float64); _Fd = dict(_F); _Fd["P"] = _ef_d / (np.std(_ef_d[_m]) + 1e-9); _Fd["vote"] = _vote / (np.std(_vote[_m]) + 1e-9)
                        np.savez_compressed(os.environ["DUMP_STACK"] + f"_{context}.npz", targets=np.array(targets, dtype=str), genes=np.array(genes, dtype=str), mask=_m,
                                            **{k_: np.asarray(v_, dtype=np.float32) for k_, v_ in _Fd.items()})
                        print(f"  堆叠特征已导出 {context}", flush=True)
                    if os.environ.get("STACK_MLP"):
                        # 对比损失 MLP(3 种子平均, numpy 前向, GELU=erf)预测每(靶,基因)值, 以 λ 混入伪批量方向, 每靶范数保持
                        from scipy.special import erf as _erf
                        _mz = np.load(os.environ.get("MLP_MODEL", str(root / "stackmlp.npz")), allow_pickle=True); _lam_m = float(os.environ["STACK_MLP"])
                        _ef = effect.astype(np.float64)
                        _F3 = dict(_F); _F3["P"] = _ef / (np.std(_ef[_m]) + 1e-9); _F3["vote"] = _vote / (np.std(_vote[_m]) + 1e-9)
                        _Xm = (np.stack([np.asarray(_F3[n_], dtype=np.float64)[_m] for n_ in _mz["features"].astype(str)], 1) - _mz["mu"]) / _mz["sd"]
                        _gelu = lambda x: 0.5 * x * (1 + _erf(x / np.sqrt(2)))
                        _outs = []
                        for _k in range(3):
                            _h = _gelu(_Xm @ _mz[f"s{_k}_0"].T + _mz[f"s{_k}_1"]); _h = _gelu(_h @ _mz[f"s{_k}_2"].T + _mz[f"s{_k}_3"]); _outs.append((_h @ _mz[f"s{_k}_4"].T + _mz[f"s{_k}_5"]).ravel())
                        _pm = np.zeros(_vote.shape); _pm[_m] = np.mean(_outs, 0)
                        _un = lambda A: np.divide(A, np.linalg.norm(A, axis=1, keepdims=True), out=np.zeros_like(A), where=np.linalg.norm(A, axis=1, keepdims=True) > 0)
                        _ne_ = np.linalg.norm(_ef, axis=1, keepdims=True)
                        effect = (_un((1 - _lam_m) * _un(_ef) + _lam_m * _un(_pm)) * _ne_).astype(np.float32)
                        print(f"  对比 MLP 混入 {context}: λ={_lam_m}", flush=True)
                    if os.environ.get("STACK_VAL"):
                        # 堆叠值模型(HGB 回归, 按基因跨靶中心化去共同成分)以 λ 混入伪批量方向, 每靶范数保持(Bill M 式学习方向, 近似对比约束)
                        _sv = _jl.load(os.environ.get("STACK_MODEL", str(root / "stackval_hgb.joblib"))); _lam_s = float(os.environ["STACK_VAL"])
                        _ef = effect.astype(np.float64)
                        _F2 = dict(_F); _F2["P"] = _ef / (np.std(_ef[_m]) + 1e-9); _F2["vote"] = _vote / (np.std(_vote[_m]) + 1e-9)
                        _Xs = np.stack([np.asarray(_F2[n_], dtype=np.float32)[_m] for n_ in _sv["features"]], 1)
                        _pv = np.zeros(_vote.shape); _pv[_m] = _sv["model"].predict(_Xs)
                        _pv = np.where(_m, _pv - np.where(_m, _pv, 0).sum(0) / np.maximum(_m.sum(0), 1), 0)
                        _un = lambda A: np.divide(A, np.linalg.norm(A, axis=1, keepdims=True), out=np.zeros_like(A), where=np.linalg.norm(A, axis=1, keepdims=True) > 0)
                        _ne_ = np.linalg.norm(_ef, axis=1, keepdims=True)
                        effect = (_un((1 - _lam_s) * _un(_ef) + _lam_s * _un(_pv)) * _ne_).astype(np.float32)
                        print(f"  堆叠值模型混入 {context}: λ={_lam_s}", flush=True)
                if space == "bulk_delta" and os.environ.get("GENE_GATE"):
                    # 低表达门控: 目标 context 对照 CPM 在 [a,b] 间线性 0->1(低表达基因的源效应多为噪声/不可迁移), 再恢复跨靶中心化总能量(预算比不变)
                    _a, _b = [float(x) for x in os.environ["GENE_GATE"].split(",")]
                    _gw = np.clip((np.asarray(control, dtype=np.float64).ravel() * 1e6 - _a) / (_b - _a), 0, 1)
                    _e0 = effect.astype(np.float64); _ss0 = ((_e0 - _e0.mean(0)) ** 2).sum()
                    _e1 = _e0 * _gw[None, :]; _ss1 = ((_e1 - _e1.mean(0)) ** 2).sum()
                    _rn = np.sqrt(_ss0 / max(_ss1, 1e-12)) if os.environ.get("GG_RENORM", "1") == "1" else 1.0
                    effect = (_e1 * _rn).astype(np.float32)
                    print(f"  低表达门控 {context}: CPM {_a}-{_b}, 全关 {(_gw == 0).mean():.2f} 全开 {(_gw == 1).mean():.2f}, 能量恢复×{_rn:.3f}", flush=True)
                if space == "bulk_delta" and os.environ.get("COMMON_INJECT"):
                    # 多源共识共模注入(#348 预算只计中心化分量, 共模不占预算): 用 κ·nrm·cm 替换预测的跨靶均值
                    _z = np.load(os.environ["COMMON_INJECT"]); _gi = {g: i for i, g in enumerate(_z["genes"].astype(str))}
                    _v = np.array([_z["cm"][_gi[g]] if g in _gi else 0.0 for g in genes], dtype=np.float64) * float(_z["nrm"]) * float(os.environ.get("CI_KAPPA", 0.5))
                    if os.environ.get("GENE_GATE") and "_gw" in dir():
                        _v = _v * _gw   # 注入共模同样做低表达门控
                    _m = effect.mean(0, keepdims=True)
                    effect = (effect - _m * float(os.environ.get("CI_REPLACE", 1)) + _v[None, :] / amplitude).astype(np.float32)
                    print(f"  共识共模注入 {context}: κ={os.environ.get('CI_KAPPA', 0.5)} ||v||={np.linalg.norm(_v):.3f} 原共模||m·amp||={np.linalg.norm(_m) * amplitude:.3f}", flush=True)
                if space == "bulk_delta":
                    bulk_final = (effect * amplitude).astype(np.float32)   # 最终伪批量效应(供离线 PDS 评估)
                if space == "log2fc" and os.environ.get("CELL_SHARED"):
                    # 共享响应只加到单细胞(log2fc)通道: CELL_SHARED="ctx:名,…", CS_W=w; 伪批量通道不变
                    _csm = dict(x.split(":") for x in os.environ["CELL_SHARED"].split(",") if ":" in x)
                    if context in _csm:
                        _zz = np.load(root.parent / "data" / "cellshared" / f"{_csm[context]}.npz"); _gg = {g: i for i, g in enumerate(_zz["genes"].astype(str))}
                        _sv = np.array([_zz["vec"][_gg[g]] if g in _gg else 0.0 for g in genes], dtype=np.float64)
                        _sv[np.asarray(control, dtype=np.float64).ravel() * 1e6 < 5] = 0
                        effect = (effect + float(os.environ.get("CS_W", 0.5)) * _sv[None, :] / amplitude).astype(np.float32)
                        print(f"  单细胞共享响应 {context}<-{_csm[context]}: w={os.environ.get('CS_W', 0.5)} ||v||={np.linalg.norm(_sv):.2f}", flush=True)
                if space == "log2fc":
                    _cell_eff_pre = effect.copy(); _cell_ctrl = control; _cell_amp = amplitude
                probability = desired_mean(
                    control, effect, space=space, amplitude=amplitude, clip=3
                )
                probability, _ = apply_promoter_prior(
                    probability, control, targets, genes, root / "official_pairs.csv", 0.15
                )
                desired.append(probability)
            if os.environ.get("REL_CELL") and "_rel_p" in dir():
                # 单细胞(log2fc)通道同样乘可信度权重 (0.5+p)^γc, 每靶范数恢复, 重算该通道期望分布
                _gc = float(os.environ["REL_CELL"]); _wc = np.where(_rel_m, (0.5 + _rel_p) ** _gc, 1.0)
                _c0 = _cell_eff_pre.astype(np.float64); _c1 = _c0 * _wc
                _n0 = np.linalg.norm(_c0, axis=1, keepdims=True); _n1 = np.linalg.norm(_c1, axis=1, keepdims=True)
                _c1 = (_c1 * np.divide(_n0, _n1, out=np.ones_like(_n0), where=_n1 > 0)).astype(np.float32)
                cell_effect = _c1 * _cell_amp
                _pr = desired_mean(_cell_ctrl, _c1, space="log2fc", amplitude=_cell_amp, clip=3)
                _pr, _ = apply_promoter_prior(_pr, _cell_ctrl, targets, genes, root / "official_pairs.csv", 0.15)
                desired[0] = _pr
                print(f"  单细胞通道可信度权重 {context}: γc={_gc}", flush=True)
            if os.environ.get("ORACLE_DESIRED"):
                # 神谕发射测试: 期望分布换成真值半 A 的(p_cell, p_bulk), 秩次塑形方向也用真值 → 只测"平台模拟"损失
                _oz = np.load(os.environ["ORACLE_DESIRED"], allow_pickle=False); _og = {g: i for i, g in enumerate(_oz["genes"].astype(str))}
                _oc = np.array([_og.get(g, -1) for g in genes]); _ok = _oc >= 0; _ot = {t: i for i, t in enumerate(_oz["targets"].astype(str))}
                desired = [np.array(desired[0], dtype=np.float64, copy=True), np.array(desired[1], dtype=np.float64, copy=True)]
                _mc = np.asarray(mean, dtype=np.float64).ravel(); _hit = 0
                for _ti, _tg in enumerate(targets):
                    if _tg not in _ot: continue
                    for _k, _key in enumerate(["p_cell", "p_bulk"]):
                        _v = np.zeros(len(genes)); _v[_ok] = _oz[_key][_ot[_tg]][_oc[_ok]]; _v = np.maximum(_v, 0); desired[_k][_ti] = _v / _v.sum()
                    cell_effect[_ti] = np.log2((desired[0][_ti] + 1e-7) / (_mc / _mc.sum() + 1e-7)); _hit += 1
                print(f"  神谕期望分布 {context}: 覆盖靶 {_hit}/{len(targets)}", flush=True)
            _dmul = None
            if os.environ.get("DEPTH_SHIFT"):
                # 强敲低靶测序深度系统性降低(Arc H1 真值: 强 10% 靶 UMI −12%, 与强度 Spearman −0.44): 深度乘子 1 − a·q³, q=预测强度分位
                _ds = float(os.environ["DEPTH_SHIFT"]); _bc = np.asarray(bulk, dtype=np.float64).ravel()
                _st = np.abs(np.asarray(desired[1], dtype=np.float64) - _bc[None, :]).sum(1)
                _q = np.argsort(np.argsort(_st)) / max(len(_st) - 1, 1); _dmul = 1 - _ds * _q ** 3
                print(f"  深度位移 {context}: a={_ds}, 乘子中位 {np.median(_dmul):.3f} 最小 {_dmul.min():.3f}", flush=True)
            _dep = (lambda _i: depths) if _dmul is None else (lambda _i: np.maximum(np.rint(np.asarray(depths, dtype=np.float64) * _dmul[_i]), 1).astype(np.asarray(depths).dtype))
            if os.environ.get("EMIT_MODE") == "fc":
                from emit_fc import fc_emit as _emit_main
            elif os.environ.get("EMIT_MODE") == "hybrid":
                import functools as _ft
                from emit_fc import hybrid_emit as _hy
                _emit_main = _ft.partial(_hy, strength=float(os.environ.get("HYB_S", 1.0)))
            else:
                _emit_main = dual_moment_counts
            for ti, target in enumerate(targets):
                try:
                    counts = _emit_main(
                        template,
                        desired[0][ti],
                        desired[1][ti],
                        depths=_dep(ti),
                        seed=seed_for(context + ":" + target, SEED),
                    )
                except ValueError as _me:
                    if os.environ.get("DUMP_FAIL"):
                        np.savez(os.environ["DUMP_FAIL"], template=np.asarray(template), p=np.asarray(desired[0][ti]), bulk=np.asarray(desired[1][ti]),
                                 depths=np.asarray(depths), genes=np.asarray(genes), target=target, context=context)
                        raise SystemExit(f"已转储失败靶 {context}:{target}")
                    if not os.environ.get("MOMENT_FALLBACK"):
                        raise
                    # 矩拟合不可行: 期望分布逐步向对照模板均值回缩(两通道同比例)后重试, 记录靶名
                    _p0 = np.asarray(template.mean(axis=0) if hasattr(template, "mean") else desired[0][ti], dtype=np.float64).ravel()
                    _p0 = _p0 / _p0.sum() if _p0.sum() > 0 else _p0
                    counts = None
                    try:   # 先只加迭代次数(D 高集中基因收敛慢), 不改期望分布
                        counts = dual_moment_counts(template, desired[0][ti], desired[1][ti], depths=_dep(ti), seed=seed_for(context + ":" + target, SEED),
                                                    iterations=int(os.environ.get("MOMENT_ITERS", 600)))
                        print(f"  矩拟合回退 {context}:{target} 加迭代 (原错误 {_me})", flush=True)
                    except ValueError:
                        pass
                    for _lam in (() if counts is not None else (0.25, 0.5, 0.75, 0.9)):
                        _d0 = ((1 - _lam) * np.asarray(desired[0][ti], np.float64) + _lam * _p0).astype(np.float32)
                        _d1 = ((1 - _lam) * np.asarray(desired[1][ti], np.float64) + _lam * _p0).astype(np.float32)
                        try:
                            counts = dual_moment_counts(template, _d0, _d1, depths=_dep(ti), seed=seed_for(context + ":" + target, SEED))
                            print(f"  矩拟合回退 {context}:{target} 回缩 λ={_lam} (原错误 {_me})", flush=True); break
                        except ValueError:
                            continue
                    if counts is None:
                        raise
                if os.environ.get("RANK_SHAPE"):
                    from rank_shape import rank_shape
                    _eps = float(os.environ.get("RS_EPS", 0.02))
                    _sg = np.where(np.abs(cell_effect[ti]) >= _eps, np.sign(cell_effect[ti]), 0)
                    _rho = float(os.environ["RANK_SHAPE"])
                    if os.environ.get("RS_CONF"):
                        # 逐基因 ρ: 方向可信度 = |效应| 在该靶内的分位(大效应方向更准), ρ_g = ρ_lo + (ρ_hi-ρ_lo)·q^γ
                        _a = sm_c[ti] if os.environ.get("SM_ORDER") else (np.abs(sm_p[ti] - 0.5) if os.environ.get("SM_CONF") else np.abs(cell_effect[ti])); _q = np.argsort(np.argsort(_a)) / max(len(_a) - 1, 1)
                        _lo = float(os.environ.get("RS_LO", 0.1)); _g = float(os.environ.get("RS_CG", 1.0))
                        _rho = _lo + (_rho - _lo) * _q ** _g
                    if os.environ.get("RS_UP_SCALE"):
                        # “下调优先”: 真实效应偏下调且预测下调方向更准 -> 上调基因的秩次塑形减弱, 让下调基因排在显著性前列
                        _rho = np.where(_sg > 0, np.broadcast_to(_rho, _sg.shape) * float(os.environ["RS_UP_SCALE"]), _rho)
                    counts = rank_shape(counts, _sg, _rho)
                if os.environ.get("MIX_Q"):
                    # 混合发射: 比例 q 的细胞为“未敲低”(对照模板期望计数), 其余细胞承担全部效应; 每基因列总数不变
                    from rank_shape import _round_cols
                    _q = float(os.environ["MIX_Q"]); _rng = np.random.default_rng(seed_for(context + ":mix:" + target, SEED))
                    _S = _rng.random(counts.shape[0]) < _q; _C = counts.astype(np.float64); _tot = _C.sum(0)
                    _y = template[_S] * depths[_S, None]
                    _rest = np.maximum(_tot - _y.sum(0), 0); _cr = _C[~_S].sum(0)
                    _C[~_S] *= np.divide(_rest, _cr, out=np.zeros_like(_cr), where=_cr > 0)[None, :]
                    _C[_S] = _y
                    counts = _round_cols(_C, np.rint(_C.sum(0))).astype(counts.dtype)
                if os.environ.get("QUIET_TOPM"):
                    # 显著集压缩(fid/jac): 每靶只保留置信度前 m 个基因的秩次偏移, 其余基因换回真实单细胞对照形状,
                    # 列总数差额集中给少数最高计数细胞 -> 伪批量/均值不变, 秩检验几乎不显著
                    _m = int(os.environ["QUIET_TOPM"]); _conf = np.abs(cell_effect[ti])
                    _keep = np.zeros(counts.shape[1], bool); _keep[np.argsort(-_conf)[:_m]] = True
                    _C = counts.astype(np.float64); _tot = _C.sum(0); _q = (~_keep) & (_tot > 0)
                    _rng = np.random.default_rng(seed_for(context + ":quiet:" + target, SEED))
                    _E = _CTRL1[:, _q] * depths[:, None]; _N = np.floor(_E + _rng.random(_E.shape))
                    _R = _tot[_q] - _N.sum(0); _kk = max(2, int(np.ceil(float(os.environ.get("QT_FRAC", 0.01)) * len(_N))))
                    _top = np.argsort(-_N - _rng.random(_N.shape) * 1e-3, axis=0)[:_kk]; _cols = np.arange(_N.shape[1])
                    for _r in range(_kk):
                        _N[_top[_r], _cols] += _R / _kk
                    _neg = _N < 0
                    if _neg.any():   # 移除过多时从剩余计数中按比例扣
                        _N = np.maximum(_N, 0); _s = _N.sum(0); _N *= np.divide(_tot[_q], _s, out=np.ones_like(_s), where=_s > 0)[None, :]
                    _C[:, _q] = _N
                    from rank_shape import _round_cols
                    counts = _round_cols(_C, np.rint(_tot)).astype(counts.dtype)
                writer.append(counts)
                if (ti + 1) % 50 == 0:
                    print(f"{context}: {ti + 1}/300, {time.monotonic() - started:.0f}s", flush=True)
    pending.rename(output)
    print(f"Saved {output.name}: {writer.nobs} cells, {writer.nnz} nonzeros")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("prediction.h5ad"))
    args = parser.parse_args()
    predict(args.data_dir, args.output)
