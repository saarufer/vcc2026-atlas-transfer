"""data_final 映射(A=D, B=E, C=F) 的预测 -> 官方 D/E/F 标签。Run: relabel_final.py IN.h5ad OUT.h5ad"""
import sys, anndata as ad, numpy as np, pandas as pd
a = ad.read_h5ad(sys.argv[1]); m = {"A": "D", "B": "E", "C": "F"}
for col in [c for c in a.obs.columns if c in ("context", "cell_context")]:
    a.obs[col] = pd.Categorical(a.obs[col].astype(str).map(lambda x: m.get(x, x)))
a.obs_names = [n.replace("A_", "D_", 1) if n.startswith("A_") else n.replace("B_", "E_", 1) if n.startswith("B_") else n.replace("C_", "F_", 1) if n.startswith("C_") else n for n in a.obs_names.astype(str)]
assert set(a.obs["context"].unique()) == {"D", "E", "F"}, a.obs["context"].unique()
a.write_h5ad(sys.argv[2]); print("relabeled", a.shape, a.obs["context"].value_counts().to_dict(), a.obs.columns.tolist())
