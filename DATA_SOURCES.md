# VCC2026 数据来源清单(Data sources)

按下表获取所用公开数据; "用途"标注其在最终管线中的角色。
Public data used by this method; the "用途 (role)" column gives each dataset's role in the final pipeline.

## 比赛数据 Challenge data
| 数据 | 获取方式 | 用途 |
|---|---|---|
| VCC2026 验证集对照 (A/B/C) | `vcc datasets download controls` (需注册 virtualcellchallenge.org) | val 阶段输入 |
| VCC2026 最终测试集对照 (D/E/F) | `vcc datasets download controls-final` (vcc-cli ≥ 0.3.0) | 最终提交输入 |
| VCC2025 H1 全量 (train/val/test, Arc Flex) | `gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/` (公开, 无需凭据) | 融合源 H1_2025; 本地 H1t 折真值 |

## 扰动图谱 Perturbation atlases
| 数据 | 出处 / 编号 | 用途 |
|---|---|---|
| Replogle 2022 K562 全基因组 / K562 必需 / RPE1 | figshare+ 10.25452/figshare.plus.20029387 | 主融合源 K562_GWPS; K562e 折 |
| X-Atlas/Orion HCT116、HEK293T | Hugging Face `Xaira-Therapeutics/X-Atlas-Orion` | 融合源; HCT116 本地折真值 |
| VIPerturb-seq K562 Flex 全基因组 | Zenodo 18460279 (VIPerturb-seq, Nature 2026) | 投票源 VIPflexGW; K562v 折真值 |
| KOLF2.1J iPSC 扰动图谱 (Pan-genome / Strong / Chromatin) | figshare+ 10.25452/figshare.plus.27261219.v1 (Doctor et al.) | 投票源 KOLFpan; KOLFstrong |
| HipSci iPSC CRISPRi 图谱 (Feng et al., Cell Genomics 2026) | figshare 26819743 (LFC), 27989294 (计数) | 投票源 HipSci |
| H1 全基因组 CRISPRi | GEO GSE295214 | 投票源 H1g |
| DLD-1 CRISPRi (Genentech) | GEO GSE337988 | 融合源 DLD1 |
| McFaline-Figueroa 胶质瘤激酶组 (A172/T98G/U87MG) | GEO GSE225775; PerturBench 处理版 mcfaline23_gxe_processed | 投票源 McFaline_* |
| Jiang et al. 六细胞系 CRISPRi | PerturBench 处理版 jiang24_processed (原始 GEO GSE281048) | 投票源 Jiang24_* |
| Nadig 2025 (TRADE) HepG2 / Jurkat | GEO GSE264667 | 本地零样本折真值 |
| Marson 实验室 CD4 T 细胞全基因组 CRISPRi | GWCD4i pseudobulk (Marson lab, CZI 发布) | CD4 先验 |
| A549 CRISPRi | GEO GSE337804 | 评估用(未进最终源池) |
| Jurkat CRISPRi | GEO GSE249595 | 评估用(数据质量不足, 未用) |
| ENCODE HepG2 敲低 | ENCODE | 评估用(未进最终源池) |
| scPerturb 集合 | Zenodo 13350497 / 10044268 | 探索 |
| sci-Plex (Srivatsan 2020) | GEO GSE139944 | 探索 |
| Tahoe-100M | Hugging Face `tahoebio/Tahoe-100M` | 探索 |
| ELEM-1200RS (bulk CRISPR KO, 63 细胞系) | Hugging Face `ELEM-Bio/ELEM-1200RS` (需申请) | 探索(未进最终源池) |
| DepMap Perturb-seq 试点 | figshare 10.6084/m9.figshare.33273600 | 探索 |

## 模型与注释 Models and annotations
| 数据 | 出处 | 用途 |
|---|---|---|
| Arc PIE (replogle_xdataset) | GitHub `ArcInstitute/pie`; Hugging Face `arcinstitute/PIE_*` | 零样本先验 |
| DepMap 24Q4 (表达, CRISPRGeneEffect) | figshare 27993248 | 细胞系身份识别(CCLE 比对) |
| GENCODE v47 TSS | GENCODE | 启动子邻近表 |
| MSigDB | MSigDB | 探索 |
| cell-eval2 | GitHub `ArcInstitute/cell-eval2` / PyPI | 本地打分 |

## 派生数据 Derived
源统计 (`*_full_statistics(_var).npz`)、本地折真值、PIE 预测、HGB 模型均可由上述原始数据用
本仓库的 `builders/` 重建; 两个 HGB 模型已随仓库发布。
最终提交 flens10221804 (SHA256 eca597642e18667f…) 可由 `configs/final.env` 逐字节复现。
