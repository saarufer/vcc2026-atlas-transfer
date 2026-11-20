# vcc2026-atlas-transfer

**Zero-shot CRISPRi response prediction in unseen cell lines by multi-atlas transfer and moment-matched count emission**
Team *Saarufer* — Arc Institute Virtual Cell Challenge 2026 ([中文版](README_zh.md))

| | Overall | pds | mse | nmae | fid | reach | jac |
|---|---|---|---|---|---|---|---|
| **Final (test D/E/F)** — rank **48 / 806** | **0.2529** | 0.765 | 0.298 | 0.211 | 0.056 | 0.178 | 0.010 |
| Best validation (A/B/C) | 0.2694 | 0.809 | 0.368 | 0.203 | 0.009 | 0.217 | 0.010 |

The method uses **only public perturbation data** and no deep model trained end to end. Every prediction is a transfer of measured CRISPRi responses from public atlases into the target cell line, rendered as integer counts.

---

## 1. Task

For each of three cell lines that have never been perturbed in any public dataset, predict 400 single cells (raw integer counts, 18,533 genes) for each of 300 CRISPRi knockdowns. The only target-side input is the line's own non-targeting control cells. Predictions are scored by `cell-eval2` (`vcc2026` preset) on six metrics, each rescaled so that 0 = "predict the cell line's true mean perturbation response for every target" and 1 = a split-half experimental replicate, then averaged.

Cell-line identities were not given. Matching control-cell expression profiles to CCLE, we identified validation A/B/C as Jurkat, HeLa and CAL-33, and final D/E/F as MOLM-13-like (AML), U-87 MG (glioblastoma) and a fibroblast/mesenchymal-like line.

## 2. Method overview

```
public CRISPRi atlases ──► per-source effects (centred) ──► all-source direction vote ──► reliability weights (HGB)
                                                                    │                         │
                         PIE zero-shot prior ─────────────────────► fusion ◄── value model (HGB)
                                                                    │
           target-line control cells ──► control-PC lever, low-expression gate, per-target amplitude κ
                                                                    │
                 MSE budget calibration (from control cells only) ──► amplitude
                                                                    │
                       dual-moment emission from real control cells ──► 400 integer-count cells / target
```

### 2.1 Sources
Responses for the 300 targets are taken from public CRISPRi Perturb-seq atlases:
- **Fusion sources:** Replogle K562 genome-wide, X-Atlas/Orion HCT116 and HEK293T, Arc H1 (VCC 2025 data), DLD-1 (GSE337988).
- **Voting sources:** VIPerturb-seq K562 genome-wide (Flex), KOLF2.1J iPSC pan-genome atlas, HipSci iPSC CRISPRi map, H1 genome-wide (GSE295214), McFaline-Figueroa glioma kinome screens (A172/T98G/U87MG), Jiang et al. six-line screens.
- **Priors:** Arc's PIE zero-shot model, CD4 T-cell CRISPRi, and a promoter-neighbour table (CRISPRi silences genes sharing a promoter with the target).

Every source is reduced to per-target pseudobulk statistics (`builders/`). The response is computed in two spaces: a per-cell log2 fold-change channel and a pseudobulk Δ channel.

### 2.2 Effect construction (`pipeline/predict_amp.py`)
1. **Source-internal centring.** Each source's mean response across all of its targets is subtracted. This removes lab/platform-specific shared components, which do not transfer between labs.
2. **All-source direction vote.** Each source contributes the unit vector of its response, weighted by √(cells). The vote is mixed into both channels (λ = 0.5) with the per-target norm preserved. Diversity of sources mattered far more than the precision of any single source.
3. **Per-target amplitude κ.** The mean pairwise cosine between sources measures how transferable a target's response is, and amplitude scales with κ^0.5 (clipped to [0.5, 3]).
4. **Reliability weights.** A gradient-boosted classifier, trained on held-out cell lines, predicts the probability that the voted direction of each (target, gene) is correct from 12 scale-free features (source agreement, z-scores, expression, vote rank…). The weights are (0.5 + p)^5, with norm restored.
5. **Value model.** A gradient-boosted regressor on the same features is centred per gene across targets and mixed into the pseudobulk direction (λ = 0.2).
6. **PIE prior.** Arc's PIE predictions are mixed into the single-cell channel (w = 0.5) and the pseudobulk channel (w = 0.1), on expressed genes only. PIE conditions on the target line's own control cells (log1p CP10k); the cell-line text key had almost no effect when replaced by a related proxy.
7. **Control-PC lever.** For the weakest 80 % of targets, the pseudobulk effect is down-weighted along the top 50 principal components of the target line's control cells. Those axes carry generic variation that makes weak targets look alike and hurts discrimination (PDS).
8. **Low-expression gate** (CPM 5–50) on the pseudobulk channel, plus power-lift and confidence-weighted rank shaping on the single-cell channel. These shape the per-cell distribution that the differential-expression metrics see, independently of the pseudobulk mean.

### 2.3 Amplitude from the MSE budget
`cell-eval2`'s sampling-corrected MSE subtracts a correction from each perturbation's error. That correction is capped by the prediction's own across-perturbation spread *B*, at a level we call the *claim*. Energy placed in the prediction is therefore "free" up to the claim, so the MSE-optimal amplitude satisfies **B ≈ claim**. The claim depends only on the jackknife sampling noise of 400-cell groups, which can be estimated **from the target line's control cells alone** (`scoring/claim_from_controls.py`). On a real Arc-platform dataset, perturbed groups have the same sampling noise as controls (ratio 1.03). We set each line's amplitude so that B = 0.98 × claim. This replaces leaderboard-tuned amplitudes with a rule that transfers to unseen cell lines without feedback.

### 2.4 Count emission
Each target's 400 cells are emitted from the target line's real control cells (two-cell pooled templates). A dual-moment fit matches two things simultaneously: the per-cell mean CPM (single-cell channel) and the depth-weighted pseudobulk (pseudobulk channel). The cells are then integerised, preserving each cell's sequencing depth. Because the two channels are fitted independently, the differential-expression metrics and the pseudobulk metrics can be optimised separately.

## 3. What we learned

- **Source diversity beats single-source precision.** Equal-weight unit-vector voting over many sources was consistently better than inverse-variance meta-analysis, similarity-based source weights or up-weighting the closest source. Concentrating weight on one source loses the noise cancellation of the vote.
- **The common-mode response is not transferable across labs.** The mean response of a cell line across all perturbations is the largest MSE error component. Estimates from other labs' atlases had cosine ≈ 0.2–0.45 with the truth, near 0 in the metric's own log1p(5e4·CPM) space. That held even for a same-cell-line source from another lab, and even when the deviation from the atlas average was modelled from control-profile features. Injecting it improves MSE slightly but damages perturbation discrimination.
- **Platform emulation, measured with an oracle bench.** Feeding the *true* pseudobulk of one half of a real Arc dataset into our emitter and scoring against the other half isolates the loss caused by emission alone. PDS, reach and nmae reached replicate level, and the official MSE is not limited by emission variance. Differential-expression significance (jac, fid) did show an emulation gap. Emitters that close it under perfect biology lowered scores with real predictions: realistic single-cell noise drowns weak, imperfect signals.
- **The measured quantity matters.** Signed-rank, noise-unit (z) and knockdown-potency re-expressions of source effects all transferred worse than plain unit vectors. The direction signal is carried by a minority of large-effect genes.
- **Validation without feedback.** With no score feedback in the final phase, we calibrated local folds (H1 thinned to official depth, HCT116, K562-Flex) against earlier official scores. Two local signals predicted official changes reliably: the sign of pseudobulk-MSE change on HCT116, and the sign of PDS change on H1. Gains that appeared only in the local DE metrics did not survive. We adopted a change only if it passed both signals, and froze the final model once a full ablation audit found 7 of 8 tuned components robust on a fold never used for tuning.

Data provenance: see [DATA_SOURCES.md](DATA_SOURCES.md).

## 4. Repository layout

```
pipeline/   predict_amp.py (prediction), model.py (source loading, centring, fusion, emission), rank_shape.py, sign_features.py, precision_fuse.py, emit_fc.py (experimental emitters)
builders/   source-statistics builders for each atlas; training scripts for the two HGB models
scoring/    mse_budget.py (budget diagnostic), claim_from_controls.py, relabel_final.py
models/     relweight_hgb.joblib, stackval_hgb.joblib
configs/    final.env — the exact switch set of the final submission
```

## 5. Reproduction

1. Download the challenge control cells (`vcc datasets download controls-final`) and the public atlases listed above into `vcc_data/` (see the paths in `builders/`).
2. Build source statistics for the final target panel with the scripts in `builders/`, into a data directory. In that directory, `context_A/B/C.h5ad` link to the D/E/F control files.
3. Run the prediction with the final configuration:
   ```bash
   set -a; source configs/final.env; set +a
   python pipeline/predict_amp.py --data-dir <data_dir> --output pred.h5ad
   python scoring/relabel_final.py pred.h5ad pred_DEF.h5ad        # A/B/C -> D/E/F labels
   vcc prep pred_DEF.h5ad -g gene_names.csv --perts pert_counts.csv -o prediction.vcc
   ```
   Every feature is controlled by an environment variable; switches not listed in `configs/final.env` are off. The pipeline is deterministic: we re-ran it and reproduced the submitted file byte for byte.

Dependencies: numpy, scipy, pandas, anndata, h5py, scikit-learn, joblib, pyarrow (`pipeline/requirements.txt`); `cell-eval2` for local scoring.

## 6. Acknowledgements and licence

The pipeline builds on the AtlasShift atlas-transfer baseline (MIT licence, see `LICENSE.upstream`). We thank the authors of all public datasets used: Replogle et al., Xaira X-Atlas/Orion, VIPerturb-seq, the KOLF2.1J perturbation atlas, the HipSci CRISPRi map (Feng et al.), GSE295214, GSE337988, McFaline-Figueroa et al., Jiang et al., and Marson lab CD4 T-cell screens. We also thank Arc Institute for the challenge, PIE and `cell-eval2`. Several components were inspired by the open exchange of ideas among challenge participants.

Released under the MIT licence (`LICENSE`).
