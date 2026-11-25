# Deep-learning exploration

([中文版](DEEP_LEARNING_zh.md))

The final pipeline is a structured atlas-transfer model, not an end-to-end deep network. That choice came out of two deep-learning lines that we built, trained on GPU and evaluated against the transfer baseline under the same protocol. This document records those experiments: what we built, how we tested it, and why neither line went into the final model.

Hardware: a single RTX 5090 (32 GB, sm_120) for development, with every design also checked to fit a 16 GB V100. Software: PyTorch 2.x with CUDA 13, FP16 mixed precision.

---

## 1. Why try deep learning

The transfer pipeline places measured responses from other cell lines into the target line. That works well for the direction of a response (PDS), but it cannot learn how a response *changes* with cellular context. Early on, the sampling-corrected MSE was stuck near the baseline, which suggested that a model of absolute expression conditioned on the target line might be needed. We tested that hypothesis in two ways:

1. **STATE**: a published set-based perturbation transformer, retrained on public CRISPRi data.
2. **Intervention-VC**: our own conditional flow-matching generator, plus a learned source-fusion model.

## 2. Line A: STATE (set-transformer perturbation model)

### 2.1 Setup
- The model is the open-source STATE transition model (Arc Institute, `arc-state`). It runs in an isolated environment, with the `transformers` version pinned for checkpoint compatibility.
- Architecture, ~155M parameters:
  - perturbation encoder (1.4M);
  - basal-cell encoder (11.9M);
  - bidirectional Llama backbone (50.4M);
  - output projection (13.3M);
  - full-gene up/down projection (78.4M).
- **Data contract:**
  - All datasets were mapped to a unified gene axis and unified `obs` fields.
  - Expression was normalised to log1p(CP10k). Without that, a 2.9× sequencing-depth difference between two sources dominated the signal.
  - Cell-line context comes from the basal (control) cells. A cell-type one-hot cannot generalise to an unseen line.
- **Balanced sampler**, stratified context → target → cells, so that the largest atlas does not dominate training.
- A 100-step smoke test ran at 19 it/s with loss falling 8.21 → 8.15. Full runs used about 3.4 GB of VRAM.

### 2.2 Evaluation protocol
- **Leave-one-cell-line-out (LOCO).** Every training fold excludes the held-out line completely.
- The model predicts single cells. These are aggregated to pseudobulk and compared with the truth in log space. Two read-outs:
  - **direct**: the predicted expression as is;
  - **Δ-reanchored**: the predicted response added to the held-out line's real control mean.
- Each read-out is reported as a ratio of the error of a mean-response baseline. A ratio below 1 means the model beats the baseline.
- The pre-registered decision rule: the model must beat the **donor-transfer baseline that also excludes the held-out line**. Beating a naive baseline was not enough.

### 2.3 Results

| Fold | Training data | Direct | Reanchored | Verdict |
|---|---|---|---|---|
| 1 | H1 (151 perturbations) → K562 | – | 1.41 → 1.65 (worsening) | fails; source too small |
| 2 | K562 (6,611 perturbations) → H1 | 4.10 | 1.07 | reanchoring essential |
| 2 (split) | seen / unseen targets | – | 1.071 / 1.070 | open-vocabulary works |
| decisive | K562 + HCT116 → HEK293T | – | **1.381** vs donor **1.219** | **STATE worse** |

What these results show:
- **Reanchoring is essential.** The direct output carries a large absolute-level error (4.10). Only the predicted response, added to real control cells, is usable.
- **Source richness matters more than the architecture.** Training on 151 perturbations failed. Training on 6,611 got close to the baseline.
- **Open-vocabulary generalisation works.** Targets never seen in training were predicted as well as seen targets (1.070 vs 1.071), so the gene-embedding route to unseen perturbations is sound.
- **The decisive test was negative.** We chose the fairest fold: same platform and a related epithelial lineage. There, STATE was 13 % worse than donor transfer. A paired bootstrap over 2,500 targets gave a gap of +0.162 (95 % CI [+0.158, +0.166]), and STATE was worse on 98 % of targets. Adding a second training context (HCT116) barely helped.
- **On the real six-metric scorer**, STATE averaged −0.156 relative to the transfer pipeline (PDS −0.037).
- **Pretrained weights.** The published checkpoint turned out to be demonstration weights. We also evaluated frozen foundation-model cell embeddings (SE-600M) as inputs, and they did not change the conclusion.

The line was closed, with this precise conclusion: *at public-data scale and with full-gene reanchoring, a set transformer adds no value over donor transfer*. That is a statement about this data regime, not a refutation of the approach.

### 2.4 Side result: differential-expression calibration
We diagnosed the model outputs with the official DE definitions:
- CPM, BH-adjusted Wilcoxon;
- gene universe restricted to control CPM > 5;
- target genes excluded.

The diagnosis showed that predictions called about 8× too many DE genes. One cause was **generic collapse**, where the same ~86 genes were called significant in more than half of all perturbations. We traced it to deterministic emission: a small, consistent fold-change applied to hundreds of cells is always "significant" under a rank test. Adding stochastic emission (heterogeneity or negative binomial) removed the artefact. However, the oracle ceiling for DE calibration was small (Jaccard 0.048 → 0.064), so we did not pursue it as a separate lever. The lesson carried into the final emitter: noise in emission is part of the model.

## 3. Line B: Intervention-VC (flow-matching generator and learned fusion)

### 3.1 Design
A conditional generative model of perturbed cells:

  P(X_pert | C_target, g, m, d, t)

where C_target is a set of control cells from the target line, g is the target gene, m the modality, d the dose and t the time.

- **ContextEncoder**: attention-pooled set encoder over control cells.
- **InterventionEmbed**: learned gene embedding plus modality, dose and time embeddings.
- **VelocityNet**: a flow-matching velocity field over all 18,533 genes.
  - Naive gene×gene attention would need about 5 GiB per cell per layer, or about 164 GiB for 32 cells.
  - We avoid that by letting genes interact through **M = 128 learned module tokens** (M×M attention). The gene dimension appears only in the input and output projections.
- **Loss** = flow-matching + population-mean + ranking terms, with a measured-gene mask so that genes missing from a dataset do not contribute.
- **Portability**:
  - FP16 AMP with GradScaler, and loss reductions in FP32;
  - no BF16 and no FlashAttention-2, so the model runs on V100 (sm_70).

### 3.2 Stage A: memory profiling and acceptance
The model has 70.1M parameters (d_model 512, 6 generator layers, 2 context layers). Acceptance tests:
- T1: forward pass gives full-gene output;
- T2: all 141 parameter tensors receive gradients;
- T3: changing unmeasured genes leaves the loss unchanged (Δloss = 0), so the mask really enters the loss;
- T4: an FP16 AMP step gives a finite gradient norm.

All four passed. Peak memory was measured on real training steps:

| batch (episodes) × cells | peak allocated |
|---|---|
| 8 × 32 | 1.50 GB |
| 32 × 64 | 1.64 GB |
| 32 × 128 | 1.80 GB |
| 64 × 64 | 2.41 GB |

Every configuration fits a 16 GB V100 with wide margin. One bug was caught here: the first draft flattened M×d_model straight to genes, which gave 1.2 billion parameters. The fix was to pool back to d_model before the output projection.

### 3.3 Stage B: population models
- **Episode builder.** It reads raw counts and samples context → target → cells, rotating cells across epochs.
- **Population branch.** It predicts log fold-change(context, target) by decoding a target embedding FiLM-modulated by the context profile.
  - 39.6M parameters, trained on 12 public contexts (64,616 episodes) with weighted MSE plus within-context InfoNCE.
  - A shuffled-target negative control scored at chance, which confirms that the model uses perturbation identity.
  - It still stayed below the transfer baseline on a held-out line (0.647 vs 0.715 discrimination).
- **Additive decomposition** into a shared target effect plus a context interaction, with an interaction penalty and context dropout: no improvement.
- **Diagnosis.** Predictions reconstructed from the learned embeddings were blurred compared with the raw source data (0.59–0.64 vs 0.74). The gap was **loss of per-target detail**, not context generalisation.
- Training fix: a zero-initialised output layer combined with an L2-normalised InfoNCE produced gradients of about 1e12, so GradScaler skipped every step. Small-std initialisation and an epsilon fixed it.

### 3.4 Stage C: learned reference fusion
This model does not regenerate responses. It learns **how to weight measured source responses**:
- α = softmax(MLP[query profile, source profile, source response, strength, leave-one-out agreement, coverage]);
- plus a learned gain;
- trained by leave-one-context-out meta-learning.

Results:
- It was the first model with validation MSE below the zero predictor (0.1008 vs 0.1068), and validation discrimination rose from 0.80 to 0.86.
- We needed a fast, noise-aware evaluator, so we built **PDSSim**. It simulates count thinning exactly at pseudobulk level (a sum of Binomials is a Binomial), runs in seconds, and reproduced the real-scorer ranking. It also showed that a single real evaluation has seed SD ≈ 0.015, so we made decisions only on multi-seed paired comparisons.
- 20-seed paired comparison against the hand-tuned fusion:
  - HCT116 fold: +0.003 ± 0.002;
  - HEK293T fold: −0.004 ± 0.002;
  - both clearly better than uniform averaging (+0.02).
- Re-training with an objective aligned to the official PDS geometry (target columns removed, within-context cosine ranking) gave no further gain.
- Group-wise leave-study-out tests found that the learned weights were more robust than hand-tuned routing on some lineages, but not better on average.

So the learned fusion reaches hand-tuned quality from data alone, but it does not exceed it. Its main lessons fed the final model:
- many diverse sources beat any single source;
- leave-one-out agreement is a safe way to weight a source;
- reliability is better learned per (target, gene) than per source.

That last point became the gradient-boosted reliability and value models in the final pipeline.

## 4. GPU in the final pipeline
- **PIE** zero-shot predictions (Arc's perturbation model) were run on GPU, conditioned on each target line's own control cells.
- **Local scoring** used `cell-eval2` with its GPU DE engine. We verified that the GPU and CPU engines agree to three decimals on all six metrics, so local DE metrics match the official engine.
- Neural alternatives to the boosted models were also tried and did not help (for example, a contrastive MLP for reliability weights).

## 5. Lessons
1. **At public-data scale, structure beats capacity.** There are fewer than ten same-modality CRISPRi cell lines, with a few thousand overlapping targets. Both a 155M-parameter transformer and a 70M-parameter generator lost to transferring measured responses directly.
2. **Predict responses, not states.** Every working variant reanchors a predicted *change* onto the target line's real control cells.
3. **Data richness decides.** 151 → 6,611 training perturbations moved the reanchored error from 1.65 to 1.07. Additional contexts help only when their lineage and platform are informative.
4. **Generalisation to unseen targets is not the bottleneck.** Seen and unseen targets scored the same. The bottleneck is how much a response changes between cell lines.
5. **Evaluation is part of the model.**
   - Proxy metrics that kept the target gene itself ranked methods wrongly.
   - Single-seed evaluations could not resolve differences below 0.03.
   - Pre-registered decision rules and paired multi-seed tests stopped us from adopting noise.
6. **Emission noise matters.** Deterministic emission creates artefacts in the significance-based metrics. Realistic noise removes them, but it also hides weak signal, so the emitter and the predictor have to be tuned together.

We expect the balance to shift when much larger private or newly released atlases cover many more cell lines. The architectures here (set-encoded context, module-token flow field, learned fusion) are ready for that regime.
