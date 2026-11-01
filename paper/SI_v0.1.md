# Supplementary Information
**Bounded stage-adaptive memory for remaining useful life prediction of turbofan engines**
**Review draft v0.1 — 2026-11-01.** Tables are generated from `evidence/` and `results/`; numbers are not hand-entered.
## S1 Data audit and preprocessing

| Subset | Train engines/rows | Test engines/rows | Conditions | Fault modes | Max test RUL |
|---|---|---|---|---|---|
| FD001 | 100 / 20,631 | 100 / 13,096 | 1 | 1 | 145 |
| FD002 | 260 / 53,759 | 259 / 33,991 | 6 | 1 | 194 |
| FD003 | 100 / 24,720 | 100 / 16,596 | 1 | 2 | 145 |
| FD004 | 249 / 61,249 | 248 / 41,214 | 6 | 2 | 195 |

The unified sensor subset is {2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21} (15 sensors). The constant sensors in FD001 are {1, 5, 10, 16, 18, 19}; FD002 and FD004 retain all 21 sensors but use the same 15-sensor subset. Windows: L=30, stride 1, RUL cap 125. Normalization: global z-score for FD001/FD003; k-means(k=6) on the three settings plus within-cluster z-score for FD002/FD004. Splits: engine-level 70/15/15, split seed 2026.
## S2 Secondary test metrics
**Table S1.** PHM score (sum, a1=13/a2=10) and unclipped-RMSE on the official test split (five seeds).

| Model | FD001 Score | FD002 Score | FD003 Score | FD004 Score | FD001 raw | FD002 raw | FD003 raw | FD004 raw |
|---|---|---|---|---|---|---|---|---|
| B1 CNN | 652.0 | 1581.8 | 1987.5 | 2186.3 | 17.694 | 28.761 | 19.418 | 29.446 |
| B2 LSTM | 462.6 | 1220.7 | 537.7 | 1192.0 | 15.444 | 27.772 | 15.691 | 26.792 |
| B3 BiGRU | 393.9 | 1005.2 | 404.9 | 1329.2 | 15.544 | 27.024 | 15.237 | 26.570 |
| B4 Transformer | 322.5 | 935.1 | 499.4 | 1396.1 | 15.044 | 26.396 | 15.478 | 26.375 |
| B5 TCN | 365.2 | 961.7 | 342.6 | 1249.7 | 15.118 | 26.609 | 14.211 | 26.108 |
| B8 fixed-rule | 433.3 | 964.9 | 433.2 | 1353.2 | 15.834 | 27.088 | 15.607 | 26.784 |
| Proposed | 491.7 | 953.7 | 385.7 | 1219.3 | 15.868 | 26.836 | 14.604 | 26.655 |

## S3 Per-seed test RMSE
**Table S2.** Per-seed test RMSE (cap 125) for the proposed model, the TCN, and the fixed-rule control.

| Subset | Model | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|---|
| FD001 | Proposed | 14.551 | 15.375 | 15.591 | 14.362 | 13.993 |
| FD001 | B5 TCN | 13.277 | 14.522 | 14.079 | 14.124 | 13.726 |
| FD001 | B8 fixed-rule | 14.450 | 14.864 | 15.231 | 14.658 | 14.100 |
| FD002 | Proposed | 14.009 | 13.956 | 14.109 | 14.751 | 14.307 |
| FD002 | B5 TCN | 14.046 | 14.117 | 13.498 | 14.012 | 14.058 |
| FD002 | B8 fixed-rule | 14.550 | 14.263 | 14.392 | 14.811 | 14.024 |
| FD003 | Proposed | 12.578 | 13.200 | 13.758 | 13.044 | 13.847 |
| FD003 | B5 TCN | 13.467 | 11.977 | 12.534 | 12.977 | 12.876 |
| FD003 | B8 fixed-rule | 13.829 | 14.006 | 14.192 | 14.600 | 14.588 |
| FD004 | Proposed | 14.152 | 14.718 | 15.629 | 14.301 | 13.982 |
| FD004 | B5 TCN | 13.942 | 14.564 | 14.607 | 14.220 | 14.640 |
| FD004 | B8 fixed-rule | 14.947 | 14.941 | 15.221 | 14.570 | 14.404 |

## S4 Gate-split screening
The gate split is a held-out 15% of the training engines. It is used for component screening, not as a second test set. **Table S3.** Gate-truncation RMSE (three truncation points per engine; mean over five seeds).

| Model | FD002 | FD004 |
|---|---|---|
| B1 CNN | 18.080 | 16.571 |
| B2 LSTM | 16.228 | 12.999 |
| B5 TCN | 14.829 | 12.671 |
| B8 fixed-rule | 15.474 | 12.919 |
| Proposed | 15.588 | 12.533 |

Gate screening used B1, B2, B5, B8, and the proposed model; B3/B4 were not part of the pre-registered gate controls.

## S5 Full ablation
**Table S4.** FD002 gate ablation (three seeds). Δ is the paired engine-level difference from full; brackets are 95% bootstrap CIs.

| Variant | RMSE | Δ% | 95% CI |
|---|---|---|---|
| C8 | 15.538 | +2.05 | [-0.20, +0.77] |
| L1 | 15.690 | +3.04 | [-0.07, +0.87] |
| dim48 | 15.162 | +1.21 | [-0.32, +0.61] |
| fixed | 15.438 | — | — |
| full | 15.428 | — | — |
| gamma05 | 15.437 | +0.30 | [-0.26, +0.35] |
| no_soft_cap | 18.496 | +23.72 | [+1.93, +4.32] |
| no_spectral | 15.427 | -0.00 | [-0.00, +0.00] |
| shuffle_ae | 15.455 | +0.16 | [-0.15, +0.20] |
| uni | 15.419 | +0.97 | [-0.40, +0.67] |
| w_o_Mv | 16.015 | +6.25 | [+0.14, +1.50] |

**Table S5.** FD004 gate ablation (two seeds).

| Variant | RMSE | Δ% | 95% CI |
|---|---|---|---|
| fixed | 12.618 | — | — |
| full | 12.210 | — | — |
| no_soft_cap | 15.035 | +29.87 | [+1.64, +4.39] |
| shuffle_ae | 12.194 | +0.43 | [-0.35, +0.40] |
| uni | 12.231 | +0.55 | [-0.56, +0.67] |
| w_o_Mv | 13.125 | +7.02 | [+0.12, +1.36] |

## S6 Validation-bias calibration (sensitivity)
A single additive offset was fitted on the validation truncation predictions and applied to the gate predictions. This analysis is not used in the main table. **Table S6.** Gate RMSE before/after the offset.

| Subset | Model | Before | After | Change |
|---|---|---|---|---|
| FD002 | Proposed | 15.588 | 15.330 | -1.66% |
| FD002 | B5 TCN | 14.829 | 14.492 | -2.28% |
| FD004 | Proposed | 12.533 | 12.478 | -0.44% |
| FD004 | B5 TCN | 12.671 | 12.533 | -1.09% |

## S7 Noise robustness
**Table S7.** Relative RMSE increase under test-time Gaussian noise on the 15 sensor channels (gate split; five seeds; descriptive only).

| Subset | Model | σ=0.1 | σ=0.2 |
|---|---|---|---|
| FD002 | Proposed | +1.53% | +4.56% |
| FD002 | B5 TCN | +2.09% | +6.27% |
| FD002 | B8 fixed-rule | +1.40% | +4.60% |
| FD004 | Proposed | +1.89% | +5.51% |
| FD004 | B5 TCN | +2.36% | +6.32% |
| FD004 | B8 fixed-rule | +1.91% | +5.42% |

## S8 Stability and operator checks
**Table S8.** Maximum state-norm ratio and operator bounds.

| Check | Value |
|---|---|
| bi-SRNL default gamma=0.025 | 0.480308 |
| B8 fixed-rule default | 0.478707 |
| gamma=5 controlled | 0.476148 |
| gamma=5 uncontrolled | 6.15125e+08 |
| Trained FD002 | 0.997898 |
| Trained FD004 | 0.999553 |
| max ||A||/α (FD002 trained) | 1.0000033 |
| max ||B||/(1−α) (FD002 trained) | 0.9981701 |
| max (||A||+||B||) (FD002 trained) | 0.9999922 |

The ratio max||A||/α is 1.0000033, within 3.3×10⁻⁶ of the bound; this is SVD floating-point error, not a violation.

## S9 Hyperparameters and reproduction
- Window 30, cap 125, batch 256, wd 1e-4, cosine T_max=40, clip 1.0, ≤40 epochs, patience 15.
- Proposed and B8: lr 3e-4, stem 1, chunk 16, dim 64 (4 heads × 16), 2 layers, bidirectional.
- B1–B5: lr 1e-3; same split/window/label/early stopping.
- Five seeds (0–4); split seed 2026; gate truncation seed 2027.
- Main matrix: 140 runs, ~8.5 GPU-hours; ablations ~7.8 GPU-hours; single 8 GB GPU.
- Commands: see the repository README (`code/train_e2.py`, `code/eval_all_test.py`, `code/analyze_main.py`, `code/make_figures.py`).
## S10 Score coefficient note
Saxena et al. (2008, Eq. 11) write \(d=\hat{y}-y\), \(a_1=10\), \(a_2=13\), while the same paragraph states that late predictions are penalized more heavily. The two statements are inconsistent. We use the community convention \(a_1=13\) for \(d<0\) and \(a_2=10\) for \(d\ge0\), which matches the stated intent; the main claims rest on RMSE.
