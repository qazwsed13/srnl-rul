# Bounded stage-adaptive memory for turbofan RUL prediction

Code, prediction files, and evidence for a manuscript under preparation (not yet submitted or published). The study asks whether a five-memory self-referential scan can serve as a 1D remaining-useful-life (RUL) encoder with bounded dynamics and stage-adaptive step sizes.

**Author:** Yimeng Chai (ORCID 0009-0009-7131-7674), School of Mechanical Engineering, North China University of Water Resources and Electric Power. Supervisor and full author list will be added at submission.

## What this repository contains

- `code/` — data pipeline, models (CNN, LSTM, BiGRU, Transformer, TCN, fixed-rule SRNL, proposed stem+SRNL), training, one-shot test evaluation, statistics, ablations, mechanism and noise analyses, figure scripts.
- `evidence/` — data audit, sensor subset, implementation notes, score-coefficient check, main and ablation tables, mechanism and robustness summaries.
- `results/` — one-shot test predictions (`pred_test_e2b_test_*.csv`), test metrics, training summaries, ablation/mechanism/robustness/stability summaries, cached split metadata.
- `figures/` — architecture, main RMSE, mechanism, noise, stability, and ablation figures (PDF/PNG).
- `paper/SI_v0.1.md` — supplementary tables (secondary metrics, per-seed values, gate, full ablation, calibration, noise, stability).
- `seeds.yaml`, `env.txt` — configurations and environment.

**Not included:** C-MAPSS raw data (NASA terms; use the downloader), model checkpoints (large; regenerable from seeds), and the manuscript text.

## Data

The C-MAPSS data are not redistributed. Download the official NASA archive and extract the nested `CMAPSSData.zip`:

```bash
python code/download_cmapss.py
```

This creates `data/CMAPSSData/` with the 12 train/test/RUL files. Source: `https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip` (12,429,152 bytes). The dataset is simulated turbofan data released for the PHM08 challenge; cite Saxena et al. (2008).

## Environment

Python 3.8, PyTorch 2.4.1+cu121, NumPy 1.24.4, pandas 2.0.3, SciPy 1.10.1, matplotlib. A single 8 GB GPU was used. See `env.txt`.

## Reproduction

```bash
# 1. data
python code/download_cmapss.py

# 2. train the main matrix (4 subsets x 7 models x 5 seeds; ~8.5 GPU-hours)
#    example for one run:
python code/train_e2.py --fd FD002 --model ours_srnl --seed 0 \
  --epochs 40 --patience 15 --lr 3e-4 --chunk 16 --stem 1 --tag e2b
#    full loop: see the command pattern in the manuscript / evidence package.

# 3. one-shot test evaluation (after all training)
python code/eval_all_test.py --prefix e2b --tag e2b_test --confirm

# 4. main table and paired statistics
python code/analyze_main.py --tag e2b_test

# 5. figures
python code/make_figures.py
python code/make_fig_ablation.py
```

The repository already contains the predictions and summaries used in the paper. `analyze_main.py` recomputes the main table from `results/pred_test_*` and writes `evidence/results_main.csv` and `evidence/stats.json`.

## Main test results (RMSE, RUL capped at 125; five seeds)

| Model | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| B1 CNN | 16.553 | 16.264 | 18.429 | 17.794 |
| B2 LSTM | 14.389 | 15.356 | 14.476 | 15.032 |
| B3 BiGRU | 14.331 | 14.379 | 13.891 | 14.665 |
| B4 Transformer | **13.869** | 14.070 | 14.158 | 14.554 |
| B5 TCN | 13.945 | **13.946** | **12.766** | **14.394** |
| B8 fixed-rule SRNL | 14.661 | 14.408 | 14.243 | 14.816 |
| Proposed (stem+SRNL) | 14.774 | 14.226 | 13.286 | 14.556 |

The proposed model is **not** the most accurate on any subset. Its pooled RMSE is 1.1–6.5% above the strongest baseline. At the engine level, no significant difference from the strongest baseline was detected on any subset (paired Wilcoxon + Holm; FD002 is borderline, p=0.088). It is significantly better than the fixed-rule control on FD002 (Holm p=0.0068). See `evidence/results_main.md` for the paired intervals and `evidence/results_ablation.md` for the component analysis.

## Caveats

- The study is offline and window-level; the bidirectional scan is not suitable for streaming without a causal variant.
- The data are simulated turbofan trajectories, not fleet data.
- The multi-condition claim is scoped to FD002/FD004; on FD001/FD003 the model is mid-pack.
- The stability claim covers the memory branch and its readout, not the convolutional stem, the residual skip, or the full network. The final RUL is bounded only by the sigmoid output.
- The gate split is used for component screening; it cannot resolve stage adaptation because its RUL distribution is shifted toward early life.

## License and citation

Code: MIT (see `LICENSE`). Figures and tables: CC BY 4.0. The C-MAPSS data remain under the NASA terms; they are not included.

If you use this repository, cite the manuscript (details to be added) and Saxena et al. (2008) for the dataset. A `CITATION.cff` is provided with a placeholder title.

## Contact

Yimeng Chai, ORCID 0009-0009-7131-7674. Please open an issue for questions about the code.
