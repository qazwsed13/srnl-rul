# -*- coding: utf-8 -*-
"""Main-table statistics from one-shot test predictions (E2b).

Inputs: results/<test_tag>_<fd>_<model>_s<seed>.json (test_cap125/test_raw)
        results/pred_test_<test_tag>_<fd>_<model>_s<seed>.csv
Outputs: evidence/results_main.csv, evidence/stats.json
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

G2 = Path(__file__).resolve().parents[1]
RESULTS = G2 / "results"
EVID = G2 / "evidence"
BASELINES = ["B1_cnn", "B2_lstm", "B3_bigru", "B4_transformer", "B5_tcn", "B8_srnl_fixed"]
OURS = "ours_srnl"
FDS = ("FD001", "FD002", "FD003", "FD004")


def per_engine_rmse(path, cap=None):
    d = pd.read_csv(path)
    y = d.rul_true.values.astype(float)
    if cap is not None:
        y = np.minimum(y, cap)
    d["se"] = (d.rul_pred.values - y) ** 2
    return d.groupby("engine").se.mean() ** 0.5


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)
        adj[idx] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="e2b_test")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    rows = []
    for fd in FDS:
        for m in BASELINES + [OURS]:
            for seed in range(5):
                p = RESULTS / ("%s_%s_%s_s%d.json" % (args.tag, fd, m, seed))
                if not p.exists():
                    continue
                d = json.loads(p.read_text(encoding="utf-8"))
                rows.append({"fd": fd, "model": m, "seed": seed,
                             "rmse_cap125": d["test_cap125"]["rmse"],
                             "score_cap125": d["test_cap125"]["score_sum"],
                             "rmse_raw": d["test_raw"]["rmse"],
                             "score_raw": d["test_raw"]["score_sum"]})
    if not rows:
        raise SystemExit("no test results")

    EVID.mkdir(exist_ok=True)
    with open(EVID / "results_main.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    agg = {}
    for fd in FDS:
        for m in BASELINES + [OURS]:
            v = [r for r in rows if r["fd"] == fd and r["model"] == m]
            if v:
                agg[(fd, m)] = {
                    "n": len(v),
                    "rmse_cap125_mean": float(np.mean([r["rmse_cap125"] for r in v])),
                    "rmse_cap125_sd": float(np.std([r["rmse_cap125"] for r in v], ddof=1)),
                    "rmse_cap125_values": [r["rmse_cap125"] for r in v],
                    "score_cap125_mean": float(np.mean([r["score_cap125"] for r in v])),
                    "rmse_raw_mean": float(np.mean([r["rmse_raw"] for r in v])),
                }

    print("%-7s %-14s %10s %8s %10s" % ("fd", "model", "RMSEcap125", "sd", "RMSEraw"))
    for (fd, m), a in agg.items():
        print("%-7s %-14s %10.3f %8.3f %10.3f" % (
            fd, m, a["rmse_cap125_mean"], a["rmse_cap125_sd"], a["rmse_raw_mean"]))

    comps = {}
    for fd in FDS:
        if (fd, OURS) not in agg:
            continue
        pe = {}
        for m in BASELINES + [OURS]:
            frames = []
            for seed in range(5):
                p = RESULTS / ("pred_test_%s_%s_%s_s%d.csv" % (args.tag, fd, m, seed))
                if p.exists():
                    frames.append(per_engine_rmse(p, cap=125))
            if frames:
                pe[m] = pd.concat(frames, axis=1).mean(axis=1)
        if OURS not in pe:
            continue
        pvals, keys = [], []
        for base in BASELINES:
            if base not in pe:
                continue
            j = pd.concat([pe[OURS].rename("ours"), pe[base].rename("base")], axis=1).dropna()
            d = (j.ours - j.base).values
            if len(d) < 5:
                continue
            stat = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
            rng = np.random.RandomState(0)
            bs = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(args.n_boot)])
            comps[(fd, base)] = {
                "n_engines": int(len(d)),
                "delta": float(d.mean()),
                "delta_pct": float(d.mean() / j.base.mean() * 100.0),
                "ci_lo": float(np.percentile(bs, 2.5)),
                "ci_hi": float(np.percentile(bs, 97.5)),
                "wilcoxon_stat": float(stat.statistic),
                "p": float(stat.pvalue),
                "wins": int((d < 0).sum()),
                "losses": int((d > 0).sum()),
            }
            pvals.append(float(stat.pvalue))
            keys.append((fd, base))
        if pvals:
            adj = holm(np.array(pvals))
            for k, pa in zip(keys, adj):
                comps[k]["p_holm"] = float(pa)

    out = {"tag": args.tag,
           "aggregate": {"%s|%s" % k: v for k, v in agg.items()},
           "comparisons": {"%s|%s" % k: v for k, v in comps.items()},
           "note": "RMSE cap125 is the main metric (training/output space); raw is supplementary"}
    (EVID / "stats.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                     encoding="utf-8")

    print("\npaired engine-level vs baselines (negative delta = ours better):")
    for (fd, base), c in comps.items():
        print("  %s vs %-14s delta %+7.2f (%+5.2f%%) CI [%+.2f, %+.2f] p=%.4f holm=%.4f wins %d/%d"
              % (fd, base, c["delta"], c["delta_pct"], c["ci_lo"], c["ci_hi"],
                 c["p"], c["p_holm"], c["wins"], c["n_engines"]))
    print("saved", EVID / "results_main.csv", EVID / "stats.json")


if __name__ == "__main__":
    main()

