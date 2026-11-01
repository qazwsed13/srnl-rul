# -*- coding: utf-8 -*-
"""Aggregate FD004 key ablations (2 seeds) vs full, with paired bootstrap CI."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

G2 = Path(__file__).resolve().parents[1]
RESULTS = G2 / "results"
EVID = G2 / "evidence"
SEEDS = (0, 1)
VARIANTS = ["no_soft_cap", "w_o_Mv", "shuffle_ae", "uni"]


def per_engine(path):
    d = pd.read_csv(path)
    d["se"] = (d.rul_pred - d.rul_true) ** 2
    return d.groupby("engine").se.mean() ** 0.5


def paired(a, b, n_boot=5000, seed=0):
    j = pd.concat([a.rename("v"), b.rename("full")], axis=1).dropna()
    d = (j.v - j.full).values
    rng = np.random.RandomState(seed)
    bs = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return {"n": int(len(d)), "delta": float(d.mean()),
            "ci_lo": float(np.percentile(bs, 2.5)),
            "ci_hi": float(np.percentile(bs, 97.5)),
            "wins": int((d < 0).sum()), "losses": int((d > 0).sum())}


def main():
    rows = []
    for v in VARIANTS:
        for s in SEEDS:
            p = RESULTS / ("e3fd4_FD004_%s_s%d.json" % (v, s))
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8"))
                rows.append({"variant": v, "seed": s, "gate": d["gate_trunc"]["rmse"],
                             "val": d["val_trunc"]["rmse"], "best_ep": d["best_epoch"],
                             "epochs": d["epochs_run"], "nan": d["nan_steps"],
                             "params": d["params"]})
    for name, model in (("full", "ours_srnl"), ("fixed", "B8_srnl_fixed")):
        for s in SEEDS:
            p = RESULTS / ("e2b_FD004_%s_s%d.json" % (model, s))
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8"))
                rows.append({"variant": name, "seed": s, "gate": d["gate_trunc"]["rmse"],
                             "val": d["val_trunc"]["rmse"], "best_ep": d["best_epoch"],
                             "epochs": d["epochs_run"], "nan": d["nan_steps"],
                             "params": d["params"]})
    if not rows:
        raise SystemExit("no FD004 ablation results")
    df = pd.DataFrame(rows)
    df.to_csv(EVID / "results_ablation_fd004.csv", index=False)

    agg = {}
    print("%-14s %8s %8s %4s %s" % ("variant", "gate mean", "gate sd", "n", "per-seed"))
    for v, g in df.groupby("variant"):
        vals = g.gate.values
        agg[v] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)),
                  "n": len(vals), "values": [float(x) for x in vals]}
        print("%-14s %8.3f %8.3f %4d %s" % (v, agg[v]["mean"], agg[v]["sd"], len(vals),
                                            " ".join("%.3f" % x for x in vals)))

    full_frames = []
    for s in SEEDS:
        p = RESULTS / ("pred_gate_trunc_e2b_FD004_ours_srnl_s%d.csv" % s)
        if p.exists():
            full_frames.append(per_engine(p))
    full = pd.concat(full_frames, axis=1).mean(axis=1) if full_frames else None
    comps = {}
    if full is not None:
        print("\npaired vs full (negative = variant better):")
        for v in VARIANTS:
            frames = []
            for s in SEEDS:
                p = RESULTS / ("pred_gate_trunc_e3fd4_FD004_%s_s%d.csv" % (v, s))
                if p.exists():
                    frames.append(per_engine(p))
            if not frames:
                continue
            pe = pd.concat(frames, axis=1).mean(axis=1)
            c = paired(pe, full)
            c["delta_pct"] = c["delta"] / full.mean() * 100
            comps[v] = c
            print("  %-14s delta %+6.2f (%+5.2f%%) CI [%+.2f, %+.2f] wins %d/%d"
                  % (v, c["delta"], c["delta_pct"], c["ci_lo"], c["ci_hi"],
                     c["wins"], c["n"]))
    out = {"fd": "FD004", "aggregate": agg, "paired_vs_full": comps,
           "note": "FD004 gate; full/fixed from e2b seeds 0-1; variants 2 seeds"}
    (RESULTS / "e3_fd004_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved", EVID / "results_ablation_fd004.csv", RESULTS / "e3_fd004_summary.json")


if __name__ == "__main__":
    main()
