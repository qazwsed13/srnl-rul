# -*- coding: utf-8 -*-
"""Aggregate E3 ablations (FD002), with paired engine-level comparison to full."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

G2 = Path(__file__).resolve().parents[1]
RESULTS = G2 / "results"
EVID = G2 / "evidence"
FULL_PREFIX = "e2b"
FULL_MODEL = "ours_srnl"
SEEDS = (0, 1, 2)


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
    for p in sorted(RESULTS.glob("e3_FD002_*_s*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        rows.append({"variant": d["model"], "seed": d["seed"],
                     "gate": d["gate_trunc"]["rmse"],
                     "val": d["val_trunc"]["rmse"],
                     "best_ep": d["best_epoch"], "epochs": d["epochs_run"],
                     "ms": d["ms_per_iter_mean"], "nan": d["nan_steps"]})
    # B8 fixed-rule stem from E2b, seeds 0-2, as the fixed control
    for s in SEEDS:
        p = RESULTS / ("%s_FD002_B8_srnl_fixed_s%d.json" % (FULL_PREFIX, s))
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            rows.append({"variant": "fixed", "seed": s,
                         "gate": d["gate_trunc"]["rmse"],
                         "val": d["val_trunc"]["rmse"],
                         "best_ep": d["best_epoch"], "epochs": d["epochs_run"],
                         "ms": d["ms_per_iter_mean"], "nan": d["nan_steps"]})
    # full model from E2b, same seeds, for the aggregate table
    for s in SEEDS:
        p = RESULTS / ("%s_FD002_%s_s%d.json" % (FULL_PREFIX, FULL_MODEL, s))
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            rows.append({"variant": "full", "seed": s,
                         "gate": d["gate_trunc"]["rmse"],
                         "val": d["val_trunc"]["rmse"],
                         "best_ep": d["best_epoch"], "epochs": d["epochs_run"],
                         "ms": d["ms_per_iter_mean"], "nan": d["nan_steps"]})
    if not rows:
        raise SystemExit("no E3 results")

    df = pd.DataFrame(rows)
    df.to_csv(EVID / "results_ablation.csv", index=False)
    print("%-14s %8s %8s %6s %s" % ("variant", "gate mean", "gate sd", "n", "per-seed"))
    agg = {}
    for v, g in df.groupby("variant"):
        vals = g.gate.values
        agg[v] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)),
                  "n": len(vals), "values": [float(x) for x in vals]}
        print("%-14s %8.3f %8.3f %6d %s" % (v, agg[v]["mean"], agg[v]["sd"], len(vals),
                                            " ".join("%.3f" % x for x in vals)))

    # paired vs full (seeds 0-2)
    full_frames = []
    for s in SEEDS:
        p = RESULTS / ("pred_gate_trunc_%s_FD002_%s_s%d.csv" % (FULL_PREFIX, FULL_MODEL, s))
        if p.exists():
            full_frames.append(per_engine(p))
    full = pd.concat(full_frames, axis=1).mean(axis=1) if full_frames else None
    comps = {}
    if full is not None:
        print("\npaired vs full (negative = variant better):")
        for v in sorted(agg):
            if v == "full":
                continue
            frames = []
            for s in SEEDS:
                p = RESULTS / ("pred_gate_trunc_e3_FD002_%s_s%d.csv" % (v, s))
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

    out = {"aggregate": agg, "paired_vs_full": comps,
           "note": "FD002 gate; full = e2b ours seeds 0-2; E3 variants 3 seeds"}
    (RESULTS / "e3_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                            encoding="utf-8")
    print("saved", EVID / "results_ablation.csv", RESULTS / "e3_summary.json")


if __name__ == "__main__":
    main()
