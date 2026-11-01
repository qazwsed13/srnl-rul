# -*- coding: utf-8 -*-
"""Gate analysis: aggregate RMSE + paired engine-level bootstrap CI.

Reads results named "<prefix>_<fd>_<model>_s<seed>.json" and the matching
pred_gate_trunc_<prefix>_<fd>_<model>_s<seed>.csv files.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

G2 = Path(__file__).resolve().parents[1]
RESULTS = G2 / "results"
BASELINES = ["B1_cnn", "B2_lstm", "B5_tcn", "B8_srnl_fixed"]
OURS = "ours_srnl"


def per_engine_rmse(path):
    d = pd.read_csv(path)
    d["se"] = (d.rul_pred - d.rul_true) ** 2
    return d.groupby("engine").se.mean() ** 0.5


def paired(a, b, n_boot=10000, seed=0):
    j = pd.concat([a.rename("ours"), b.rename("base")], axis=1).dropna()
    d = (j.ours - j.base).values
    rng = np.random.RandomState(seed)
    bs = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return {
        "n_engines": int(len(d)),
        "ours_mean_engine_rmse": float(j.ours.mean()),
        "base_mean_engine_rmse": float(j.base.mean()),
        "delta": float(d.mean()),
        "ci_lo": float(np.percentile(bs, 2.5)),
        "ci_hi": float(np.percentile(bs, 97.5)),
        "wins": int((d < 0).sum()),
        "losses": int((d > 0).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="e2a_v1")
    ap.add_argument("--base-prefix", default=None)
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    base_prefix = args.base_prefix or args.prefix

    rows = []
    for pref in ([args.prefix] if base_prefix == args.prefix else [args.prefix, base_prefix]):
        for p in sorted(RESULTS.glob("%s_*.json" % pref)):
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("model") not in BASELINES + [OURS]:
                continue
            if pref != args.prefix and d.get("model") == OURS:
                continue
            rows.append({"fd": d["fd"], "model": d["model"], "seed": d["seed"],
                         "gate_trunc_rmse": d["gate_trunc"]["rmse"],
                         "gate_trunc_score": d["gate_trunc"]["score_sum"],
                         "val_trunc_rmse": d["val_trunc"]["rmse"],
                         "best_epoch": d["best_epoch"], "epochs_run": d["epochs_run"],
                         "nan_steps": d["nan_steps"], "params": d["params"],
                         "ms_per_iter": d["ms_per_iter_mean"]})
    if not rows:
        raise SystemExit("no rows for prefix %s" % args.prefix)
    with open(RESULTS / ("results_gate_%s.csv" % args.prefix), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    agg, paired_out = {}, {}
    for fd in ("FD001", "FD002", "FD003", "FD004"):
        for m in BASELINES + [OURS]:
            v = [r["gate_trunc_rmse"] for r in rows if r["fd"] == fd and r["model"] == m]
            if v:
                agg[(fd, m)] = {"mean": float(np.mean(v)), "n": len(v), "values": v}
        seeds = sorted(set(r["seed"] for r in rows if r["fd"] == fd))
        for base in BASELINES:
            ps = []
            for s in seeds:
                pa = RESULTS / ("pred_gate_trunc_%s_%s_%s_s%d.csv" % (args.prefix, fd, OURS, s))
                pb = RESULTS / ("pred_gate_trunc_%s_%s_%s_s%d.csv" % (base_prefix, fd, base, s))
                if pa.exists() and pb.exists():
                    ps.append(paired(per_engine_rmse(pa), per_engine_rmse(pb), args.n_boot, s))
            if ps:
                paired_out[(fd, base)] = {
                    "delta_mean_over_seeds": float(np.mean([x["delta"] for x in ps])),
                    "ci_lo_min": float(min(x["ci_lo"] for x in ps)),
                    "ci_hi_max": float(max(x["ci_hi"] for x in ps)),
                    "per_seed": ps,
                }

    print("%-7s %-14s %8s %4s %s" % ("fd", "model", "mean", "n", "per-seed"))
    for (fd, m), a in agg.items():
        print("%-7s %-14s %8.3f %4d %s" % (fd, m, a["mean"], a["n"],
                                           " ".join("%.3f" % x for x in a["values"])))

    decision = {}
    for fd in ("FD001", "FD002", "FD003", "FD004"):
        if (fd, OURS) not in agg:
            continue
        ours = agg[(fd, OURS)]["mean"]
        best_m = min(BASELINES, key=lambda m: agg[(fd, m)]["mean"] if (fd, m) in agg else 9e9)
        best = agg[(fd, best_m)]["mean"]
        d = paired_out.get((fd, best_m))
        decision[fd] = {
            "ours": ours, "best_baseline": best, "best_baseline_name": best_m,
            "delta_vs_best_pct": (ours - best) / best * 100.0,
            "paired_vs_best": d,
        }
        for base in ("B5_tcn", "B8_srnl_fixed"):
            if (fd, base) in agg:
                decision[fd]["delta_vs_%s_pct" % base] = (
                    (ours - agg[(fd, base)]["mean"]) / agg[(fd, base)]["mean"] * 100.0)

    worse_both = all(decision[fd]["delta_vs_best_pct"] >= 1.0 for fd in decision)
    ci_worse_both = all(
        decision[fd]["paired_vs_best"]["ci_lo_min"] > 0
        for fd in decision if decision[fd]["paired_vs_best"] is not None)
    ambiguous = any(abs(decision[fd]["delta_vs_best_pct"]) < 1.0 for fd in decision)
    verdict = "KILL_PRE_REGISTERED" if worse_both else ("NEED_SEED_2" if ambiguous else "GO")
    out = {
        "prefix": args.prefix,
        "rows": rows,
        "aggregate": {"%s|%s" % k: v for k, v in agg.items()},
        "paired": {"%s|%s" % k: v for k, v in paired_out.items()},
        "decision": decision,
        "verdict": verdict,
        "aggregate_rule_kill": bool(worse_both),
        "paired_ci_supports_kill": bool(worse_both and ci_worse_both),
        "rule": "pre-registered: >=1% worse on aggregate gate RMSE on both subsets => kill",
    }
    (RESULTS / ("%s_summary.json" % args.prefix)).write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\npaired engine-level delta (ours - baseline; negative = ours better):")
    for fd in decision:
        p = decision[fd]["paired_vs_best"]
        if p:
            print("  %s vs %s: delta %+.3f CI [%.3f, %.3f] wins %d/%d"
                  % (fd, decision[fd]["best_baseline_name"], p["delta_mean_over_seeds"],
                     p["ci_lo_min"], p["ci_hi_max"],
                     p["per_seed"][0]["wins"], p["per_seed"][0]["n_engines"]))
    print("\ndecision:")
    for fd, d in decision.items():
        print("  %s: ours %.3f | best %s %.3f | delta %+.2f%% | vs B5 %+.2f%% | vs B8 %+.2f%%"
              % (fd, d["ours"], d["best_baseline_name"], d["best_baseline"],
                 d["delta_vs_best_pct"], d.get("delta_vs_B5_tcn_pct", float("nan")),
                 d.get("delta_vs_B8_srnl_fixed_pct", float("nan"))))
    print("verdict: %s | aggregate kill=%s | paired CI supports kill=%s"
          % (verdict, worse_both, worse_both and ci_worse_both))


if __name__ == "__main__":
    main()

