# -*- coding: utf-8 -*-
"""Aggregate mechanism JSONs across seeds/subsets."""
import json
from pathlib import Path

import numpy as np

G2 = Path(__file__).resolve().parents[1]
RESULTS = G2 / "results"


def main():
    out = {}
    for fd in ("FD002", "FD004"):
        runs = []
        for s in range(5):
            p = RESULTS / ("mechanism_%s_gate_s%d.json" % (fd, s))
            if p.exists():
                runs.append(json.loads(p.read_text(encoding="utf-8")))
        if not runs:
            continue
        bands = sorted(runs[0]["by_rul_band"])
        agg = {"n_runs": len(runs), "corr_alpha_rul": [], "corr_eta_rul": [],
               "bands": {}, "by_cond": {}}
        for r in runs:
            agg["corr_alpha_rul"].append(r["corr_alpha_rul"])
            agg["corr_eta_rul"].append(r["corr_eta_rul"])
        for b in bands:
            a = [r["by_rul_band"][b]["alpha_mean"] for r in runs if b in r["by_rul_band"]]
            e = [r["by_rul_band"][b]["eta_mean"] for r in runs if b in r["by_rul_band"]]
            agg["bands"][b] = {"alpha_mean": float(np.mean(a)), "alpha_sd": float(np.std(a)),
                               "eta_mean": float(np.mean(e)), "eta_sd": float(np.std(e)),
                               "n_runs": len(a)}
        conds = sorted(runs[0]["by_cond"])
        for c in conds:
            a = [r["by_cond"][c]["alpha_mean"] for r in runs if c in r["by_cond"]]
            e = [r["by_cond"][c]["eta_mean"] for r in runs if c in r["by_cond"]]
            agg["by_cond"][c] = {"alpha_mean": float(np.mean(a)), "eta_mean": float(np.mean(e))}
        out[fd] = agg

        print("\n==", fd, "runs", len(runs), "==")
        print("corr alpha-RUL %.3f ± %.3f | eta-RUL %.3f ± %.3f"
              % (np.mean(agg["corr_alpha_rul"]), np.std(agg["corr_alpha_rul"]),
                 np.mean(agg["corr_eta_rul"]), np.std(agg["corr_eta_rul"])))
        for b in bands:
            v = agg["bands"][b]
            print("  RUL %-8s alpha %.4f ± %.4f | eta %.4f ± %.4f"
                  % (b, v["alpha_mean"], v["alpha_sd"], v["eta_mean"], v["eta_sd"]))
        ac = [v["alpha_mean"] for v in agg["by_cond"].values()]
        ec = [v["eta_mean"] for v in agg["by_cond"].values()]
        print("  condition spread: alpha range %.4f, eta range %.4f"
              % (max(ac) - min(ac), max(ec) - min(ec)))

    (RESULTS / "mechanism_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nsaved", RESULTS / "mechanism_summary.json")


if __name__ == "__main__":
    main()
