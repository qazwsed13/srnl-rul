# -*- coding: utf-8 -*-
"""Aggregate E5a noise robustness CSV across seeds."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

G2 = Path(__file__).resolve().parents[1]
RESULTS = G2 / "results"


def main():
    out = {}
    for fd in ("FD002", "FD004"):
        for model in ("ours_srnl", "B5_tcn", "B8_srnl_fixed"):
            frames = []
            for s in range(5):
                p = RESULTS / ("e5_noise_%s_%s_s%d.csv" % (fd, model, s))
                if p.exists():
                    frames.append(pd.read_csv(p))
            if not frames:
                continue
            d = pd.concat(frames)
            g = d.groupby("sigma").agg(rmse=("rmse", "mean"), sd=("rmse", "std"),
                                       score=("score", "mean")).reset_index()
            base = float(g.loc[g.sigma == 0, "rmse"].iloc[0])
            g["rel_pct"] = (g.rmse - base) / base * 100
            out["%s|%s" % (fd, model)] = g.to_dict(orient="list")
            print("\n%s %s (baseline %.3f)" % (fd, model, g.loc[g.sigma == 0, "rmse"].iloc[0]))
            for _, r in g.iterrows():
                print("  sigma %.2f  RMSE %.3f ± %.3f  (%+.2f%%)  Score %.1f"
                      % (r.sigma, r.rmse, r.sd, r.rel_pct, r.score))
    (RESULTS / "robust_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nsaved", RESULTS / "robust_summary.json")


if __name__ == "__main__":
    main()
