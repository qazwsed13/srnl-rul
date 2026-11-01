# -*- coding: utf-8 -*-
"""Diagnose the current metrics: label distributions, outlier dominance, bias by RUL band."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

G2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(G2 / "code"))
import data as D  # noqa: E402
from train_e2 import build_trunc  # noqa: E402

RESULTS = G2 / "results"


def label_stats(y):
    y = np.asarray(y, float)
    return {"n": len(y), "mean": float(y.mean()), "sd": float(y.std()),
            "p10": float(np.percentile(y, 10)), "median": float(np.median(y)),
            "p90": float(np.percentile(y, 90)), "min": float(y.min()), "max": float(y.max())}


def pred_diag(path):
    d = pd.read_csv(path)
    d["err"] = d.rul_pred - d.rul_true
    d["se"] = d.err ** 2
    d["ae"] = d.err.abs()
    by = d.groupby("engine").agg(se=("se", "mean"), ae=("ae", "mean"),
                                 bias=("err", "mean"), n=("se", "size"))
    by["rmse"] = np.sqrt(by.se)
    top = by.se.sort_values(ascending=False).head(5)
    total = by.se.sum()
    late = d[d.rul_true < 50]
    early = d[d.rul_true >= 100]
    return {
        "n": len(d), "engines": int(d.engine.nunique()),
        "rmse": float(np.sqrt(d.se.mean())),
        "mean_engine_rmse": float(by.rmse.mean()),
        "mae": float(d.ae.mean()),
        "median_ae": float(d.ae.median()),
        "bias": float(d.err.mean()),
        "top5_mse_share": float(top.sum() / total),
        "top5_engines": {int(k): float(np.sqrt(v)) for k, v in top.items()},
        "late_bias": float(late.err.mean()) if len(late) else None,
        "late_mae": float(late.ae.mean()) if len(late) else None,
        "early_bias": float(early.err.mean()) if len(early) else None,
        "early_mae": float(early.ae.mean()) if len(early) else None,
    }


def main():
    print("== official test RUL ==")
    for fd in ("FD001", "FD002", "FD003", "FD004"):
        y = D.load_rul(fd)
        s = label_stats(y)
        print(fd, "n=%d mean=%.1f median=%.1f p10=%.1f p90=%.1f max=%.0f >125=%d"
              % (s["n"], s["mean"], s["median"], s["p10"], s["p90"], s["max"],
                 int((y > 125).sum())))

    print("\n== gate/val trunc labels ==")
    for fd in ("FD002", "FD004"):
        meta = json.loads((RESULTS / "cache" / ("%s_meta.json" % fd)).read_text(encoding="utf-8"))
        prep = D.Preprocessor.from_dict(meta["preprocessor"])
        df = D.load_frame(fd, "train")
        for split in ("val", "gate"):
            _, y, _ = build_trunc(df, prep, meta["split"][split], 30, 125.0, 3, 2027)
            s = label_stats(y)
            print("%s %s n=%d mean=%.1f median=%.1f p10=%.1f p90=%.1f"
                  % (fd, split, s["n"], s["mean"], s["median"], s["p10"], s["p90"]))

    print("\n== prediction diagnosis (2-seed mean files; per seed printed) ==")
    pairs = [
        ("FD002", "ours stem", "pred_gate_trunc_e2a_v2_FD002_ours_srnl_s%d.csv"),
        ("FD002", "TCN", "pred_gate_trunc_e2a_v1_FD002_B5_tcn_s%d.csv"),
        ("FD004", "ours stem", "pred_gate_trunc_e2a_v2_FD004_ours_srnl_s%d.csv"),
        ("FD004", "TCN", "pred_gate_trunc_e2a_v1_FD004_B5_tcn_s%d.csv"),
    ]
    for fd, name, pattern in pairs:
        for seed in (0, 1):
            p = RESULTS / (pattern % seed)
            if p.exists():
                d = pred_diag(p)
                print("%s %-10s s%d rmse=%.2f mean_eng=%.2f mae=%.2f medAE=%.1f "
                      "bias=%+.1f top5share=%.0f%% late_bias=%s early_bias=%s top5=%s"
                      % (fd, name, seed, d["rmse"], d["mean_engine_rmse"], d["mae"],
                         d["median_ae"], d["bias"], 100 * d["top5_mse_share"],
                         ("%+.1f" % d["late_bias"]) if d["late_bias"] is not None else "na",
                         ("%+.1f" % d["early_bias"]) if d["early_bias"] is not None else "na",
                         {k: round(v, 1) for k, v in list(d["top5_engines"].items())[:3]}))


if __name__ == "__main__":
    main()
