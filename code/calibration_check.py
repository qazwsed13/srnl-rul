# -*- coding: utf-8 -*-
"""Check whether a val-fitted global bias correction transfers to the gate split."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

G2 = Path(__file__).resolve().parents[1]
RESULTS = G2 / "results"
sys.path.insert(0, str(G2 / "code"))

PAIRS = [
    ("FD002", "ours_stem", "e2a_v2", "ours_srnl"),
    ("FD002", "TCN", "e2a_v1", "B5_tcn"),
    ("FD004", "ours_stem", "e2a_v2", "ours_srnl"),
    ("FD004", "TCN", "e2a_v1", "B5_tcn"),
]


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def main():
    for fd, label, pref, model in PAIRS:
        rows = []
        for seed in (0, 1):
            pv = RESULTS / ("pred_val_trunc_%s_%s_%s_s%d.csv" % (pref, fd, model, seed))
            pg = RESULTS / ("pred_gate_trunc_%s_%s_%s_s%d.csv" % (pref, fd, model, seed))
            if not (pv.exists() and pg.exists()):
                continue
            v = pd.read_csv(pv)
            g = pd.read_csv(pg)
            bias = float((v.rul_pred - v.rul_true).mean())
            before = rmse(g.rul_pred, g.rul_true)
            after = rmse(g.rul_pred - bias, g.rul_true)
            rows.append((seed, bias, before, after))
            print("%s %-9s s%d val_bias %+5.2f  gate %.3f -> %.3f (%+.2f%%)"
                  % (fd, label, seed, bias, before, after, (after - before) / before * 100))
        if rows:
            print("  mean delta %+.2f%%" % np.mean([(a - b) / b * 100 for _, _, b, a in rows]))


if __name__ == "__main__":
    main()
