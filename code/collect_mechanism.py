# -*- coding: utf-8 -*-
"""Collect per-token alpha/eta from a trained SRNL checkpoint (E4 mechanism).

Runs on the gate split of the training engines (never on official test).
The scan method is wrapped at runtime; models.py is not modified.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

G2 = Path(__file__).resolve().parents[1]
RESULTS = G2 / "results"
sys.path.insert(0, str(G2 / "code"))
import data as D  # noqa: E402
from models import SRNLScan, build_model  # noqa: E402


def collect(model, x, device, bs=256):
    """Return per-window mean/last alpha and eta, averaged over heads, per scan."""
    caps = {}
    originals = {}
    for name, mod in model.named_modules():
        if isinstance(mod, SRNLScan):
            originals[name] = mod._step_params

            def wrapped(Sx, log=None, _orig=mod._step_params, _name=name):
                out = _orig(Sx, log)
                alpha, eta = out[2], out[3]
                caps.setdefault(_name, []).append(
                    (alpha.detach().mean(-1).cpu(), eta.detach().mean(-1).cpu()))
                return out

            mod._step_params = wrapped
    rows = []
    try:
        with torch.no_grad():
            for i in range(0, len(x), bs):
                caps.clear()
                xb = torch.from_numpy(x[i:i + bs]).to(device)
                model(xb)
                n = len(xb)
                per = {}
                for name in sorted(caps):
                    a = torch.cat([z[0] for z in caps[name]], dim=1)  # (B,L)
                    e = torch.cat([z[1] for z in caps[name]], dim=1)
                    per[name] = (a, e)
                for j in range(n):
                    row = {"idx": i + j}
                    for name, (a, e) in per.items():
                        row[name + "_alpha_mean"] = float(a[j].mean())
                        row[name + "_alpha_last"] = float(a[j, -1])
                        row[name + "_eta_mean"] = float(e[j].mean())
                        row[name + "_eta_last"] = float(e[j, -1])
                    rows.append(row)
    finally:
        for name, mod in model.named_modules():
            if isinstance(mod, SRNLScan) and name in originals:
                mod._step_params = originals[name]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", default="FD002")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="gate")
    ap.add_argument("--chunk", type=int, default=16)
    args = ap.parse_args()

    ckpt = args.ckpt or str(RESULTS / ("ckpt_e2a_v2_%s_ours_srnl_s%d.pt" % (args.fd, args.seed)))
    meta = json.loads((RESULTS / "cache" / ("%s_meta.json" % args.fd)).read_text(encoding="utf-8"))
    prep = D.Preprocessor.from_dict(meta["preprocessor"])
    df = D.load_frame(args.fd, "train")
    x, y, e, last = D.build_windows(df, prep, L=30, cap=125, stride=1)
    m = np.isin(e, meta["split"][args.split])
    x, y, e = x[m], y[m], e[m]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model("ours_srnl", x.shape[-1], chunk=args.chunk, stem=True).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    feat = collect(model, x, device)
    feat["engine"] = e
    feat["rul"] = y
    # operating condition of each window = cluster of its last cycle (conditions are
    # interleaved within an engine; an engine-mode label collapses everything to one cluster)
    conds = []
    for eid, g in df.groupby("engine", sort=True):
        s = g[["set1", "set2", "set3"]].values
        z = (s - prep.set_mean) / prep.set_std
        lab = ((z[:, None, :] - prep.centroids[None, :, :]) ** 2).sum(-1).argmin(1)
        for end in range(29, len(g)):
            conds.append(int(lab[end]))
    conds = np.asarray(conds)[m]
    if len(conds) != len(feat):
        raise ValueError("cond length %d != windows %d" % (len(conds), len(feat)))
    feat["cond"] = conds

    out = RESULTS / ("mechanism_%s_%s_s%d.csv" % (args.fd, args.split, args.seed))
    feat.to_csv(out, index=False)

    # summary: alpha/eta by condition and RUL band (layer 0 forward scan as primary)
    name = "encoder.layers.0.fwd"
    summary = {"fd": args.fd, "split": args.split, "seed": args.seed, "n": int(len(feat)),
               "scan": name, "by_cond": {}, "by_rul_band": {}}
    if name + "_alpha_mean" in feat.columns:
        for c, g in feat.groupby("cond"):
            summary["by_cond"][int(c)] = {
                "n": int(len(g)),
                "alpha_mean": float(g[name + "_alpha_mean"].mean()),
                "eta_mean": float(g[name + "_eta_mean"].mean()),
                "alpha_sd": float(g[name + "_alpha_mean"].std()),
            }
        bands = [(0, 25), (25, 50), (50, 100), (100, 126)]
        for lo, hi in bands:
            g = feat[(feat.rul >= lo) & (feat.rul < hi)]
            if len(g):
                summary["by_rul_band"]["%d-%d" % (lo, hi)] = {
                    "n": int(len(g)),
                    "alpha_mean": float(g[name + "_alpha_mean"].mean()),
                    "eta_mean": float(g[name + "_eta_mean"].mean()),
                }
        summary["corr_alpha_rul"] = float(feat[name + "_alpha_mean"].corr(feat["rul"]))
        summary["corr_eta_rul"] = float(feat[name + "_eta_mean"].corr(feat["rul"]))
    (RESULTS / ("mechanism_%s_%s_s%d.json" % (args.fd, args.split, args.seed))).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("saved", out)


if __name__ == "__main__":
    main()
