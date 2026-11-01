# -*- coding: utf-8 -*-
"""E5a: test-time Gaussian noise robustness on the gate split (no retraining).

Noise is added to the 15 sensor channels only (settings stay clean), 3 draws per
sigma. Uses the gate engines, so the official test is not reused for selection.
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
import metrics as M  # noqa: E402
from models import build_model  # noqa: E402
from train_e2 import build_trunc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", default="FD002")
    ap.add_argument("--model", default="ours_srnl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stem", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--sigmas", default="0,0.01,0.02,0.05,0.1,0.2")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--n-sensors", type=int, default=15)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.loads((RESULTS / "cache" / ("%s_meta.json" % args.fd)).read_text(encoding="utf-8"))
    prep = D.Preprocessor.from_dict(meta["preprocessor"])
    df = D.load_frame(args.fd, "train")
    x, y, e = build_trunc(df, prep, meta["split"]["gate"], 30, 125.0, 3, 2027)

    model = build_model(args.model, x.shape[-1], chunk=args.chunk,
                        stem=bool(args.stem)).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    rows = []
    sigmas = [float(s) for s in args.sigmas.split(",")]
    with torch.no_grad():
        for sigma in sigmas:
            rmses, scores = [], []
            for draw in range(args.draws):
                rng = np.random.RandomState(1000 + draw)
                xn = x.copy()
                if sigma > 0:
                    xn[:, :, :args.n_sensors] += rng.normal(
                        0, sigma, size=xn[:, :, :args.n_sensors].shape).astype(np.float32)
                preds = []
                for i in range(0, len(xn), 512):
                    xb = torch.from_numpy(xn[i:i + 512]).to(device)
                    preds.append(model(xb).cpu().numpy())
                p = np.concatenate(preds)
                rmses.append(M.rmse(p, y))
                scores.append(M.phm_score(p, y))
            rows.append({"fd": args.fd, "model": args.model, "seed": args.seed,
                         "sigma": sigma, "rmse": float(np.mean(rmses)),
                         "rmse_sd": float(np.std(rmses)), "score": float(np.mean(scores))})
    out = RESULTS / ("e5_noise_%s_%s_s%d.csv" % (args.fd, args.model, args.seed))
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("saved", out)


if __name__ == "__main__":
    main()
