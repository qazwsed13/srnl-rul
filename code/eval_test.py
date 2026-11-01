# -*- coding: utf-8 -*-
"""One-shot official-test evaluation for the final table (E2b).

Test must not be used before this step (conception/02 §3.3). The checkpoint
comes from train_e2.py, which early-stops on the val split only.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import data as D
import metrics as M
from models import build_model, count_params

G2 = Path(__file__).resolve().parents[1]
RESULTS = G2 / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="e2b")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--cap", type=float, default=125.0)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--norm", type=int, default=0)
    ap.add_argument("--stem", type=int, default=0)
    ap.add_argument("--sensor-set", default="fd001")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.loads((RESULTS / "cache" / ("%s_meta.json" % args.fd)).read_text(encoding="utf-8"))
    prep = D.Preprocessor.from_dict(meta["preprocessor"])
    df = D.load_frame(args.fd, "test")
    x, eng = D.build_test_last_windows(df, prep, L=args.window)
    y = D.load_rul(args.fd)
    if len(x) != len(y) or not np.array_equal(eng, np.arange(1, len(y) + 1)):
        raise ValueError("test engine order mismatch: %d windows, %d RUL" % (len(x), len(y)))

    model = build_model(args.model, x.shape[-1], cap=args.cap,
                        chunk=args.chunk, norm=bool(args.norm),
                        stem=bool(args.stem)).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), 512):
            xb = torch.from_numpy(x[i:i + 512]).to(device)
            preds.append(model(xb).detach().cpu().numpy())
    pred = np.concatenate(preds)
    rep_cap = M.regression_report(pred, np.minimum(y, args.cap))
    rep_raw = M.regression_report(pred, y)

    stem = "%s_%s_%s_s%d" % (args.tag, args.fd, args.model, args.seed)
    pd.DataFrame({"engine": eng, "rul_true": y, "rul_pred": pred}).to_csv(
        RESULTS / ("pred_test_%s.csv" % stem), index=False)
    out = {"fd": args.fd, "model": args.model, "seed": args.seed,
           "test_cap125": rep_cap, "test_raw": rep_raw,
           "params": count_params(model), "ckpt": str(args.ckpt),
           "date": time.strftime("%Y-%m-%d %H:%M:%S"),
           "note": "official test used once; cap125 = labels clipped at 125 (training/output space), raw = original labels"}
    (RESULTS / ("%s.json" % stem)).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    print("[%s] %s n=%d cap125 RMSE %.3f Score %.1f | raw RMSE %.3f Score %.1f"
          % (args.tag, stem, rep_cap["n"], rep_cap["rmse"], rep_cap["score_sum"],
             rep_raw["rmse"], rep_raw["score_sum"]))


if __name__ == "__main__":
    main()
