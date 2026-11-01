# -*- coding: utf-8 -*-
"""Trained-checkpoint norm ratio on the gate split (final model)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

G2 = Path(__file__).resolve().parents[1]
RESULTS = G2 / "results"
sys.path.insert(0, str(G2 / "code"))
import data as D  # noqa: E402
from models import build_model, new_log  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", default="FD002")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.loads((RESULTS / "cache" / ("%s_meta.json" % args.fd)).read_text(encoding="utf-8"))
    prep = D.Preprocessor.from_dict(meta["preprocessor"])
    df = D.load_frame(args.fd, "train")
    x, y, e, last = D.build_windows(df, prep, L=30, cap=125, stride=1)
    m = np.isin(e, meta["split"]["gate"])
    x = x[m]
    ckpt = RESULTS / ("ckpt_e2b_%s_ours_srnl_s%d.pt" % (args.fd, args.seed))
    model = build_model("ours_srnl", x.shape[-1], chunk=16, stem=True).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    log = new_log()
    with torch.no_grad():
        for i in range(0, len(x), 512):
            xb = torch.from_numpy(x[i:i + 512]).to(device)
            model(xb, log)
    out = {"fd": args.fd, "seed": args.seed, "n_windows": int(len(x)),
           "max_ratio": log["max_ratio"], "final_norm": log["final_norm"],
           "finite": log["finite"]}
    print(json.dumps(out, indent=2))
    (RESULTS / ("trained_norm_%s_s%d.json" % (args.fd, args.seed))).write_text(
        json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()