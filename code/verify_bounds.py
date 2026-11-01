# -*- coding: utf-8 -*-
"""Numerically check the operator bounds of Lemmas 2-3 on a trained checkpoint."""
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
from models import SRNLScan, build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", default="FD002")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batches", type=int, default=4)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.loads((RESULTS / "cache" / ("%s_meta.json" % args.fd)).read_text(encoding="utf-8"))
    prep = D.Preprocessor.from_dict(meta["preprocessor"])
    df = D.load_frame(args.fd, "train")
    x, y, e, last = D.build_windows(df, prep, L=30, cap=125, stride=1)
    m = np.isin(e, meta["split"]["gate"])
    x = x[m][: args.batches * 256]
    ckpt = RESULTS / ("ckpt_e2b_%s_ours_srnl_s%d.pt" % (args.fd, args.seed))
    model = build_model("ours_srnl", x.shape[-1], chunk=16, stem=True).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    stats = {"max_A_over_alpha": 0.0, "max_B_over_one_minus_alpha": 0.0,
             "max_A_plus_B": 0.0, "n": 0}
    for name, mod in model.named_modules():
        if isinstance(mod, SRNLScan):
            orig = mod._step_params

            def wrapped(Sx, log=None, _orig=orig, _name=name):
                out = _orig(Sx, log)
                k, delta, alpha, eta = out[0], out[1], out[2], out[3]
                d = k.shape[-1]
                I = torch.eye(d, device=k.device, dtype=k.dtype)
                A = alpha[..., None, None] * I - eta[..., None, None] * torch.einsum(
                    "bchi,bchj->bchij", k, k)
                B = -eta[..., None, None] * torch.einsum("bchi,bchj->bchij", delta, k)
                a = torch.linalg.matrix_norm(A, ord=2)
                b = torch.linalg.matrix_norm(B, ord=2)
                al = alpha
                while al.dim() < a.dim():
                    al = al.unsqueeze(-1)
                stats["max_A_over_alpha"] = max(
                    stats["max_A_over_alpha"], float((a / al.clamp_min(1e-12)).max()))
                stats["max_B_over_one_minus_alpha"] = max(
                    stats["max_B_over_one_minus_alpha"],
                    float((b / (1 - al).clamp_min(1e-12)).max()))
                stats["max_A_plus_B"] = max(stats["max_A_plus_B"], float((a + b).max()))
                stats["n"] += int(a.numel())
                return out

            mod._step_params = wrapped
    with torch.no_grad():
        for i in range(0, len(x), 256):
            xb = torch.from_numpy(x[i:i + 256]).to(device)
            model(xb)
    print(json.dumps(stats, indent=2))
    out = {"fd": args.fd, "seed": args.seed, **stats,
           "pass": bool(stats["max_A_over_alpha"] <= 1 + 1e-5
                        and stats["max_B_over_one_minus_alpha"] <= 1 + 1e-5
                        and stats["max_A_plus_B"] <= 1 + 1e-5)}
    (RESULTS / ("bound_check_%s_s%d.json" % (args.fd, args.seed))).write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("PASS" if out["pass"] else "FAIL")


if __name__ == "__main__":
    main()
