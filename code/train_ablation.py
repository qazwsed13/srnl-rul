# -*- coding: utf-8 -*-
"""E3 ablation runner, reusing train_e2's data/early-stop/evaluation code.

Example:
  python train_ablation.py --fd FD002 --variant no_soft_cap --seed 0 --tag e3
"""
import argparse
import json
import time
from pathlib import Path

import torch

import train_e2
from models_ablation import build_ablation

G2 = Path(__file__).resolve().parents[1]
RESULTS = G2 / "results"

BASE = {"rule": "self", "soft_cap": True, "spectral": True, "use_v": True,
        "shuffle_ae": False, "bidirectional": True, "layers": 2, "dim": 64,
        "chunk": 16, "gamma": 0.025, "alpha_init": 0.95, "stem": True}

VARIANTS = {
    "full": dict(BASE),
    "fixed": dict(BASE, rule="fixed"),
    "no_soft_cap": dict(BASE, soft_cap=False),
    "no_spectral": dict(BASE, spectral=False),
    "no_control": dict(BASE, soft_cap=False, spectral=False),
    "w_o_Mv": dict(BASE, use_v=False),
    "shuffle_ae": dict(BASE, shuffle_ae=True),
    "uni": dict(BASE, bidirectional=False),
    "C8": dict(BASE, chunk=8),
    "C32": dict(BASE, chunk=32),
    "L1": dict(BASE, layers=1),
    "L3": dict(BASE, layers=3),
    "dim48": dict(BASE, dim=48),
    "dim80": dict(BASE, dim=80),
    "gamma05": dict(BASE, gamma=0.05),
    "alpha09": dict(BASE, alpha_init=0.9),
    "alpha099": dict(BASE, alpha_init=0.99),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", default="FD002")
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="e3")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--cap", type=float, default=125.0)
    ap.add_argument("--split-seed", type=int, default=2026)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    spec = VARIANTS[args.variant]
    args.chunk = int(spec.get("chunk", 16))
    args.norm = 0
    args.stem = int(bool(spec.get("stem", False)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    orig = train_e2.build_model

    def factory(name, in_dim, cap=125.0, chunk=16, norm=False, stem=False):
        return build_ablation(spec, in_dim, cap)

    train_e2.build_model = factory
    try:
        t0 = time.strftime("%Y-%m-%d %H:%M:%S")
        print("[%s] %s %s seed %d start %s" % (args.tag, args.fd, args.variant, args.seed, t0), flush=True)
        out, gt, vt, ga = train_e2.run(args.fd, "ours_srnl", args.seed, args, device)
    finally:
        train_e2.build_model = orig

    out["model"] = args.variant
    out["variant_spec"] = spec
    out["started"] = t0
    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    stem = "%s_%s_%s_s%d" % (args.tag, args.fd, args.variant, args.seed)
    train_e2.write_pred(RESULTS / ("pred_gate_trunc_%s.csv" % stem), *gt)
    train_e2.write_pred(RESULTS / ("pred_val_trunc_%s.csv" % stem), *vt)
    (RESULTS / ("%s.json" % stem)).write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    ckpt_src = RESULTS / ("ckpt_%s_%s_ours_srnl_s%d.pt" % (args.tag, args.fd, args.seed))
    ckpt_dst = RESULTS / ("ckpt_%s.pt" % stem)
    if ckpt_src.exists():
        ckpt_src.replace(ckpt_dst)
    print("[%s] %s gate %.3f val %.3f best ep %d | %.1f ms/iter | %.0f MB"
          % (args.tag, stem, out["gate_trunc"]["rmse"], out["val_trunc"]["rmse"],
             out["best_epoch"], out["ms_per_iter_mean"], out["peak_mem_mb"]), flush=True)


if __name__ == "__main__":
    main()
