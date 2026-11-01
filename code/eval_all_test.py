# -*- coding: utf-8 -*-
"""Run eval_test.py once for every E2b checkpoint (after all training is done)."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

G2 = Path(__file__).resolve().parents[1]
RESULTS = G2 / "results"
CODE = G2 / "code"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="e2b")
    ap.add_argument("--tag", default="e2b_test")
    ap.add_argument("--confirm", action="store_true",
                    help="required: official test is evaluated once")
    args = ap.parse_args()
    if not args.confirm:
        raise SystemExit("refusing to touch official test without --confirm")

    done = 0
    for p in sorted(RESULTS.glob("%s_*.json" % args.prefix)):
        if p.name.endswith("_summary.json"):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if "gate_trunc" not in d:
            continue
        stem = p.stem
        ckpt = RESULTS / ("ckpt_%s.pt" % stem)
        if not ckpt.exists():
            print("MISSING ckpt", ckpt)
            continue
        cfg = d.get("config", {})
        cmd = [sys.executable, str(CODE / "eval_test.py"),
               "--fd", d["fd"], "--model", d["model"], "--seed", str(d["seed"]),
               "--ckpt", str(ckpt), "--tag", args.tag,
               "--chunk", str(cfg.get("chunk", 16)),
               "--stem", str(int(cfg.get("conv_stem", False)))]
        print("RUN", " ".join(cmd[-8:]), flush=True)
        subprocess.run(cmd, check=True)
        done += 1
    print("evaluated", done, "checkpoints")


if __name__ == "__main__":
    main()
