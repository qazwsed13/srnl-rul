# -*- coding: utf-8 -*-
"""E2 training: multi-condition FD002/FD004, early stopping on val, gate predictions.

Protocol fixed before running (conception/02 §3.3 G2):
  - 70/15/15 engine-level split (split seed 2026); train -> val (early stop) -> gate (go/no-go).
  - early stopping metric: RMSE over all val windows.
  - gate metric: 3 random truncation points per gate engine (seed 2027 + engine id),
    which mimics the test protocol (one truncated trajectory per engine).
  - official test is NOT touched here.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import data as D
import metrics as M
from models import build_model, count_params, new_log

G2 = Path(__file__).resolve().parents[1]
RESULTS = G2 / "results"
CACHE = RESULTS / "cache"


def load_prepared(fd, L=30, cap=125.0, split_seed=2026):
    CACHE.mkdir(parents=True, exist_ok=True)
    npz = CACHE / ("%s_L%d.npz" % (fd, L))
    mp = CACHE / ("%s_meta.json" % fd)
    if npz.exists() and mp.exists():
        z = np.load(npz)
        meta = json.loads(mp.read_text(encoding="utf-8"))
        return z["x"], z["y"], z["e"], z["last"], meta

    sel = D.select_sensor_subset()
    df = D.load_frame(fd, "train")
    split = D.split_engines(df["engine"].unique(), seed=split_seed)
    prep = D.Preprocessor.fit(df[df["engine"].isin(split["train"])],
                              sel["sensors"], n_clusters=6, seed=0)
    x, y, e, last = D.build_windows(df, prep, L=L, cap=cap, stride=1)
    np.savez_compressed(npz, x=x, y=y, e=e, last=last)
    meta = {
        "fd": fd,
        "window": L,
        "cap": cap,
        "split_seed": split_seed,
        "split": {k: [int(v) for v in vs] for k, vs in split.items()},
        "preprocessor": prep.to_dict(),
        "sensor_selection": sel,
        "n_windows": int(len(x)),
        "input_dim": int(x.shape[-1]),
    }
    mp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return x, y, e, last, meta


def build_trunc(df, prep, engines, L=30, cap=125.0, k=3, seed=2027):
    """k random truncation points per engine, label = min(max_cycle - t, cap)."""
    feats = prep.transform(df)
    xs, ys, es = [], [], []
    eng = set(int(v) for v in engines)
    for eid, g in df.groupby("engine", sort=True):
        if int(eid) not in eng:
            continue
        f = feats[g.index.values]
        n = len(g)
        cyc = g["cycle"].values
        maxc = int(cyc.max())
        rng = np.random.RandomState(seed + int(eid))
        ends = rng.randint(L - 1, n, size=k)
        for t in ends:
            xs.append(f[t - L + 1:t + 1])
            ys.append(min(maxc - int(cyc[t]), cap))
            es.append(int(eid))
    return (np.stack(xs).astype(np.float32),
            np.asarray(ys, np.float32),
            np.asarray(es, np.int64))


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batch_index(n, bs, shuffle, rng):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def merge_log(dst, src):
    if src is None:
        return
    dst["max_ratio"] = max(dst["max_ratio"], src["max_ratio"])
    dst["final_norm"] = max(dst["final_norm"], src["final_norm"])
    dst["finite"] = dst["finite"] and src["finite"]


def evaluate(model, x, y, device, bs=512, want_log=False):
    model.eval()
    preds = []
    log = new_log()
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = torch.from_numpy(x[i:i + bs]).to(device)
            lg = new_log() if want_log else None
            p = model(xb, lg)
            merge_log(log, lg)
            preds.append(p.detach().cpu().numpy())
    return np.concatenate(preds), log


def run(fd, name, seed, args, device):
    x, y, e, last, meta = load_prepared(fd, args.window, args.cap, args.split_seed)
    split = meta["split"]
    prep = D.Preprocessor.from_dict(meta["preprocessor"])
    x_tr, y_tr, _, _ = D.subset_windows(x, y, e, last, split["train"])
    x_va, y_va, _, _ = D.subset_windows(x, y, e, last, split["val"])
    x_ga, y_ga, e_ga, _ = D.subset_windows(x, y, e, last, split["gate"])

    df = D.load_frame(fd, "train")
    x_vt, y_vt, e_vt = build_trunc(df, prep, split["val"], args.window, args.cap, 3, 2027)
    x_gt, y_gt, e_gt = build_trunc(df, prep, split["gate"], args.window, args.cap, 3, 2027)

    set_seed(seed)
    model = build_model(name, x_tr.shape[-1], chunk=args.chunk,
                        norm=bool(args.norm), stem=bool(args.stem)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    rng = np.random.RandomState(seed)
    best_rmse, best_ep, best_state = float("inf"), -1, None
    history, it_times, nan_steps, gstep = [], [], 0, 0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        model.train()
        losses = []
        diverged = False
        for idx in batch_index(len(x_tr), args.bs, True, rng):
            xb = torch.from_numpy(x_tr[idx]).to(device)
            yb = torch.from_numpy(y_tr[idx] / args.cap).to(device)
            t_it = time.perf_counter()
            pred = model(xb)
            loss = torch.mean((pred / args.cap - yb) ** 2)
            lv = float(loss.detach())
            if not math.isfinite(lv):
                nan_steps += 1
                diverged = True
                print("  non-finite loss at epoch %d, stop and restore best" % (ep + 1),
                      flush=True)
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t_it
            gstep += 1
            if gstep > args.warmup:
                it_times.append(dt)
            losses.append(lv)
        if diverged:
            break
        sched.step()
        pv, _ = evaluate(model, x_va, y_va, device)
        va_rmse = M.rmse(pv, y_va)
        history.append({"epoch": ep + 1, "train_mse_norm": float(np.mean(losses)),
                        "val_rmse_all": va_rmse})
        if va_rmse < best_rmse - 1e-6:
            best_rmse, best_ep = va_rmse, ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep + 1 - best_ep >= args.patience:
            print("  early stop at epoch %d (best %d, val_rmse %.3f)"
                  % (ep + 1, best_ep, best_rmse), flush=True)
            break
        if (ep + 1) % args.log_every == 0:
            print("  %s s%d ep %d  train_mse %.5f  val_rmse %.3f  best %.3f@%d"
                  % (name, seed, ep + 1, history[-1]["train_mse_norm"],
                     va_rmse, best_rmse, best_ep), flush=True)
    train_s = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(),
               RESULTS / ("ckpt_%s_%s_%s_s%d.pt" % (args.tag, fd, name, seed)))

    pv, _ = evaluate(model, x_va, y_va, device)
    pvt, log_v = evaluate(model, x_vt, y_vt, device, want_log=True)
    pga, _ = evaluate(model, x_ga, y_ga, device)
    pgt, _ = evaluate(model, x_gt, y_gt, device)
    ms = [t * 1000.0 for t in it_times]
    peak_mb = (torch.cuda.max_memory_allocated() / 1024 ** 2) if device == "cuda" else float("nan")
    out = {
        "fd": fd,
        "model": name,
        "seed": seed,
        "params": count_params(model),
        "best_epoch": best_ep,
        "epochs_run": len(history),
        "train_seconds": train_s,
        "ms_per_iter_mean": float(np.mean(ms)) if ms else None,
        "peak_mem_mb": float(peak_mb),
        "nan_steps": nan_steps,
        "history": history,
        "val_all": M.regression_report(pv, y_va),
        "val_trunc": M.regression_report(pvt, y_vt),
        "gate_all": M.regression_report(pga, y_ga),
        "gate_trunc": M.regression_report(pgt, y_gt),
        "memory_log": log_v,
    }
    return out, (e_gt, y_gt, pgt), (e_vt, y_vt, pvt), (e_ga, y_ga, pga)


def write_pred(path, eng, true, pred):
    with open(path, "w", encoding="utf-8") as f:
        f.write("engine,rul_true,rul_pred\n")
        for a, b, c in zip(eng, true, pred):
            f.write("%d,%.4f,%.4f\n" % (int(a), float(b), float(c)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--cap", type=float, default=125.0)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--norm", type=int, default=0)
    ap.add_argument("--stem", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=2026)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--tag", default="e2a")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.strftime("%Y-%m-%d %H:%M:%S")
    print("[%s] %s %s seed %d start %s" % (args.tag, args.fd, args.model, args.seed, t0), flush=True)
    out, gt, vt, ga = run(args.fd, args.model, args.seed, args, device)
    out["started"] = t0
    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out["config"] = {
        "epochs_max": args.epochs, "patience": args.patience, "batch_size": args.bs,
        "lr": args.lr, "weight_decay": args.wd, "window": args.window, "cap": args.cap,
        "chunk": args.chunk, "layer_norm": bool(args.norm), "conv_stem": bool(args.stem),
        "split_seed": args.split_seed, "gate_trunc_k": 3, "gate_trunc_seed": 2027,
        "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "note": "gate split from train engines; official test untouched",
    }
    stem = "%s_%s_%s_s%d" % (args.tag, args.fd, args.model, args.seed)
    write_pred(RESULTS / ("pred_gate_trunc_%s.csv" % stem), *gt)
    write_pred(RESULTS / ("pred_val_trunc_%s.csv" % stem), *vt)
    write_pred(RESULTS / ("pred_gate_all_%s.csv" % stem), *ga)
    (RESULTS / ("%s.json" % stem)).write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[%s] %s gate_trunc RMSE %.3f | val_trunc %.3f | best ep %d | %.1f ms/iter | %.0f MB"
          % (args.tag, stem, out["gate_trunc"]["rmse"], out["val_trunc"]["rmse"],
             out["best_epoch"], out["ms_per_iter_mean"], out["peak_mem_mb"]), flush=True)


if __name__ == "__main__":
    main()


