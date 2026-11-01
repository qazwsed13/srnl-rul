# -*- coding: utf-8 -*-
"""C-MAPSS data pipeline for G2 (E1)."""
from pathlib import Path

import numpy as np
import pandas as pd

G2 = Path(__file__).resolve().parents[1]
DATA_DIR = G2 / "data" / "CMAPSSData"

COLS = (["engine", "cycle"] + ["set%d" % i for i in (1, 2, 3)]
        + ["s%d" % i for i in range(1, 22)])
ALL_FDS = ("FD001", "FD002", "FD003", "FD004")


def load_frame(fd, split):
    path = DATA_DIR / ("%s_%s.txt" % (split, fd))
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")
    if df.shape[1] != 26:
        raise ValueError("%s %s: expected 26 columns, got %d" % (fd, split, df.shape[1]))
    df.columns = COLS
    return df


def load_rul(fd):
    return pd.read_csv(DATA_DIR / ("RUL_%s.txt" % fd), header=None).values.ravel().astype(np.float32)


def constant_sensors(df, tol=1e-8):
    std = df[["s%d" % i for i in range(1, 22)]].std(ddof=0).values
    return [i + 1 for i, s in enumerate(std) if s < tol]


def select_sensor_subset():
    """FD001-train non-constant sensors; drop any all-zero sensor on FD002/FD004 train.

    The four-subset variance union is returned for the sensitivity table only
    (conception/02 §3.2); the model input uses the FD001 rule.
    """
    info = {}
    union = set()
    for fd in ALL_FDS:
        const = set(constant_sensors(load_frame(fd, "train")))
        info[fd] = sorted(const)
        union |= set(range(1, 22)) - const

    base = set(range(1, 22)) - set(info["FD001"])
    dropped_zero = []
    for fd in ("FD002", "FD004"):
        df = load_frame(fd, "train")
        for s in sorted(base):
            if np.abs(df["s%d" % s].values).max() < 1e-12:
                dropped_zero.append([fd, s])
                base.discard(s)

    return {
        "sensors": sorted(base),
        "constant_by_fd": {k: v for k, v in info.items()},
        "variance_union": sorted(union),
        "dropped_all_zero": dropped_zero,
    }


def kmeans_np(x, k, seed=0, iters=100):
    """Small k-means++ (no sklearn in this env); x is (n, d) float64."""
    rng = np.random.RandomState(seed)
    n = x.shape[0]
    centers = [x[rng.randint(n)]]
    d2 = ((x - centers[0]) ** 2).sum(1)
    for _ in range(1, k):
        prob = d2 / max(d2.sum(), 1e-12)
        centers.append(x[rng.choice(n, p=prob)])
        d2 = np.minimum(d2, ((x - centers[-1]) ** 2).sum(1))
    c = np.stack(centers)
    labels = None
    for _ in range(iters):
        dist = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        labels = dist.argmin(1)
        new_c = np.stack([
            x[labels == j].mean(0) if (labels == j).any() else c[j] for j in range(k)
        ])
        if np.allclose(new_c, c, atol=1e-9):
            c = new_c
            break
        c = new_c
    return c, labels


class Preprocessor(object):
    """Single-condition: global z-score. Multi-condition: k-means(k=6) + per-cluster z-score."""

    def __init__(self, sensor_ids, single, set_mean, set_std, mean, std,
                 centroids=None, cluster_mean=None, cluster_std=None):
        self.sensor_ids = list(sensor_ids)
        self.single = bool(single)
        self.set_mean = np.asarray(set_mean, np.float64)
        self.set_std = np.asarray(set_std, np.float64)
        self.mean = np.asarray(mean, np.float64)
        self.std = np.asarray(std, np.float64)
        self.centroids = None if centroids is None else np.asarray(centroids, np.float64)
        self.cluster_mean = None if cluster_mean is None else np.asarray(cluster_mean, np.float64)
        self.cluster_std = None if cluster_std is None else np.asarray(cluster_std, np.float64)

    @classmethod
    def fit(cls, train_df, sensor_ids, n_clusters=6, seed=0):
        sens = ["s%d" % s for s in sensor_ids]
        sets = ["set1", "set2", "set3"]
        s = train_df[sets].values.astype(np.float64)
        set_mean = s.mean(0)
        set_std = s.std(0)
        single = bool(np.all(set_std < 1e-6))
        set_std = np.where(set_std < 1e-8, 1.0, set_std)

        x = train_df[sens].values.astype(np.float64)
        mean = x.mean(0)
        std = np.where(x.std(0) < 1e-6, 1.0, x.std(0))

        if single:
            return cls(sensor_ids, True, set_mean, set_std, mean, std)

        z = (s - set_mean) / set_std
        centroids, labels = kmeans_np(z, n_clusters, seed=seed)
        cm = np.zeros((n_clusters, len(sens)))
        cs = np.zeros((n_clusters, len(sens)))
        for j in range(n_clusters):
            m = labels == j
            if m.sum() >= 10:
                cm[j] = x[m].mean(0)
                cs[j] = x[m].std(0)
            else:
                cm[j] = mean
                cs[j] = std
        cs = np.where(cs < 1e-6, 1.0, cs)
        return cls(sensor_ids, False, set_mean, set_std, mean, std,
                   centroids, cm, cs)

    def transform(self, df):
        sens = ["s%d" % s for s in self.sensor_ids]
        sets = ["set1", "set2", "set3"]
        x = df[sens].values.astype(np.float64)
        s = df[sets].values.astype(np.float64)
        z = (s - self.set_mean) / self.set_std
        if self.single:
            xn = (x - self.mean) / self.std
        else:
            zc = (s - self.set_mean) / self.set_std
            dist = ((zc[:, None, :] - self.centroids[None, :, :]) ** 2).sum(-1)
            lab = dist.argmin(1)
            xn = (x - self.cluster_mean[lab]) / self.cluster_std[lab]
        return np.concatenate([xn, z], axis=1).astype(np.float32)

    def to_dict(self):
        d = {
            "sensor_ids": self.sensor_ids,
            "single_condition": self.single,
            "setting_mean": self.set_mean.tolist(),
            "setting_std": self.set_std.tolist(),
            "sensor_mean": self.mean.tolist(),
            "sensor_std": self.std.tolist(),
        }
        if self.centroids is not None:
            d["centroids"] = self.centroids.tolist()
            d["cluster_mean"] = self.cluster_mean.tolist()
            d["cluster_std"] = self.cluster_std.tolist()
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(d["sensor_ids"], d["single_condition"],
                   d["setting_mean"], d["setting_std"],
                   d["sensor_mean"], d["sensor_std"],
                   d.get("centroids"), d.get("cluster_mean"), d.get("cluster_std"))


def build_windows(df, prep, L=30, cap=125, stride=1):
    """Sliding windows ending at each cycle >= L; label = min(max_cycle - t, cap)."""
    feats = prep.transform(df)
    xs, ys, es, last = [], [], [], []
    for eid, g in df.groupby("engine", sort=True):
        f = feats[g.index.values]
        maxc = int(g["cycle"].max())
        rul = np.minimum(maxc - g["cycle"].values, cap)
        n = len(g)
        for end in range(L - 1, n, stride):
            xs.append(f[end - L + 1:end + 1])
            ys.append(rul[end])
            es.append(eid)
            last.append(end == n - 1)
    return (np.stack(xs).astype(np.float32),
            np.asarray(ys, np.float32),
            np.asarray(es, np.int64),
            np.asarray(last, bool))


def build_test_last_windows(df, prep, L=30):
    """One window per test engine: last L cycles (left-pad with first row if short)."""
    feats = prep.transform(df)
    xs, es = [], []
    for eid, g in df.groupby("engine", sort=True):
        f = feats[g.index.values]
        if len(f) >= L:
            w = f[-L:]
        else:
            w = np.concatenate([np.repeat(f[:1], L - len(f), 0), f], 0)
        xs.append(w)
        es.append(eid)
    return np.stack(xs).astype(np.float32), np.asarray(es, np.int64)


def split_engines(engines, seed=2026, frac=(0.70, 0.15, 0.15)):
    e = np.array(sorted(set(int(x) for x in engines)))
    rng = np.random.RandomState(seed)
    rng.shuffle(e)
    n = len(e)
    n_tr = int(round(frac[0] * n))
    n_va = int(round(frac[1] * n))
    return {"train": np.sort(e[:n_tr]),
            "val": np.sort(e[n_tr:n_tr + n_va]),
            "gate": np.sort(e[n_tr + n_va:])}


def subset_windows(x, y, e, last, engines):
    m = np.isin(e, np.asarray(engines))
    return x[m], y[m], e[m], last[m]


if __name__ == "__main__":
    sel = select_sensor_subset()
    print("unified sensors:", sel["sensors"])
    print("count:", len(sel["sensors"]))
    for fd in ALL_FDS:
        print(fd, "constant:", sel["constant_by_fd"][fd])
    print("variance union:", sel["variance_union"], "count:", len(sel["variance_union"]))
    print("dropped all-zero:", sel["dropped_all_zero"])
