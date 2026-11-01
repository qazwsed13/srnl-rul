# -*- coding: utf-8 -*-
"""Fig. 5: ablation deltas for FD002 and FD004 (gate)."""
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

G2 = Path(__file__).resolve().parents[1]
FIG = G2 / "figures"
HELPER = Path(r"C:\Users\柴一萌\.qoder\skills\scientific-figure-pro\scripts\scientific_figure_pro.py")
spec = importlib.util.spec_from_file_location("sfp", HELPER)
sfp = importlib.util.module_from_spec(spec)
sys.modules["sfp"] = sfp
spec.loader.exec_module(sfp)

LABELS = {"full": "Full", "fixed": "Fixed-rule", "no_soft_cap": "w/o soft cap",
          "w_o_Mv": "w/o M$^v$", "shuffle_ae": "Shuffle η/α", "uni": "Unidirectional",
          "no_spectral": "w/o spectral", "C8": "C = 8", "L1": "1 layer",
          "dim48": "width 48", "gamma05": "γ = 0.05"}


def load(path):
    return json.loads((G2 / "results" / path).read_text(encoding="utf-8"))


def panel(ax, d, title):
    vals = []
    for v, a in d["aggregate"].items():
        if v == "full":
            continue
        p = d.get("paired_vs_full", {}).get(v)
        delta = p["delta_pct"] if p else (a["mean"] / d["aggregate"]["full"]["mean"] - 1) * 100
        vals.append((v, delta))
    vals.sort(key=lambda z: z[1])
    names = [LABELS.get(v, v) for v, _ in vals]
    xs = [max(0.0, x) for _, x in vals]
    colors = []
    for v, x in vals:
        if v == "no_soft_cap":
            colors.append("#B64342")
        elif v == "w_o_Mv":
            colors.append("#E9A6A1")
        elif v == "fixed":
            colors.append("#42949E")
        else:
            colors.append("#CFCECE")
    y = np.arange(len(vals))
    ax.barh(y, xs, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Δ RMSE vs full (%)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    for yi, (_, x) in zip(y, vals):
        ax.text(max(x, 0) + 0.5, yi, "%+.1f" % x, va="center", fontsize=8)


def main():
    d2 = load("e3_summary.json")
    d4 = load("e3_fd004_summary.json")
    sfp.apply_publication_style(sfp.FigureStyle(font_size=11, axes_linewidth=1.5))
    fig, axes = sfp.create_subplots(1, 2, figsize=(10, 4.2))
    panel(axes[0], d2, "FD002 (3 seeds)")
    panel(axes[1], d4, "FD004 (2 seeds)")
    fig.tight_layout()
    sfp.finalize_figure(fig, FIG / "fig_ablation", formats=["pdf", "png"], dpi=300)
    print("saved", FIG / "fig_ablation.png")


if __name__ == "__main__":
    main()