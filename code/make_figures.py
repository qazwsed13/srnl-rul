# -*- coding: utf-8 -*-
"""Create manuscript figures from the evidence JSONs (scientific-figure-pro style)."""
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

FDS = ["FD001", "FD002", "FD003", "FD004"]
MODELS = ["B1_cnn", "B2_lstm", "B3_bigru", "B4_transformer", "B5_tcn", "B8_srnl_fixed", "ours_srnl"]
LABELS = ["B1 CNN", "B2 LSTM", "B3 BiGRU", "B4 Transformer", "B5 TCN", "B8 fixed-rule", "Ours"]
COLORS = ["#8C8C8C", "#9A4D8E", "#42949E", "#6BAED6", "#B64342", "#CFCECE", "#0F4D92"]


def load(name):
    return json.loads((G2 / "evidence" / name).read_text(encoding="utf-8")) if (G2 / "evidence" / name).exists() else json.loads((G2 / "results" / name).read_text(encoding="utf-8"))


def fig_main_rmse():
    d = load("stats.json")["aggregate"]
    series = []
    for m in MODELS:
        series.append([d["%s|%s" % (fd, m)]["rmse_cap125_mean"] for fd in FDS])
    sfp.apply_publication_style(sfp.FigureStyle(font_size=12, axes_linewidth=1.8))
    fig, axes = sfp.create_subplots(1, 1, figsize=(11, 4.2))
    ax = axes[0]
    sfp.make_grouped_bar(ax, FDS, series, LABELS, ylabel="RMSE (cycles, cap 125)",
                         colors=COLORS)
    ax.set_ylim(11.5, 19.8)
    ax.legend(ncol=7, loc="upper center", bbox_to_anchor=(0.5, 1.18),
              frameon=False, fontsize=9, handlelength=1.2, columnspacing=1.0)
    ax.yaxis.grid(True, alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    sfp.finalize_figure(fig, FIG / "fig_main_rmse", formats=["pdf", "png"], dpi=300)


def fig_mechanism():
    d = load("mechanism_summary.json")
    mid = np.array([12.5, 37.5, 75.0, 113.0])
    bands = ["0-25", "25-50", "50-100", "100-126"]
    sfp.apply_publication_style(sfp.FigureStyle(font_size=12, axes_linewidth=1.8))
    fig, axes = sfp.create_subplots(1, 2, figsize=(9, 3.6))
    for ax, key, ylab in ((axes[0], "alpha", "retention α"),
                          (axes[1], "eta", "step size η")):
        for fd, color in (("FD002", "#0F4D92"), ("FD004", "#42949E")):
            means = [d[fd]["bands"][b][key + "_mean"] for b in bands]
            sds = [d[fd]["bands"][b][key + "_sd"] for b in bands]
            ax.errorbar(mid, means, yerr=sds, marker="o", capsize=3, linewidth=2,
                        color=color, label=fd)
        ax.set_xticks(mid)
        ax.set_xticklabels(bands)
        ax.set_xlabel("RUL band (cycles)")
        ax.set_ylabel(ylab)
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)
    sfp.finalize_figure(fig, FIG / "fig_mechanism", formats=["pdf", "png"], dpi=300)


def fig_noise():
    d = load("robust_summary.json")
    sfp.apply_publication_style(sfp.FigureStyle(font_size=12, axes_linewidth=1.8))
    fig, axes = sfp.create_subplots(1, 2, figsize=(9, 3.6))
    for ax, fd in zip(axes, ("FD002", "FD004")):
        for model, label, color in (("ours_srnl", "Ours", "#0F4D92"),
                                    ("B5_tcn", "B5 TCN", "#B64342"),
                                    ("B8_srnl_fixed", "B8 fixed-rule", "#8C8C8C")):
            v = d["%s|%s" % (fd, model)]
            ax.plot(v["sigma"], v["rel_pct"], marker="o", linewidth=2,
                    color=color, label=label)
        ax.set_xlabel("noise σ (z-scored sensors)")
        ax.set_ylabel("RMSE increase vs clean (%)")
        ax.set_title(fd, fontsize=12)
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)
    sfp.finalize_figure(fig, FIG / "fig_noise", formats=["pdf", "png"], dpi=300)


def fig_stability():
    d = json.loads((G2 / "results" / "e6_stability.json").read_text(encoding="utf-8"))
    checks = {c["tag"]: c for c in d["checks"]}
    cases = [("bi-SRNL default gamma=0.025", "default γ", "#0F4D92"),
             ("B8 fixed-rule default", "fixed-rule", "#CFCECE"),
             ("gamma=5 controlled", "γ=5 controlled", "#42949E"),
             ("gamma=5 uncontrolled", "γ=5 uncontrolled", "#B64342")]
    vals = [checks[c[0]]["max_ratio"] for c in cases]
    sfp.apply_publication_style(sfp.FigureStyle(font_size=12, axes_linewidth=1.8))
    fig, axes = sfp.create_subplots(1, 1, figsize=(7, 3.8))
    ax = axes[0]
    bars = ax.bar([c[1] for c in cases], vals, color=[c[2] for c in cases],
                  edgecolor="black", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_ylabel("max ||S_new|| / ||S_old||")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_ylim(0.1, 1e10)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.6, "%.2f" % v if v < 10 else "%.1e" % v,
                ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", labelsize=9)
    sfp.finalize_figure(fig, FIG / "fig_stability", formats=["pdf", "png"], dpi=300)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig_main_rmse()
    fig_mechanism()
    fig_noise()
    fig_stability()
    print("figures saved to", FIG)
