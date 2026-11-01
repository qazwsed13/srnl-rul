# -*- coding: utf-8 -*-
"""Build the manuscript Table 3 (FD002 + FD004 key ablations)."""
import csv
import json
from pathlib import Path

G2 = Path(__file__).resolve().parents[1]
RESULTS = G2 / "results"
EVID = G2 / "evidence"

ORDER = ["full", "fixed", "no_soft_cap", "w_o_Mv", "shuffle_ae", "uni",
         "no_spectral", "C8", "L1", "dim48", "gamma05"]


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main():
    d2 = load("e3_summary.json")
    d4 = load("e3_fd004_summary.json")
    rows = []
    for v in ORDER:
        a2 = d2["aggregate"].get(v)
        a4 = d4["aggregate"].get(v)
        p2 = d2.get("paired_vs_full", {}).get(v)
        p4 = d4.get("paired_vs_full", {}).get(v)
        if not (a2 or a4):
            continue
        rows.append({
            "variant": v,
            "fd002_gate": "%.3f" % a2["mean"] if a2 else "",
            "fd002_delta_pct": "%+.2f" % p2["delta_pct"] if p2 else "",
            "fd002_ci": "[%+.2f, %+.2f]" % (p2["ci_lo"], p2["ci_hi"]) if p2 else "",
            "fd004_gate": "%.3f" % a4["mean"] if a4 else "",
            "fd004_delta_pct": "%+.2f" % p4["delta_pct"] if p4 else "",
            "fd004_ci": "[%+.2f, %+.2f]" % (p4["ci_lo"], p4["ci_hi"]) if p4 else "",
        })
    with open(EVID / "table3_ablation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(open(EVID / "table3_ablation.csv", encoding="utf-8").read())


if __name__ == "__main__":
    main()
