"""Appendix figures: evidence of method rather than headline findings.

  A1  every sensitivity cell (26/26)
  A2  false-accusation rate against homogeneity, and the population-size arm whose
      preregistered prediction was falsified
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "results"
INK, GRID = "#1b1b1f", "#d8d8de"
RED, GREEN = "#c92a2a", "#2b8a3e"
SERIES = {"D1 Cramer's V": "#4c6ef5", "D2 marginal MI": "#12b886",
          "D3 conditional MI": "#e8590c", "D4 net influence": "#9c36b5"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.edgecolor": GRID, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

LABEL = {"k (actions)": "k", "m (contexts)": "m", "persistence": "persist.",
         "T (rounds)": "T", "n (agents)": "n",
         "lambda (cartel strength)": "λ", "kappa_C (cartel baseline)": "κ_C"}


def fig_a1() -> None:
    sens = json.loads((OUT / "report_robustness.json").read_text())["arm5_sensitivity"]
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    y, ticks, labels = 0, [], []
    for sweep, cells in sens.items():
        if sweep.startswith("_"):
            continue
        for val, d in cells.items():
            for key, colour in (("level", RED), ("pattern", GREEN)):
                a, lo, hi = d[key]["auroc"], d[key]["lo"], d[key]["hi"]
                ax.plot([lo, hi], [y, y], color=colour, lw=1.4, alpha=0.5)
                ax.plot([a], [y], "o", color=colour, ms=4.5)
            ticks.append(y)
            labels.append(f"{LABEL[sweep]} = {val}")
            y += 1
        y += 0.6
    ax.axvline(0.5, color=INK, ls="--", lw=1.1, alpha=0.75)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlim(-0.03, 1.03)
    ax.invert_yaxis()
    ax.set_xlabel("accuracy  (AUROC)")
    ax.set_title("A1. The sign holds in all 26 sensitivity cells",
                 fontsize=10.5, loc="left", pad=22)
    ax.text(0, 1.025, "red = scores the top pair alone   ·   green = compares pairs",
            transform=ax.transAxes, fontsize=8, color=INK, alpha=0.75)
    fig.tight_layout()
    fig.savefig(OUT / "figureA1.png", bbox_inches="tight")
    print(f"wrote {OUT / 'figureA1.png'}")


def fig_a2() -> None:
    r1 = json.loads((OUT / "report.json").read_text())
    r3 = json.loads((OUT / "report_population.json").read_text())
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 3.9))

    rates = r1["monoculture_curve"]["rates"]
    kappas = sorted(float(k) for k in rates)
    for name, colour in SERIES.items():
        axL.plot(kappas, [rates[str(k)][name] for k in kappas], "o-",
                 color=colour, lw=1.8, ms=4, label=name)
    axL.axhline(0.05, color=INK, ls="--", lw=1, alpha=0.65)
    axL.text(0.26, 0.08, "nominal 5%", fontsize=7.5, color=INK, alpha=0.75)
    axL.set_xlabel("how alike the agents are")
    axL.set_ylabel("false-accusation rate")
    axL.set_ylim(-0.03, 1.05)
    axL.set_title("A2a. No agreement exists in any run here",
                  fontsize=10.5, loc="left", pad=8)
    axL.legend(frameon=False, fontsize=7.5, loc="center right")

    ns = r3["ns"]
    for name, colour in SERIES.items():
        axR.plot(ns, [r3["results"][str(n)]["false_accusation_rate"][name] for n in ns],
                 "o-", color=colour, lw=1.8, ms=4)
    axR.axhline(0.70, color="#c92a2a", ls=":", lw=1.4)
    axR.text(5.0, 0.73, "published n=3 reference (0.70)", fontsize=7.5, color="#c92a2a")
    axR.set_xlabel("agents in the group")
    axR.set_ylabel("false-accusation rate")
    axR.set_ylim(-0.03, 1.05)
    axR.set_title("A2b. Our preregistered prediction, falsified",
                  fontsize=10.5, loc="left", pad=8)

    fig.tight_layout()
    fig.savefig(OUT / "figureA2.png", bbox_inches="tight")
    print(f"wrote {OUT / 'figureA2.png'}")


if __name__ == "__main__":
    fig_a1()
    fig_a2()
