"""Figure 3: robustness of the sign, and the studentisation repair."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "results"
INK, GRID = "#1b1b1f", "#d8d8de"
RED, GREEN, BLUE = "#c92a2a", "#2b8a3e", "#1864ab"

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


def main() -> int:
    r = json.loads((OUT / "report_robustness.json").read_text())
    sens, mech = r["arm5_sensitivity"], r["arm6_mechanism"]

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(12.2, 4.3), gridspec_kw={"width_ratios": [1.45, 1]})

    # ---- panel A: every sensitivity cell -------------------------------
    y, ticks, labels = 0, [], []
    for sweep, cells in sens.items():
        if sweep.startswith("_"):
            continue
        for val, d in cells.items():
            for key, colour in (("level", RED), ("pattern", GREEN)):
                a, lo, hi = d[key]["auroc"], d[key]["lo"], d[key]["hi"]
                axA.plot([lo, hi], [y, y], color=colour, lw=1.4, alpha=0.5)
                axA.plot([a], [y], "o", color=colour, ms=4.5)
            ticks.append(y)
            labels.append(f"{LABEL[sweep]} = {val}")
            y += 1
        y += 0.6

    axA.axvline(0.5, color=INK, ls="--", lw=1.1, alpha=0.75)
    axA.set_yticks(ticks)
    axA.set_yticklabels(labels, fontsize=7)
    axA.set_xlim(-0.03, 1.03)
    axA.set_xlabel("AUROC   (cartel vs. matched honest monoculture)")
    axA.invert_yaxis()
    axA.set_title("A. The sign holds everywhere: 26/26 cells",
                  fontsize=10.5, loc="left", pad=22)
    axA.text(0, 1.035, "red = level-reading statistic   ·   green = pattern-reading",
             transform=axA.transAxes, fontsize=8, color=INK, alpha=0.75)

    # ---- panel B: the variance mechanism and its repair ----------------
    ks = sorted(float(k) for k in mech if not k.startswith("_"))
    raw = [mech[str(k)]["raw_contrast"] for k in ks]
    sd = [mech[str(k)]["sd_gamma"] for k in ks]
    stud = [mech[str(k)]["studentised_contrast"] for k in ks]

    axB.plot(ks, [v / raw[0] for v in raw], "o-", color=RED, lw=1.9, ms=5,
             label="raw contrast")
    axB.plot(ks, [v / sd[0] for v in sd], "s--", color=INK, lw=1.4, ms=4,
             alpha=0.65, label="sd of γ")
    axB.plot(ks, [v / stud[0] for v in stud], "o-", color=BLUE, lw=1.9, ms=5,
             label="studentised contrast")
    axB.axhline(1.0, color=INK, ls=":", lw=1, alpha=0.6)
    axB.set_xlabel("κ  —  how alike the agents' priors are")
    axB.set_ylabel("statistic, relative to its value at κ = 0.25")
    axB.set_title("B. Why thresholds don't transfer, and the fix",
                  fontsize=10.5, loc="left", pad=22)
    axB.text(0, 1.035, "a flat line means one threshold works across homogeneity regimes",
             transform=axB.transAxes, fontsize=8, color=INK, alpha=0.75)
    axB.legend(frameon=False, fontsize=8, loc="upper left")
    axB.annotate(f"×{raw[-1] / raw[0]:.1f}", xy=(ks[-1], raw[-1] / raw[0]),
                 xytext=(-30, 6), textcoords="offset points",
                 fontsize=8.5, color=RED)
    axB.annotate(f"×{stud[-1] / stud[0]:.1f}", xy=(ks[-1], stud[-1] / stud[0]),
                 xytext=(-30, -16), textcoords="offset points",
                 fontsize=8.5, color=BLUE)

    fig.tight_layout()
    path = OUT / "figure3.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
