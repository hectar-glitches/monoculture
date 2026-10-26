"""Figure 3: two repairs that Figure 2 does not already show.

Figure 2 covers the null-class repair. This covers the other two:

  A. Observe the shared context. Strongest, but least available for language agents,
     because much of what they share is the model prior and cannot be logged.
  B. Divide by the spread. Weakest, but free, and it is what makes one threshold
     usable across different degrees of homogeneity.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "results"
INK, GRID = "#1b1b1f", "#d8d8de"
RED, BLUE, GREY = "#c92a2a", "#1864ab", "#495057"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.edgecolor": GRID, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def main() -> int:
    r1 = json.loads((OUT / "report.json").read_text())
    mech = json.loads((OUT / "report_robustness.json").read_text())["arm6_mechanism"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.4, 4.0))

    # ---- A: observing the shared context -------------------------------
    phis = sorted(float(p) for p in r1["phi_sweep"])
    series = [("D3 conditional MI", BLUE, "compares against the context"),
              ("D2 marginal MI", RED, "ignores the context")]
    for name, colour, note in series:
        ys = [r1["phi_sweep"][str(p)][name]["auroc"] for p in phis]
        lo = [r1["phi_sweep"][str(p)][name]["lo"] for p in phis]
        hi = [r1["phi_sweep"][str(p)][name]["hi"] for p in phis]
        axA.plot(phis, ys, "o-", color=colour, lw=2.0, ms=5, label=note)
        axA.fill_between(phis, lo, hi, color=colour, alpha=0.15, lw=0)

    axA.axhline(0.5, color=INK, ls="--", lw=1.1, alpha=0.7)
    axA.text(0.02, 0.515, "chance", fontsize=7.5, color=INK, alpha=0.75)
    axA.set_xlabel("share of the agents' common context the auditor can see")
    axA.set_ylabel("accuracy  (AUROC)")
    axA.set_ylim(-0.03, 1.06)
    axA.set_title("A. Logging what agents have in common", fontsize=10.5,
                  loc="left", pad=20)
    axA.text(0, 1.035, "works completely, but for language agents it is only "
                       "partly available",
             transform=axA.transAxes, fontsize=8, color=INK, alpha=0.75)
    axA.legend(frameon=False, fontsize=8.5, loc="center right")

    # ---- B: dividing by the spread -------------------------------------
    ks = sorted(float(k) for k in mech if not k.startswith("_"))
    raw = [mech[str(k)]["raw_contrast"] for k in ks]
    stud = [mech[str(k)]["studentised_contrast"] for k in ks]
    axB.plot(ks, [v / raw[0] for v in raw], "o-", color=RED, lw=2.0, ms=5,
             label="raw score")
    axB.plot(ks, [v / stud[0] for v in stud], "o-", color=BLUE, lw=2.0, ms=5,
             label="divided by the spread")
    axB.axhline(1.0, color=INK, ls="--", lw=1.1, alpha=0.7)
    axB.set_xlabel("how alike the agents are")
    axB.set_ylabel("score, relative to its value in a diverse group")
    axB.set_title("B. Making one threshold work everywhere", fontsize=10.5,
                  loc="left", pad=20)
    axB.text(0, 1.035, "a flat line means a threshold set in one group transfers "
                       "to another",
             transform=axB.transAxes, fontsize=8, color=INK, alpha=0.75)
    axB.legend(frameon=False, fontsize=8.5, loc="upper left")
    axB.annotate(f"×{raw[-1] / raw[0]:.1f}", xy=(ks[-1], raw[-1] / raw[0]),
                 xytext=(-32, 2), textcoords="offset points", fontsize=9, color=RED)
    axB.annotate(f"×{stud[-1] / stud[0]:.1f}", xy=(ks[-1], stud[-1] / stud[0]),
                 xytext=(-32, -14), textcoords="offset points", fontsize=9, color=BLUE)

    fig.tight_layout()
    path = OUT / "figure3.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
