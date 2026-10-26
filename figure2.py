"""Figure 2: which aggregation forms invert, and the null-class repair."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "results"
INK, GRID = "#1b1b1f", "#d8d8de"
RED, GREEN = "#c92a2a", "#2b8a3e"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.edgecolor": GRID, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.grid": True, "axes.grid.axis": "x",
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

SHORT = {
    "A1 Peak (max over agents)": "Peak\nmax over agents",
    "A2 Alignment (mean pairwise)": "Alignment\nmean pairwise",
    "A3 Split (top-k vs rest)": "Split\ntop-k vs rest",
    "A4 GLRT / independence null": "GLRT\nindependence null",
    "A5 GLRT / homogeneous null": "GLRT\nhomogeneous null",
}


def main() -> int:
    r = json.loads((OUT / "report_aggregation.json").read_text())["results"]
    names = list(SHORT)
    ys = range(len(names))[::-1]

    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    for y, name in zip(ys, names):
        d = r[name]
        a, lo, hi = d["auroc"], d["lo"], d["hi"]
        colour = RED if d["reads"] == "level" else GREEN
        ax.barh(y, a - 0.5, left=0.5, height=0.55, color=colour, alpha=0.85)
        ax.plot([lo, hi], [y, y], color=INK, lw=1.3, alpha=0.8)
        label = f"{a:.3f}"
        ax.text(a + (0.025 if a >= 0.5 else -0.025), y, label, va="center",
                ha="left" if a >= 0.5 else "right", fontsize=8.5, color=INK)

    ax.axvline(0.5, color=INK, lw=1.1, ls="--", alpha=0.7)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([SHORT[n] for n in names], fontsize=8.5)
    ax.set_xlim(-0.16, 1.14)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("accuracy  (AUROC)   —  pair mechanism vs. shared prior, context unobserved")
    ax.set_title("Group-aggregation form decides whether a detector works or inverts",
                 fontsize=11, loc="left", pad=26)
    ax.text(0, 1.055, "red = scores the top pair alone   ·   "
                      "green = compares pairs against each other",
            transform=ax.transAxes, fontsize=8.5, color=INK, alpha=0.75)
    ax.text(0.5, len(names) - 0.45, "chance", fontsize=8,
            color=INK, alpha=0.7, ha="center")

    # Bracket only the two GLRT rows, which are the controlled comparison.
    ax.annotate("", xy=(1.085, 1.0), xytext=(1.085, 0.0),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0, alpha=0.55))
    ax.text(1.10, 0.5, "same test,\nsame data,\nonly the null\nclass differs",
            fontsize=7.5, color=INK, va="center", alpha=0.85)

    fig.tight_layout()
    path = OUT / "figure2.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
