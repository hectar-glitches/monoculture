"""Figure 1: the mechanism, with no statistics in it.

Sort every pair in a group by how often its two agents agree, then plot that profile.

A pair-specific mechanism lifts ONE pair. A shared prior lifts ALL of them. The top of
the two profiles is the same height -- that is the pair you would investigate, and it
tells you nothing. Everything that distinguishes the two groups is in the pairs you
weren't looking at.

Labels follow the paper's terminology: the groups are named after the generator that made
them ("pair mechanism", "shared prior"), not after a verdict.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from processes import World, simulate, matched_kappa_I

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

KAPPA_C, LAM, CARTEL, N_RUNS = 0.40, 0.35, (0, 1), 300
W = World(n_agents=6, k=4, m=8, persistence=0.7, T=400)
KAPPA_I = matched_kappa_I(KAPPA_C, LAM, W.k)

INK, GRID = "#1b1b1f", "#d8d8de"
C_PAIR, C_PRIOR = "#1864ab", "#e8590c"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.edgecolor": GRID, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def profiles(kappa, lam, cartel, seed0):
    """Per-run sorted vector of pairwise agreement rates."""
    pairs = list(itertools.combinations(range(W.n_agents), 2))
    out = np.empty((N_RUNS, len(pairs)))
    for r in range(N_RUNS):
        A, _ = simulate(W, kappa, lam=lam, cartel=cartel, seed=seed0 + r)
        rates = [float((A[:, i] == A[:, j]).mean()) for i, j in pairs]
        out[r] = np.sort(rates)[::-1]
    return out


def main() -> int:
    P = profiles(KAPPA_C, LAM, CARTEL, 8_100_000)     # one pair linked
    S = profiles(KAPPA_I, 0.0, None, 8_200_000)       # nobody linked, shared prior

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(11.6, 4.2), gridspec_kw={"width_ratios": [1.65, 1]})

    # ---------- panel A: the profiles ----------
    x = np.arange(1, P.shape[1] + 1)
    for M, colour, label in ((P, C_PAIR, "pair mechanism  (one pair linked)"),
                             (S, C_PRIOR, "shared prior  (nobody linked)")):
        med = np.median(M, axis=0)
        lo, hi = np.percentile(M, 10, axis=0), np.percentile(M, 90, axis=0)
        axA.plot(x, med, "o-", color=colour, lw=2.0, ms=5, label=label)
        axA.fill_between(x, lo, hi, color=colour, alpha=0.15, lw=0)

    # The construction sets every linked pair to the same agreement rate in both
    # groups. Drawing it makes the selection effect visible.
    from processes import diag_mass
    matched = LAM + (1 - LAM) * W.k * diag_mass(KAPPA_C, W.k)
    axA.axhline(matched, color=INK, ls="--", lw=1.1, alpha=0.65)
    axA.text(len(x) - 0.2, matched + 0.006,
             f"a linked pair, by construction = {matched:.3f}",
             fontsize=7.5, color=INK, alpha=0.8, ha="right")

    axA.set_xticks(x)
    axA.set_xlabel("pairs of agents, ranked by how often they agree")
    axA.set_ylabel("agreement rate")
    axA.set_title("A. One mechanism lifts one pair. A shared model lifts all of them.",
                  fontsize=10.5, loc="left", pad=30)
    axA.legend(frameon=False, fontsize=8.5, loc="center right")

    axA.annotate("both built to sit on the dashed line —\n"
                 "but the shared-prior top pair sits above it,\n"
                 "because it is the best of 15 rather than the only one",
                 xy=(1.05, np.median(S, axis=0)[0]), xytext=(2.4, 0.655),
                 fontsize=7.8, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1.0, alpha=0.6))
    mid = len(x) // 2 + 2
    axA.annotate("everything that tells\nthem apart is here",
                 xy=(mid, (np.median(P, axis=0)[mid - 1] +
                           np.median(S, axis=0)[mid - 1]) / 2),
                 xytext=(mid - 1.6, 0.195), fontsize=8, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1.0, alpha=0.6))
    axA.set_ylim(0.20, 0.70)

    # ---------- panel B: what each kind of detector reads ----------
    stats = {
        "top pair alone\n(what most detectors score)":
            (P[:, 0], S[:, 0]),
        "top pair minus the rest\n(comparing pairs)":
            (P[:, 0] - np.median(P[:, 1:], axis=1),
             S[:, 0] - np.median(S[:, 1:], axis=1)),
    }
    xs = np.arange(len(stats))
    w = 0.34
    for off, colour, lab, idx in ((-w / 2, C_PAIR, "pair mechanism", 0),
                                  (w / 2, C_PRIOR, "shared prior", 1)):
        vals = [np.median(v[idx]) for v in stats.values()]
        errs = [[np.median(v[idx]) - np.percentile(v[idx], 10) for v in stats.values()],
                [np.percentile(v[idx], 90) - np.median(v[idx]) for v in stats.values()]]
        axB.bar(xs + off, vals, w, color=colour, alpha=0.9, label=lab)
        axB.errorbar(xs + off, vals, yerr=errs, fmt="none", ecolor=INK,
                     elinewidth=1.0, capsize=3, alpha=0.7)

    axB.set_xticks(xs)
    axB.set_xticklabels(list(stats), fontsize=8)
    axB.set_ylabel("value of the statistic")
    axB.set_title("B. Read one pair and they match.\n     Compare pairs and they don't.",
                  fontsize=10.5, loc="left", pad=8)
    axB.legend(frameon=False, fontsize=8.5)
    axB.axhline(0, color=INK, lw=0.8, alpha=0.5)

    fig.tight_layout()
    path = OUT / "figure1.png"
    fig.savefig(path, bbox_inches="tight")

    summary = {
        "n_runs": N_RUNS, "n_agents": W.n_agents, "kappa_C": KAPPA_C, "lam": LAM,
        "kappa_I": KAPPA_I,
        "top_pair": {"pair_mechanism": float(np.median(P[:, 0])),
                     "shared_prior": float(np.median(S[:, 0]))},
        "other_pairs_median": {
            "pair_mechanism": float(np.median(P[:, 1:])),
            "shared_prior": float(np.median(S[:, 1:]))},
        "contrast": {
            "pair_mechanism": float(np.median(P[:, 0] - np.median(P[:, 1:], axis=1))),
            "shared_prior": float(np.median(S[:, 0] - np.median(S[:, 1:], axis=1)))},
    }
    (OUT / "report_mechanism.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {path}")
    print(json.dumps(summary["top_pair"], indent=2))
    print(json.dumps(summary["other_pairs_median"], indent=2))
    print(json.dumps(summary["contrast"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
