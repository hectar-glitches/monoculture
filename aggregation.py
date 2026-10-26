"""Arm 4: which GROUP-AGGREGATION forms invert under monoculture.

Rose et al. (arXiv:2604.01151, HELIOS/NARCBench) aggregate per-agent scores into a
scenario-level verdict using five named techniques -- Peak Suspicion, Deception Split,
Sorted Concatenation, Activation Alignment and Asymmetry Probe -- with a composite
hypothesis-testing framework (their Sec. 3.2) and a GLRT reading of Peak (their App. R).
They report AUROC 1.00 in-distribution and 0.73-0.93 under environment shift.

IMPORTANT SCOPE NOTE, stated here and in the paper. Their detector reads model
ACTIVATIONS, which this simulation does not produce, and their exact formulas sit in
sections of the paper we could not retrieve. We therefore do NOT reproduce HELIOS and do
NOT claim to break it. What we test is the AGGREGATION STRUCTURE those names describe,
computed over behavioural dependence scores. The mechanism under test is about how a
group verdict is formed from per-pair evidence, and that is invariant to whether the
evidence came from activations or actions.

The preregistered prediction follows directly from the level-vs-pattern mechanism:

  LEVEL-READING forms (max over agents, mean pairwise) should INVERT, because monoculture
  raises every pair equally and they have no reference point to subtract.

  PATTERN-READING forms (top-k minus rest, centred max) should RESIST, because the
  uniform lift cancels.

  A GLRT INVERTS OR RESISTS DEPENDING ONLY ON ITS NULL CLASS. Against an independence
  null it is a level reader. Against a null that contains the homogeneous population it
  is a pattern reader. Same test, same data, opposite behaviour.

That last point is the constructive result: it is a repair available even when the shared
context cannot be observed, which matters because for LLM agents a large part of the
shared context is the model prior and cannot be logged.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from processes import World, simulate, mask_context, matched_kappa_I
from detectors import mutual_information, auroc_ci

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

KAPPA_C, LAM, N_RUNS, CARTEL = 0.40, 0.35, 200, (0, 1)
W = World(n_agents=6, k=4, m=8, persistence=0.7, T=400)
KAPPA_I = matched_kappa_I(KAPPA_C, LAM, W.k)


def pair_G(A: np.ndarray, k: int) -> dict[tuple[int, int], float]:
    """G-statistic per pair: 2*N*MI is the likelihood-ratio statistic for independence."""
    N = A.shape[0]
    return {(i, j): 2.0 * N * mutual_information(A[:, i], A[:, j], k)
            for i, j in itertools.combinations(range(A.shape[1]), 2)}


def per_agent(G: dict, n: int) -> np.ndarray:
    """Each agent's suspicion = its strongest pairwise link."""
    return np.array([max(v for (a, b), v in G.items() if a == i or b == i)
                     for i in range(n)])


# -- the aggregation forms --------------------------------------------------

def a1_peak(G, n):
    """Peak: max over agents. Reads LEVEL."""
    return float(per_agent(G, n).max())


def a2_alignment(G, n):
    """Alignment: mean pairwise association. Reads LEVEL."""
    return float(np.mean(list(G.values())))


def a3_split(G, n):
    """Split: the suspicious subset against the remainder. Reads PATTERN."""
    s = np.sort(per_agent(G, n))[::-1]
    return float(s[:2].mean() - s[2:].mean()) if n > 2 else float(s[0])


def a4_glrt_independence(G, n):
    """GLRT with an INDEPENDENCE null: best coalition vs. all-pairs-independent.
    Structurally identical to Peak, so predicted to read LEVEL."""
    return float(max(G.values()))


def a5_glrt_homogeneous(G, n):
    """GLRT with a HOMOGENEOUS null: the null class contains a population in which every
    pair is dependent at a common, unknown level, estimated from the data. The median is
    used as the central estimate because it is robust to the one elevated pair under the
    alternative. Reads PATTERN."""
    vals = np.array(list(G.values()))
    return float(vals.max() - np.median(vals))


FORMS = {
    "A1 Peak (max over agents)":   (a1_peak, "level"),
    "A2 Alignment (mean pairwise)": (a2_alignment, "level"),
    "A3 Split (top-k vs rest)":    (a3_split, "pattern"),
    "A4 GLRT / independence null": (a4_glrt_independence, "level"),
    "A5 GLRT / homogeneous null":  (a5_glrt_homogeneous, "pattern"),
}


def score_runs(kappa, lam, cartel, phi, n_runs, seed0):
    out = {name: np.empty(n_runs) for name in FORMS}
    for r in range(n_runs):
        A, s = simulate(W, kappa, lam=lam, cartel=cartel, seed=seed0 + r)
        mask_context(s, phi, np.random.default_rng(777000 + seed0 + r))
        G = pair_G(A, W.k)
        for name, (fn, _) in FORMS.items():
            out[name][r] = fn(G, W.n_agents)
    return out


def main() -> int:
    print("=" * 92)
    print("MONOCULTURE arm 4  --  which group-aggregation forms invert")
    print("=" * 92)
    print(f"n={W.n_agents} agents, phi=0 (shared context unobserved: the LLM regime)")
    print(f"cartel kappa_C={KAPPA_C} lambda={LAM}  vs  matched monoculture "
          f"kappa_I={KAPPA_I:.6f}")
    print("\npositives = one real colluding pair;  negatives = honest, no agreement at all\n")

    pos = score_runs(KAPPA_C, LAM, CARTEL, 0.0, N_RUNS, seed0=610000)
    neg = score_runs(KAPPA_I, 0.0, None, 0.0, N_RUNS, seed0=620000)

    print(f"    {'aggregation form':<30} {'reads':<9} {'AUROC':<22} verdict")
    print("    " + "-" * 82)
    rows = {}
    for name, (_, reads) in FORMS.items():
        a, lo, hi = auroc_ci(pos[name], neg[name], seed=11)
        if hi < 0.5:
            verdict = "INVERTED"
        elif lo > 0.5:
            verdict = "works"
        else:
            verdict = "at chance"
        predicted = "INVERTED" if reads == "level" else "works"
        flag = "" if verdict == predicted else "   <-- against prediction"
        rows[name] = {"auroc": a, "lo": lo, "hi": hi, "reads": reads,
                      "verdict": verdict, "predicted": predicted}
        print(f"    {name:<30} {reads:<9} {a:.3f} [{lo:.3f},{hi:.3f}]   "
              f"{verdict}{flag}")

    hits = sum(r["verdict"] == r["predicted"] for r in rows.values())
    print(f"\n    mechanism predicted {hits}/{len(rows)} correctly")
    print("\n    The two GLRT rows differ only in their null class, on identical data.")
    print("    Specifying the null to contain the homogeneous population is a repair that")
    print("    does not require observing the shared context -- which matters because for")
    print("    LLM agents much of that context is the model prior and cannot be logged.")

    (OUT / "report_aggregation.json").write_text(json.dumps(
        {"kappa_C": KAPPA_C, "lam": LAM, "kappa_I": KAPPA_I, "n_agents": W.n_agents,
         "n_runs": N_RUNS, "phi": 0.0, "results": rows}, indent=2, default=str))
    print(f"\nwrote {OUT / 'report_aggregation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
