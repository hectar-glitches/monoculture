"""Arms 5-7: robustness, mechanism verification, and specification sensitivity.

  5  Parameter sensitivity. Every headline number so far sits at one point in parameter
     space. Vary each parameter one at a time and check the SIGN holds: level-reading
     statistics below 0.5, pattern-reading statistics above it.

  6  Mechanism verification. Amendment 1 claimed the contrast cancels the uniform lift's
     MEAN but not its VARIANCE, which is why population size did not help. That was an
     explanation fitted to the data, not a measurement. Measure it, and test whether
     studentising by the spread restores threshold transfer.

  7  Specification sensitivity. Arm 4's per-agent suspicion was defined as the agent's
     strongest pairwise link. That was our choice, not Rose et al.'s. Check the inversion
     does not depend on it.

Runs at $0.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from processes import World, simulate, mask_context, matched_kappa_I
from detectors import mutual_information, conditional_mutual_information, auroc_ci
from aggregation import pair_G

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

BASE = dict(n_agents=4, k=4, m=8, persistence=0.7, T=400)
KAPPA_C, LAM, CARTEL = 0.40, 0.35, (0, 1)
N_SENS = 100          # runs per class per sensitivity cell
PHI = 0.0


# --------------------------------------------------------------------------
# Two representative statistics: one of each kind
# --------------------------------------------------------------------------

def level_stat(A: np.ndarray, k: int) -> float:
    """Max over pairs of marginal dependence. The form that inverts."""
    return float(max(pair_G(A, k).values()))


def pattern_stat(A: np.ndarray, k: int) -> float:
    """Max minus median over pairs: the homogeneous-null GLRT. The form that works."""
    v = np.array(list(pair_G(A, k).values()))
    return float(v.max() - np.median(v))


# --------------------------------------------------------------------------
# Arm 5: parameter sensitivity
# --------------------------------------------------------------------------

def cell_auroc(world_kw: dict, kappa_C: float, lam: float, n_runs: int,
               seed0: int) -> tuple[tuple, tuple, float]:
    w = World(**world_kw)
    kappa_I = matched_kappa_I(kappa_C, lam, w.k)
    pos_l, pos_p = np.empty(n_runs), np.empty(n_runs)
    neg_l, neg_p = np.empty(n_runs), np.empty(n_runs)
    for r in range(n_runs):
        A, _ = simulate(w, kappa_C, lam=lam, cartel=CARTEL, seed=seed0 + r)
        pos_l[r], pos_p[r] = level_stat(A, w.k), pattern_stat(A, w.k)
        B, _ = simulate(w, kappa_I, seed=seed0 + 50_000 + r)
        neg_l[r], neg_p[r] = level_stat(B, w.k), pattern_stat(B, w.k)
    return (auroc_ci(pos_l, neg_l, n_boot=600, seed=5),
            auroc_ci(pos_p, neg_p, n_boot=600, seed=5), kappa_I)


def arm5() -> dict:
    print("\n" + "=" * 94)
    print("ARM 5  --  parameter sensitivity: does the SIGN hold off the baseline point?")
    print("=" * 94)

    # m must stay divisible by k so the focal map remains balanced, which the
    # analytic matching assumes.
    sweeps = [
        ("k (actions)",      [dict(k=kk, m=2 * kk) for kk in (3, 4, 6, 8)],
         lambda d: d["k"]),
        ("m (contexts)",     [dict(m=mm) for mm in (4, 8, 16)], lambda d: d["m"]),
        ("persistence",      [dict(persistence=pp) for pp in (0.0, 0.3, 0.7, 0.9)],
         lambda d: d["persistence"]),
        ("T (rounds)",       [dict(T=tt) for tt in (100, 200, 400, 800)],
         lambda d: d["T"]),
        ("n (agents)",       [dict(n_agents=nn) for nn in (3, 4, 6, 8)],
         lambda d: d["n_agents"]),
    ]

    rows, seed = {}, 1_000_000
    for label, variants, keyf in sweeps:
        print(f"\n  {label}")
        print(f"    {'value':>10} | {'kappa_I':>8} | {'level stat':>22} | "
              f"{'pattern stat':>22}")
        rows[label] = {}
        for v in variants:
            kw = {**BASE, **v}
            (la, llo, lhi), (pa, plo, phi_), kI = cell_auroc(
                kw, KAPPA_C, LAM, N_SENS, seed)
            seed += 200_000
            rows[label][str(keyf(v))] = {
                "kappa_I": kI,
                "level": {"auroc": la, "lo": llo, "hi": lhi},
                "pattern": {"auroc": pa, "lo": plo, "hi": phi_},
            }
            print(f"    {keyf(v):>10} | {kI:>8.4f} | "
                  f"{la:.3f} [{llo:.2f},{lhi:.2f}]".rjust(22) +
                  f" | {pa:.3f} [{plo:.2f},{phi_:.2f}]".rjust(25))

    # Cartel-strength sweeps need their own loop: they change kappa_C / lambda.
    for label, pairs in (("lambda (cartel strength)",
                          [(KAPPA_C, l) for l in (0.15, 0.25, 0.35, 0.50)]),
                         ("kappa_C (cartel baseline)",
                          [(c, LAM) for c in (0.30, 0.40, 0.50)])):
        print(f"\n  {label}")
        print(f"    {'value':>10} | {'kappa_I':>8} | {'level stat':>22} | "
              f"{'pattern stat':>22}")
        rows[label] = {}
        for kc, lm in pairs:
            (la, llo, lhi), (pa, plo, phi_), kI = cell_auroc(
                BASE, kc, lm, N_SENS, seed)
            seed += 200_000
            key = lm if "lambda" in label else kc
            rows[label][str(key)] = {
                "kappa_I": kI,
                "level": {"auroc": la, "lo": llo, "hi": lhi},
                "pattern": {"auroc": pa, "lo": plo, "hi": phi_},
            }
            print(f"    {key:>10} | {kI:>8.4f} | "
                  f"{la:.3f} [{llo:.2f},{lhi:.2f}]".rjust(22) +
                  f" | {pa:.3f} [{plo:.2f},{phi_:.2f}]".rjust(25))

    # Verdict: sign consistency across every cell.
    cells = [c for s in rows.values() for c in s.values()]
    lvl_inv = sum(c["level"]["hi"] < 0.5 for c in cells)
    pat_ok = sum(c["pattern"]["lo"] > 0.5 for c in cells)
    print(f"\n  level statistic inverted (CI entirely below 0.5) in "
          f"{lvl_inv}/{len(cells)} cells")
    print(f"  pattern statistic worked (CI entirely above 0.5) in "
          f"{pat_ok}/{len(cells)} cells")
    rows["_summary"] = {"n_cells": len(cells), "level_inverted": lvl_inv,
                        "pattern_worked": pat_ok}
    return rows


# --------------------------------------------------------------------------
# Arm 6: is the mechanism really mean-cancellation without variance-cancellation?
# --------------------------------------------------------------------------

def arm6(n_runs: int = 150) -> dict:
    print("\n" + "=" * 94)
    print("ARM 6  --  mechanism check: does the contrast cancel the mean but not the "
          "variance?")
    print("=" * 94)
    print("  pure monoculture, no agreement in any run, shared context unobserved\n")
    print(f"    {'kappa':>6} | {'mean g':>9} | {'sd g':>9} | {'raw contrast':>13} | "
          f"{'studentised':>12}")

    w = World(**BASE)
    out = {}
    for kap in (0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85):
        means, sds, raw, stud = [], [], [], []
        for r in range(n_runs):
            A, s = simulate(w, kap, seed=3_000_000 + int(kap * 1000) * 500 + r)
            s_hat = mask_context(s, PHI, np.random.default_rng(r))
            prev, nxt, ctx = A[:-1], A[1:], s_hat[:-1]
            g = np.array([
                conditional_mutual_information(prev[:, i], nxt[:, j], ctx, w.k)
                for i, j in itertools.permutations(range(w.n_agents), 2)])
            mu, sd = float(g.mean()), float(g.std())
            means.append(mu)
            sds.append(sd)
            raw.append(float(g.max() - np.median(g)))
            stud.append(float((g.max() - np.median(g)) / sd) if sd > 0 else 0.0)
        out[kap] = {"mean_gamma": float(np.mean(means)), "sd_gamma": float(np.mean(sds)),
                    "raw_contrast": float(np.mean(raw)),
                    "studentised_contrast": float(np.mean(stud))}
        print(f"    {kap:>6.2f} | {np.mean(means):>9.5f} | {np.mean(sds):>9.5f} | "
              f"{np.mean(raw):>13.5f} | {np.mean(stud):>12.4f}")

    ks = sorted(out)
    rng_raw = out[ks[-1]]["raw_contrast"] / max(out[ks[0]]["raw_contrast"], 1e-12)
    rng_std = out[ks[-1]]["studentised_contrast"] / max(
        out[ks[0]]["studentised_contrast"], 1e-12)
    sd_growth = out[ks[-1]]["sd_gamma"] / max(out[ks[0]]["sd_gamma"], 1e-12)
    print(f"\n  sd of gamma grows {sd_growth:.1f}x from kappa=0.25 to 0.85")
    print(f"  raw contrast grows {rng_raw:.1f}x over the same range "
          f"-> threshold does not transfer")
    print(f"  studentised contrast grows {rng_std:.1f}x "
          f"-> {'transfers' if rng_std < 2.0 else 'still does not transfer'}")
    out["_summary"] = {"sd_growth": sd_growth, "raw_growth": rng_raw,
                       "studentised_growth": rng_std}
    return out


# --------------------------------------------------------------------------
# Arm 7: does arm 4 depend on how per-agent suspicion was defined?
# --------------------------------------------------------------------------

def arm7(n_runs: int = 150) -> dict:
    print("\n" + "=" * 94)
    print("ARM 7  --  specification sensitivity of the per-agent score")
    print("=" * 94)

    w = World(**{**BASE, "n_agents": 6})
    kappa_I = matched_kappa_I(KAPPA_C, LAM, w.k)
    aggs = {"max": np.max, "mean": np.mean, "sum": np.sum}

    def scores(kappa, lam, cartel, seed0):
        res = {f"{nm}-{form}": np.empty(n_runs)
               for nm in aggs for form in ("peak", "split")}
        for r in range(n_runs):
            A, _ = simulate(w, kappa, lam=lam, cartel=cartel, seed=seed0 + r)
            G = pair_G(A, w.k)
            for nm, fn in aggs.items():
                per = np.array([fn([v for (a, b), v in G.items() if a == i or b == i])
                                for i in range(w.n_agents)])
                srt = np.sort(per)[::-1]
                res[f"{nm}-peak"][r] = srt[0]
                res[f"{nm}-split"][r] = srt[:2].mean() - srt[2:].mean()
        return res

    pos = scores(KAPPA_C, LAM, CARTEL, 4_000_000)
    neg = scores(kappa_I, 0.0, None, 4_500_000)

    print(f"\n    {'per-agent score':>16} | {'Peak (level)':>22} | {'Split (pattern)':>22}")
    out = {}
    for nm in aggs:
        pa, plo, phi_ = auroc_ci(pos[f"{nm}-peak"], neg[f"{nm}-peak"], n_boot=800, seed=7)
        sa, slo, shi = auroc_ci(pos[f"{nm}-split"], neg[f"{nm}-split"], n_boot=800, seed=7)
        out[nm] = {"peak": {"auroc": pa, "lo": plo, "hi": phi_},
                   "split": {"auroc": sa, "lo": slo, "hi": shi}}
        print(f"    {nm:>16} | {pa:.3f} [{plo:.2f},{phi_:.2f}]".rjust(42) +
              f" | {sa:.3f} [{slo:.2f},{shi:.2f}]".rjust(25))

    inv = all(out[nm]["peak"]["hi"] < 0.5 for nm in aggs)
    ok = all(out[nm]["split"]["lo"] > 0.5 for nm in aggs)
    print(f"\n  Peak inverted under all three definitions: {inv}")
    print(f"  Split worked under all three definitions:   {ok}")
    out["_summary"] = {"peak_inverted_all": bool(inv), "split_worked_all": bool(ok)}
    return out


def main() -> int:
    report = {"baseline": BASE, "kappa_C": KAPPA_C, "lam": LAM,
              "arm5_sensitivity": arm5(), "arm6_mechanism": arm6(),
              "arm7_specification": arm7()}
    (OUT / "report_robustness.json").write_text(
        json.dumps(report, indent=2, default=str))
    print(f"\nwrote {OUT / 'report_robustness.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
