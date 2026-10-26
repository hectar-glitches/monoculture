"""Arm 2: lagged coordination, so net influence is tested inside its design envelope.

The contemporaneous arm found D4 (Bonjour net influence) failing its phi=1 positive
control at AUROC 0.651. The preregistration forbids reporting a null from a detector
that cannot pass its own control, so the question is whether D4 is broken or simply
mismatched: net influence is a LAGGED statistic, and the arm-1 cartel coordinated
within a round.

Here the cartel coordinates across time instead. If D4 now passes at phi=1 and still
collapses at phi=0, the identifiability result holds across coordination mechanisms
and D4 was mismatched, not broken.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from processes import (World, simulate_lagged, mask_context, match_lagged_kappa,
                       lagged_pair_dist, tv_distance)
from detectors import DETECTORS, run_level_score, auroc_ci

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

KAPPA_C = 0.40
LAM = 0.35
N_RUNS = 200
CARTEL = (0, 1)
PHIS = [0.0, 0.25, 0.5, 0.75, 1.0]
W = World(n_agents=4, k=4, m=8, persistence=0.7, T=400)


def score_runs(kappa: float, lam: float, cartel, phi: float, n_runs: int,
               seed0: int) -> dict[str, np.ndarray]:
    scores = {name: np.empty(n_runs) for name in DETECTORS}
    for r in range(n_runs):
        A, s = simulate_lagged(W, kappa, lam=lam, cartel=cartel, seed=seed0 + r)
        rng = np.random.default_rng(300000 + seed0 + r)
        s_hat = mask_context(s, phi, rng)
        for name, fn in DETECTORS.items():
            scores[name][r] = run_level_score(fn(A, s_hat, W.k))
    return scores


def main() -> int:
    print("=" * 84)
    print("MONOCULTURE arm 2  --  LAGGED coordination (net influence's design envelope)")
    print("=" * 84)

    print("\n[0] solving the lagged matching numerically")
    kappa_I = match_lagged_kappa(W, KAPPA_C, LAM, CARTEL)
    print(f"    matched monoculture concentration kappa_I = {kappa_I:.6f}")

    big = World(**{**W.__dict__, "T": 400_000})
    A_C, _ = simulate_lagged(big, KAPPA_C, lam=LAM, cartel=CARTEL, seed=21)
    A_I, _ = simulate_lagged(big, kappa_I, seed=22)
    tv = tv_distance(lagged_pair_dist(A_I, *CARTEL, W.k),
                     lagged_pair_dist(A_C, *CARTEL, W.k))
    print(f"    empirical TV of lagged pair distributions = {tv:.4f}  (threshold 0.01)")
    if tv >= 0.01:
        print("    ** FAIL: matching not achieved, nothing downstream is reportable **")
        return 1
    print("    PASS")

    print("\n[1] AUROC by context observability")
    print(f"    {'phi':>5} | " + " | ".join(f"{n:^22}" for n in DETECTORS))
    sweep = {}
    for phi in PHIS:
        pos = score_runs(KAPPA_C, LAM, CARTEL, phi, N_RUNS, seed0=4000)
        neg = score_runs(kappa_I, 0.0, None, phi, N_RUNS, seed0=70000)
        row, cells = {}, []
        for name in DETECTORS:
            a, lo, hi = auroc_ci(pos[name], neg[name], seed=3)
            row[name] = {"auroc": a, "lo": lo, "hi": hi}
            cells.append(f"{a:.3f} [{lo:.2f},{hi:.2f}]".center(22))
        sweep[phi] = row
        print(f"    {phi:>5.2f} | " + " | ".join(cells))

    d4_hi = sweep[1.0]["D4 net influence"]
    d4_lo = sweep[0.0]["D4 net influence"]
    passed = d4_hi["auroc"] > 0.75
    print(f"\n    D4 positive control at phi=1: AUROC {d4_hi['auroc']:.3f} "
          f"[{d4_hi['lo']:.3f},{d4_hi['hi']:.3f}] -> "
          f"{'PASS -- D4 was mismatched in arm 1, not broken' if passed else '** STILL FAILS **'}")
    print(f"    D4 at phi=0: AUROC {d4_lo['auroc']:.3f} "
          f"[{d4_lo['lo']:.3f},{d4_lo['hi']:.3f}]")

    (OUT / "report_lagged.json").write_text(json.dumps(
        {"kappa_I": kappa_I, "tv": tv, "phi_sweep": sweep,
         "d4_control_pass": bool(passed)}, indent=2, default=str))
    print(f"\nwrote {OUT / 'report_lagged.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
