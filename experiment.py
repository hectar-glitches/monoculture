"""The experiment, in the order the preregistration demands.

  0  verify the matched construction          (if this fails, nothing else is reportable)
  1  phi = 1 positive control                 (if this fails, the detectors are broken)
  2  phi = 0 headline                         (the identifiability collapse)
  3  phi sweep                                (how much context logging buys how much power)
  4  kappa sweep, no agreement anywhere       (the monoculture false-accusation curve)

Runs at $0. No API key, no network, no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from processes import (World, simulate, mask_context, matched_kappa_I, verify_matching,
                       empirical_pair_dist, pair_distribution, tv_distance)
from detectors import DETECTORS, CONDITIONAL, run_level_score, auroc, auroc_ci

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

KAPPA_C = 0.40      # cartel members' baseline prior concentration
LAM = 0.35          # cartel coordination probability
N_RUNS = 200        # runs per class
CARTEL = (0, 1)
PHIS = [0.0, 0.25, 0.5, 0.75, 1.0]

W = World(n_agents=4, k=4, m=8, persistence=0.7, T=400)
KAPPA_I = matched_kappa_I(KAPPA_C, LAM, W.k)


def score_runs(kappa: float, lam: float, cartel, phi: float, n_runs: int,
               seed0: int) -> dict[str, np.ndarray]:
    scores = {name: np.empty(n_runs) for name in DETECTORS}
    for r in range(n_runs):
        A, s = simulate(W, kappa, lam=lam, cartel=cartel, seed=seed0 + r)
        rng = np.random.default_rng(100000 + seed0 + r)
        s_hat = mask_context(s, phi, rng)
        for name, fn in DETECTORS.items():
            scores[name][r] = run_level_score(fn(A, s_hat, W.k))
    return scores


def main() -> int:
    report: dict = {"params": {"kappa_C": KAPPA_C, "lam": LAM, "kappa_I": KAPPA_I,
                               "n_runs": N_RUNS, "world": W.__dict__}}
    print("=" * 84)
    print("MONOCULTURE  --  agreement vs shared priors")
    print("=" * 84)
    print(f"cartel pair runs at kappa_C={KAPPA_C}, lambda={LAM}")
    print(f"matched monoculture concentration kappa_I={KAPPA_I:.6f}")

    # --- 0. the construction ------------------------------------------------
    print("\n[0] verifying the matched construction")
    v = verify_matching(KAPPA_C, LAM, W.k)
    print(f"    analytic TV(P_I, P_C) = {v['tv_analytic']:.2e}  "
          f"(diag {v['diag_I']:.6f} vs {v['diag_C']:.6f})")
    ok_analytic = v["tv_analytic"] < 1e-9

    A_I, _ = simulate(W, KAPPA_I, seed=7)
    A_C, _ = simulate(W, KAPPA_C, lam=LAM, cartel=CARTEL, seed=7)
    big = World(**{**W.__dict__, "T": 200_000})
    B_I, _ = simulate(big, KAPPA_I, seed=11)
    B_C, _ = simulate(big, KAPPA_C, lam=LAM, cartel=CARTEL, seed=11)
    tv_emp = tv_distance(empirical_pair_dist(B_I, 0, 1, W.k),
                         empirical_pair_dist(B_C, 0, 1, W.k))
    print(f"    empirical TV at T=200,000  = {tv_emp:.4f}  (threshold 0.01)")
    ok_emp = tv_emp < 0.01
    report["construction"] = {**v, "tv_empirical": tv_emp,
                              "pass": bool(ok_analytic and ok_emp)}
    print(f"    {'PASS' if ok_analytic and ok_emp else '** FAIL -- stop here **'}")
    if not (ok_analytic and ok_emp):
        (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str))
        return 1

    # --- 1-3. phi sweep, with phi=1 as the positive control -----------------
    print("\n[1-3] AUROC by context observability "
          "(positives = real cartel, negatives = honest monoculture)")
    print(f"    {'phi':>5} | " + " | ".join(f"{n:^22}" for n in DETECTORS))
    sweep = {}
    for phi in PHIS:
        pos = score_runs(KAPPA_C, LAM, CARTEL, phi, N_RUNS, seed0=1000)
        neg = score_runs(KAPPA_I, 0.0, None, phi, N_RUNS, seed0=50000)
        row = {}
        cells = []
        for name in DETECTORS:
            a, lo, hi = auroc_ci(pos[name], neg[name], seed=3)
            row[name] = {"auroc": a, "lo": lo, "hi": hi}
            cells.append(f"{a:.3f} [{lo:.2f},{hi:.2f}]".center(22))
        sweep[phi] = row
        print(f"    {phi:>5.2f} | " + " | ".join(cells))
    report["phi_sweep"] = sweep

    pc = {n: sweep[1.0][n]["auroc"] for n in DETECTORS}
    pc_ok = all(pc[n] > 0.75 for n in CONDITIONAL)
    print(f"\n    positive control (phi=1, conditional detectors > 0.75): "
          f"{'PASS' if pc_ok else '** FAIL -- detectors broken, null not reportable **'}")
    report["positive_control_pass"] = bool(pc_ok)

    print("\n    headline (phi=0):")
    for name in DETECTORS:
        c = sweep[0.0][name]
        chance = c["lo"] <= 0.5 <= c["hi"]
        side = "below" if c["auroc"] < 0.5 else "above"
        print(f"      {name:<20} AUROC {c['auroc']:.3f} [{c['lo']:.3f},{c['hi']:.3f}]  "
              f"{'indistinguishable from chance' if chance else f'{side} chance'}")

    # --- 4. monoculture curve: no agreement anywhere ------------------------
    print("\n[4] false-accusation rate vs prior similarity (no agreement in ANY run)")
    kappas = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    calib = score_runs(0.25, 0.0, None, 0.0, N_RUNS, seed0=90000)
    thresholds = {n: float(np.percentile(calib[n], 95)) for n in DETECTORS}
    print(f"    thresholds frozen at the 95th percentile of kappa=0.25 "
          f"(so FPR=0.05 there by construction)")
    print(f"    {'kappa':>6} | " + " | ".join(f"{n:^18}" for n in DETECTORS))
    curve = {}
    for kap in kappas:
        sc = score_runs(kap, 0.0, None, 0.0, N_RUNS, seed0=200000 + int(kap * 1000))
        rates = {n: float(np.mean(sc[n] > thresholds[n])) for n in DETECTORS}
        curve[kap] = rates
        print(f"    {kap:>6.2f} | " +
              " | ".join(f"{rates[n]:^18.3f}" for n in DETECTORS))
    report["monoculture_curve"] = {"thresholds": thresholds, "rates": curve}

    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {OUT / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
