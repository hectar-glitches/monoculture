"""Arm 3: why the contrast works, and when it stops working.

Arm 2 showed net influence is invariant to the monoculture confound: it scores
Gamma(i;j) = gamma(i->j) - max_{l != i,j} gamma(l->j), and a shared prior lifts every
pair's dependence roughly equally, so the subtraction cancels it.

That cancellation is a maximum over n-2 alternative explanations. With n=3 there is
exactly ONE alternative, so the contrast is a single noisy draw and cancellation is
unreliable. This predicts that net influence's false-accusation rate under monoculture
should fall steeply with population size.

The prediction is checkable against a published number: the only public application of
net influence to LLM agents ran THREE agents and reported innocent same-model pairs
flagged in 7 of 10 games (Raj-Taware, collusion-net-influence, Oct 2026). If this arm
reproduces a high rate at n=3 and a low one at n>=5, that failure was a population-size
artifact rather than a property of the statistic.

No agreement exists in any run scored here. Every flag is a false accusation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from processes import World, simulate_lagged, mask_context
from detectors import DETECTORS, run_level_score

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

NS = [3, 4, 5, 6, 8, 10]
KAPPA_LOW = 0.25      # diverse population: used only to calibrate thresholds
KAPPA_HIGH = 0.79     # monoculture, matched level from arm 2
N_RUNS = 300
PHI = 0.0             # the LLM regime: shared model prior, unobservable


def flag_rates(n: int, kappa: float, n_runs: int, seed0: int,
               thresholds: dict[str, float] | None) -> dict:
    w = World(n_agents=n, k=4, m=8, persistence=0.7, T=400)
    scores = {name: np.empty(n_runs) for name in DETECTORS}
    for r in range(n_runs):
        A, s = simulate_lagged(w, kappa, seed=seed0 + r)      # no cartel, ever
        s_hat = mask_context(s, PHI, np.random.default_rng(900000 + seed0 + r))
        for name, fn in DETECTORS.items():
            scores[name][r] = run_level_score(fn(A, s_hat, w.k))
    if thresholds is None:
        return {n_: float(np.percentile(scores[n_], 95)) for n_ in DETECTORS}
    return {n_: float(np.mean(scores[n_] > thresholds[n_])) for n_ in DETECTORS}


def main() -> int:
    print("=" * 84)
    print("MONOCULTURE arm 3  --  false accusation vs population size")
    print("=" * 84)
    print(f"no agreement in any run; thresholds frozen at the 95th percentile of a")
    print(f"diverse population (kappa={KAPPA_LOW}), then applied to a monoculture "
          f"(kappa={KAPPA_HIGH})")
    print(f"\n    {'n':>3} | " + " | ".join(f"{d:^18}" for d in DETECTORS))

    out = {}
    for n in NS:
        th = flag_rates(n, KAPPA_LOW, N_RUNS, 10_000 + n * 1000, None)
        rates = flag_rates(n, KAPPA_HIGH, N_RUNS, 500_000 + n * 1000, th)
        out[n] = {"thresholds": th, "false_accusation_rate": rates}
        print(f"    {n:>3} | " + " | ".join(f"{rates[d]:^18.3f}" for d in DETECTORS))

    d4 = [out[n]["false_accusation_rate"]["D4 net influence"] for n in NS]
    print(f"\n    D4 false-accusation rate: n=3 -> {d4[0]:.3f}, n=10 -> {d4[-1]:.3f}")
    print(f"    published n=3 reference point (Raj-Taware, same-model pairs): 0.700")
    print(f"    monotone decreasing in n: {all(a >= b - 0.02 for a, b in zip(d4, d4[1:]))}")

    (OUT / "report_population.json").write_text(json.dumps(
        {"ns": NS, "kappa_low": KAPPA_LOW, "kappa_high": KAPPA_HIGH,
         "n_runs": N_RUNS, "results": out}, indent=2, default=str))
    print(f"\nwrote {OUT / 'report_population.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
