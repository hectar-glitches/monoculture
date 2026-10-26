"""Matched generative processes.

Process I: independent agents sharing a prior (monoculture). No agreement anywhere.
Process C: one pair under an explicit agreement.

The construction solves for the shared-prior concentration that makes the two
observationally identical once the shared context is marginalised out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UNKNOWN = -1


# --------------------------------------------------------------------------
# The matching
# --------------------------------------------------------------------------

def diag_mass(kappa: float, k: int) -> float:
    """Per-cell diagonal mass of the context-marginalised pair distribution under
    independent draws from the shared policy."""
    q = (1.0 - kappa) / (k - 1)
    return (kappa ** 2 + (k - 1) * q ** 2) / k


def solve_kappa(target_D: float, k: int) -> float:
    """Invert diag_mass on [1/k, 1]. D is strictly increasing there, so bisect."""
    lo, hi = 1.0 / k, 1.0 - 1e-12
    if target_D <= diag_mass(lo, k):
        return lo
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if diag_mass(mid, k) < target_D:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def matched_kappa_I(kappa_C: float, lam: float, k: int) -> float:
    """The monoculture concentration whose pair distribution equals that of a
    cartel pair running (kappa_C, lam)."""
    return solve_kappa(lam / k + (1 - lam) * diag_mass(kappa_C, k), k)


def pair_distribution(kappa: float, k: int, lam: float = 0.0) -> np.ndarray:
    """Analytic k x k joint distribution of one pair, context marginalised."""
    q = (1.0 - kappa) / (k - 1)
    P = np.full((k, k), 0.0)
    # Group contexts by focal action; each focal action is equally likely.
    for a in range(k):
        p = np.full(k, q)
        p[a] = kappa
        P += np.outer(p, p) / k
    P *= (1 - lam)
    P += lam * np.eye(k) / k
    return P


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

@dataclass
class World:
    n_agents: int = 4
    k: int = 4            # actions
    m: int = 8            # contexts
    persistence: float = 0.7
    T: int = 400

    def focal(self) -> np.ndarray:
        """Balanced focal map: each action is focal for m/k contexts."""
        return np.array([s % self.k for s in range(self.m)])


def _context_chain(w: World, rng: np.random.Generator) -> np.ndarray:
    """Vectorised persistent chain: resample on reset, forward-fill otherwise."""
    reset = rng.random(w.T) >= w.persistence
    reset[0] = True
    fresh = rng.integers(w.m, size=w.T)
    last_reset = np.maximum.accumulate(np.where(reset, np.arange(w.T), 0))
    return fresh[last_reset]


def _policy_draw_all(focal_seq: np.ndarray, kappa: float, k: int, n: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Draw [T, n] actions from pi(.|s) for every round at once.

    pi puts mass kappa on the focal action and spreads 1-kappa uniformly over the
    other k-1, so sample the branch first and then an index among the non-focal
    actions, shifting past the focal index.
    """
    T = focal_seq.shape[0]
    take_focal = rng.random((T, n)) < kappa
    other = rng.integers(k - 1, size=(T, n))
    f = focal_seq[:, None]
    other = other + (other >= f)          # skip the focal index
    return np.where(take_focal, np.broadcast_to(f, (T, n)), other)


def simulate(w: World, kappa: float, lam: float = 0.0,
             cartel: tuple[int, int] | None = None,
             seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Returns (actions [T, n_agents], contexts [T]).

    lam > 0 with a cartel pair gives Process C; lam = 0 gives Process I.
    """
    rng = np.random.default_rng(seed)
    s = _context_chain(w, rng)
    A = _policy_draw_all(w.focal()[s], kappa, w.k, w.n_agents, rng)

    if cartel is not None and lam > 0:
        i, j = cartel
        coordinate = rng.random(w.T) < lam
        joint = rng.integers(w.k, size=w.T)
        A[coordinate, i] = joint[coordinate]
        A[coordinate, j] = joint[coordinate]

    return A, s


def simulate_lagged(w: World, kappa: float, lam: float = 0.0,
                    cartel: tuple[int, int] | None = None,
                    seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Process C_lag: the cartel coordinates ACROSS TIME rather than within a round.

    With probability lam, agent j copies agent i's previous action. Net influence
    (Bonjour et al.) is a lagged statistic, so a contemporaneous cartel is outside its
    design envelope; this variant tests it on the coordination type it was built for.
    """
    rng = np.random.default_rng(seed)
    s = _context_chain(w, rng)
    A = _policy_draw_all(w.focal()[s], kappa, w.k, w.n_agents, rng)

    if cartel is not None and lam > 0:
        i, j = cartel
        follow = rng.random(w.T) < lam
        # Agent i's column is never rewritten, so this is order-independent and
        # equivalent to the sequential update.
        A[1:, j] = np.where(follow[1:], A[:-1, i], A[1:, j])
    return A, s


def lagged_pair_dist(A: np.ndarray, i: int, j: int, k: int) -> np.ndarray:
    """Empirical joint of (a_i[t], a_j[t+1])."""
    P = np.zeros((k, k))
    np.add.at(P, (A[:-1, i], A[1:, j]), 1.0)
    return P / P.sum()


def match_lagged_kappa(w: World, kappa_C: float, lam: float, cartel: tuple[int, int],
                       T_match: int = 400_000, seed: int = 5) -> float:
    """Numerically solve for the monoculture concentration whose LAGGED pair
    distribution matches the lagged cartel's.

    No closed form here, so bisect on the lagged diagonal mass, which is monotone in
    kappa (more concentration on the focal action plus a persistent context means more
    lagged agreement). The result is verified by TV distance like the analytic case.
    """
    big = World(**{**w.__dict__, "T": T_match})
    A_C, _ = simulate_lagged(big, kappa_C, lam=lam, cartel=cartel, seed=seed)
    target = float(np.trace(lagged_pair_dist(A_C, cartel[0], cartel[1], w.k)))

    def diag_at(kap: float) -> float:
        A_I, _ = simulate_lagged(big, kap, seed=seed + 1)
        return float(np.trace(lagged_pair_dist(A_I, cartel[0], cartel[1], w.k)))

    lo, hi = 1.0 / w.k, 0.995
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        if diag_at(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mask_context(s: np.ndarray, phi: float, rng: np.random.Generator) -> np.ndarray:
    """Auditor sees the context with probability phi, UNKNOWN otherwise."""
    if phi >= 1.0:
        return s.copy()
    if phi <= 0.0:
        return np.full_like(s, UNKNOWN)
    out = s.copy()
    out[rng.random(s.shape[0]) >= phi] = UNKNOWN
    return out


# --------------------------------------------------------------------------
# Verification of the construction itself
# --------------------------------------------------------------------------

def tv_distance(P: np.ndarray, Q: np.ndarray) -> float:
    return 0.5 * float(np.abs(P - Q).sum())


def empirical_pair_dist(A: np.ndarray, i: int, j: int, k: int) -> np.ndarray:
    P = np.zeros((k, k))
    np.add.at(P, (A[:, i], A[:, j]), 1.0)
    return P / P.sum()


def verify_matching(kappa_C: float, lam: float, k: int = 4) -> dict:
    """Analytic check: does the matched monoculture pair equal the cartel pair?"""
    kI = matched_kappa_I(kappa_C, lam, k)
    P_I = pair_distribution(kI, k, lam=0.0)
    P_C = pair_distribution(kappa_C, k, lam=lam)
    return {"kappa_C": kappa_C, "lam": lam, "kappa_I": kI,
            "tv_analytic": tv_distance(P_I, P_C),
            "diag_I": float(P_I[0, 0]), "diag_C": float(P_C[0, 0])}
