"""Detector battery. All pairwise; run-level score is the max over pairs.

D1 Cramer's V          marginal association
D2 Marginal MI         I(a_i ; a_j)
D3 Conditional MI      I(a_i ; a_j | s_hat)
D4 Net influence       Bonjour et al. 2022, Eq. 3-5, conditioned on s_hat

D1/D2 never condition on context. D3/D4 do. The design predicts only the latter
recover power as context observability rises.
"""

from __future__ import annotations

import itertools

import numpy as np

from processes import UNKNOWN


# --------------------------------------------------------------------------
# Information-theoretic primitives (plug-in estimators, natural log => nats)
# --------------------------------------------------------------------------

def _joint_counts(x: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    C = np.zeros((k, k))
    np.add.at(C, (x, y), 1.0)
    return C


def mutual_information(x: np.ndarray, y: np.ndarray, k: int) -> float:
    if x.size == 0:
        return 0.0
    C = _joint_counts(x, y, k)
    N = C.sum()
    if N == 0:
        return 0.0
    P = C / N
    px = P.sum(1, keepdims=True)
    py = P.sum(0, keepdims=True)
    denom = px @ py
    nz = (P > 0) & (denom > 0)
    return float((P[nz] * np.log(P[nz] / denom[nz])).sum())


def conditional_mutual_information(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                                   k: int) -> float:
    """I(X;Y|Z), averaging per-stratum MI weighted by stratum mass.

    UNKNOWN strata are pooled into a single stratum: an auditor who cannot see the
    context cannot condition on it, which is exactly the regime under test.
    """
    total, N = 0.0, x.size
    if N == 0:
        return 0.0
    for v in np.unique(z):
        mask = z == v
        w = mask.sum() / N
        if mask.sum() < 2:
            continue
        total += w * mutual_information(x[mask], y[mask], k)
    return float(total)


def cramers_v(x: np.ndarray, y: np.ndarray, k: int) -> float:
    C = _joint_counts(x, y, k)
    N = C.sum()
    if N == 0:
        return 0.0
    px = C.sum(1, keepdims=True) / N
    py = C.sum(0, keepdims=True) / N
    E = N * (px @ py)
    nz = E > 0
    chi2 = float((((C[nz] - E[nz]) ** 2) / E[nz]).sum())
    r = min(np.count_nonzero(C.sum(1)), np.count_nonzero(C.sum(0)))
    if r < 2:
        return 0.0
    return float(np.sqrt(chi2 / (N * (r - 1))))


# --------------------------------------------------------------------------
# Detectors
# --------------------------------------------------------------------------

def d1_cramers_v(A: np.ndarray, s_hat: np.ndarray, k: int) -> dict:
    n = A.shape[1]
    return {(i, j): cramers_v(A[:, i], A[:, j], k)
            for i, j in itertools.combinations(range(n), 2)}


def d2_marginal_mi(A: np.ndarray, s_hat: np.ndarray, k: int) -> dict:
    n = A.shape[1]
    return {(i, j): mutual_information(A[:, i], A[:, j], k)
            for i, j in itertools.combinations(range(n), 2)}


def d3_conditional_mi(A: np.ndarray, s_hat: np.ndarray, k: int) -> dict:
    n = A.shape[1]
    return {(i, j): conditional_mutual_information(A[:, i], A[:, j], s_hat, k)
            for i, j in itertools.combinations(range(n), 2)}


def d4_net_influence(A: np.ndarray, s_hat: np.ndarray, k: int) -> dict:
    """Bonjour et al. 2022.

    gamma(i->j) = I(a_i[t] ; a_j[t+1] | s_hat[t])
    Gamma(i;j)  = gamma(i->j) - max_{l != i,j} gamma(l->j)

    Symmetrised over the pair, since a colluding pair should influence each other
    more than any outsider influences either of them.
    """
    n = A.shape[1]
    prev, nxt, ctx = A[:-1], A[1:], s_hat[:-1]

    gamma = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            gamma[i, j] = conditional_mutual_information(prev[:, i], nxt[:, j], ctx, k)

    out = {}
    for i, j in itertools.combinations(range(n), 2):
        others_j = [gamma[l, j] for l in range(n) if l not in (i, j)]
        others_i = [gamma[l, i] for l in range(n) if l not in (i, j)]
        gij = gamma[i, j] - (max(others_j) if others_j else 0.0)
        gji = gamma[j, i] - (max(others_i) if others_i else 0.0)
        out[(i, j)] = float(max(gij, gji))
    return out


DETECTORS = {
    "D1 Cramer's V":    d1_cramers_v,
    "D2 marginal MI":   d2_marginal_mi,
    "D3 conditional MI": d3_conditional_mi,
    "D4 net influence": d4_net_influence,
}

CONDITIONAL = {"D3 conditional MI", "D4 net influence"}


def run_level_score(pairwise: dict) -> float:
    """An auditor flags the most suspicious pair."""
    return max(pairwise.values()) if pairwise else 0.0


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score(positive) > score(negative)), ties counted as 1/2."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    comparisons = pos[:, None] - neg[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0))
                 / (pos.size * neg.size))


def auroc_ci(pos: np.ndarray, neg: np.ndarray, n_boot: int = 2000,
             seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    point = auroc(pos, neg)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = auroc(rng.choice(pos, pos.size, replace=True),
                         rng.choice(neg, neg.size, replace=True))
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
