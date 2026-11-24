"""Exact spectral analysis of the gyroscopic VMS with a loop delay.

Realised interconnection with sensor delay d_s and actuator delay d_a:

    q''(t) - n z'(t - d_a) + q(t) = 0,
    z''(t) + n q'(t - d_s) + z(t) = 0.

Characteristic function (only the loop delay d = d_a + d_s matters):

    Delta(s; d) = (s^2 + 1)^2 + n^2 s^2 exp(-s d)
                = F(s; h) * conj-branch,   F(s; h) = s^2 + 1 - i n s exp(-s h),  h = d/2.

Main facts (Theorem 'delay' in the paper):
  * imaginary-axis crossings happen only at s = +-i Om+ and s = +-i Om-;
  * crossings at Om+ are always destabilising, at Om- always stabilising;
  * the number of roots in Re s > 0 is
        N(d) = 2 * (1 + floor(d Om+/(2 pi)) - floor(d Om-/(2 pi)))  >= 2,
    hence the loop is exponentially unstable for every d > 0;
  * first-order: Re s_+ = n Om+^2 h / S + O(h^2),  Re s_- = -n Om-^2 h / S + O(h^2).
"""
from __future__ import annotations

import numpy as np

from .params import VMSParams


def char_full(s, n, d):
    return (s**2 + 1) ** 2 + n**2 * s**2 * np.exp(-s * d)


def dchar_full(s, n, d):
    e = np.exp(-s * d)
    return 4 * s * (s**2 + 1) + n**2 * (2 * s - d * s**2) * e


def first_order_rates(p: VMSParams, d: float):
    """(growth rate of + mode, decay rate of - mode), both positive numbers."""
    h = d / 2
    return p.n * p.Op**2 * h / p.S, p.n * p.Om**2 * h / p.S


def n_unstable_formula(p: VMSParams, d: float) -> int:
    if d <= 0:
        return 0
    return int(2 * (1 + np.floor(d * p.Op / (2 * np.pi)) - np.floor(d * p.Om / (2 * np.pi))))


def crossing_delays(p: VMSParams, d_max: float):
    """Loop delays at which roots cross the imaginary axis.
    Returns (destabilising list, stabilising list)."""
    # per branch: h = m pi / Om+ (destab) and h = m pi / Om- (stab), m >= 1; d = 2h
    m = np.arange(1, 1000)
    dp = 2 * m * np.pi / p.Op
    dm = 2 * m * np.pi / p.Om
    return dp[dp <= d_max], dm[dm <= d_max]


def spectrum_collocation(p: VMSParams, d: float, N: int = 40):
    """Rightmost characteristic roots via Chebyshev collocation of the
    infinitesimal generator (Breda, Maset, Vermiglio 2005), followed by
    Newton refinement on Delta(s; d).  Whole loop delay put on the sensor
    channel (spectrum depends only on d)."""
    n = p.n
    A0 = np.array([[0, 1, 0, 0], [-1, 0, 0, n], [0, 0, 0, 1], [0, 0, -1, 0]], float)
    A1 = np.zeros((4, 4))
    A1[3, 1] = -n
    if d == 0:
        return np.linalg.eigvals(A0 + A1)
    # Chebyshev nodes on [-d, 0]: theta_j = d (cos(j pi/N) - 1)/2, j = 0..N
    j = np.arange(N + 1)
    x = np.cos(np.pi * j / N)
    c = np.ones(N + 1)
    c[0] = c[-1] = 2
    c *= (-1) ** j
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1 / c) / (dX + np.eye(N + 1))
    D -= np.diag(D.sum(axis=1))
    D *= 2 / d  # map [-1,1] -> [-d,0]
    k = 4
    M = np.zeros(((N + 1) * k, (N + 1) * k))
    M[:k, :k] = A0
    M[:k, N * k:(N + 1) * k] = A1  # theta_N = -d
    M[k:, :] = np.kron(D[1:, :], np.eye(k))
    ev = np.linalg.eigvals(M)
    # keep reasonably converged part (|Im| moderate) and refine
    ev = ev[np.abs(ev) < 0.5 * N / d + 5]
    out = []
    for s in ev:
        for _ in range(30):
            f = char_full(s, n, d)
            df = dchar_full(s, n, d)
            step = f / df
            s = s - step
            if abs(step) < 1e-13:
                break
        if abs(char_full(s, n, d)) < 1e-8:
            out.append(s)
    out = np.array(out)
    # de-duplicate
    uniq = []
    for s in out[np.argsort(-out.real)]:
        if all(abs(s - u) > 1e-7 for u in uniq):
            uniq.append(s)
    return np.array(uniq)


def spectral_abscissa(p: VMSParams, d: float, N: int = 40) -> float:
    return float(spectrum_collocation(p, d, N).real.max())
