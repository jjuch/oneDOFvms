"""Exact sampled-data (zero-order-hold) analysis of the gyroscopic VMS.

Implementation model (Assumption 'sampled implementation' in the paper):
  * host velocity q' is sampled at t_k = k Ts (optionally delayed by k_s samples);
  * the virtual oscillator is integrated exactly over [t_k, t_{k+1}) with its
    coupling input -n q'_k held constant;
  * the actuator force n z'_k is held constant over [t_k, t_{k+1})
    (optionally delayed by k_a samples).

Over one period, with xi = (q, q', z, z') and G the coupling matrix,
    xi(t_{k+1}) = e^{A0 Ts} xi_k + Gamma G xi_k  (no extra delays)
which gives a linear map Phi(Ts).  Its effective generator satisfies
    (1/Ts) log Phi(Ts) = A_c - (Ts/2) G A_c + O(Ts^2),
i.e. the ZOH acts, to first order, as a delay Ts/2 on *each* channel
(loop delay Ts).  Hence |lambda| > 1 for small Ts > 0 (Corollary 'ZOH').
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .params import VMSParams


def matrices(p: VMSParams):
    n = p.n
    A0 = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]], float)
    G = np.zeros((4, 4))
    G[1, 3] = n
    G[3, 1] = -n
    return A0, G


def phi(p: VMSParams, Ts: float, k_s: int = 0, k_a: int = 0) -> np.ndarray:
    """Lifted one-step map including k_s sensor and k_a actuator sample delays.
    Augmented state: xi_k, buffered sensor samples, buffered actuator commands."""
    A0, G = matrices(p)
    E = expm(A0 * Ts)
    # Gamma = int_0^Ts e^{A0 s} ds
    M = np.zeros((8, 8))
    M[:4, :4] = A0 * Ts
    M[:4, 4:] = np.eye(4) * Ts
    Gam = expm(M)[:4, 4:]
    Gs = np.zeros((4, 4))
    Gs[3, 1] = -p.n  # sensor-driven input to virtual
    Ga = np.zeros((4, 4))
    Ga[1, 3] = p.n  # actuator input to host
    if k_s == 0 and k_a == 0:
        return E + Gam @ G
    # augmented state: [xi_k, xi_{k-1}, ..., xi_{k-K}], K = max(k_s, k_a)
    K = max(k_s, k_a)
    dim = 4 * (K + 1)
    P = np.zeros((dim, dim))
    P[:4, :4] = E
    P[:4, 4 * k_s:4 * k_s + 4] += Gam @ Gs
    P[:4, 4 * k_a:4 * k_a + 4] += Gam @ Ga
    for i in range(K):
        P[4 * (i + 1):4 * (i + 2), 4 * i:4 * (i + 1)] = np.eye(4)
    return P


def effective_rates(p: VMSParams, Ts: float, k_s: int = 0, k_a: int = 0):
    """(1/Ts) log|lambda| for the eigenvalues closest to the nominal modes,
    returned as (rate of + mode, rate of - mode)."""
    ev = np.linalg.eigvals(phi(p, Ts, k_s, k_a))
    s = np.log(ev.astype(complex)) / Ts
    sp = s[np.argmin(np.abs(s - 1j * p.Op))]
    sm = s[np.argmin(np.abs(s - 1j * p.Om))]
    return sp.real, sm.real


def first_order_rates(p: VMSParams, Ts: float, k_s: int = 0, k_a: int = 0):
    """Equivalent loop delay d_eq = Ts (two half-sample holds) + (k_s + k_a) Ts."""
    d_eq = Ts * (1 + k_s + k_a)
    h = d_eq / 2
    return p.n * p.Op**2 * h / p.S, -p.n * p.Om**2 * h / p.S
