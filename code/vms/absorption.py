"""Absorption set of the resonant orbit (Theorem 'absorption').

On the resonant torus an orbit is fixed by (A+, A-, psi).  Writing
rho = A-/A+ = exp(-e_A) and r = tau/sigma = Om+/Om-, the host origin
q = q' = 0 lies on the orbit iff 1 <= rho <= r and psi belongs to the finite set
Psi*(e_A) computed below.
"""
from __future__ import annotations

import numpy as np

from .params import VMSParams


def psi_star(eA: float, p: VMSParams) -> np.ndarray:
    """Absorbing relative phases (mod 2 pi) for log-amplitude ratio eA.
    Empty array if eA is outside [-ln(tau/sigma), 0]."""
    rho = np.exp(-eA)
    r = p.Op / p.Om
    if rho < 1 - 1e-12 or rho > r + 1e-12:
        return np.array([])
    c2 = np.clip((rho**2 - 1) / (r**2 - 1), 0.0, 1.0)
    cp = np.sqrt(c2)
    sp = np.sqrt(1 - c2)
    out = []
    for sc in (+1, -1):
        for ss in (+1, -1):
            thp = np.arctan2(ss * sp, sc * cp)
            thm = np.arctan2(-np.sin(thp) / rho, -r * np.cos(thp) / rho)
            out.append(np.angle(np.exp(1j * (p.sigma * thp - p.tau * thm))))
    out = np.array(out)
    # unique modulo 2 pi
    u = []
    for v in out:
        if all(abs(np.angle(np.exp(1j * (v - w)))) > 1e-9 for w in u):
            u.append(v)
    return np.sort(np.array(u))


def min_host_radius(eA: float, psi: np.ndarray, p: VMSParams, A_plus: float = 1.0,
                    M: int = 6000) -> np.ndarray:
    """min over the resonant orbit of sqrt(q^2 + q'^2), vectorised in psi."""
    Am = A_plus * np.exp(-eA)
    s = np.linspace(0, 2 * np.pi * p.sigma, M, endpoint=False)  # theta-
    psi = np.atleast_1d(psi)[:, None]
    thm = s[None, :]
    thp = (psi + p.tau * thm) / p.sigma
    q = A_plus * np.sin(thp) + Am * np.sin(thm)
    qd = A_plus * p.Op * np.cos(thp) + Am * p.Om * np.cos(thm)
    return np.sqrt(q**2 + qd**2).min(axis=1)
