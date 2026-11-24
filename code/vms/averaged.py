"""First-order averaged model of the closed loop (Section 'averaged dynamics').

State: (e_H, e_A, zeta_A, psi, xi).  With the scale-invariant laws

    u   = -K_E (H_r / Hhat^2) ehat_H ghat,
    s   = -(K_A / Hhat) (ehat_A + zeta_A) mhat,       zeta_A' = k_IA ehat_A,
    c   = -(k_P sin(psi - psi*) + xi) / beta_psi,     xi'     = k_I sin(psi - psi*),

and an equivalent loop delay d_eq (h = d_eq / 2), the averaged dynamics are

    e_H'   = -2 K_E nu(a) (e_H + delta_H) + omega_H(a) (1 + e_H)
    e_A'   = -(K_A/S)(e_A + zeta_A + delta_A) - (2 K_E / S^2) chi(a) (e_H + delta_H)/(1 + e_H) + alpha_Sigma
    zeta_A'= k_IA (e_A + delta_A)
    psi'   = -k_P sin(psi - psi*) - xi + beta_psi d_s
    xi'    = k_I sin(psi - psi*)

with a = energy share of the + mode = Om+ e^{2 e_A} / (Om+ e^{2 e_A} + Om-),
nu(a)    = (1/(2 S^2)) (Om+^2 a^2 + Om-^2 (1-a)^2 + 4 a (1-a)),
chi(a)   = (a (Om+^2 - 2) - (1-a)(Om-^2 - 2)) / 4,
omega_H  = (2 n h / S) (Om+^2 a - Om-^2 (1 - a)),
alpha_Sigma = n h (Om+^2 + Om-^2) / S,   beta_psi = 2 sqrt(tau sigma) / S.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .params import VMSParams


def share(eA, p: VMSParams):
    e = np.exp(2 * eA)
    return p.Op * e / (p.Op * e + p.Om)


def nu(a, p):
    return (p.Op**2 * a**2 + p.Om**2 * (1 - a) ** 2 + 4 * a * (1 - a)) / (2 * p.S**2)


def nu_bounds(p):
    a = np.linspace(0, 1, 20001)
    v = nu(a, p)
    return v.min(), v.max()


def chi(a, p):
    return (a * (p.Op**2 - 2) - (1 - a) * (p.Om**2 - 2)) / 4


def omega_H(a, p, h):
    return 2 * p.n * h / p.S * (p.Op**2 * a - p.Om**2 * (1 - a))


def alpha_sigma(p, h):
    return p.n * h * (p.Op**2 + p.Om**2) / p.S


def beta_psi(p):
    return 2 * np.sqrt(p.tau * p.sigma) / p.S


def rhs(t, y, p, KE, KA, KIA, kP, kI, h, dH=0.0, dA=0.0, ds=0.0, psi_ref=None):
    eH, eA, zA, psi, xi = y
    a = share(eA, p)
    psi_ref = p.psi_abs if psi_ref is None else psi_ref
    deH = -2 * KE * nu(a, p) * (eH + dH) + omega_H(a, p, h) * (1 + eH)
    deA = (-(KA / p.S) * (eA + zA + dA)
           - 2 * KE / p.S**2 * chi(a, p) * (eH + dH) / (1 + eH)
           + alpha_sigma(p, h))
    dzA = KIA * (eA + dA)
    e = np.sin(psi - psi_ref)
    dpsi = -kP * e - xi + beta_psi(p) * ds
    dxi = kI * e
    return [deH, deA, dzA, dpsi, dxi]


def simulate_averaged(p, y0, T, **kw):
    sol = solve_ivp(rhs, (0, T), y0, args=(p,), kwargs=None, dense_output=True,
                    max_step=1.0, rtol=1e-8, atol=1e-10,
                    **{}) if False else solve_ivp(
        lambda t, y: rhs(t, y, p, **kw), (0, T), y0, max_step=1.0, rtol=1e-8, atol=1e-10,
        dense_output=True)
    return sol


def steady_state(p, KE, KA, h, KIA=0.0):
    """Equilibrium of (e_H, e_A) without measurement errors (zeta_A = 0 if KIA=0)."""
    from scipy.optimize import fsolve

    def f(z):
        eH, eA = z
        r = rhs(0, [eH, eA, 0.0, 0.0, 0.0], p, KE, KA, 0.0, 0.0, 0.0, h)
        return r[:2]

    return fsolve(f, [0.0, 0.0])
