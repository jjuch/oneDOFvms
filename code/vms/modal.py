"""Exact modal-polar coordinates for the gyroscopic VMS.

The linear invertible map T : (q, q', z, z') -> (X+, Y+, X-, Y-),

    X+ = (q' + Om- z)/S,   Y+ = (Om- q - z')/S,
    X- = (q' - Om+ z)/S,   Y- = (Om+ q + z')/S,

defines complex modal coordinates zeta_pm = X_pm + i Y_pm = A_pm exp(i theta_pm).
Along the nominal flow zeta_pm' = i Om_pm zeta_pm.  The map is a coordinate
change on all of R^4 (not only on nominal trajectories).
"""
from __future__ import annotations

import numpy as np

from .params import VMSParams


def T_matrix(p: VMSParams) -> np.ndarray:
    S, Op, Om = p.S, p.Op, p.Om
    # state ordering x = (q, qd, z, zd)
    return np.array(
        [
            [0.0, 1.0, Om, 0.0],  # X+
            [Om, 0.0, 0.0, -1.0],  # Y+
            [0.0, 1.0, -Op, 0.0],  # X-
            [Op, 0.0, 0.0, 1.0],  # Y-
        ]
    ) / S


def T_inv(p: VMSParams) -> np.ndarray:
    Op, Om = p.Op, p.Om
    # q = Y+ + Y-, qd = Op X+ + Om X-, z = X+ - X-, zd = -Op Y+ + Om Y-
    return np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [Op, 0.0, Om, 0.0],
            [1.0, 0.0, -1.0, 0.0],
            [0.0, -Op, 0.0, Om],
        ]
    )


def zeta(x: np.ndarray, p: VMSParams) -> tuple[np.ndarray, np.ndarray]:
    """Complex modal coordinates for state(s) x with shape (..., 4)."""
    m = x @ T_matrix(p).T
    return m[..., 0] + 1j * m[..., 1], m[..., 2] + 1j * m[..., 3]


def polar(x: np.ndarray, p: VMSParams):
    """Return (A+, A-, theta+, theta-)."""
    zp, zm = zeta(x, p)
    return np.abs(zp), np.abs(zm), np.angle(zp), np.angle(zm)


def energy(x: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum(np.asarray(x) ** 2, axis=-1)


def host_energy(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return 0.5 * (x[..., 0] ** 2 + x[..., 1] ** 2)


def energy_from_amplitudes(Ap, Am, p: VMSParams):
    """H = (S/2)(Om+ A+^2 + Om- A-^2)."""
    return 0.5 * p.S * (p.Op * Ap**2 + p.Om * Am**2)


def errors(x: np.ndarray, p: VMSParams, Hr: float):
    """Return (e_H, e_A, psi) where psi = sigma theta+ - tau theta- (wrapped)."""
    Ap, Am, thp, thm = polar(x, p)
    eH = energy(x) / Hr - 1.0
    eA = np.log(Ap / Am)
    psi = np.angle(np.exp(1j * (p.sigma * thp - p.tau * thm)))
    return eH, eA, psi


def nominal_state(t, p: VMSParams, Ap=1.0, Am=1.0, php=0.0, phm=0.0):
    """Nominal trajectory (ECC eqs. (6)-(9))."""
    t = np.asarray(t)
    thp = p.Op * t + php
    thm = p.Om * t + phm
    q = Ap * np.sin(thp) + Am * np.sin(thm)
    qd = Ap * p.Op * np.cos(thp) + Am * p.Om * np.cos(thm)
    z = Ap * np.cos(thp) - Am * np.cos(thm)
    zd = -Ap * p.Op * np.sin(thp) + Am * p.Om * np.sin(thm)
    return np.stack([q, qd, z, zd], axis=-1)


def nominal_A(p: VMSParams, n_override: float | None = None) -> np.ndarray:
    """Closed-loop matrix of the nominal interconnection."""
    n = p.n if n_override is None else n_override
    return np.array(
        [[0, 1, 0, 0], [-1, 0, 0, n], [0, 0, 0, 1], [0, -n, -1, 0]], dtype=float
    )


def system_matrix(n: float, omega_z: float = 1.0) -> np.ndarray:
    """Interconnection with virtual stiffness omega_z^2 (host stiffness 1)."""
    return np.array(
        [[0, 1, 0, 0], [-1, 0, 0, n], [0, 0, 0, 1], [0, -n, -omega_z**2, 0]], dtype=float
    )


def projector(n: float, omega_z: float = 1.0):
    """Complex modal projector P (2x4) for the frozen system A(n, omega_z):
    zeta = P @ x, zeta_pm' = i w_pm zeta_pm along A(n, omega_z).
    Rows are left eigenvectors for +i w_+ and +i w_-, normalised such that the
    q'-coefficient is real and equal to 1/(w_+ + w_-).  For omega_z = 1 this
    reproduces T exactly (zeta_pm = X_pm + i Y_pm).  Returns (P, w_+, w_-)."""
    A = system_matrix(n, omega_z)
    ev, L = np.linalg.eig(A.T)  # left eigenvectors of A
    idx = np.where(ev.imag > 0)[0]
    idx = idx[np.argsort(-ev.imag[idx])]
    wp, wm = ev.imag[idx[0]], ev.imag[idx[1]]
    S = wp + wm
    rows = []
    for j in idx:
        l = L[:, j]
        l = l / l[1] / S
        rows.append(l)
    return np.array(rows), wp, wm


def mu_bar(Ap, Am, p: VMSParams):
    """Average of g^2 = (q' z')^2 over the (non-resonant to order 4) torus:
    Om+^4 A+^4/8 + Om-^4 A-^4/8 + A+^2 A-^2/2."""
    return p.Op**4 * Ap**4 / 8 + p.Om**4 * Am**4 / 8 + Ap**2 * Am**2 / 2
