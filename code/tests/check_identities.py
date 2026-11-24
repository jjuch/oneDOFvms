"""Numerical verification of the identities used in the paper.

Run:  python -m tests.check_identities   (from the code/ directory)
Every check prints PASS/FAIL with the max residual.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from vms import VMSParams, modal

rng = np.random.default_rng(0)
ok_all = True


def report(name, res, tol=1e-9):
    global ok_all
    ok = res < tol
    ok_all &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name:60s} residual={res:.2e}")


for (tau, sig) in [(11, 9), (6, 5), (4, 3), (41, 35)]:
    p = VMSParams(tau, sig)
    print(f"--- (tau, sigma) = ({tau}, {sig}), n = {p.n:.6f}")
    report("Om+ Om- = 1", abs(p.Op * p.Om - 1))
    report("Om+ - Om- = n", abs(p.Op - p.Om - p.n))
    T, Ti = modal.T_matrix(p), modal.T_inv(p)
    report("T^-1 T = I", np.abs(Ti @ T - np.eye(4)).max())
    A = modal.nominal_A(p)
    # modal dynamics: T A T^-1 = blockdiag(Om+ J, Om- J) with J=[[0,-1],[1,0]]
    J = np.array([[0, -1], [1, 0]])
    Mdl = np.zeros((4, 4))
    Mdl[:2, :2] = p.Op * J
    Mdl[2:, 2:] = p.Om * J
    report("T A T^-1 = diag(Om+ J, Om- J)", np.abs(T @ A @ Ti - Mdl).max())
    # energy identity H = S/2 (Om+ A+^2 + Om- A-^2), on random states
    x = rng.normal(size=(1000, 4))
    Ap, Am, _, _ = modal.polar(x, p)
    report("H = S/2 (Om+ A+^2 + Om- A-^2)",
           np.abs(modal.energy(x) - modal.energy_from_amplitudes(Ap, Am, p)).max())
    # m = q q' + z z' = S A+ A- sin(theta+ + theta-)
    _, _, thp, thm = modal.polar(x, p)
    m = x[:, 0] * x[:, 1] + x[:, 2] * x[:, 3]
    report("q q' + z z' = S A+ A- sin(th+ + th-)",
           np.abs(m - p.S * Ap * Am * np.sin(thp + thm)).max())
    # g = q' z' modal expression
    g = x[:, 1] * x[:, 3]
    g_mod = (-(p.Op**2) * Ap**2 / 2 * np.sin(2 * thp) + p.Om**2 * Am**2 / 2 * np.sin(2 * thm)
             + Ap * Am * np.sin(thm - thp))
    report("q' z' modal expression", np.abs(g - g_mod).max())
    # average of g^2 over one common period on a nominal orbit
    Ap0, Am0 = 0.7, 0.4
    t = np.linspace(0, p.P0, 200001)
    xs = modal.nominal_state(t, p, Ap0, Am0, 0.3, -1.1)
    avg = np.trapezoid((xs[:, 1] * xs[:, 3]) ** 2, t) / p.P0
    report("<g^2>_P0 = mu_bar(A)  (needs tau+sigma>4)", abs(avg - modal.mu_bar(Ap0, Am0, p)), 1e-6)
    # absorption lemma: A+=A-, psi = psi_abs  => orbit hits host origin
    psi = p.psi_abs
    # choose theta-(0)=0, theta+(0)=psi/sigma  (so sigma th+ - tau th- = psi)
    t = np.linspace(0, p.P0, 400001)
    xs = modal.nominal_state(t, p, 1.0, 1.0, psi / p.sigma, 0.0)
    report("absorption: min host radius = 0 at psi = psi_abs",
           np.sqrt(2 * modal.host_energy(xs)).min(), 2e-3)

# perturbed modal dynamics: zeta+' = i Om+ zeta+ + (D1 - i D2)/S etc.
p = VMSParams(11, 9)
x = rng.normal(size=4)
c, w, eps, fq, fz = 0.03, -0.02, 0.05, 0.01, -0.02
q, qd, z, zd = x
D1 = (c + w) * zd + fq
D2 = -(c - w) * qd - eps * z + fz
xdot = modal.nominal_A(p) @ x + np.array([0, D1, 0, D2])
zp_dot_true = (modal.T_matrix(p) @ xdot)
zp, zm = modal.zeta(x, p)
pred_p = 1j * p.Op * zp + (D1 - 1j * D2) / p.S
pred_m = 1j * p.Om * zm + (D1 + 1j * D2) / p.S
report("perturbed modal eq. (zeta+)", abs(zp_dot_true[0] + 1j * zp_dot_true[1] - pred_p))
report("perturbed modal eq. (zeta-)", abs(zp_dot_true[2] + 1j * zp_dot_true[3] - pred_m))

# first-order frequency sensitivities
h = 1e-6
for name, pert, dp_pred, dm_pred in [
    ("dOm/dn (common coupling)", "n", p.Op / p.S, -p.Om / p.S),
    ("dOm/deps (virtual stiffness)", "eps", 1 / (2 * p.S), 1 / (2 * p.S)),
]:
    A0 = modal.nominal_A(p)
    Ap_ = A0.copy()
    if pert == "n":
        Ap_[1, 3] += h
        Ap_[3, 1] -= h
    else:
        Ap_[3, 2] -= h
    ev0 = np.sort(np.abs(np.linalg.eigvals(A0).imag))[::2]
    ev1 = np.sort(np.abs(np.linalg.eigvals(Ap_).imag))[::2]
    d = (ev1 - ev0) / h  # sorted ascending: (Om-, Om+)
    report(name, max(abs(d[1] - dp_pred), abs(d[0] - dm_pred)), 1e-4)

print("\nALL PASS" if ok_all else "\nSOME CHECKS FAILED")
