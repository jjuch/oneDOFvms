"""Numerical cross-checks of the main theorems.

Run from code/:  python -m tests.check_theorems
  * Theorem 'delay': closed-form unstable-root count vs spectral collocation;
  * Proposition 'zoh': lifted ZOH rates vs first-order prediction;
  * Theorem 'absorption': predicted absorbing phases vs brute-force minimum;
  * Proposition 'authority': ratio formula vs eigenvalues.
"""
import numpy as np

from vms import VMSParams, delay, zoh, modal
from vms.absorption import psi_star, min_host_radius

ok_all = True


def report(name, ok, info=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name:62s} {info}")


for tau, sig in [(11, 9), (6, 5), (4, 3)]:
    p = VMSParams(tau, sig)
    ds = np.concatenate([np.linspace(0.01, 1, 5), np.linspace(1.3, 30, 25)])
    bad = 0
    for d in ds:
        r = delay.spectrum_collocation(p, d, N=60)
        bad += int((r.real > 1e-9).sum() != delay.n_unstable_formula(p, d))
    report(f"({tau},{sig}) delay: N(d) formula vs collocation", bad == 0, f"mismatches={bad}/{len(ds)}")
    r = delay.spectrum_collocation(p, 0.01, N=30)
    sp = r[np.argmin(np.abs(r - 1j * p.Op))].real
    report(f"({tau},{sig}) delay: first-order rate (d=0.01)",
           abs(sp / delay.first_order_rates(p, 0.01)[0] - 1) < 1e-2, f"ratio={sp / delay.first_order_rates(p, 0.01)[0]:.4f}")
    for ks in (0, 1, 2):
        e = zoh.effective_rates(p, 0.01, ks, ks)[0]
        f = zoh.first_order_rates(p, 0.01, ks, ks)[0]
        report(f"({tau},{sig}) ZOH: rate vs d_eq=(1+2k)Ts, k={ks}", abs(e / f - 1) < 1e-2, f"ratio={e / f:.4f}")
    worst = 0.0
    lr = np.log(tau / sig)
    for eA in np.linspace(-lr, 0, 7):
        worst = max(worst, min_host_radius(eA, psi_star(eA, p), p, M=60000).max())
    report(f"({tau},{sig}) absorption: radius at Psi*(e_A)", worst < 2e-4, f"max={worst:.1e}")
    ps = np.linspace(-np.pi, np.pi, 1441)
    outside = min(min_host_radius(0.03, ps, p).min(), min_host_radius(-lr - 0.03, ps, p).min())
    report(f"({tau},{sig}) absorption: no absorption outside band", outside > 1e-2, f"min={outside:.3f}")
    # authority: r + 1/r = (1 + wz^2 + kappa)/wz
    for wz, kap in [(1.0, p.n**2), (1.03, 1.02 * p.n**2), (0.97, 0.9 * p.n**2)]:
        A = np.array([[0, 1, 0, 0], [-1, 0, 0, np.sqrt(kap)], [0, 0, 0, 1], [0, -np.sqrt(kap), -wz**2, 0]])
        w = np.sort(np.abs(np.linalg.eigvals(A).imag))[::2]
        rr = w[1] / w[0]
        report(f"({tau},{sig}) authority: ratio formula wz={wz}", abs(rr + 1 / rr - (1 + wz**2 + kap) / wz) < 1e-10)

print("\nALL PASS" if ok_all else "\nSOME CHECKS FAILED")
