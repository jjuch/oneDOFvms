"""Figure: resonant tuning (CCTA framework) and the new relative-phase view.

(a) host phase portrait for (11,9) impulse response with convex envelope;
(b) same for (6,5) with non-zero inscribed circle;
(c) inscribed radius r_res(psi)/rho0 vs relative phase psi for A+ = A-.
"""
import numpy as np

from common import C, FULL, plt, save
from vms import VMSParams, modal


def envelope(p, rho0, M=721):
    ph = np.linspace(0, 2 * np.pi, M)
    out = np.zeros((M, 2))
    for Om in (p.Op, p.Om):
        den = np.sqrt(np.cos(ph) ** 2 + Om**2 * np.sin(ph) ** 2)
        out[:, 0] += np.cos(ph) / den
        out[:, 1] += Om**2 * np.sin(ph) / den
    return rho0 * out


def rres_of_psi(p, psis, M=40001):
    t = np.linspace(0, p.P0, M)
    out = []
    for psi in psis:
        x = modal.nominal_state(t, p, 1.0, 1.0, psi / p.sigma, 0.0)
        out.append(np.sqrt(2 * modal.host_energy(x)).min())
    return np.array(out)


def main():
    fig, ax = plt.subplots(1, 3, figsize=(FULL, 2.2), gridspec_kw={"width_ratios": [1, 1, 1.35], "wspace": 0.35})
    for k, (tau, sig) in enumerate([(11, 9), (6, 5)]):
        p = VMSParams(tau, sig)
        rho0 = 1.0 / p.S
        t = np.linspace(0, p.P0, 20001)
        x = modal.nominal_state(t, p, rho0, rho0)
        a = ax[k]
        a.plot(x[:, 0], x[:, 1], color=C["blue"], lw=0.5)
        e = envelope(p, rho0)
        a.plot(e[:, 0], e[:, 1], color=C["red"], lw=0.9)
        r = np.sqrt(2 * modal.host_energy(x)).min()
        th = np.linspace(0, 2 * np.pi, 200)
        if r > 1e-6:
            a.plot(r * np.cos(th), r * np.sin(th), "k--", lw=0.8)
        a.plot(0, 0, "k+", ms=5)
        a.set_aspect("equal")
        a.set_xlabel(r"$q$")
        a.set_ylabel(r"$\dot q$")
        a.set_title(rf"({'ab'[k]}) $({tau},{sig})$: $r_\mathrm{{res}}={r:.3f}$")
    a = ax[2]
    psis = np.linspace(-np.pi, np.pi, 721)
    for (tau, sig), col in zip([(11, 9), (6, 5), (4, 3), (41, 35)],
                               [C["blue"], C["orange"], C["green"], C["purple"]]):
        p = VMSParams(tau, sig)
        rr = rres_of_psi(p, psis)
        a.plot(psis, rr / 1.0, color=col, label=rf"$({tau},{sig})$")
        pa = np.angle(np.exp(1j * p.psi_abs))
        a.plot([pa], [0], "o", color=col, ms=3)
    a.axvline(0, color=C["grey"], lw=0.5, ls=":")
    a.set_xlabel(r"relative phase $\psi=\sigma\theta_+-\tau\theta_-$ [rad]")
    a.set_ylabel(r"$r_\mathrm{res}/A$")
    a.set_xlim(-np.pi, np.pi)
    a.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    a.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    a.legend(loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=0.8)
    a.set_ylim(0, 0.215)
    a.yaxis.labelpad = 1
    a.set_title(r"(c) $r_\mathrm{res}(\psi)$ for $A_+=A_-=A$")
    a.text(0.03, 0.04, r"impulse: $\psi=0$", transform=a.transAxes, fontsize=7)
    save(fig, "fig_tuning.pdf")


if __name__ == "__main__":
    main()
