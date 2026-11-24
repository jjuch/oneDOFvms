"""Figure: absorption set in the (e_A, psi) plane for (tau, sigma) = (11, 9).

Colour: log10 of the minimal host radius along the resonant orbit (A+ = 1).
Dashed: the predicted absorbing set Psi*(e_A), e_A in [-ln(tau/sigma), 0].
"""
import numpy as np

from common import C, COL, plt, save
from vms import VMSParams
from vms.absorption import min_host_radius, psi_star


def main(tau=11, sig=9):
    p = VMSParams(tau, sig)
    lr = np.log(tau / sig)
    eAs = np.linspace(-lr - 0.12, 0.12, 141)
    psis = np.linspace(-np.pi, np.pi, 241)
    Z = np.array([min_host_radius(e, psis, p, M=3000) for e in eAs])
    fig, ax = plt.subplots(figsize=(COL, 2.4))
    cs = ax.pcolormesh(psis, eAs, np.log10(np.maximum(Z, 1e-4)), shading="auto",
                       cmap="viridis", rasterized=True)
    cb = fig.colorbar(cs, ax=ax, pad=0.02)
    cb.set_label(r"$\log_{10}\min_{\mathrm{orbit}}\sqrt{q^2+\dot q^2}$")
    # predicted curve
    pts = []
    for e in np.linspace(-lr, 0, 400):
        for v in psi_star(e, p):
            pts.append((v, e))
    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], ".", ms=0.8, color=C["red"])
    ax.axhline(0, color="w", lw=0.5, ls="--")
    ax.axhline(-lr, color="w", lw=0.5, ls="--")
    ax.plot(0, 0, "o", ms=4, mfc="none", mec="w")
    ax.annotate("impulse", (0, 0), (0.5, 0.07), color="w", fontsize=7,
                arrowprops=dict(arrowstyle="-", color="w", lw=0.5))
    ax.set_xlabel(r"$\psi$ [rad]")
    ax.set_ylabel(r"$e_A=\ln(A_+/A_-)$")
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
    ax.text(-3.05, -lr + 0.008, r"$-\ln(\tau/\sigma)$", color="w", fontsize=7)
    save(fig, "fig_absorption.pdf")


if __name__ == "__main__":
    main()
