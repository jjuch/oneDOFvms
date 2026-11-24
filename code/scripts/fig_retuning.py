"""Figure: retuning authority of the virtual stiffness vs the common coupling.

For a frozen interconnection with coupling product kappa = (n+a)(n+b) and
virtual stiffness omega_z^2, the modal ratio r satisfies
        r + 1/r = (1 + omega_z^2 + kappa) / omega_z.
(a) r vs omega_z for several kappa: the target tau/sigma is reachable only if
    kappa <= n^2 + n^4/4 and the sensitivity vanishes at omega_z = sqrt(1+kappa);
(b) r vs common coupling trim c (omega_z = 1): monotone, O(1) authority.
"""
import numpy as np

from common import C, FULL, plt, save
from vms import VMSParams


def ratio(kappa, wz):
    y = (1 + wz**2 + kappa) / wz
    return (y + np.sqrt(np.maximum(y**2 - 4, 0))) / 2


def main():
    p = VMSParams(11, 9)
    n = p.n
    r0 = p.tau / p.sigma
    fig, ax = plt.subplots(1, 2, figsize=(FULL, 2.1))
    wz = np.linspace(0.9, 1.15, 600)
    a = ax[0]
    cols = [C["green"], C["blue"], C["orange"], C["red"]]
    for rel, col in zip([-0.02, 0.0, 0.002, 0.02], cols):
        kap = (n * (1 + rel)) ** 2
        a.plot(wz, ratio(kap, wz), color=col, label=rf"$\kappa=n^2(1{rel:+.1%})^2$".replace("%", r"\%"))
    a.axhline(r0, color="k", lw=0.6, ls="--")
    a.axvline(np.sqrt(1 + n**2), color=C["grey"], lw=0.5, ls=":")
    a.text(np.sqrt(1 + n**2) + 0.003, r0 - 0.012, r"$\sqrt{1+\kappa}$", fontsize=7)
    a.set_xlabel(r"virtual frequency $\omega_z$")
    a.set_ylabel(r"modal ratio $r=\omega_+/\omega_-$")
    a.set_ylim(r0 - 0.03, r0 + 0.05)
    a.legend(frameon=False, fontsize=6, loc="upper center")
    a.set_title(r"(a) retuning by $\omega_z$: reachable iff $\kappa\leq n^2+n^4/4$")
    a = ax[1]
    c = np.linspace(-0.02, 0.02, 400)
    for rel, col in zip([-0.02, 0.0, 0.02], [C["green"], C["blue"], C["red"]]):
        kap = (n * (1 + rel) + c) ** 2
        a.plot(c, ratio(kap, 1.0), color=col, label=rf"$d_s={rel:+.0%}\,n$".replace("%", r"\%"))
    a.axhline(r0, color="k", lw=0.6, ls="--")
    a.set_xticks([-0.02, -0.01, 0, 0.01, 0.02])
    a.set_xlabel(r"common-coupling trim $c$")
    a.set_ylabel(r"$r$ at $\omega_z=1$")
    a.legend(frameon=False, fontsize=6)
    a.set_title(r"(b) retuning by common coupling: $\partial r/\partial c=2\Omega_+^2/S$")
    fig.tight_layout()
    save(fig, "fig_retuning.pdf")


if __name__ == "__main__":
    main()
