"""Figure: exact spectral consequences of loop delay and zero-order hold.

(a) spectral abscissa vs loop delay d, with the number of unstable roots N(d)
    from the closed-form count and the imaginary-axis crossing delays;
(b) small-delay regime: real parts of the + and - modes (exact quasi-polynomial
    roots, lifted ZOH map) against the first-order prediction.
"""
import numpy as np

from common import C, FULL, plt, save
from vms import VMSParams, delay, zoh


def main():
    p = VMSParams(11, 9)
    fig, ax = plt.subplots(1, 2, figsize=(FULL, 2.3))
    # (a)
    ds = np.linspace(0.02, 40, 700)
    absc = np.array([delay.spectral_abscissa(p, d, N=50) for d in ds])
    Nf = np.array([delay.n_unstable_formula(p, d) for d in ds])
    a = ax[0]
    a.plot(ds, absc, color=C["blue"], label="spectral abscissa")
    a.set_xlabel(r"loop delay $d$ [s]")
    a.set_ylabel(r"$\max\,\mathrm{Re}\,s$ [1/s]", color=C["blue"])
    a.set_ylim(bottom=0)
    b = a.twinx()
    b.step(ds, Nf, where="post", color=C["red"], lw=0.8)
    b.set_ylabel(r"unstable roots $N(d)$", color=C["red"])
    b.set_ylim(0, 8)
    dp, dm = delay.crossing_delays(p, 40)
    for x in dp:
        a.axvline(x, color=C["red"], lw=0.4, ls="--")
    for x in dm:
        a.axvline(x, color=C["green"], lw=0.4, ls=":")
    a.set_title(r"(a) $N(d)=2(1+\lfloor d\Omega_+/2\pi\rfloor-\lfloor d\Omega_-/2\pi\rfloor)\geq2$")
    # (b)
    a = ax[1]
    dsm = np.linspace(0.0, 0.3, 31)[1:]
    rp, rm = [], []
    for d in dsm:
        r = delay.spectrum_collocation(p, d, N=30)
        rp.append(r[np.argmin(np.abs(r - 1j * p.Op))].real)
        rm.append(r[np.argmin(np.abs(r - 1j * p.Om))].real)
    fp, fm = delay.first_order_rates(p, 1.0)
    dl = np.linspace(0, 0.3, 2)
    a.plot(dl, fp * dl, color=C["red"], lw=0.8, label=r"first order $n\Omega_\pm^2 d/(2S)$")
    a.plot(dl, -fm * dl, color=C["red"], lw=0.8)
    a.plot(dsm, rp, "o", ms=2.5, color=C["blue"], label=r"exact, mode $+$ (pure delay)")
    a.plot(dsm, rm, "s", ms=2.5, color=C["green"], label=r"exact, mode $-$ (pure delay)")
    for (ks, ka), mk in [((0, 0), "^"), ((1, 1), "v")]:
        Tss = np.linspace(0.01, 0.3 / (1 + ks + ka), 8)
        zr = np.array([zoh.effective_rates(p, T, ks, ka) for T in Tss])
        a.plot(Tss * (1 + ks + ka), zr[:, 0], mk, ms=3, mfc="none", color=C["orange"],
               label=rf"lifted ZOH map, $k_s=k_a={ks}$, $d=(1+k_s+k_a)T_s$")
        a.plot(Tss * (1 + ks + ka), zr[:, 1], mk, ms=3, mfc="none", color=C["orange"])
    a.axhline(0, color="k", lw=0.4)
    a.set_xlabel(r"(equivalent) loop delay $d$ [s]")
    a.set_ylabel(r"$\mathrm{Re}\,s_\pm$ [1/s]")
    a.legend(frameon=False, loc="upper left", fontsize=6)
    a.set_ylim(-0.016, 0.042)
    a.set_title("(b) mode $+$ destabilised, mode $-$ damped")
    fig.tight_layout()
    save(fig, "fig_delay.pdf")


if __name__ == "__main__":
    main()
