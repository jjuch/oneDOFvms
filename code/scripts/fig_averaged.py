"""Figure: validation of the first-order averaged model against the exact
sampled-data simulation (delay + ZOH + gain errors, no noise / play).

(a) e_A: without integral action it settles at S alpha_Sigma / K_A, with
    integral action it is driven to zero;
(b) psi and trim c: the phase loop locks psi to psi* and c -> -d_s.
"""
import numpy as np

from common import C, FULL, plt, save
from vms import VMSParams, averaged as av
from vms.sim import Gains, Imperfections, simulate


def window_mean(t, y, P0):
    dt = t[1] - t[0]
    k = int(round(P0 / dt))
    c = np.cumsum(np.insert(y, 0, 0.0))
    m = (c[k:] - c[:-k]) / k
    return t[k // 2: k // 2 + len(m)], m


def window_circmean(t, y, P0):
    tt, cr = window_mean(t, np.cos(y), P0)
    _, sr = window_mean(t, np.sin(y), P0)
    return tt, np.arctan2(sr, cr)


def main():
    p = VMSParams(11, 9)
    Ts, ks, ka = 0.03, 1, 1
    h = Ts * (1 + ks + ka) / 2
    KE, KA = 1.0, 0.3
    T = 2500.0
    fig, ax = plt.subplots(1, 2, figsize=(FULL, 2.2))
    imp0 = Imperfections(Ts=Ts, ks=ks, ka=ka, gs=0, ga=0, noise=0, play=0)
    for KIA, col in [(0.0, C["orange"]), (0.01, C["blue"])]:
        r = simulate(p, imp0, Gains(KE=KE, KA=KA, KIA=KIA, normalized=True), T=T)
        _, eA, _ = r.errors(projector=False)
        tw, mw = window_mean(r.t, eA, p.P0)
        ax[0].plot(tw, mw, color=col, lw=1.0, label=rf"simulation, $k_{{IA}}={KIA}$")
        sol = av.simulate_averaged(p, [0, 0, 0, 0, 0], T, KE=KE, KA=KA, KIA=KIA, kP=0, kI=0, h=h)
        tt = np.linspace(0, T, 800)
        ax[0].plot(tt, sol.sol(tt)[1], "--", color="k", lw=0.7)
    ax[0].axhline(p.S * av.alpha_sigma(p, h) / KA, color=C["grey"], ls=":", lw=0.6)
    ax[0].text(50, p.S * av.alpha_sigma(p, h) / KA * 1.04, r"$S\alpha_\Sigma/K_A$", fontsize=7)
    ax[0].plot([], [], "k--", lw=0.7, label="averaged model")
    ax[0].set_xlabel("$t$ [s]")
    ax[0].set_ylabel(r"$e_A$ (mean over $P_0$)")
    ax[0].legend(frameon=False, fontsize=6, loc="center right")
    ax[0].set_title(r"(a) amplitude loop, $d_{eq}=0.09$ s")
    # (b) phase loop with common gain error
    gs, ga = -0.03, 0.065
    ds = p.n * (ga + gs) / 2
    imp1 = Imperfections(Ts=Ts, ks=ks, ka=ka, gs=gs, ga=ga, noise=0, play=0)
    kP, kI, ton = 0.02, 2e-4, 300.0
    r = simulate(p, imp1, Gains(KE=KE, KA=KA, KIA=0.01, normalized=True, phase_mode="trim",
                                kP=kP, kI=kI, t_phase_on=ton), T=T)
    _, _, psi = r.errors(projector=True)
    tw, mw = window_circmean(r.t, psi - p.psi_abs, p.P0)
    ax[1].plot(tw, mw, color=C["blue"], lw=1.0, label=r"$\psi-\psi^\star$, simulation")

    def rhs(t, y):
        on = t >= ton
        return av.rhs(t, y, p, KE, KA, 0.01, kP if on else 0.0, kI if on else 0.0, h, ds=ds)

    from scipy.integrate import solve_ivp
    sol = solve_ivp(rhs, (0, T), [0, 0, 0, p.psi_abs, 0], max_step=1.0, dense_output=True)
    tt = np.linspace(0, T, 1500)
    ax[1].plot(tt, np.angle(np.exp(1j * (sol.sol(tt)[3] - p.psi_abs))), "k--", lw=0.7,
               label="averaged model")
    b = ax[1].twinx()
    b.plot(r.t[::20], r.meta["c_trim"][::20] / p.n, color=C["green"], lw=0.8)
    b.plot(tt, -sol.sol(tt)[4] / av.beta_psi(p) / p.n * (tt >= ton), ":", color="k", lw=0.7)
    b.axhline(-ds / p.n, color=C["green"], lw=0.4, ls=":")
    b.set_ylabel(r"trim $c/n$", color=C["green"])
    ax[1].axvline(ton, color=C["grey"], lw=0.4)
    ax[1].set_xlabel("$t$ [s]")
    ax[1].set_ylabel(r"$\psi-\psi^\star$ [rad]")
    ax[1].legend(frameon=False, fontsize=6, loc="upper right")
    ax[1].set_title(r"(b) phase loop, $d_s=1.75\%\,n$ (on at $t=300$ s)")
    fig.tight_layout()
    save(fig, "fig_averaged.pdf")


if __name__ == "__main__":
    main()
