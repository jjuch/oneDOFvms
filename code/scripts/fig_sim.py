"""Figures for Section 'Simulations'.

fig_sim_portraits.pdf : subsystem energies (top) and late host phase plane
                        (bottom) for the successive controller layers;
fig_sim_errors.pdf    : e_H, e_A, psi, retuning action and per-cycle minimum
                        host energy, ECC vs proposed.
"""
import numpy as np

from common import C, FULL, plt, save
from run_simulations import CASES, P, run
from fig_averaged import window_circmean, window_mean


def cycle_min(t, Hq, P0):
    dt = t[1] - t[0]
    k = int(round(P0 / dt))
    idx = np.arange(0, len(t) - k, k)
    return t[idx] + P0 / 2, np.array([Hq[i:i + k].min() for i in idx])


def portraits():
    names = ["imperfect", "E", "EA", "ECC", "proposed"]
    titles = ["(a) uncompensated", "(b) $u$", "(c) $u,s$", "(d) $u,s,\\omega_z$ (ECC)",
              "(e) proposed"]
    fig, ax = plt.subplots(2, 5, figsize=(FULL, 2.9), gridspec_kw={"height_ratios": [1, 1.25]})
    for j, name in enumerate(names):
        r = run(name)
        t, x, Hr = r["t"], r["x"], r["Hr"]
        Hq = 0.5 * (x[:, 0] ** 2 + x[:, 1] ** 2) / Hr
        Hz = 0.5 * (x[:, 2] ** 2 + x[:, 3] ** 2) / Hr
        a = ax[0, j]
        st = 10
        a.plot(t[::st], Hq[::st], color=C["blue"], lw=0.35, label=r"$H_q/H_r$")
        a.plot(t[::st], Hz[::st], color=C["orange"], lw=0.35, label=r"$H_z/H_r$")
        if name == "imperfect":
            a.set_yscale("log")
        else:
            a.set_ylim(0, 1.25)
        a.set_title(titles[j])
        a.set_xlabel("$t$ [s]")
        a.set_xticks([0, 1500, 3000])
        if j == 0:
            a.set_ylabel("energy")
        b = ax[1, j]
        L = t >= t[-1] - 2 * P.P0
        b.plot(x[L, 0], x[L, 1], color=C["blue"], lw=0.35)
        b.plot(0, 0, "k+", ms=4)
        b.set_aspect("equal", adjustable="datalim")
        b.set_xlabel("$q$")
        if j == 0:
            b.set_ylabel(r"$\dot q$")
        if name != "imperfect":
            b.set_xlim(-1.15, 1.15)
            b.set_ylim(-1.15, 1.15)
    ax[0, 1].legend(frameon=False, loc="upper right", handlelength=1.0)
    fig.tight_layout(h_pad=0.4, w_pad=0.3)
    save(fig, "fig_sim_portraits.pdf")


def errors():
    rE, rP = run("ECC"), run("proposed")
    fig, ax = plt.subplots(3, 2, figsize=(FULL, 3.6), sharex=True)
    st = 5
    for r, col, lab in [(rE, C["orange"], "ECC [$\\omega_z$ batch]"), (rP, C["blue"], "proposed [trim]")]:
        t = r["t"]
        for k, key in enumerate(["eH", "eA"]):
            ax[0, k].plot(t[::st], r[key][::st], color=col, lw=0.3, alpha=0.25)
            tw, mw = window_mean(t, r[key], P.P0)
            ax[0, k].plot(tw, mw, color=col, lw=1.0, label=lab)
        ax[1, 0].plot(t[::st], np.angle(np.exp(1j * (r["psi"][::st] - P.psi_abs))), ".", ms=0.3,
                      color=col, alpha=0.15)
        tw, mw = window_circmean(t, r["psi"] - P.psi_abs, P.P0)
        ax[1, 0].plot(tw, mw, ".", ms=0.8, color=col)
        x, Hr = r["x"], r["Hr"]
        Hq = 0.5 * (x[:, 0] ** 2 + x[:, 1] ** 2) / Hr
        tc, mc = cycle_min(t, Hq, P.P0)
        ax[1, 1].semilogy(tc, mc, "o-", ms=2, lw=0.6, color=col)
    ax[2, 0].plot(rE["t"], rE["omega_z"] - 1, color=C["orange"], lw=0.7, label=r"$\omega_z-1$ (ECC)")
    ax[2, 0].plot(rP["t"], rP["c_trim"] / P.n, color=C["blue"], lw=0.7, label=r"$c/n$ (proposed)")
    ds = P.n * (0.065 - 0.03) / 2
    ax[2, 0].axhline(-ds / P.n, color=C["blue"], ls=":", lw=0.6)
    ax[2, 0].axhline(np.sqrt(1 + P.n**2) - 1, color=C["orange"], ls=":", lw=0.6)
    tt = rP["t"]
    ax[2, 1].plot(tt[::st], rP["u"][::st], color=C["green"], lw=0.3, label="$u$")
    ax[2, 1].plot(tt[::st], rP["s"][::st], color=C["purple"], lw=0.3, label="$s$")
    ax[0, 0].set_ylabel("$e_H$")
    ax[0, 0].set_ylim(-0.15, 0.15)
    ax[0, 1].set_ylabel("$e_A$")
    ax[0, 1].set_ylim(-0.6, 0.6)
    ax[1, 0].set_ylabel(r"$\psi-\psi^\star$ [rad]")
    ax[1, 0].set_ylim(-np.pi, np.pi)
    ax[1, 1].set_ylabel(r"$\min_{\mathrm{cycle}} H_q/H_r$")
    ax[2, 0].set_ylabel("retuning action")
    ax[2, 1].set_ylabel("proposed $u$, $s$")
    ax[0, 0].legend(frameon=False, loc="upper right")
    ax[2, 0].legend(frameon=False, loc="upper right", fontsize=6)
    ax[2, 1].legend(frameon=False, loc="upper right", fontsize=6)
    for a in ax[2]:
        a.set_xlabel("$t$ [s]")
    for k, a in enumerate(ax.flat):
        a.text(0.01, 0.97, f"({'abcdef'[k]})", transform=a.transAxes, va="top", fontsize=8,
               fontweight="bold")
    fig.tight_layout(h_pad=0.3)
    save(fig, "fig_sim_errors.pdf")


if __name__ == "__main__":
    portraits()
    errors()
