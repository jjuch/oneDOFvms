"""Sampled-data closed-loop simulator of the imperfect gyroscopic VMS.

Everything runs on the controller grid t_k = k Ts and is *exact* between
samples (host and virtual oscillators are linear and driven by held inputs),
so no ODE integrator error enters the comparison.

Realised loop at step k
-----------------------
  measurement : yhat_k = (1+gs) (q, q')(t_{k-ks}) + noise
  controller  : u_k, s_k, omega_z (energy / amplitude / phase-frequency loops)
  virtual     : z'' + (n + s_k - u_k) qhat'_k + omega_z^2 z = 0   on [t_k, t_k+1)
  actuator    : F = (1+ga) * play( (n + s_k + u_k) z'_k  delayed ka samples )
  host        : q'' + q = F                                        on [t_k, t_k+1)

The energy and amplitude loops are those of the ECC paper (eqs. (37), (53));
the phase loop is either 'none', 'freq' (ECC batch frequency adaptation,
eq. (82), with a least-squares phase-slope estimate in place of I/Q
demodulation) or 'phase' (relative-phase locking proposed in the journal
version).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .params import VMSParams
from . import modal


@dataclass
class Imperfections:
    Ts: float = 0.03
    ks: int = 1  # sensor delay [samples]
    ka: int = 1  # actuator delay [samples]
    gs: float = -0.03  # sensor gain error
    ga: float = 0.065  # actuator gain error
    noise: float = 0.002  # std of additive sensor noise (q and q')
    play: float = 0.003  # actuator play (backlash) half-width
    seed: int = 1

    @staticmethod
    def ideal(Ts=0.03):
        return Imperfections(Ts=Ts, ks=0, ka=0, gs=0.0, ga=0.0, noise=0.0, play=0.0)


@dataclass
class Gains:
    KE: float = 0.0
    KA: float = 0.0
    phase_mode: str = "none"  # 'none' | 'freq' | 'phase' (via omega_z) | 'trim' (via common coupling)
    # ECC frequency loop
    KO: float = 1.0
    N_batch: int = 2084  # ~ one common period P0 / Ts for (11,9), Ts=0.03
    # proposed phase loop
    kP: float = 0.02
    kI: float = 2e-4
    psi_ref: float | None = None  # default: absorbing phase
    t_phase_on: float = 0.0
    omega_bounds: tuple = (0.8, 1.2)
    trim_bound: float = 0.05  # |c_trim| bound for 'trim' mode
    sat: float = np.inf  # |u|, |s| saturation
    KIA: float = 0.0  # integral gain on e_A (removes delay-induced modal drift)
    normalized: bool = False  # scale-invariant laws: divide by Hhat instead of Hr
    adaptive_projection: bool = True  # modal projector of A(n, omega_z)


@dataclass
class SimResult:
    t: np.ndarray
    x: np.ndarray  # true state (q, q', z, z') at t_k
    u: np.ndarray
    s: np.ndarray
    omega_z: np.ndarray
    F: np.ndarray
    p: VMSParams
    Hr: float
    meta: dict = field(default_factory=dict)

    def errors(self, projector: bool = True):
        """(e_H, e_A, psi) along the run.  With projector=True the modal
        coordinates of the frozen system A(n, omega_z(t)) are used (this removes
        the coordinate ripple caused by omega_z != 1)."""
        if not projector:
            return modal.errors(self.x, self.p, self.Hr)
        eH = modal.energy(self.x) / self.Hr - 1.0
        zp = np.empty(len(self.t), complex)
        zm = np.empty(len(self.t), complex)
        w_last, P = None, None
        C = self.meta.get("c_trim", np.zeros(len(self.t)))
        for k, (xk, w, c) in enumerate(zip(self.x, self.omega_z, C)):
            key = w + 10.0 * c
            if w_last is None or abs(key - w_last) > 1e-6:
                P, _, _ = modal.projector(self.p.n + c, w)
                w_last = key
            zp[k], zm[k] = P @ xk
        eA = np.log(np.abs(zp) / np.abs(zm))
        psi = np.angle(np.exp(1j * (self.p.sigma * np.angle(zp) - self.p.tau * np.angle(zm))))
        return eH, eA, psi

    def late(self, frac=0.25):
        return self.t >= self.t[-1] * (1 - frac)

    def metrics(self, frac=0.25):
        eH, eA, psi = self.errors()
        L = self.late(frac)
        Hq = modal.host_energy(self.x)
        ratio, wp, wm = estimate_ratio(self.t[L], self.x[L], self.p, return_freqs=True)
        # per-common-period minimum of the host energy (median over late windows)
        dt = self.t[1] - self.t[0]
        Pk = int(round(self.p.P0 / dt))
        HqL = Hq[L]
        mins = [HqL[i:i + Pk].min() for i in range(0, len(HqL) - Pk + 1, Pk)]
        That = 0.5 * (2 * np.pi * self.p.tau / wp + 2 * np.pi * self.p.sigma / wm)
        return {
            "median_cycle_min_Hq": float(np.median(mins) / self.Hr) if mins else np.nan,
            "rms_eH": float(np.sqrt(np.mean(eH[L] ** 2))),
            "rms_eA": float(np.sqrt(np.mean(eA[L] ** 2))),
            "min_Hq_over_Hr": float(Hq[L].min() / self.Hr),
            "ratio": ratio if np.sqrt(np.mean(eA[L] ** 2)) < 1 else np.nan,
            "eOmega": (float(abs(ratio - self.p.tau / self.p.sigma))
                       if np.isfinite(ratio) and np.sqrt(np.mean(eA[L] ** 2)) < 1 else np.nan),
            "psi_spread": float(np.abs(np.angle(
                np.exp(1j * (psi[L] - np.angle(np.mean(np.exp(1j * psi[L]))))))).max()),
            "closure": closure_error(self.t[L], self.x[L], self.p, That),
        }


def _rot(omega, dt):
    c, s = np.cos(omega * dt), np.sin(omega * dt)
    return np.array([[c, s / omega], [-omega * s, c]])


def _osc_step(x, omega, f, dt):
    """Exact step of y'' + omega^2 y = f (f constant)."""
    xe = np.array([f / omega**2, 0.0])
    return _rot(omega, dt) @ (x - xe) + xe


def _play(x, y_prev, r):
    if r <= 0:
        return x
    if x - r > y_prev:
        return x - r
    if x + r < y_prev:
        return x + r
    return y_prev


def estimate_ratio(t, x, p: VMSParams, return_freqs=False):
    """Ratio of modal frequencies from the least-squares slope of the
    unwrapped modal phases (a simple stand-in for I/Q demodulation)."""
    if len(t) < 10:
        return (np.nan, np.nan, np.nan) if return_freqs else np.nan
    _, _, thp, thm = modal.polar(x, p)
    wp = np.polyfit(t, np.unwrap(thp), 1)[0]
    wm = np.polyfit(t, np.unwrap(thm), 1)[0]
    return (wp / wm, wp, wm) if return_freqs else wp / wm


def closure_error(t, x, p: VMSParams, period=None):
    """Normalised state mismatch after one common period (default: the
    designed P0; pass the period inferred from the estimated modal lines)."""
    if len(t) < 3:
        return np.nan
    dt = t[1] - t[0]
    period = p.P0 if (period is None or not np.isfinite(period)) else period
    k = int(round(period / dt))
    if k >= len(t):
        return np.nan
    num = np.sum(np.sum((x[k:] - x[:-k]) ** 2, axis=1))
    den = np.sum(np.sum(x[:-k] ** 2, axis=1))
    return float(num / den)


def simulate(p: VMSParams, imp: Imperfections, gains: Gains, T: float = 3000.0,
             v0: float = 1.0, x0=None) -> SimResult:
    rng = np.random.default_rng(imp.seed)
    Ts = imp.Ts
    K = int(round(T / Ts))
    n = p.n
    x = np.zeros(4) if x0 is None else np.array(x0, float)
    if x0 is None:
        x[1] = v0
    Hr = modal.energy(x)
    Tm = modal.T_matrix(p)
    P_proj, _, _ = modal.projector(n, 1.0)
    w_proj = 1.0
    psi_ref = p.psi_abs if gains.psi_ref is None else gains.psi_ref

    meas_buf = [np.zeros(2)] * (imp.ks + 1)
    act_buf = [0.0] * (imp.ka + 1)
    F_prev = 0.0
    omega_z = 1.0
    xi = 0.0  # phase-loop integrator (in units of psi rate)
    zA = 0.0  # amplitude-loop integrator
    c_trim = 0.0  # common-coupling trim (phase loop, 'trim' mode)
    beta_psi = 2 * np.sqrt(p.tau * p.sigma) / p.S  # d psi'/d c
    batch_t, batch_thp, batch_thm = [], [], []

    X = np.empty((K + 1, 4))
    U = np.zeros(K + 1)
    Sg = np.zeros(K + 1)
    W = np.ones(K + 1)
    Fr = np.zeros(K + 1)
    C = np.zeros(K + 1)
    X[0] = x
    for k in range(K):
        tk = k * Ts
        # --- sensing (delayed, gain error, noise) ---
        y = (1 + imp.gs) * x[:2] + imp.noise * rng.normal(size=2)
        meas_buf.append(y)
        meas_buf.pop(0)
        qh, qdh = meas_buf[0]
        z, zd = x[2], x[3]
        xh = np.array([qh, qdh, z, zd])
        # --- energy loop ---
        eH = modal.energy(xh) / Hr - 1.0
        g = qdh * zd
        Hh = max(modal.energy(xh), 1e-9)
        if gains.normalized:
            u = -gains.KE * Hr / Hh**2 * eH * g
        else:
            u = -gains.KE / Hr * eH * g
        # --- amplitude loop ---
        m = qh * qdh + z * zd
        key = omega_z + 10.0 * c_trim
        if gains.adaptive_projection and abs(key - w_proj) > 1e-6:
            P_proj, _, _ = modal.projector(n + c_trim, omega_z)
            w_proj = key
        zp, zm = P_proj @ xh
        Ap, Am = abs(zp), abs(zm)
        eA = np.log(max(Ap, 1e-12) / max(Am, 1e-12))
        zA += Ts * gains.KIA * eA
        s = -gains.KA / (Hh if gains.normalized else Hr) * (eA + zA) * m
        u = float(np.clip(u, -gains.sat, gains.sat))
        s = float(np.clip(s, -gains.sat, gains.sat))
        # --- phase / frequency loop ---
        thp, thm = np.angle(zp), np.angle(zm)
        if gains.phase_mode == "freq":
            batch_t.append(tk)
            batch_thp.append(thp)
            batch_thm.append(thm)
            if len(batch_t) >= gains.N_batch:
                bt = np.array(batch_t)
                wp = np.polyfit(bt, np.unwrap(batch_thp), 1)[0]
                wm = np.polyfit(bt, np.unwrap(batch_thm), 1)[0]
                eO = p.Om * (wp - p.Op) - p.Op * (wm - p.Om)
                if tk >= gains.t_phase_on:
                    omega_z = omega_z + gains.KO * p.S / n * eO
                    omega_z = float(np.clip(omega_z, *gains.omega_bounds))
                batch_t, batch_thp, batch_thm = [], [], []
        elif gains.phase_mode == "phase" and tk >= gains.t_phase_on:
            psi = p.sigma * thp - p.tau * thm
            e = np.sin(psi - psi_ref)
            xi += Ts * gains.kI * e
            eps = (gains.kP * e + xi) / p.b_psi
            omega_z = float(np.clip(np.sqrt(max(1 + eps, 1e-6)), *gains.omega_bounds))
        elif gains.phase_mode == "trim" and tk >= gains.t_phase_on:
            psi = p.sigma * thp - p.tau * thm
            e = np.sin(psi - psi_ref)
            xi += Ts * gains.kI * e
            c_trim = float(np.clip(-(gains.kP * e + xi) / beta_psi,
                                   -gains.trim_bound, gains.trim_bound))
        # --- actuator (delayed, play, gain) ---
        Fc = (n + c_trim + s + u) * zd
        act_buf.append(Fc)
        act_buf.pop(0)
        F = (1 + imp.ga) * _play(act_buf[0], F_prev, imp.play)
        F_prev = F / (1 + imp.ga) if imp.ga != -1 else F
        # --- propagate exactly over one sample ---
        x[:2] = _osc_step(x[:2], 1.0, F, Ts)
        x[2:] = _osc_step(x[2:], omega_z, -(n + c_trim + s - u) * qdh, Ts)
        X[k + 1] = x
        U[k], Sg[k], W[k], Fr[k], C[k] = u, s, omega_z, F, c_trim
    t = np.arange(K + 1) * Ts
    U[K], Sg[K], W[K], Fr[K], C[K] = U[K - 1], Sg[K - 1], W[K - 1], Fr[K - 1], C[K - 1]
    return SimResult(t, X, U, Sg, W, Fr, p, Hr, meta={"imp": imp, "gains": gains, "c_trim": C})
