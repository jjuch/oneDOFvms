"""Nominal parameters of the gyroscopic virtual mechanical system (VMS).

Normalised model (host frequency 1, virtual frequency 1):

    q'' - n z' + q = 0,
    z'' + n q' + z = 0,

with modal frequencies Om_plus > Om_minus > 0, Om_plus * Om_minus = 1,
Om_plus - Om_minus = n.  A coprime resonant pair (tau, sigma), tau > sigma,
fixes n = (tau - sigma) / sqrt(tau * sigma).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd, pi, sqrt


@dataclass(frozen=True)
class VMSParams:
    tau: int = 11
    sigma: int = 9

    def __post_init__(self):
        if not (self.tau > self.sigma > 0):
            raise ValueError("need tau > sigma > 0")
        if gcd(self.tau, self.sigma) != 1:
            raise ValueError("(tau, sigma) must be coprime")

    @property
    def n(self) -> float:
        """Gyroscopic coupling realising Om+/Om- = tau/sigma."""
        return (self.tau - self.sigma) / sqrt(self.tau * self.sigma)

    @property
    def S(self) -> float:
        """S = Om+ + Om- = sqrt(n^2 + 4) = (tau + sigma)/sqrt(tau sigma)."""
        return (self.tau + self.sigma) / sqrt(self.tau * self.sigma)

    @property
    def Op(self) -> float:
        return self.tau / sqrt(self.tau * self.sigma)

    @property
    def Om(self) -> float:
        return self.sigma / sqrt(self.tau * self.sigma)

    @property
    def P0(self) -> float:
        """Common period of the resonant torus flow."""
        return 2 * pi * sqrt(self.tau * self.sigma)

    @property
    def b_psi(self) -> float:
        """Sensitivity of the relative-phase rate to virtual stiffness detuning:
        d(psi)/dt = -b_psi * eps + ...,  eps = omega_z^2 - 1."""
        return (self.tau - self.sigma) / (2 * self.S)

    @property
    def psi_abs(self) -> float:
        """Relative phase (mod 2 pi) for which the closed orbit passes through
        the host origin when A+ = A- (absorption).  See Lemma 'absorption'."""
        return ((self.tau + self.sigma) * pi / 2) % (2 * pi)

    @property
    def absorbing_impulse(self) -> bool:
        """CCTA Theorem 1: impulse response reaches host origin iff
        tau - sigma = 2 mod 4."""
        return (self.tau - self.sigma) % 4 == 2


NOMINAL = VMSParams(11, 9)
