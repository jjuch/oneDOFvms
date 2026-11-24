# Working notes — 1-DOF gyroscopic VMS journal paper

Last updated: 2026-11-20 (first full draft).

## 1. Story line (one paragraph)

Design (CCTA): coprime modal ratio (τ, σ) ⇒ closed Lissajous orbits; impulse response
absorbs iff τ−σ ≡ 2 (mod 4). **New geometry:** exact modal–polar coordinates
(H, e_A, ψ); absorption iff −ln(τ/σ) ≤ e_A ≤ 0 and ψ ∈ Ψ*(e_A) — every pair absorbs at
the right phase. **Failure:** any loop delay ⇒ exponential instability (closed-form root
count), fast mode pumped, slow mode damped; ZOH = extra loop delay T_s; static gains only
detune ψ; ω_z retuning tolerates only n²/8 coupling error. **Restoration:** energy loop
(normalised), amplitude loop with integral action, phase-locked loop through the common
coupling; averaged dynamics triangular and ISS; semiglobal practical ISS by averaging.

## 2. Notation (keep consistent with `paper/macros.tex`)

| Symbol | Meaning |
|---|---|
| x = (q, q̇, z, ż) | host and virtual states (ECC sign convention, z ↦ −z vs CCTA) |
| n | gyroscopic coupling, n = (τ−σ)/√(τσ) |
| Ω± | modal frequencies, Ω₊Ω₋ = 1, Ω₊−Ω₋ = n, S = Ω₊+Ω₋ = √(n²+4) |
| P₀ | common period 2π√(τσ) |
| ζ± = X± + iY± = A± e^{iθ±} | exact modal coordinates (Lemma 1, eq. (3)) |
| H, H_r, e_H | total energy, reference H(x(0)), e_H = H/H_r − 1 |
| e_A | ln(A₊/A₋) |
| ψ, ψ* | σθ₊ − τθ₋, target (τ+σ)π/2 |
| a | energy share of + mode, S Ω₊ A₊²/(2H) |
| g, m | q̇ż (energy regressor), qq̇ + zż = S A₊A₋ sin(θ₊+θ₋) |
| η_q, η_z, η_c, η_w | static gain errors, common/antisymmetric parts |
| w (= u in ECC), s, c_ψ | antisymmetric action, fast common action, phase trim |
| d = d_a + d_y, d_eq | loop delay, equivalent loop delay (1+k_y+k_a)T_s |
| α± | first-order modal rates nΩ±² d/(2S); α_Σ = α₊+α₋ |
| β_ψ, b_ψ | ψ̇ sensitivity to common coupling (2√(τσ)/S) and to ε = ω_z²−1 ((τ−σ)/(2S)) |
| ν(a), ν_min, ν_max | normalised excitation ⟨g²⟩/H², ν_min = Ω₋²/(2S²) |

## 3. Assumptions ledger

| ID | Statement | Used in |
|---|---|---|
| A-res | (τ, σ) coprime, τ > σ, **τ+σ ≥ 5** (no resonant harmonics of order ≤ 4) | Lemma 3 (PE), Lemma 4 (averaging) |
| A-n | n < √2 (concavity of ν) — always true for practical pairs | Lemma 3 |
| A-samp (Ass. 1) | exact/symplectic integration of the virtual oscillator, ZOH actuator, integer sample delays | Prop. 2, simulations |
| A-meas (Ass. 2) | measurement errors = bounded biases δ_H, δ_A, δ_ψ + zero-mean noise (2nd order); multiplicative errors in ĝ, m̂ absorbed in gains | Lemma 4, Thm 3 |
| A-scale | slow-adaptation scaling (K_E, K_A, k_P) = O(ε), (k_IA, k_I) = O(ε²), d, η = O(ε) | Lemma 4, Thm 4 |
| A-gain | K_E > n d S Ω₊⁴ (energy gain beats delay anti-damping) | Thm 3(a) |
| A-dom | ‖δ_H‖ ≤ 1 − 2h₀, h₀ ∈ (0, ½) | Thm 3(a) invariance |

## 4. Results status

| Result | Status | Numerical check |
|---|---|---|
| Lemma 1 (modal–polar coords), Lemma 2 (perturbed modal eqs) | proved | `tests/check_identities.py` |
| Thm 1 (absorption set) + Cor. 1 (mod 4) | proved | `tests/check_theorems.py`, `fig_absorption` |
| Thm 2 (delay ⇒ instability, N(d)) | proved (crossing analysis) | `tests/check_theorems.py`, `fig_delay` |
| Cor. 2 (modal signature of delay) | proved to first order | `fig_averaged` |
| Prop. 2 (ZOH = delay T_s) | proved to first order; all-T_s claim open (G3) | `tests/check_theorems.py` |
| Prop. 3 (static mismatch) | proved | — |
| Prop. 4 (retuning authority) | proved | `tests/check_theorems.py`, `fig_retuning` |
| Prop. 5 (ideal loops) | proved (LaSalle) | — |
| Lemma 3 (structural excitation) | proved | `tests/check_identities.py` |
| Lemma 4 (averaged loop) | proved at first order (term-by-term averages) | `fig_averaged` (e_A within 1–3 %) |
| Thm 3 (ISS of averaged loop) | proved; phase part only asymptotic (G5) | — |
| Thm 4 (SGP-ISS of actual loop) | **sketch**; delay/ZOH inside averaging open (G6) | simulations |

## 5. Open gaps (mirror of `\gap{}` markers in the PDF)

| ID | Gap | Suggested route | Owner |
|---|---|---|---|
| G1 | Optimal absorbing target (apex vs interior of Ψ*) | sensitivity of min H_q to (δe_A, δψ); worst case over ISS residual set | |
| G2 | Rigour of CCTA Thm 2 (inscribed-radius localisation) | explicit bounds or numerical certificate | |
| G3 | ZOH instability for all T_s | symbolic Jury / Schur–Cohn on the quartic of Φ(T_s) | |
| G4 | Second-order accuracy of e_H steady state | second-order averaging or bound | |
| G5 | Almost-global ISS of the phase loop | Angeli (2004) framework | |
| G6 | Delay and sampling inside averaging theorem | (i) FDE averaging (Lehman–Weibel 1999) or (ii) exact lifted discrete-time + Nešić–Teel | |
| G7 | Non-small gains (sims use K_E = 10) | scalar LTV energy equation + shape-drift bound | |
| G8 | Phase estimation / ripple (≈0.5 rad) | ripple budget; I/Q comparison; filtering | |
| G9 | Harvesting stage and reference energy with host damping | decaying/estimated H_r | |
| G10 | Fair ECC baseline (original I/Q estimator) + TDPC baseline | re-run with I/Q; add Hannaford–Ryu | |
| G11 | Experimental validation | 1-DOF voice-coil rig | |
| — | Verify Automatica page limit; current draft ≈ 13 pages + refs in compat layout | condense Sec. 3.3, move proofs to supplementary? | |
| — | Verify `% VERIFY` bib entries; confirm Nešić–Teel 2001 theorem number | | |
| — | The CCTA 2026 paper on "first x poles" of delay systems that a co-author remembered: identify and cite if relevant (Remark 4) | | |

## 6. Design decisions (and why)

1. **ω_z adaptation dropped, replaced by common-coupling trim** (Prop. 4): reachable set
   of ω_z retuning tolerates only n²/8 ≈ 0.5 % positive coupling error for (11,9); the
   ECC example violates this, and the ECC loop hunts (Fig. 7e). Trim authority is 2/n ≈ 10×
   larger and monotone.
2. **Integral action on e_A**: delay induces a *state-independent* drift α_Σ of e_A; a
   proportional loop leaves e_A* = Sα_Σ/K_A (validated to 1 %).
3. **Normalised laws** (divide by Ĥ instead of H_r): homogeneous of degree 0, makes the
   averaged coefficients depend on e_A only; identical to ECC laws at H = H_r.
4. **Target (e_A, ψ) = (0, ψ*)** kept for continuity with ECC; interior targets may be
   more robust in e_A (G1). The apex is quadratically insensitive to phase errors.
5. **Windowed contraction → ISS**: replaced the ECC windowed bounds by (a) exact LaSalle
   arguments for the ideal loops, (b) a structural PE lemma (no excitation assumption),
   (c) averaging + ISS cascade.

## 7. Corrections to the conference papers (already noted in the text)

* ECC eq. (48): second amplitude rate has opposite sign (energy conservation).
* ECC eq. (52): A± → √(2H_r)/S, not √(2H_r/S).
* CCTA Sec. V-A: smallest δ = 6 pair with beat ≥ 10 is (35, 29) (T_min ≈ 16.7 s), not (41, 35).
* CCTA Thm 1 is a statement about the impulse phase ψ = 0, not about the pair (Thm 1 here).

## 8. Simulation set-up (ECC example) and headline numbers

(11,9), T_s = 0.03 s, one-sample sensor/actuator delays (d_eq = 0.09 s), sensor gain −3 %,
actuator gain +6.5 %, noise σ = 0.002, play 0.003, T = 3000 s, metrics on last 25 %.
See `paper/tables/sim_metrics.tex` (generated). Proposed controller: ratio error
7.5e−6, closure 2.5e−4, median per-period min H_q/H_r 1.2e−5 (ECC: 3.8e−4, 0.089, 1.7e−5
with hunting ω_z; energy–amplitude only: 3.5e−3, 7.6e−3, 4.9e−4).
