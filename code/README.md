# code/

Python package `vms` and scripts reproducing every number, figure and table of the paper.

| Module | Content | Paper |
|---|---|---|
| `vms/params.py` | resonant pair, n, Ω±, S, P₀, ψ*, β_ψ, b_ψ | Sec. 2–3 |
| `vms/modal.py` | exact modal–polar coordinates, frozen-system projector, energies | Lemma 1 |
| `vms/absorption.py` | absorbing phases Ψ*(e_A), minimal host radius on an orbit | Thm 1 |
| `vms/delay.py` | quasi-polynomial, closed-form root count, collocation spectrum | Thm 2 |
| `vms/zoh.py` | exact lifted ZOH map with sample delays | Prop. 2 |
| `vms/averaged.py` | first-order averaged closed loop | Lemma 4, Thm 3 |
| `vms/sim.py` | exact sampled-data simulator with delay, ZOH, gain error, noise, play; ECC and proposed controllers | Sec. 6 |

| Script | Output |
|---|---|
| `scripts/fig_tuning.py` | `figures/fig_tuning.pdf` |
| `scripts/fig_absorption.py` | `figures/fig_absorption.pdf` |
| `scripts/fig_delay.py` | `figures/fig_delay.pdf` |
| `scripts/fig_retuning.py` | `figures/fig_retuning.pdf` |
| `scripts/fig_averaged.py` | `figures/fig_averaged.pdf` |
| `scripts/run_simulations.py` | `results/*.pkl`, `paper/tables/sim_metrics.tex` |
| `scripts/fig_sim.py` | `figures/fig_sim_portraits.pdf`, `figures/fig_sim_errors.pdf` |
| `scripts/make_all.py` | everything |

Tests: `python -m tests.check_identities`, `python -m tests.check_theorems` (run from `code/`).
The scripts add `code/` to `sys.path` themselves; tests need `PYTHONPATH=.` or running
from `code/` with `python -m`.
