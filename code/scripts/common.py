"""Shared plotting style and paths."""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(CODE)
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(CODE, "results")
TAB = os.path.join(ROOT, "paper", "tables")
for d in (FIG, RES, TAB):
    os.makedirs(d, exist_ok=True)
sys.path.insert(0, CODE)

# Automatica: column width 8.4 cm, text width 17.4 cm (verify with autart.cls)
COL = 3.31
FULL = 6.85

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 0.9,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "mathtext.fontset": "cm",
})

# colour-blind-safe palette (Okabe-Ito)
C = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
    "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "black": "#000000",
    "grey": "#7f7f7f",
}


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))
