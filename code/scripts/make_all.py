"""Regenerate every figure and table of the paper.

    cd code && python scripts/make_all.py            # uses cached simulations
    cd code && python scripts/make_all.py --force    # re-run all simulations
"""
import subprocess
import sys
import time
import os

HERE = os.path.dirname(os.path.abspath(__file__))
steps = [
    ["run_simulations.py"] + (["--force"] if "--force" in sys.argv else []),
    ["fig_tuning.py"],
    ["fig_absorption.py"],
    ["fig_delay.py"],
    ["fig_retuning.py"],
    ["fig_averaged.py"],
    ["fig_sim.py"],
]
for s in steps:
    t0 = time.time()
    print(">>", " ".join(s))
    subprocess.run([sys.executable, os.path.join(HERE, s[0])] + s[1:], check=True, cwd=HERE)
    print(f"   done in {time.time() - t0:.1f} s")
