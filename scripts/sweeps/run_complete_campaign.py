#!/usr/bin/env python3
"""
Runs the extended sweeps end to end:
  1. extended_ma_sweeps.py           larger Ma values for the ratio sweeps
  2. sweep_critical_Ma_extended.py   critical-Ma grid for A_1 = 1, B_1 = 1, and Ma_c vs Pe
  3. analyze_and_plot_extended.py    figures from the above
A failed stage is reported and the next one still runs.
"""

import os
import sys
import traceback

# stages live in this folder and in ../plotting
script_dir = os.path.dirname(os.path.abspath(__file__))
for d in (script_dir, os.path.join(script_dir, "..", "plotting")):
    if d not in sys.path:
        sys.path.insert(0, d)

import extended_ma_sweeps
import sweep_critical_Ma_extended
import analyze_and_plot_extended


def main():
    stages = [
        ("extended Ma sweeps", extended_ma_sweeps.run_extended_sweeps),
        ("critical Ma grid (A_1=1, B_1=1)", sweep_critical_Ma_extended.run_pusher_puller_grid),
        ("critical Ma vs Pe", sweep_critical_Ma_extended.run_pe_scaling_sweeps),
        ("plots", analyze_and_plot_extended.main),
    ]
    for name, stage_fn in stages:
        print(f"\n--- {name} ---")
        try:
            stage_fn()
        except Exception as e:
            print(f"{name} failed: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
