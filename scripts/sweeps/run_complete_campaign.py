#!/usr/bin/env python3
"""
run_complete_campaign.py

Master automation script that executes the complete 5-hour scientific campaign:
1. extended_ma_sweeps.py: Sweeps larger and larger Ma values to find full dynamic ranges.
2. sweep_critical_Ma_extended.py: Computes 2D critical Ma grid for pusher/puller squirmers and multi-Pe scaling.
3. analyze_and_plot_extended.py: Analyzes optimal ranges, generates universal scaling collapse and publication plots.
"""

import os
import sys
import time
import traceback

# Stages live in this folder (sweeps) and in ../plotting
script_dir = os.path.dirname(os.path.abspath(__file__))
for d in (script_dir, os.path.join(script_dir, "..", "plotting")):
    if d not in sys.path:
        sys.path.insert(0, d)

import extended_ma_sweeps
import sweep_critical_Ma_extended
import analyze_and_plot_extended

def main():
    stages = [
        ('Stage 1: Extended Ma Sweeps', lambda: extended_ma_sweeps.run_extended_sweeps()),
        ('Stage 2A: Pusher/Puller Grid', lambda: sweep_critical_Ma_extended.run_pusher_puller_grid()),
        ('Stage 2B: Pe Scaling Sweeps', lambda: sweep_critical_Ma_extended.run_pe_scaling_sweeps()),
        ('Stage 3: Analysis & Plots', lambda: analyze_and_plot_extended.main()),
    ]
    for name, stage_fn in stages:
        try:
            print(f'\n{"="*60}')
            print(f' {name}')
            print(f'{"="*60}')
            stage_fn()
            print(f' {name} — COMPLETED')
        except Exception as e:
            print(f' {name} — FAILED: {e}')
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()
