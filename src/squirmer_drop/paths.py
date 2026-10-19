"""
Central definition of where data, figures and per-run outputs live in the repository.
All scripts import their locations from here instead of building paths relative to themselves.
"""

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Committed summary datasets and publication figures
DATA_DIR = os.path.join(REPO_ROOT, "data")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")
RATIO_SWEEPS_DIR = os.path.join(DATA_DIR, "ratio_sweeps")
CRITICAL_MA_DIR = os.path.join(DATA_DIR, "critical_Ma")
CRITICAL_MA_GRID_DIR = os.path.join(CRITICAL_MA_DIR, "grid")
PARTICLE_DATA_DIR = os.path.join(DATA_DIR, "particle")
RATIO_FIGURES_DIR = os.path.join(FIGURES_DIR, "ratio")
CRITICAL_MA_FIGURES_DIR = os.path.join(FIGURES_DIR, "critical_Ma")

# Solver verification against analytical limits
VERIFICATION_DIR = os.path.join(REPO_ROOT, "verification")
VERIFICATION_DATA_DIR = os.path.join(VERIFICATION_DIR, "data")
VERIFICATION_FIGURES_DIR = os.path.join(VERIFICATION_DIR, "figures")

# Bulky regenerable outputs (gitignored): per-run state vectors and exploratory plots
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs")
RUNS_DIR = os.path.join(OUTPUT_DIR, "runs")
OUTPUT_FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")


def mode_label(A_1, B_1):
    """Label used for a squirmer mode in file and folder names, e.g. 'A1=0_B1=1'."""
    return f"A1={A_1:.0f}_B1={B_1:.0f}"


def critical_ma_dir(A_1, B_1):
    """Directory holding critical-Ma CSVs for a squirmer mode."""
    return os.path.join(CRITICAL_MA_DIR, mode_label(A_1, B_1))


def critical_ma_lambda_figures_dir(A_1, B_1, lambda_val):
    """Per-lambda exploratory plots from the critical-Ma search (gitignored)."""
    return os.path.join(OUTPUT_FIGURES_DIR, "critical_Ma", mode_label(A_1, B_1), f"lambda={lambda_val}")
