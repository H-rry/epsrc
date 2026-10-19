import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# Paths to the two summary datasets
from squirmer_drop.paths import critical_ma_dir, CRITICAL_MA_FIGURES_DIR

analysis_dir_A1_B1 = critical_ma_dir(1, 1)
analysis_dir_A0_B1 = critical_ma_dir(0, 1)

path1 = os.path.join(analysis_dir_A1_B1, "critical_Ma_summary.csv")
path2 = os.path.join(analysis_dir_A0_B1, "critical_Ma_summary.csv")

def read_summary(path):
    Lambda_list = []
    Ma_Us = []
    Ma_Ud = []
    if not os.path.exists(path):
        return None, None, None
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            Lambda_list.append(float(row['lambda']))
            Ma_Us.append(float(row['critical_Ma_Us']))
            Ma_Ud.append(float(row['critical_Ma_Ud']))
    sorted_indices = np.argsort(Lambda_list)
    return (np.array(Lambda_list)[sorted_indices],
            np.array(Ma_Us)[sorted_indices],
            np.array(Ma_Ud)[sorted_indices])

# Read datasets
L1, Us1, Ud1 = read_summary(path1)
L2, Us2, Ud2 = read_summary(path2)

# Set up styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Inter', 'Roboto', 'Arial', 'DejaVu Sans'],
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#f8fafc',
    'axes.edgecolor': '#cbd5e1',
    'axes.labelcolor': '#1e293b',
    'xtick.color': '#475569',
    'ytick.color': '#475569',
    'text.color': '#1e293b',
    'grid.color': '#e2e8f0',
    'grid.linestyle': '--'
})

output_dir = CRITICAL_MA_FIGURES_DIR
os.makedirs(output_dir, exist_ok=True)

# PLOT COMPARISON GRAPH (All 4 curves)
if L1 is not None and L2 is not None:
    fig, ax = plt.subplots(figsize=(10, 8))
    # Solid lines for A1=1 B1=1
    ax.plot(L1, Us1, 'o-', color='#2563eb', linewidth=2.0, markersize=8, label=r"Squirmer Velocity $U_s$ ($A_1=1, B_1=1$)")
    ax.plot(L1, Ud1, 's-', color='#ef4444', linewidth=2.0, markersize=8, label=r"Drop Velocity $U_d$ ($A_1=1, B_1=1$)")
    # Dashed lines for A1=0 B1=1
    ax.plot(L2, Us2, 'o--', color='#3b82f6', linewidth=2.0, markersize=8, markerfacecolor='none', markeredgewidth=1.5, label=r"Squirmer Velocity $U_s$ ($A_1=0, B_1=1$)")
    ax.plot(L2, Ud2, 's--', color='#f87171', linewidth=2.0, markersize=8, markerfacecolor='none', markeredgewidth=1.5, label=r"Drop Velocity $U_d$ ($A_1=0, B_1=1$)")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"Viscosity Ratio $\lambda$ (Log Scale)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel(r"Critical Marangoni Number $Ma_{\text{crit}}$ (Log Scale)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title("Validity Range of Analytical Solution: Comparison\nCritical Marangoni Number vs. $\\lambda$ (Relative Error = 2%)", fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=True, facecolor='#ffffff', fontsize=11, loc='best')
    plt.tight_layout()

    comp_path = os.path.join(output_dir, "critical_Ma_comparison_A_1_B_1_vs_A_0_B_1.png")
    plt.savefig(comp_path, dpi=250)
    plt.close(fig)
    print(f"Saved comparison plot to: {comp_path}")

# plot combined graph (a1=1, b1=1)
if L1 is not None:
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.plot(L1, Us1, 'o-', color='#2563eb', linewidth=2.0, markersize=8, label=r"Squirmer Velocity $U_s$")
    ax.plot(L1, Ud1, 's-', color='#ef4444', linewidth=2.0, markersize=8, label=r"Drop Velocity $U_d$")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"Viscosity Ratio $\lambda$ (Log Scale)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel(r"Critical Marangoni Number $Ma_{\text{crit}}$ (Log Scale)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title("Validity Range of Analytical Solutions ($A_1=1, B_1=1$)\nCritical Marangoni Number vs. $\\lambda$ (Relative Error = 2%)", fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=True, facecolor='#ffffff', fontsize=11, loc='best')
    plt.tight_layout()

    combined_path1 = os.path.join(output_dir, "Ma_critical_combined_A1_B1.png")
    plt.savefig(combined_path1, dpi=250)
    plt.close(fig)

# plot combined graph (a1=0, b1=1)
if L2 is not None:
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.plot(L2, Us2, 'o-', color='#2563eb', linewidth=2.0, markersize=8, label=r"Squirmer Velocity $U_s$")
    ax.plot(L2, Ud2, 's-', color='#ef4444', linewidth=2.0, markersize=8, label=r"Drop Velocity $U_d$")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"Viscosity Ratio $\lambda$ (Log Scale)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel(r"Critical Marangoni Number $Ma_{\text{crit}}$ (Log Scale)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title("Validity Range of Analytical Solutions ($A_1=0, B_1=1$)\nCritical Marangoni Number vs. $\\lambda$ (Relative Error = 2%)", fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=True, facecolor='#ffffff', fontsize=11, loc='best')
    plt.tight_layout()

    combined_path2 = os.path.join(output_dir, "Ma_critical_combined_A0_B1.png")
    plt.savefig(combined_path2, dpi=250)
    plt.close(fig)

print("All summary, comparison, and combined plots generated successfully.")
