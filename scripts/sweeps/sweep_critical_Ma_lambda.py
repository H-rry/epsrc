"""
Critical Marangoni number vs viscosity ratio for the neutral squirmer (A_1=0, B_1=1).

For each lambda, finds the Ma at which the O(Pe Ma) asymptotic solution deviates from the
coupled solver by 2% (averaged over k), for both U_s and U_d. Writes the summary CSV to
data/critical_Ma/A1=0_B1=1/ and summary plots to figures/critical_Ma/.
"""

import os
import numpy as np

from squirmer_drop import coupling as ch
from squirmer_drop.paths import critical_ma_dir, mode_label, CRITICAL_MA_FIGURES_DIR

def main():
    import matplotlib.pyplot as plt

    # RUN_MODE: 'test' for 5 lambdas, 'sweep' for high-res sweep of 100 lambdas
    RUN_MODE = 'test'

    # Define squirmer mode for neutral squirmer (A_1=0, B_1=1)
    N = 5
    A_n = np.zeros(N); A_n[0] = 0.0
    B_n = np.zeros(N); B_n[0] = 1.0

    if RUN_MODE == 'test':
        Lambda_list = [0.1, 0.5, 1.0, 2.0, 10.0]
    else:
        Lambda_list = [0.01, 0.05, 0.2, 0.5, 1, 2, 5, 10, 50, 100, 200]

    analysis_dir = critical_ma_dir(A_n[0], B_n[0])
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(CRITICAL_MA_FIGURES_DIR, exist_ok=True)
    label = mode_label(A_n[0], B_n[0])

    critical_mas_Us = []
    critical_mas_Ud = []

    for idx, L in enumerate(Lambda_list):
        L_rounded = round(L, 4)
        print(f"lambda = {L_rounded} ({idx+1}/{len(Lambda_list)})")

        print("--- Searching for Us critical Marangoni ---")
        Ma_crit_Us, _ = ch.find_Ma_critical_squirmer(lambda_val=L, vel_type='Us', starting_ma=0.001)
        critical_mas_Us.append(Ma_crit_Us)

        print("--- Searching for Ud critical Marangoni ---")
        Ma_crit_Ud, _ = ch.find_Ma_critical_squirmer(lambda_val=L, vel_type='Ud', starting_ma=Ma_crit_Us)
        critical_mas_Ud.append(Ma_crit_Ud)

    summary_path = os.path.join(analysis_dir, "critical_Ma_summary.csv")
    with open(summary_path, 'w') as f:
        f.write("lambda,critical_Ma_Us,critical_Ma_Ud\n")
        for idx, L in enumerate(Lambda_list):
            f.write(f"{L},{critical_mas_Us[idx]},{critical_mas_Ud[idx]}\n")
    print(f"\nSaved critical Ma summary to: {summary_path}")

    # Summary Us vs lambda
    plt.close('all')
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.plot(Lambda_list, critical_mas_Us, 'o-', color='#2563eb', linewidth=2.0, markersize=6, label=r"Critical Marangoni $Ma_{\text{crit}, U_s}$")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"Viscosity Ratio $\lambda$ (Log Scale)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel(r"Critical Marangoni Number $Ma_{\text{crit}}$ (Log Scale)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title("Validity Range of Analytical Solution for Squirmer Velocity $U_s$\n" + r"Critical Marangoni $Ma_{\text{crit}}$ vs. $\lambda$ (Relative Error = 2%)", fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=True, facecolor='#ffffff', fontsize=11)
    plt.tight_layout()
    plot_path_Us = os.path.join(CRITICAL_MA_FIGURES_DIR, f"Ma_critical_Us_vs_lambda_{label}.png")
    plt.savefig(plot_path_Us, dpi=250)
    plt.close(fig)
    print(f"Saved summary Us plot to: {plot_path_Us}")

    # Summary Ud vs lambda
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.plot(Lambda_list, critical_mas_Ud, 's-', color='#ef4444', linewidth=2.0, markersize=6, label=r"Critical Marangoni $Ma_{\text{crit}, U_d}$")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"Viscosity Ratio $\lambda$ (Log Scale)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel(r"Critical Marangoni Number $Ma_{\text{crit}}$ (Log Scale)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title("Validity Range of Analytical Solution for Drop Velocity $U_d$\n" + r"Critical Marangoni $Ma_{\text{crit}}$ vs. $\lambda$ (Relative Error = 2%)", fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=True, facecolor='#ffffff', fontsize=11)
    plt.tight_layout()
    plot_path_Ud = os.path.join(CRITICAL_MA_FIGURES_DIR, f"Ma_critical_Ud_vs_lambda_{label}.png")
    plt.savefig(plot_path_Ud, dpi=250)
    plt.close(fig)
    print(f"Saved summary Ud plot to: {plot_path_Ud}")

    # Combined Us and Ud vs lambda
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.plot(Lambda_list, critical_mas_Us, 'o-', color='#2563eb', linewidth=2.0, markersize=6, label=r"Critical Squirmer Marangoni $Ma_{\text{crit}, U_s}$")
    ax.plot(Lambda_list, critical_mas_Ud, 's-', color='#ef4444', linewidth=2.0, markersize=6, label=r"Critical Drop Marangoni $Ma_{\text{crit}, U_d}$")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"Viscosity Ratio $\lambda$ (Log Scale)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel(r"Critical Marangoni Number $Ma_{\text{crit}}$ (Log Scale)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(fr"Validity Range of Analytical Solutions ($A_1={A_n[0]:.0f}, B_1={B_n[0]:.0f}$)" + "\n" + "Critical Marangoni Number vs. $\lambda$ (Relative Error = 2%)", fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=True, facecolor='#ffffff', fontsize=11)
    plt.tight_layout()
    plot_path_combined = os.path.join(CRITICAL_MA_FIGURES_DIR, f"Ma_critical_combined_vs_lambda_{label}.png")
    plt.savefig(plot_path_combined, dpi=250)
    plt.close(fig)
    print(f"Saved summary combined plot to: {plot_path_combined}")

if __name__ == "__main__":
    main()
