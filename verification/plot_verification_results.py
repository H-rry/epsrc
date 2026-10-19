import os
import numpy as np
import matplotlib.pyplot as plt
from squirmer_drop import stokes as fluid
from squirmer_drop.paths import VERIFICATION_DATA_DIR, VERIFICATION_FIGURES_DIR

# plot style
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

def generate_particle_plots(fig_dir):
    print("Generating Particle comparison plots...")
    k_vals = np.array([0.001, 0.01, 0.02, 0.1, 0.9, 0.95, 0.99, 0.999])
    lambda_val = 2.0
    N = 5

    # 1. Algebraic Verification (Sheet assumption G_1^i = -1/(12 pi lambda))
    Ud_solver_sheet = []
    Up_solver_sheet = []

    # 2. Corrected Physical System (G_1^i = -1/(12 pi lambda k))
    Ud_solver_phys = []
    Up_solver_phys = []

    for k in k_vals:
        A = fluid.assemble_A(N, k, lambda_val)

        # Sheet Solve
        b_sheet = fluid.assemble_b(N, Ma=0.0, mode='particle', lambda_val=lambda_val, k=k)
        b_sheet[5 * N + 2] = -1.0 / (12.0 * np.pi * lambda_val)
        x_sheet = np.linalg.solve(A, b_sheet)
        Ud_solver_sheet.append(x_sheet[5 * N + 1])
        Up_solver_sheet.append(x_sheet[5 * N + 1] + x_sheet[5 * N + 2])

        # Physical Solve
        b_phys = fluid.assemble_b(N, Ma=0.0, mode='particle', lambda_val=lambda_val, k=k)
        x_phys = np.linalg.solve(A, b_phys)
        Ud_solver_phys.append(x_phys[5 * N + 1])
        Up_solver_phys.append(x_phys[5 * N + 1] + x_phys[5 * N + 2])

    Ud_solver_sheet = np.array(Ud_solver_sheet)
    Up_solver_sheet = np.array(Up_solver_sheet)
    Ud_solver_phys = np.array(Ud_solver_phys)
    Up_solver_phys = np.array(Up_solver_phys)

    # Analytical Formulas (Sheet Derivations)
    denom = 12.0 * np.pi * (2.0 * (k_vals**5) * (lambda_val - 1.0) + 3.0 * lambda_val + 2.0)
    Ud_anal = (-4.0 * (k_vals**5) * lambda_val + 4.0 * (k_vals**5) + 5.0 * (k_vals**2) - 6.0 * lambda_val - 9.0) / denom

    Up_num = (-4.0*(k_vals**6)*(lambda_val**2) + 8.0*(k_vals**6)*lambda_val - 4.0*(k_vals**6)
               - 9.0*(k_vals**5)*lambda_val + 9.0*(k_vals**5) + 10.0*(k_vals**3)*lambda_val - 10.0*(k_vals**3)
               - 6.0*k_vals*(lambda_val**2) - 3.0*k_vals*lambda_val + 9.0*k_vals - 6.0*lambda_val - 4.0)
    Up_anal = Up_num / (k_vals * lambda_val * denom)


    # Create the figures
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Left Plot: Droplet Velocity Ud
    axs[0].plot(k_vals, Ud_solver_sheet, 'o', color='#2563eb', markersize=9, label='Solver (Sheet Assumption)')
    axs[0].plot(k_vals, Ud_anal, '--', color='#ef4444', linewidth=2.0, label='Analytical (Sheet Eq 16a)')
    axs[0].set_xscale('log')
    axs[0].set_xlabel('Core Size Ratio ($k$)', fontsize=12, fontweight='bold')
    axs[0].set_ylabel('Droplet Velocity ($U_d$)', fontsize=12, fontweight='bold')
    axs[0].set_title('Droplet Velocity $U_d$ Comparison', fontsize=13, fontweight='bold', pad=12)
    axs[0].legend(frameon=True, facecolor='#ffffff')
    axs[0].grid(True, which="both", alpha=0.3)

    # Right Plot: Particle Velocity Up
    axs[1].plot(k_vals, Up_solver_sheet, 'o', color='#2563eb', markersize=9, label='Solver (Sheet Assumption)')
    axs[1].plot(k_vals, Up_anal, '--', color='#ef4444', linewidth=2.0, label='Analytical (Sheet Eq 16b)')
    axs[1].set_xscale('log')
    axs[1].set_yscale('symlog')
    axs[1].set_xlabel('Core Size Ratio ($k$)', fontsize=12, fontweight='bold')
    axs[1].set_ylabel('Particle Velocity ($U_p$)', fontsize=12, fontweight='bold')
    axs[1].set_title('Particle Velocity $U_p$ Comparison', fontsize=13, fontweight='bold', pad=12)
    axs[1].legend(frameon=True, facecolor='#ffffff')
    axs[1].grid(True, which="both", alpha=0.3)

    plt.suptitle('Clean Drop Particle Limits Verification (Ma = 0, $\lambda$ = 2.0)', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    plot_path = os.path.join(fig_dir, "particle_verification_curves.png")
    plt.savefig(plot_path, dpi=250)
    print(f"Saved particle comparison plot to: {plot_path}")
    plt.close()

    # Create the corrected physical plot
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Left Plot: Droplet Velocity Ud
    axs[0].plot(k_vals, Ud_solver_phys, 's', color='#10b981', markersize=8, label='Solver (Corrected Physical)')
    axs[0].plot(k_vals, Ud_solver_sheet, '--', color='#2563eb', linewidth=1.5, label='Solver (Sheet Assumption)')
    axs[0].set_xscale('log')
    axs[0].set_xlabel('Core Size Ratio ($k$)', fontsize=12, fontweight='bold')
    axs[0].set_ylabel('Droplet Velocity ($U_d$)', fontsize=12, fontweight='bold')
    axs[0].set_title('Droplet Velocity $U_d$ comparison', fontsize=13, fontweight='bold', pad=12)
    axs[0].legend(frameon=True, facecolor='#ffffff')
    axs[0].grid(True, which="both", alpha=0.3)

    # Right Plot: Particle Velocity Up
    axs[1].plot(k_vals, Up_solver_phys, 's', color='#10b981', markersize=8, label='Solver (Corrected Physical)')
    axs[1].plot(k_vals, Up_solver_sheet, '--', color='#2563eb', linewidth=1.5, label='Solver (Sheet Assumption)')
    axs[1].set_xscale('log')
    axs[1].set_yscale('symlog')
    axs[1].set_xlabel('Core Size Ratio ($k$)', fontsize=12, fontweight='bold')
    axs[1].set_ylabel('Particle Velocity ($U_p$)', fontsize=12, fontweight='bold')
    axs[1].set_title('Particle Velocity $U_p$ Comparison', fontsize=13, fontweight='bold', pad=12)
    axs[1].legend(frameon=True, facecolor='#ffffff')
    axs[1].grid(True, which="both", alpha=0.3)

    plt.suptitle('Physical Correction vs. Sheet Assumption (Particle)', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    plot_path_corr = os.path.join(fig_dir, "particle_physical_correction.png")
    plt.savefig(plot_path_corr, dpi=250)
    print(f"Saved physical comparison plot to: {plot_path_corr}")
    plt.close()

def generate_squirmer_plots(fig_dir):
    print("Generating Squirmer comparison plots...")
    pe_vals = np.array([0.1, 0.01, 0.001, 0.0001])
    k = 0.2
    lambda_val = 2.0
    Ma = 0.1
    A_1 = 0.1
    B_1 = 0.1

    # 1. O(1) Swimmer and Drop velocities (Eq 4.3 and 4.4)
    denom_0 = (6.0 * lambda_val - 6.0) * (k**5) + 9.0 * lambda_val + 6.0
    U0_D = 10.0 * (k**3) * lambda_val * (A_1 + B_1) / denom_0
    U0_S = (-12.0 * (lambda_val - 1.0) * (A_1 + B_1/2.0) * (k**5)
            + 10.0 * (k**3) * (A_1 + B_1) * (lambda_val - 1.0)
            - 3.0 * (A_1 - 2.0*B_1) * (lambda_val + 2.0/3.0)) / denom_0

    # 2. O(Pe) Correction coefficients (Eq 4.5 and 4.6)
    denom_1 = ((lambda_val - 1.0) * (k**5) + 1.5 * lambda_val + 1.0) ** 2
    U1_D_coeff = -5.0 * lambda_val * (A_1 + B_1) * (k**3) * (1.0 - k**5) / (6.0 * denom_1)
    U1_S_coeff = -25.0 * (k**3) * lambda_val * (1.0 - k) * (k + 1.0) * (A_1 + B_1) / (12.0 * denom_1)

    # Solver values from sweeps
    UD_solver = []
    US_solver = []

    base_dir = os.path.join(VERIFICATION_DATA_DIR, "analytical_solution_verification_squirmer", "Ma=0.1")
    for Pe in pe_vals:
        path = os.path.join(base_dir, f"Pe={Pe}/N=5/data_lambda=2.0_k=0.2.csv")
        if not os.path.exists(path):
            path = os.path.join(base_dir, f"Pe={Pe}/N=5/data_lambda=2_k=0.2.csv")

        x_fluid = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split(",")
                if len(parts) >= 3 and parts[2]:
                    x_fluid.append(float(parts[2]))
        UD_solver.append(x_fluid[26])
        US_solver.append(x_fluid[26] + x_fluid[27])

    UD_solver = np.array(UD_solver)
    US_solver = np.array(US_solver)

    # Asymptotic curves
    pe_dense = np.logspace(-4, -1, 100)
    UD_asympt_dense = U0_D + pe_dense * Ma * U1_D_coeff
    US_asympt_dense = U0_S + pe_dense * Ma * U1_S_coeff

    # 1. Comparison Plot
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Drop Velocity UD
    axs[0].plot(pe_dense, UD_asympt_dense, '--', color='#ef4444', linewidth=2.0, label=r'($U_{0,D} + Pe_s \cdot Ma \cdot U_{1,D}$)')
    axs[0].plot(pe_vals, UD_solver, 'o', color='#2563eb', markersize=9, label='Solver (Numerical)')
    axs[0].set_xscale('log')
    axs[0].set_xlabel('Peclet Number ($Pe_s$)', fontsize=12, fontweight='bold')
    axs[0].set_ylabel('Drop Velocity ($U_D$)', fontsize=12, fontweight='bold')
    axs[0].set_title('Drop Velocity $U_D$ Comparison', fontsize=13, fontweight='bold', pad=12)
    axs[0].legend(frameon=True, facecolor='#ffffff')
    axs[0].grid(True, which="both", alpha=0.3)

    # Right: Squirmer Velocity US
    axs[1].plot(pe_dense, US_asympt_dense, '--', color='#ef4444', linewidth=2.0, label=r'($U_{0,S} + Pe_s \cdot Ma \cdot U_{1,S}$)')
    axs[1].plot(pe_vals, US_solver, 'o', color='#2563eb', markersize=9, label='Solver (Numerical)')
    axs[1].set_xscale('log')
    axs[1].set_xlabel('Peclet Number ($Pe_s$)', fontsize=12, fontweight='bold')
    axs[1].set_ylabel('Squirmer Velocity ($U_S$)', fontsize=12, fontweight='bold')
    axs[1].set_title('Squirmer Velocity $U_S$ Comparison', fontsize=13, fontweight='bold', pad=12)
    axs[1].legend(frameon=True, facecolor='#ffffff')
    axs[1].grid(True, which="both", alpha=0.3)

    plt.suptitle('Small Peclet Squirmer Limits Verification (Ma = 0.1, k = 0.2, $\lambda$ = 2.0)', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    plot_path = os.path.join(fig_dir, "squirmer_verification_curves.png")
    plt.savefig(plot_path, dpi=250)
    print(f"Saved squirmer comparison plot to: {plot_path}")
    plt.close()

    # 2. Log-Log Error plot
    UD_asympt_vals = U0_D + pe_vals * Ma * U1_D_coeff
    US_asympt_vals = U0_S + pe_vals * Ma * U1_S_coeff

    err_UD = np.abs(UD_solver - UD_asympt_vals)
    err_US = np.abs(US_solver - US_asympt_vals)

    plt.figure(figsize=(8, 6))
    plt.loglog(pe_vals, err_UD, 'o-', color='#10b981', linewidth=2.5, markersize=8, label=r'Drop Error ($|U_{D,\text{num}} - U_{D,\text{asympt}}|$)')
    plt.loglog(pe_vals, err_US, 's-', color='#8b5cf6', linewidth=2.5, markersize=8, label=r'Swimmer Error ($|U_{S,\text{num}} - U_{S,\text{asympt}}|$)')

    # Add a reference slope of 1.0 (indicating linear error scaling in Pe_s)
    ref_x = np.array([1e-4, 1e-1])
    ref_y = ref_x * (err_US[0] / pe_vals[0])
    plt.loglog(ref_x, ref_y, 'k:', alpha=0.5, label='Reference Slope (Slope = 1.0)')

    plt.xlabel('Peclet Number ($Pe_s$)', fontsize=12, fontweight='bold')
    plt.ylabel('Absolute Error', fontsize=12, fontweight='bold')
    plt.title('Asymptotic Convergence Error Scaling', fontsize=14, fontweight='bold', pad=15)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(frameon=True, facecolor='#ffffff')
    plt.tight_layout()

    plot_path_err = os.path.join(fig_dir, "squirmer_error_scaling.png")
    plt.savefig(plot_path_err, dpi=250)
    print(f"Saved squirmer error scaling plot to: {plot_path_err}")
    plt.close()

if __name__ == "__main__":
    fig_dir = VERIFICATION_FIGURES_DIR
    os.makedirs(fig_dir, exist_ok=True)

    generate_particle_plots(fig_dir)
    generate_squirmer_plots(fig_dir)
