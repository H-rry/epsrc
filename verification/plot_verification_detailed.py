import os
import numpy as np
import matplotlib.pyplot as plt
from squirmer_drop import stokes as fluid
from squirmer_drop import analytical_squirmer
from squirmer_drop.paths import VERIFICATION_FIGURES_DIR

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

def generate_detailed_particle_plots(fig_dir):
    print("Generating detailed Particle plots...")
    lambda_val = 2.0
    N = 5

    # Generate continuous k values
    k_dense_small = np.logspace(-4, -1, 500)
    k_dense_large = np.linspace(0.8, 0.999999, 500)

    # EVALUATE CONTINUOUS ANALYTICAL CURVES (Sheet assumptions)
    def get_sheet_analytical(k_arr):
        denom = 12.0 * np.pi * (2.0 * (k_arr**5) * (lambda_val - 1.0) + 3.0 * lambda_val + 2.0)
        Ud = (-4.0 * (k_arr**5) * lambda_val + 4.0 * (k_arr**5) + 5.0 * (k_arr**2) - 6.0 * lambda_val - 9.0) / denom
        Up_num = (-4.0*(k_arr**6)*(lambda_val**2) + 8.0*(k_arr**6)*lambda_val - 4.0*(k_arr**6)
                   - 9.0*(k_arr**5)*lambda_val + 9.0*(k_arr**5) + 10.0*(k_arr**3)*lambda_val - 10.0*(k_arr**3)
                   - 6.0*k_arr*(lambda_val**2) - 3.0*k_arr*lambda_val + 9.0*k_arr - 6.0*lambda_val - 4.0)
        Up = Up_num / (k_arr * lambda_val * denom)
        return Ud, Up

    Ud_anal_small, Up_anal_small = get_sheet_analytical(k_dense_small)
    Ud_anal_large, Up_anal_large = get_sheet_analytical(k_dense_large)

    # run numerical solver continuously
    def run_solver_continuous(k_arr):
        Ud_list = []
        Up_list = []
        for k in k_arr:
            A = fluid.assemble_A(N, k, lambda_val)
            b = fluid.assemble_b(N, Ma=0.0, mode='particle', lambda_val=lambda_val, k=k)
            x = np.linalg.solve(A, b)
            Ud_list.append(x[5 * N + 1])
            Up_list.append(x[5 * N + 1] + x[5 * N + 2])
        return np.array(Ud_list), np.array(Up_list)

    # Get continuous numerical results
    Ud_num_phys_small, Up_num_phys_small = run_solver_continuous(k_dense_small)
    Ud_num_phys_large, Up_num_phys_large = run_solver_continuous(k_dense_large)

    # Discrete verification sweep points (to show as markers)
    k_markers_small = np.array([0.001, 0.01, 0.02, 0.1])
    k_markers_large = np.unique(np.concatenate(([0.9, 0.95, 0.99, 0.999], 1.0 - np.logspace(-2, -6, 10))))

    Ud_marker_phys_small, Up_marker_phys_small = run_solver_continuous(k_markers_small)
    Ud_marker_phys_large, Up_marker_phys_large = run_solver_continuous(k_markers_large)

    # PLOT: CORRECTED PHYSICAL VELOCITIES VERIFICATION (1x3 grid)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))

    # Left Panel: Small-core Up (linear scale plotted against 1/k)
    inv_k_dense = 1.0 / k_dense_small
    inv_k_markers = 1.0 / k_markers_small

    axs[0].plot(inv_k_markers, -Up_marker_phys_small, 's', color='#f59e0b', markersize=8, label='Markers $-U_p$')

    ref_k = np.logspace(-4, -1, 10)
    ref_Up = 1.0 / (6.0 * np.pi * lambda_val * ref_k)
    axs[0].plot(1.0 / ref_k, ref_Up, 'k:', alpha=0.6, label='$1/(6\\pi\\lambda k)$')

    axs[0].set_xlabel('Inverse Core Size ($1/k$)', fontsize=11, fontweight='bold')
    axs[0].set_ylabel('Particle Velocity Magnitude ($-U_p$)', fontsize=11, fontweight='bold')
    axs[0].set_title('Small Core: Particle Velocity $U_p$ vs $1/k$', fontsize=12, fontweight='bold', pad=10)
    axs[0].grid(True, which="both", alpha=0.3)
    axs[0].legend(frameon=True, facecolor='#ffffff')

    # Middle Panel: Small-core Ud (linear scale plotted against 1/k)
    axs[1].plot(inv_k_dense, Ud_anal_small, 'r--', linewidth=1.5, label='Analytical $U_d$ (Sheet)')
    axs[1].plot(inv_k_markers, Ud_marker_phys_small, 'o', color='#ef4444', markersize=8, label='Markers $U_d$')

    small_k_limit = -(6.0 * lambda_val + 9.0) / (12.0 * np.pi * (3.0 * lambda_val + 2.0))
    axs[1].axhline(y=small_k_limit, color='k', linestyle=':', alpha=0.6, label='Limit $k \\to 0$')
    axs[1].set_xlim(5, 1000)
    axs[1].set_ylim(-0.08, -0.04)
    axs[1].set_xlabel('Inverse Core Size ($1/k$)', fontsize=11, fontweight='bold')
    axs[1].set_ylabel('Droplet Velocity ($U_d$)', fontsize=11, fontweight='bold')
    axs[1].set_title('Small Core: Droplet Velocity $U_d$ vs $1/k$', fontsize=12, fontweight='bold', pad=10)
    axs[1].grid(True, which="both", alpha=0.3)
    axs[1].legend(frameon=True, facecolor='#ffffff')

    # Right Panel: Large k (linear)
    axs[2].plot(k_dense_large, Ud_num_phys_large, '-', color='#3b82f6', linewidth=2, label='Solver $U_d$ (Physical)')
    axs[2].plot(k_dense_large, Up_num_phys_large, '-', color='#8b5cf6', linewidth=2, label='Solver $U_p$ (Physical)')
    axs[2].plot(k_markers_large, Ud_marker_phys_large, 'o', color='#ef4444', markersize=8, label='Markers $U_d$')
    axs[2].plot(k_markers_large, Up_marker_phys_large, 's', color='#f59e0b', markersize=8, label='Markers $U_p$')
    axs[2].axhline(y=-1.0 / (6.0 * np.pi), color='k', linestyle=':', alpha=0.6, label='Limit $-1/(6\\pi)$')
    axs[2].set_xlim(0.8, 1.0)
    axs[2].set_xlabel('Core Size Ratio ($k$)', fontsize=11, fontweight='bold')
    axs[2].set_ylabel('Velocity', fontsize=11, fontweight='bold')
    axs[2].set_title('Large Core Limit ($U_p \\to U_d$)', fontsize=12, fontweight='bold', pad=10)
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(frameon=True, facecolor='#ffffff')

    plt.suptitle('Physical Particle Solution Verification (Ma = 0, $\\lambda$ = 2.0)', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    plot_path = os.path.join(fig_dir, "particle_verification_detailed.png")
    plt.savefig(plot_path, dpi=250)
    print(f"Saved detailed particle comparison plot to: {plot_path}")
    plt.close()

def generate_detailed_squirmer_plots(fig_dir):
    print("Generating detailed Squirmer plots...")
    from squirmer_drop import surfactant

    pe_vals = np.array([0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1.0])
    k = 0.2
    lambda_val = 2.0
    Ma = 0.1
    A_1 = 0.1
    B_1 = 0.1
    N = 5

    # asymptotic U = U_0 + Pe * Ma * U_1 (U_1 per unit of the solver's Ma)
    U0_D = analytical_squirmer.analytical_U_D(k, 0.0, lambda_val, A_1, B_1)
    U0_S = analytical_squirmer.analytical_U_S(k, 0.0, lambda_val, A_1, B_1)
    U1_D_coeff = analytical_squirmer.analytical_U_D(k, 1.0, lambda_val, A_1, B_1, Pe=1.0) - U0_D
    U1_S_coeff = analytical_squirmer.analytical_U_S(k, 1.0, lambda_val, A_1, B_1, Pe=1.0) - U0_S

    # Run coupling on-the-fly to get numerical results for all Pe
    UD_solver = []
    US_solver = []

    A_n = 0.1 ** np.arange(1, N + 1)
    B_n = 0.1 ** np.arange(1, N + 1)
    A_mat = fluid.assemble_A(N, k, lambda_val)

    for Pe in pe_vals:
        c_n = np.zeros(N)
        omega = 0.1
        max_iter = 300
        tol = 1e-6

        x_fluid = np.zeros(5 * N + 3)
        for iteration in range(max_iter):
            b = fluid.assemble_b(N, Ma, c_n=c_n, mode='squirmer', lambda_val=lambda_val, k=k, A_n=A_n, B_n=B_n)
            x_fluid = np.linalg.solve(A_mat, b)

            b_n_new = np.zeros(N)
            # Corrected extraction for mode 1
            U_D = x_fluid[5 * N + 1]
            G_o1 = x_fluid[4]
            H_o1 = x_fluid[5 * N]
            b_n_new[0] = U_D - 1.5 * G_o1 + 1.5 * H_o1

            # Corrected extraction for modes n >= 2
            for n in range(2, N + 1):
                G_on = x_fluid[5 * (n - 1) + 4]
                b_n_new[n - 1] = -2.0 * (2.0 * n + 1.0) / (n * (n + 1.0)) * G_on

            c_n_new = surfactant.solve_c_n(b_n_new, Pe)
            diff = np.linalg.norm(c_n_new - c_n)
            if diff < tol:
                break
            c_n = omega * c_n_new + (1.0 - omega) * c_n

        UD_solver.append(x_fluid[5 * N + 1])
        US_solver.append(x_fluid[5 * N + 1] + x_fluid[5 * N + 2])

    UD_solver = np.array(UD_solver)
    US_solver = np.array(US_solver)

    # Continuous asymptotic lines (plotted up to Pe = 1.0)
    pe_dense = np.logspace(-5, 0.0, 500)
    UD_asympt_dense = U0_D + pe_dense * Ma * U1_D_coeff
    US_asympt_dense = U0_S + pe_dense * Ma * U1_S_coeff

    # Create plot
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Drop velocity UD
    axs[0].plot(pe_dense, UD_asympt_dense, '--', color='#ef4444', linewidth=2.0, label=r'$U_{0,D} + Pe_s \cdot Ma \cdot U_{1,D}$')
    axs[0].plot(pe_vals, UD_solver, 'o', color='#2563eb', markersize=8, label='Solver (Numerical)')
    axs[0].axhline(y=U0_D, color='k', linestyle=':', alpha=0.5, label='Leading-order $U_{0,D}$')
    axs[0].set_xscale('log')
    axs[0].set_xlim(1e-5, 1.0)
    axs[0].set_xlabel('Peclet Number ($Pe_s$)', fontsize=12, fontweight='bold')
    axs[0].set_ylabel('Drop Velocity ($U_D$)', fontsize=12, fontweight='bold')
    axs[0].set_title('Drop Velocity $U_D$ vs $Pe_s$', fontsize=13, fontweight='bold', pad=12)
    axs[0].legend(frameon=True, facecolor='#ffffff')
    axs[0].grid(True, which="both", alpha=0.3)

    # Right: Squirmer velocity US
    axs[1].plot(pe_dense, US_asympt_dense, '--', color='#ef4444', linewidth=2.0, label=r'$U_{0,S} + Pe_s \cdot Ma \cdot U_{1,S}$')
    axs[1].plot(pe_vals, US_solver, 'o', color='#2563eb', markersize=8, label='Solver (Numerical)')
    axs[1].axhline(y=U0_S, color='k', linestyle=':', alpha=0.5, label='Leading-order $U_{0,S}$')
    axs[1].set_xscale('log')
    axs[1].set_xlim(1e-5, 1.0)
    axs[1].set_xlabel('Peclet Number ($Pe_s$)', fontsize=12, fontweight='bold')
    axs[1].set_ylabel('Squirmer Velocity ($U_S$)', fontsize=12, fontweight='bold')
    axs[1].set_title('Squirmer Velocity $U_S$ vs $Pe_s$', fontsize=13, fontweight='bold', pad=12)
    axs[1].legend(frameon=True, facecolor='#ffffff')
    axs[1].grid(True, which="both", alpha=0.3)

    plt.suptitle('Continuous Squirmer Solutions Verification (Ma = 0.1, k = 0.2, $\\lambda$ = 2.0)', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    plot_path = os.path.join(fig_dir, "squirmer_verification_detailed.png")
    plt.savefig(plot_path, dpi=250)
    print(f"Saved detailed squirmer comparison plot to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    fig_dir = VERIFICATION_FIGURES_DIR
    os.makedirs(fig_dir, exist_ok=True)

    generate_detailed_particle_plots(fig_dir)
    generate_detailed_squirmer_plots(fig_dir)
