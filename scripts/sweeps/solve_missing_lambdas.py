import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from squirmer_drop import coupling as ch
from squirmer_drop import analytical_squirmer as tas
from squirmer_drop.paths import critical_ma_dir, critical_ma_lambda_figures_dir, CRITICAL_MA_FIGURES_DIR

analysis_dir = critical_ma_dir(1.0, 1.0)

# We need to run the critical Ma search for lambda = 0.1 and 10.0 under A_1 = 1.0, B_1 = 1.0
def get_average_error_A1_B1(Ma, lambda_val, k_list, Pe=0.1, N=5):
    Us_solver = []
    Us_anal = []
    Ud_solver = []
    Ud_anal = []

    A_n = np.zeros(N)
    B_n = np.zeros(N)
    A_n[0] = 1.0
    B_n[0] = 1.0

    for k in k_list:
        c_n, b_n, x_fluid = ch.run_coupling_iteration(
            N=N, k=k, lambda_val=lambda_val, Ma=Ma, Pe=Pe,
            max_iter=1500, omega=0.05, mode='squirmer',
            A_n=A_n, B_n=B_n
        )
        if x_fluid is None:
            return 999.0, 999.0, [], [], [], []

        U_d = x_fluid[5 * N + 1]
        delta_U = x_fluid[5 * N + 2]
        U_s_solver = U_d + delta_U

        U_s_anal = tas.analytical_U_S(k, Ma, lambda_val, 1.0, 1.0)
        U_d_anal = tas.analytical_U_D(k, Ma, lambda_val, 1.0, 1.0)

        Us_solver.append(U_s_solver)
        Us_anal.append(U_s_anal)
        Ud_solver.append(U_d)
        Ud_anal.append(U_d_anal)

    Us_solver = np.array(Us_solver)
    Us_anal = np.array(Us_anal)
    Ud_solver = np.array(Ud_solver)
    Ud_anal = np.array(Ud_anal)

    diff_Us = np.linalg.norm(Us_solver - Us_anal)
    norm_Us = np.linalg.norm(Us_solver)
    err_Us = (diff_Us / norm_Us) * 100.0 if norm_Us != 0 else 999.0

    diff_Ud = np.linalg.norm(Ud_solver - Ud_anal)
    norm_Ud = np.linalg.norm(Ud_solver)
    err_Ud = (diff_Ud / norm_Ud) * 100.0 if norm_Ud != 0 else 999.0

    return err_Us, err_Ud, Us_solver, Us_anal, Ud_solver, Ud_anal

def find_Ma_critical_A1_B1(lambda_val, vel_type='Us', starting_ma=0.001):
    k_list = np.arange(0.1, 0.9, 0.1)
    Pe = 0.1

    def evaluate_Ma(Ma_val):
        err_Us, err_Ud, Us_s, Us_a, Ud_s, Ud_a = get_average_error_A1_B1(Ma_val, lambda_val, k_list, Pe)
        if vel_type == 'Us':
            return err_Us, Us_s, Us_a
        else:
            return err_Ud, Ud_s, Ud_a

    Ma_low = starting_ma
    err_low, solver_low, anal_low = evaluate_Ma(Ma_low)

    history = {}
    history[Ma_low] = (err_low, solver_low, anal_low)

    # Bracketing
    if err_low >= 2.0:
        Ma_high = Ma_low
        err_high = err_low
        while err_low >= 2.0 and Ma_low > 1e-8:
            Ma_high = Ma_low
            err_high = err_low
            Ma_low /= 2.0
            err_low, solver_low, anal_low = evaluate_Ma(Ma_low)
            history[Ma_low] = (err_low, solver_low, anal_low)
        if err_low >= 2.0:
            return Ma_low
    else:
        Ma_high = Ma_low * 2.0
        err_high, solver_high, anal_high = evaluate_Ma(Ma_high)
        history[Ma_high] = (err_high, solver_high, anal_high)
        while err_high < 2.0 and Ma_high < ch.MA_SEARCH_MAX:
            Ma_low = Ma_high
            err_low = err_high
            Ma_high *= 2.0
            err_high, solver_high, anal_high = evaluate_Ma(Ma_high)
            history[Ma_high] = (err_high, solver_high, anal_high)
        if Ma_high >= ch.MA_SEARCH_MAX and err_high < 2.0:
            return Ma_high

    # Bisection
    for step in range(8):
        Ma_mid = 0.5 * (Ma_low + Ma_high)
        err_mid, solver_mid, anal_mid = evaluate_Ma(Ma_mid)
        if err_mid < 2.0:
            Ma_low = Ma_mid
        else:
            Ma_high = Ma_mid

    Ma_crit = 0.5 * (Ma_low + Ma_high)

    # Save the individual profiles
    lambda_dir = os.path.join(analysis_dir, f"lambda={lambda_val}")
    os.makedirs(lambda_dir, exist_ok=True)
    lambda_fig_dir = critical_ma_lambda_figures_dir(1.0, 1.0, lambda_val)
    os.makedirs(lambda_fig_dir, exist_ok=True)

    err_crit, solver_crit, anal_crit = evaluate_Ma(Ma_crit)
    history[Ma_crit] = (err_crit, solver_crit, anal_crit)

    # Save CSV
    csv_path = os.path.join(lambda_dir, f"critical_data_{vel_type}.csv")
    with open(csv_path, 'w') as f:
        f.write(f"k,{vel_type}_solver,{vel_type}_analytical,relative_error_pct\n")
        for idx, k in enumerate(k_list):
            err_point = 100.0 * abs(solver_crit[idx] - anal_crit[idx]) / abs(solver_crit[idx]) if solver_crit[idx] != 0 else 0
            f.write(f"{k},{solver_crit[idx]},{anal_crit[idx]},{err_point}\n")

    # Plot individual curves
    plt.close('all')
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sorted_mas = sorted(list(history.keys()))
    valid_mas = [m for m in sorted_mas if history[m][0] < 100.0]

    if len(valid_mas) >= 4:
        indices = [0, len(valid_mas)//3, 2*len(valid_mas)//3, len(valid_mas)-1]
        mas_to_plot = [valid_mas[i] for i in indices]
    else:
        mas_to_plot = valid_mas

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(mas_to_plot)))
    k_dense = np.linspace(0.01, 0.99, 200)

    vel_label = r"Squirmer Velocity $U_s$" if vel_type == 'Us' else r"Drop Velocity $U_d$"

    for idx, Ma_val in enumerate(mas_to_plot):
        err_val, s_vals, a_vals = history[Ma_val]
        c = colors[idx]

        if vel_type == 'Us':
            a_dense = [tas.analytical_U_S(kv, Ma_val, lambda_val, 1.0, 1.0) for kv in k_dense]
        else:
            a_dense = [tas.analytical_U_D(kv, Ma_val, lambda_val, 1.0, 1.0) for kv in k_dense]

        ax.plot(k_dense, a_dense, '--', color=c, alpha=0.7,
                label=f"Analytical (Ma={Ma_val:.3f}, Err={err_val:.2f}%)" if Ma_val != Ma_crit else f"Analytical (Ma_c={Ma_val:.3f}, Err={err_val:.2f}%)")

        ax.plot(k_list, s_vals, 'o', color=c, markersize=7,
                label=f"Solver (Ma={Ma_val:.3f})" if Ma_val != Ma_crit else f"Solver (Ma_c={Ma_val:.3f})")

    ax.set_xlabel(r"Core Size Ratio $k$", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel(vel_label, fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(fr"{vel_label} Profiles Approaching $Ma_{{\text{{crit}}}}$" + "\n" + fr"($\lambda = {lambda_val}$, Pe = 0.1, $A_1=1, B_1=1$)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, facecolor='#ffffff', fontsize=10, loc='best')

    plt.tight_layout()
    plot_path = os.path.join(lambda_fig_dir, f"{vel_type}_vs_k_approaching_Mac.png")
    plt.savefig(plot_path, dpi=250)
    plt.close(fig)
    print(f"Saved individual {vel_type} plot for lambda={lambda_val} to: {plot_path}")

    return Ma_crit

# Run search for lambda = 0.1 and 10.0
print("Running Us search for lambda = 0.1...")
Us_01 = find_Ma_critical_A1_B1(0.1, 'Us', 0.001)
print("Running Ud search for lambda = 0.1...")
Ud_01 = find_Ma_critical_A1_B1(0.1, 'Ud', Us_01)

print("Running Us search for lambda = 10.0...")
Us_10 = find_Ma_critical_A1_B1(10.0, 'Us', 0.001)
print("Running Ud search for lambda = 10.0...")
Ud_10 = find_Ma_critical_A1_B1(10.0, 'Ud', Us_10)

print(f"Lambda 0.1 Us Ma_crit: {Us_01}, Ud Ma_crit: {Ud_01}")
print(f"Lambda 10.0 Us Ma_crit: {Us_10}, Ud Ma_crit: {Ud_10}")

# Update critical_Ma_summary.csv
summary_path = os.path.join(analysis_dir, "critical_Ma_summary.csv")
os.makedirs(CRITICAL_MA_FIGURES_DIR, exist_ok=True)
data_dict = {}

# Read existing summary data if it exists
if os.path.exists(summary_path):
    with open(summary_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data_dict[float(row['lambda'])] = (float(row['critical_Ma_Us']), float(row['critical_Ma_Ud']))

# Update/Insert the new values
data_dict[0.1] = (Us_01, Ud_01)
data_dict[10.0] = (Us_10, Ud_10)

# Sort by lambda
sorted_lambdas = sorted(list(data_dict.keys()))

# Write back
with open(summary_path, 'w') as f:
    f.write("lambda,critical_Ma_Us,critical_Ma_Ud\n")
    for L in sorted_lambdas:
        f.write(f"{L},{data_dict[L][0]},{data_dict[L][1]}\n")
print(f"Updated summary CSV at: {summary_path}")

# Replot summary and combined plots for A1=1, B1=1
Lambda_list = sorted_lambdas
critical_mas_Us = [data_dict[L][0] for L in Lambda_list]
critical_mas_Ud = [data_dict[L][1] for L in Lambda_list]

# Us summary
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
plot_path_Us = os.path.join(CRITICAL_MA_FIGURES_DIR, "Ma_critical_Us_vs_lambda_A1=1_B1=1.png")
plt.savefig(plot_path_Us, dpi=250)
plt.close(fig)

# Ud summary
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
plot_path_Ud = os.path.join(CRITICAL_MA_FIGURES_DIR, "Ma_critical_Ud_vs_lambda_A1=1_B1=1.png")
plt.savefig(plot_path_Ud, dpi=250)
plt.close(fig)

# Combined summary plot
fig, ax = plt.subplots(figsize=(9, 7.5))
ax.plot(Lambda_list, critical_mas_Us, 'o-', color='#2563eb', linewidth=2.0, markersize=6, label=r"Critical Squirmer Marangoni $Ma_{\text{crit}, U_s}$")
ax.plot(Lambda_list, critical_mas_Ud, 's-', color='#ef4444', linewidth=2.0, markersize=6, label=r"Critical Drop Marangoni $Ma_{\text{crit}, U_d}$")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r"Viscosity Ratio $\lambda$ (Log Scale)", fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylabel(r"Critical Marangoni Number $Ma_{\text{crit}}$ (Log Scale)", fontsize=12, fontweight='bold', labelpad=10)
ax.set_title("Validity Range of Analytical Solutions ($A_1=1, B_1=1$)\n" + "Critical Marangoni Number vs. $\lambda$ (Relative Error = 2%)", fontsize=13, fontweight='bold', pad=15)
ax.grid(True, which="both", alpha=0.3)
ax.legend(frameon=True, facecolor='#ffffff', fontsize=11)
plt.tight_layout()
plot_path_combined = os.path.join(CRITICAL_MA_FIGURES_DIR, "Ma_critical_combined_vs_lambda_A1=1_B1=1.png")
plt.savefig(plot_path_combined, dpi=250)
plt.close(fig)

# Outputs are kept in the project directory
print(f"Outputs are stored in the project directory: {analysis_dir} and {CRITICAL_MA_FIGURES_DIR}")
