import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from squirmer_drop.analytical_squirmer import analytical_U_S, analytical_U_D
from squirmer_drop.paths import RUNS_DIR, VERIFICATION_FIGURES_DIR

# Set styling for plots
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

squirmer_dir = os.path.join(RUNS_DIR, "plots_squirmer")
output_dir = os.path.join(VERIFICATION_FIGURES_DIR, "squirmer_vs_analytical")
os.makedirs(output_dir, exist_ok=True)

# Find all data files
files = glob.glob(os.path.join(squirmer_dir, "Ma=*", "Pe=*", "N=*", "*.csv"))
print(f"Total CSV files found: {len(files)}")

# Data structure to hold parsed values:
# parsed_data[Ma][lambda_val][k_val] = (U_d, U_s)
parsed_data = {}

for filepath in files:
    parts = filepath.replace("\\", "/").split("/")
    try:
        # Extract Ma and normalize
        ma_str = parts[-4].split("=")[1]
        Ma = float(ma_str)

        try:
            n_str = parts[-2].split("=")[1]
            N = int(n_str)
        except Exception:
            N = 5

        # Extract filename and normalize lambda and k
        filename = parts[-1]
        filename_no_ext = os.path.splitext(filename)[0]
        file_parts = filename_no_ext.split("_")

        lambda_str = file_parts[1].split("=")[1]
        lambda_val = float(lambda_str)

        k_str = file_parts[2].split("=")[1]
        k_val = float(k_str)
    except Exception as e:
        print(f"Error parsing path {filepath}: {e}")
        continue

    # Read the data from CSV
    try:
        # The CSV has columns: index, c_n, x_fluid
        x_fluid = []
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                row_parts = line.strip().split(",")
                if len(row_parts) >= 3 and row_parts[2]:
                    x_fluid.append(float(row_parts[2]))

        if len(x_fluid) > 5 * N + 2:
            U_d = x_fluid[5 * N + 1]
            delta_U = x_fluid[5 * N + 2]
            U_s = U_d + delta_U

            # Round parameters to standard values for grouping
            Ma_key = round(Ma, 3)
            lambda_key = round(lambda_val, 3)
            k_key = round(k_val, 3)

            if Ma_key not in parsed_data:
                parsed_data[Ma_key] = {}
            if lambda_key not in parsed_data[Ma_key]:
                parsed_data[Ma_key][lambda_key] = {}

            parsed_data[Ma_key][lambda_key][k_key] = (U_d, U_s)
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")

# Target values requested:
target_Mas = [0.1, 1.0, 10.0]
target_lambdas = [0.1, 0.5, 1.0, 2.0, 10.0]

# Define consistent colors for changing Ma
colors_ma = {
    0.1: '#2563eb',   # Blue
    1.0: '#8b5cf6',   # Purple
    10.0: '#ef4444'   # Red
}

# Function to generate plots for a given Ma, lambda, and velocity type (Single curves)
def generate_and_save_plot(Ma, lambda_val, vel_type, A_1, B_1):
    ma_matches = [k for k in parsed_data.keys() if np.isclose(k, Ma)]
    if not ma_matches:
        return
    ma_key = ma_matches[0]

    lambda_matches = [k for k in parsed_data[ma_key].keys() if np.isclose(k, lambda_val)]
    if not lambda_matches:
        return
    lambda_key = lambda_matches[0]

    k_v_dict = parsed_data[ma_key][lambda_key]
    k_vals = sorted(list(k_v_dict.keys()))

    vel_vals = []
    for k in k_vals:
        U_d, U_s = k_v_dict[k]
        if vel_type == 'Us':
            vel_vals.append(U_s)
        else:
            vel_vals.append(U_d)

    k_dense = np.linspace(0.01, 0.99, 200)
    if vel_type == 'Us':
        analytical_vals = [analytical_U_S(k, Ma, lambda_val, A_1, B_1) for k in k_dense]
    else:
        analytical_vals = [analytical_U_D(k, Ma, lambda_val, A_1, B_1) for k in k_dense]

    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 6.5))

    color = '#2563eb' if vel_type == 'Us' else '#ef4444'
    label_name = r"Swimmer Velocity $U_s$" if vel_type == 'Us' else r"Drift Velocity $U_d$"

    ax.plot(k_dense, analytical_vals, '--', color=color, alpha=0.75, linewidth=2.0, label="Analytical (Asymptotic)")
    ax.plot(k_vals, vel_vals, 'o', color=color, markersize=8, label="Numerical (Solver)")

    ax.set_xlabel(r"Core Size Ratio $k$", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel(label_name, fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f"{label_name} vs. $k$\n(Ma = {Ma}, $\\lambda$ = {lambda_val}, Pe = 0.1)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, facecolor='#ffffff', fontsize=11)

    plt.tight_layout()
    filename = f"{vel_type}_vs_k_Ma={Ma}_lambda={lambda_val}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=250)
    plt.close(fig)

def generate_combined_Us_Ud_plot(lambda_val, A_1, B_1):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(9, 7.5))
    k_dense = np.linspace(0.01, 0.99, 200)

    for Ma in target_Mas:
        ma_matches = [k for k in parsed_data.keys() if np.isclose(k, Ma)]
        if not ma_matches:
            continue
        ma_key = ma_matches[0]

        lambda_matches = [k for k in parsed_data[ma_key].keys() if np.isclose(k, lambda_val)]
        if not lambda_matches:
            continue
        lambda_key = lambda_matches[0]

        k_v_dict = parsed_data[ma_key][lambda_key]
        k_vals = sorted(list(k_v_dict.keys()))

        uds_num = [k_v_dict[k][0] for k in k_vals]
        ups_num = [k_v_dict[k][1] for k in k_vals]

        uds_anal = [analytical_U_D(k, Ma, lambda_val, A_1, B_1) for k in k_dense]
        ups_anal = [analytical_U_S(k, Ma, lambda_val, A_1, B_1) for k in k_dense]

        c = colors_ma[Ma]

        ax.plot(k_dense, ups_anal, '--', color=c, alpha=0.8, linewidth=2.0, label=f"Analytical $U_s$ (Ma = {Ma})")
        ax.plot(k_vals, ups_num, 'o', color=c, markersize=8, label=f"Numerical $U_s$ (Ma = {Ma})")
        ax.plot(k_dense, uds_anal, ':', color=c, alpha=0.8, linewidth=2.0, label=f"Analytical $U_d$ (Ma = {Ma})")
        ax.plot(k_vals, uds_num, 's', color=c, markersize=7, label=f"Numerical $U_d$ (Ma = {Ma})")

    ax.set_xlabel(r"Core Size Ratio $k$", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel(r"Velocity ($U_s$, $U_d$)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f"Squirmer Velocities vs. $k$ ($\\lambda$ = {lambda_val}, Pe = 0.1)\nEffect of Marangoni Number (Ma)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, facecolor='#ffffff', fontsize=10, loc='best', ncol=2)

    plt.tight_layout()
    filename = f"combined_Us_Ud_lambda={lambda_val}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=250)
    plt.close(fig)

def generate_separate_combined_plot(lambda_val, vel_type, A_1, B_1):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(9, 7.5))
    k_dense = np.linspace(0.01, 0.99, 200)

    for Ma in target_Mas:
        ma_matches = [k for k in parsed_data.keys() if np.isclose(k, Ma)]
        if not ma_matches:
            continue
        ma_key = ma_matches[0]

        lambda_matches = [k for k in parsed_data[ma_key].keys() if np.isclose(k, lambda_val)]
        if not lambda_matches:
            continue
        lambda_key = lambda_matches[0]

        k_v_dict = parsed_data[ma_key][lambda_key]
        k_vals = sorted(list(k_v_dict.keys()))

        if vel_type == 'Us':
            vel_num = [k_v_dict[k][1] for k in k_vals]
            vel_anal = [analytical_U_S(k, Ma, lambda_val, A_1, B_1) for k in k_dense]
        else:
            vel_num = [k_v_dict[k][0] for k in k_vals]
            vel_anal = [analytical_U_D(k, Ma, lambda_val, A_1, B_1) for k in k_dense]

        c = colors_ma[Ma]
        label_name = r"$U_s$" if vel_type == 'Us' else r"$U_d$"

        ax.plot(k_dense, vel_anal, '--', color=c, alpha=0.8, linewidth=2.0, label=f"Analytical {label_name} (Ma = {Ma})")
        ax.plot(k_vals, vel_num, 'o', color=c, markersize=8, label=f"Numerical {label_name} (Ma = {Ma})")

    label_full = r"Swimmer Velocity $U_s$" if vel_type == 'Us' else r"Drift Velocity $U_d$"

    ax.set_xlabel(r"Core Size Ratio $k$", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel(label_full, fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f"{label_full} vs. $k$ ($\\lambda$ = {lambda_val}, Pe = 0.1)\nEffect of Marangoni Number (Ma)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, facecolor='#ffffff', fontsize=11)

    plt.tight_layout()
    filename = f"combined_{vel_type}_lambda={lambda_val}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=250)

    print(f"Saved: {filename}")

    plt.close(fig)

def generate_Us_vs_lambda_plot(target_k=0.3, A_1=1, B_1=1):
    # Plot Us against lambda for a fixed k, overlaying Ma
    plt.close('all')
    fig, ax = plt.subplots(figsize=(9, 7.5))

    lambda_dense = np.linspace(0.4, 2.2, 200)

    for Ma in target_Mas:
        ma_matches = [k for k in parsed_data.keys() if np.isclose(k, Ma)]
        if not ma_matches:
            continue
        ma_key = ma_matches[0]

        # Collect numerical lambda and Us values for the target k
        lambdas_num = []
        ups_num = []

        for lambda_key in parsed_data[ma_key].keys():
            k_matches = [k for k in parsed_data[ma_key][lambda_key].keys() if np.isclose(k, target_k)]
            if k_matches:
                k_key = k_matches[0]
                _, U_s = parsed_data[ma_key][lambda_key][k_key]
                lambdas_num.append(lambda_key)
                ups_num.append(U_s)

        # Sort values by lambda
        if lambdas_num:
            sorted_indices = np.argsort(lambdas_num)
            lambdas_num = np.array(lambdas_num)[sorted_indices]
            ups_num = np.array(ups_num)[sorted_indices]

            # Compute analytical curve
            ups_anal = [analytical_U_S(target_k, Ma, l_val, A_1, B_1) for l_val in lambda_dense]

            c = colors_ma[Ma]
            ax.plot(lambda_dense, ups_anal, '--', color=c, alpha=0.8, linewidth=2.0, label=f"Analytical (Ma = {Ma})")
            ax.plot(lambdas_num, ups_num, 'o', color=c, markersize=8, label=f"Numerical (Ma = {Ma})")

    ax.set_xlabel(r"Viscosity Ratio $\lambda$", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel(r"Swimmer Velocity $U_s$", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f"Swimmer Velocity $U_s$ vs. $\\lambda$ ($k$ = {target_k}, Pe = 0.1)\nEffect of Marangoni Number (Ma)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim(0.4, 2.2)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, facecolor='#ffffff', fontsize=11)

    plt.tight_layout()
    filename = f"Us_vs_lambda_k={target_k}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=250)

    print(f"Saved: {filename}")

    plt.close(fig)

def generate_Ud_vs_lambda_plot(target_k=0.3, A_1=1, B_1=1):
    # Plot Ud against lambda for a fixed k, overlaying Ma
    plt.close('all')
    fig, ax = plt.subplots(figsize=(9, 7.5))

    lambda_dense = np.linspace(0.4, 2.2, 200)

    for Ma in target_Mas:
        ma_matches = [k for k in parsed_data.keys() if np.isclose(k, Ma)]
        if not ma_matches:
            continue
        ma_key = ma_matches[0]

        # Collect numerical lambda and Ud values for the target k
        lambdas_num = []
        uds_num = []

        for lambda_key in parsed_data[ma_key].keys():
            k_matches = [k for k in parsed_data[ma_key][lambda_key].keys() if np.isclose(k, target_k)]
            if k_matches:
                k_key = k_matches[0]
                U_d, _ = parsed_data[ma_key][lambda_key][k_key]
                lambdas_num.append(lambda_key)
                uds_num.append(U_d)

        # Sort values by lambda
        if lambdas_num:
            sorted_indices = np.argsort(lambdas_num)
            lambdas_num = np.array(lambdas_num)[sorted_indices]
            uds_num = np.array(uds_num)[sorted_indices]

            # Compute analytical curve
            uds_anal = [analytical_U_D(target_k, Ma, l_val, A_1, B_1) for l_val in lambda_dense]

            c = colors_ma[Ma]
            ax.plot(lambda_dense, uds_anal, '--', color=c, alpha=0.8, linewidth=2.0, label=f"Analytical (Ma = {Ma})")
            ax.plot(lambdas_num, uds_num, 'o', color=c, markersize=8, label=f"Numerical (Ma = {Ma})")

    ax.set_xlabel(r"Viscosity Ratio $\lambda$", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel(r"Drift Velocity $U_d$", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f"Drift Velocity $U_d$ vs. $\\lambda$ ($k$ = {target_k}, Pe = 0.1)\nEffect of Marangoni Number (Ma)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim(0.4, 2.2)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, facecolor='#ffffff', fontsize=11)

    plt.tight_layout()
    filename = f"Ud_vs_lambda_k={target_k}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=250)

    print(f"Saved: {filename}")

    plt.close(fig)

def get_edges(arr):
    # Dynamically calculate cell boundary edges for pcolormesh
    arr = np.array(arr)
    if len(arr) < 2:
        return np.array([arr[0] - 0.5, arr[0] + 0.5])
    midpoints = 0.5 * (arr[:-1] + arr[1:])
    start = arr[0] - 0.5 * (arr[1] - arr[0])
    end = arr[-1] + 0.5 * (arr[-1] - arr[-2])
    return np.concatenate([[start], midpoints, [end]])

def generate_heatmaps_plot(vel_type='Us', Ma_target=0.1, A_1=0.0, B_1=1.0):
    # Plot Us(k, lambda) or Ud(k, lambda) heatmaps for solver results and analytical results
    ma_matches = [k for k in parsed_data.keys() if np.isclose(k, Ma_target)]
    if not ma_matches:
        print(f"Warning: Ma={Ma_target} not found for heatmap.")
        return
    ma_key = ma_matches[0]

    # Extract unique lambda and k values
    lambdas = sorted(list(parsed_data[ma_key].keys()))

    # Collect all unique ks
    ks_set = set()
    for l_key in lambdas:
        ks_set.update(parsed_data[ma_key][l_key].keys())
    ks = sorted(list(ks_set))

    if not lambdas or not ks:
        print("Warning: No lambda or k values found for heatmap.")
        return

    # Create 2D grid for solver results
    solver_grid = np.zeros((len(ks), len(lambdas)))
    for l_idx, l_val in enumerate(lambdas):
        for k_idx, k_val in enumerate(ks):
            if k_val in parsed_data[ma_key][l_val]:
                U_d, U_s = parsed_data[ma_key][l_val][k_val]
                solver_grid[k_idx, l_idx] = U_s if vel_type == 'Us' else U_d

    # Create dense grid for analytical results
    lambda_dense = np.linspace(min(lambdas), max(lambdas), 100)
    k_dense = np.linspace(min(ks), max(ks), 100)

    analytical_grid = np.zeros((len(k_dense), len(lambda_dense)))
    for i, kv in enumerate(k_dense):
        for j, lv in enumerate(lambda_dense):
            if vel_type == 'Us':
                analytical_grid[i, j] = analytical_U_S(kv, Ma_target, lv, A_1, B_1)
            else:
                analytical_grid[i, j] = analytical_U_D(kv, Ma_target, lv, A_1, B_1)

    # Set common limits for colorbar comparability
    vmin = min(np.min(solver_grid), np.min(analytical_grid))
    vmax = max(np.max(solver_grid), np.max(analytical_grid))

    label_full = r"Swimmer Velocity $U_s$" if vel_type == 'Us' else r"Drift Velocity $U_d$"
    symbol_full = r"U_s" if vel_type == 'Us' else r"U_d"

    # Target folders
    local_folder = os.path.join(output_dir, f"heatmaps_{vel_type}", f"Ma={Ma_target}")
    os.makedirs(local_folder, exist_ok=True)

    # solver heatmap
    plt.close('all')
    fig, ax = plt.subplots(figsize=(9, 7.5))

    # Convert to meshgrid boundaries dynamically
    lambda_edges = get_edges(lambdas)
    k_edges = get_edges(ks)

    mesh_l, mesh_k = np.meshgrid(lambda_edges, k_edges)
    im = ax.pcolormesh(mesh_l, mesh_k, solver_grid, cmap='viridis', vmin=vmin, vmax=vmax, edgecolor='#ffffff', linewidth=0.5)

    ax.set_xticks(lambdas)
    ax.set_xticklabels([str(l) for l in lambdas])
    ax.set_yticks(ks)
    ax.set_yticklabels([f"{k:.1f}" for k in ks])

    ax.set_xlabel(r"Viscosity Ratio $\lambda$", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel(r"Core Size Ratio $k$", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f"Solver {label_full} ${symbol_full}(k, \\lambda)$ (Ma = {Ma_target})", fontsize=13, fontweight='bold', pad=15)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(label_full, fontsize=11, fontweight='bold', labelpad=10)

    plt.tight_layout()
    solver_save_path = os.path.join(local_folder, "solver.png")
    plt.savefig(solver_save_path, dpi=250)

    print(f"Saved: {solver_save_path}")

    plt.close(fig)

    # analytical heatmap
    fig, ax = plt.subplots(figsize=(9, 7.5))

    mesh_l_dense, mesh_k_dense = np.meshgrid(lambda_dense, k_dense)
    im = ax.pcolormesh(mesh_l_dense, mesh_k_dense, analytical_grid, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')

    ax.set_xlabel(r"Viscosity Ratio $\lambda$", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel(r"Core Size Ratio $k$", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f"Analytical {label_full} ${symbol_full}(k, \\lambda)$ (Ma = {Ma_target})", fontsize=13, fontweight='bold', pad=15)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(label_full, fontsize=11, fontweight='bold', labelpad=10)

    plt.tight_layout()
    analytical_save_path = os.path.join(local_folder, "analytical.png")
    plt.savefig(analytical_save_path, dpi=250)

    print(f"Saved: {analytical_save_path}")

    plt.close(fig)

    # difference heatmap
    # Compute analytical results on the exact discrete solver coordinates
    analytical_discrete = np.zeros((len(ks), len(lambdas)))
    for l_idx, l_val in enumerate(lambdas):
        for k_idx, k_val in enumerate(ks):
            if vel_type == 'Us':
                analytical_discrete[k_idx, l_idx] = analytical_U_S(k_val, Ma_target, l_val, A_1, B_1)
            else:
                analytical_discrete[k_idx, l_idx] = analytical_U_D(k_val, Ma_target, l_val, A_1, B_1)

    difference_grid = solver_grid - analytical_discrete

    fig, ax = plt.subplots(figsize=(9, 7.5))

    # Calculate a symmetric range about zero for the diverging coolwarm colormap
    max_diff = np.max(np.abs(difference_grid))
    if max_diff == 0:
        max_diff = 1.0
    vmin_diff = -max_diff
    vmax_diff = max_diff

    im = ax.pcolormesh(mesh_l, mesh_k, difference_grid, cmap='coolwarm', vmin=vmin_diff, vmax=vmax_diff, edgecolor='#ffffff', linewidth=0.5)

    ax.set_xticks(lambdas)
    ax.set_xticklabels([str(l) for l in lambdas])
    ax.set_yticks(ks)
    ax.set_yticklabels([f"{k:.1f}" for k in ks])

    ax.set_xlabel(r"Viscosity Ratio $\lambda$", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel(r"Core Size Ratio $k$", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f"Difference ${symbol_full}(k, \\lambda)$ (Solver - Analytical) (Ma = {Ma_target})", fontsize=13, fontweight='bold', pad=15)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(fr"Velocity Difference $\Delta {symbol_full}$", fontsize=11, fontweight='bold', labelpad=10)

    plt.tight_layout()
    diff_save_path = os.path.join(local_folder, "difference.png")
    plt.savefig(diff_save_path, dpi=250)

    print(f"Saved: {diff_save_path}")

    plt.close(fig)

# Generate the combined plots
A_1 = 1
B_1 = 1
for lambda_val in target_lambdas:
    generate_separate_combined_plot(lambda_val, 'Us', A_1, B_1)
    generate_separate_combined_plot(lambda_val, 'Ud', A_1, B_1)

# Generate the new Us and Ud vs lambda plots
generate_Us_vs_lambda_plot(target_k=0.3, A_1=A_1, B_1=B_1)
generate_Ud_vs_lambda_plot(target_k=0.3, A_1=A_1, B_1=B_1)

# Generate heatmaps for all Ma targets (0.1, 1.0, 10.0) with matching constants A_1=1.0, B_1=1.0
for Ma in [0.1, 1.0, 10.0]:
    generate_heatmaps_plot(vel_type='Us', Ma_target=Ma, A_1=A_1, B_1=1.0)
    generate_heatmaps_plot(vel_type='Ud', Ma_target=Ma, A_1=A_1, B_1=1.0)

# Generate heatmaps for Ma = 100.0 with matching constants A_1=1.0, B_1=1.0
generate_heatmaps_plot(vel_type='Us', Ma_target=100.0, A_1=A_1, B_1=B_1)
generate_heatmaps_plot(vel_type='Ud', Ma_target=100.0, A_1=A_1, B_1=B_1)

print("\nAll plots successfully generated and updated!")
