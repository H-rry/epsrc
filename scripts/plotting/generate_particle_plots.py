import os
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt

# Import the analytical solutions from testing_analytical_particle
from squirmer_drop.analytical_particle import analytical_U_D, analytical_U_P
from squirmer_drop.paths import RUNS_DIR, PARTICLE_DATA_DIR, VERIFICATION_FIGURES_DIR

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

particle_dir = os.path.join(RUNS_DIR, "plots_particle")
output_dir = os.path.join(VERIFICATION_FIGURES_DIR, "particle_vs_analytical")
# Output kept in project directory (home dir copy removed)
# home_dir = os.path.expanduser("~")
os.makedirs(output_dir, exist_ok=True)

# Find all data files
files = glob.glob(os.path.join(particle_dir, "Ma=*", "Pe=*", "N=*", "*.csv"))
print(f"Total CSV files found: {len(files)}")

# Data structure to hold parsed values:
# parsed_data[Ma][Pe][lambda_val][k_val] = (U_d, U_p)
parsed_data = {}
all_data_list = []

for filepath in files:
    parts = filepath.replace("\\", "/").split("/")
    try:
        # Extract Ma, Pe, and N
        ma_str = parts[-4].split("=")[1]
        Ma = float(ma_str)
        
        pe_str = parts[-3].split("=")[1]
        Pe = float(pe_str)
        
        n_str = parts[-2].split("=")[1]
        N = int(n_str)
        
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
        
        # In particle mode, U_D is at 5*N + 1, delta_U at 5*N + 2
        ud_idx = 5 * N + 1
        delta_u_idx = 5 * N + 2
        if len(x_fluid) > delta_u_idx:
            U_d = x_fluid[ud_idx]
            delta_U = x_fluid[delta_u_idx]
            U_p = U_d + delta_U
            
            Ma_key = round(Ma, 3)
            Pe_key = round(Pe, 3)
            lambda_key = round(lambda_val, 3)
            k_key = round(k_val, 3)
            
            if Ma_key not in parsed_data:
                parsed_data[Ma_key] = {}
            if Pe_key not in parsed_data[Ma_key]:
                parsed_data[Ma_key][Pe_key] = {}
            if lambda_key not in parsed_data[Ma_key][Pe_key]:
                parsed_data[Ma_key][Pe_key][lambda_key] = {}
            
            parsed_data[Ma_key][Pe_key][lambda_key][k_key] = (U_d, U_p)
            
            all_data_list.append({
                'lambda': lambda_val,
                'k': k_val,
                'Ma': Ma,
                'Pe': Pe,
                'U_d': U_d,
                'U_p': U_p
            })
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")

# ----------------- CSV COLLATED DATA GENERATION -----------------
# Sort data: k first, then Ma, then lambda, then Pe
sorted_data = sorted(all_data_list, key=lambda x: (x['k'], x['Ma'], x['lambda'], x['Pe']))

csv_filename = "collated_particle_data.csv"
os.makedirs(PARTICLE_DATA_DIR, exist_ok=True)
local_csv_path = os.path.join(PARTICLE_DATA_DIR, csv_filename)
# Output kept in project directory (home dir copy removed)
# home_csv_path = os.path.join(home_dir, csv_filename)

with open(local_csv_path, 'w') as f:
    f.write("k,Ma,lambda,Pe,U_d,U_p\n")
    for row in sorted_data:
        f.write(f"{row['k']},{row['Ma']},{row['lambda']},{row['Pe']},{row['U_d']},{row['U_p']}\n")

# Output kept in project directory (home dir copy removed)
print(f"Saved CSV locally: {local_csv_path}")


# ----------------- PLOTTING FUNCTIONS -----------------
def generate_k_sweep_verification_plots(lambda_target=2.0, Pe_target=1.0):
    # Generates verification curves for U_d and U_p against 1/k for fixed lambda and Pe
    # Plots Ma = 0.0 numerical results for Pe = 1.0
    
    Ma_target = 0.0
    
    # Try to find if we have Ma = 0.0 keys
    ma_keys = [k for k in parsed_data.keys() if np.isclose(k, Ma_target)]
    if not ma_keys:
        print(f"Warning: No Ma = 0.0 data found in parsed_data. Only keys available: {list(parsed_data.keys())}")
        ks_num = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        has_ma_data = False
    else:
        ma_key = ma_keys[0]
        if Pe_target in parsed_data[ma_key] and lambda_target in parsed_data[ma_key][Pe_target]:
            k_v_dict = parsed_data[ma_key][Pe_target][lambda_target]
            ks_num = sorted(list(k_v_dict.keys()))
            has_ma_data = True
        else:
            ks_num = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            has_ma_data = False
            
    # Define dense spacing linear in 1/k
    min_k = min(ks_num)
    max_k = max(ks_num)
    
    inv_ks_dense = np.linspace(1.0 / max_k, 1.0 / min_k, 200)
    ks_dense = 1.0 / inv_ks_dense
    
    # Calculate analytical curves
    Ud_sheet_dense = [analytical_U_D(kv, lambda_target) for kv in ks_dense]
    Up_sheet_dense = [analytical_U_P(kv, lambda_target) for kv in ks_dense]
    
    # ----------------- PLOT 1: Ud vs 1/k -----------------
    plt.close('all')
    fig, ax = plt.subplots(figsize=(9, 7.5))
    
    # Plot analytical sheet limit
    ax.plot(inv_ks_dense, Ud_sheet_dense, '--', color='#ef4444', linewidth=2.0, label="Analytical (Sheet Eq 16a)")
    
    # Plot numerical solver results (Ma = 0.0, Pe = 1.0)
    if has_ma_data:
        ma_key = ma_keys[0]
        k_v_dict = parsed_data[ma_key][Pe_target][lambda_target]
        uds_num = [k_v_dict[k][0] for k in ks_num]
        inv_ks_num = [1.0 / k for k in ks_num]
        ax.plot(inv_ks_num, uds_num, 'o', color='#2563eb', markersize=8, label="Numerical (Ma = 0.0)")
    else:
        print("Skipping numerical plot points for Ud due to lack of Ma = 0.0 data.")
        
    ax.set_xlabel(r"Inverse Core Size Ratio $1/k$ (Linear Scale)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel(r"Droplet Velocity $U_d$", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f"Droplet Velocity $U_d$ Verification vs. $1/k$\n($\\lambda = {lambda_target}$, $Pe = {Pe_target}$, Ma = 0.0)", fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, facecolor='#ffffff', fontsize=11)
    
    plt.tight_layout()
    filename_ud = f"Ud_vs_inv_k_lambda={lambda_target}_Ma=0.0_Pe={Pe_target}.png"
    save_path_ud = os.path.join(output_dir, filename_ud)
    plt.savefig(save_path_ud, dpi=250)
    
    # Output kept in project directory (home dir copy removed)
    print(f"Saved locally: {save_path_ud}")
    plt.close(fig)
    
    # ----------------- PLOT 2: Up vs 1/k -----------------
    fig, ax = plt.subplots(figsize=(9, 7.5))
    
    # Plot analytical sheet limit
    ax.plot(inv_ks_dense, Up_sheet_dense, '--', color='#ef4444', linewidth=2.0, label="Analytical (Sheet Eq 16b)")
    
    # Plot numerical solver results (Ma = 0.0, Pe = 1.0)
    if has_ma_data:
        ma_key = ma_keys[0]
        k_v_dict = parsed_data[ma_key][Pe_target][lambda_target]
        ups_num = [k_v_dict[k][1] for k in ks_num]
        inv_ks_num = [1.0 / k for k in ks_num]
        ax.plot(inv_ks_num, ups_num, 'o', color='#2563eb', markersize=8, label="Numerical (Ma = 0.0)")
    else:
        print("Skipping numerical plot points for Up due to lack of Ma = 0.0 data.")
        
    ax.set_xlabel(r"Inverse Core Size Ratio $1/k$ (Linear Scale)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel(r"Particle Velocity $U_p$", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f"Particle Velocity $U_p$ Verification vs. $1/k$\n($\\lambda = {lambda_target}$, $Pe = {Pe_target}$, Ma = 0.0)", fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, facecolor='#ffffff', fontsize=11)
    
    plt.tight_layout()
    filename_up = f"Up_vs_inv_k_lambda={lambda_target}_Ma=0.0_Pe={Pe_target}.png"
    save_path_up = os.path.join(output_dir, filename_up)
    plt.savefig(save_path_up, dpi=250)
    
    # Output kept in project directory (home dir copy removed)
    print(f"Saved locally: {save_path_up}")
    plt.close(fig)


def generate_lambda_sweep_verification_plots(k_target=0.3, Pe_target=1.0):
    # Generates verification curves for U_d and U_p against lambda for fixed k and Pe
    # Plots Ma = 0.0 numerical results for Pe = 1.0
    
    Ma_target = 0.0
    
    # Try to find if we have Ma = 0.0 keys
    ma_keys = [k for k in parsed_data.keys() if np.isclose(k, Ma_target)]
    has_data = len(ma_keys) > 0
    
    lambdas_dense = np.logspace(-1, 1, 200)
    
    # Calculate analytical curves
    Ud_sheet_dense = [analytical_U_D(k_target, lv) for lv in lambdas_dense]
    Up_sheet_dense = [analytical_U_P(k_target, lv) for lv in lambdas_dense]
        
    # ----------------- PLOT 1: Ud vs lambda -----------------
    plt.close('all')
    fig, ax = plt.subplots(figsize=(9, 7.5))
    
    # Plot analytical sheet limit
    ax.plot(lambdas_dense, Ud_sheet_dense, '--', color='#ef4444', linewidth=2.0, label="Analytical (Sheet Eq 16a)")
    
    # Plot numerical solver results (Ma = 0.0, Pe = 1.0)
    if has_data:
        ma_key = ma_keys[0]
        if Pe_target in parsed_data[ma_key]:
            lambdas_num = []
            uds_num = []
            for lambda_key in parsed_data[ma_key][Pe_target].keys():
                k_matches = [k for k in parsed_data[ma_key][Pe_target][lambda_key].keys() if np.isclose(k, k_target)]
                if k_matches:
                    k_key = k_matches[0]
                    Ud_val, _ = parsed_data[ma_key][Pe_target][lambda_key][k_key]
                    lambdas_num.append(lambda_key)
                    uds_num.append(Ud_val)
                    
            if lambdas_num:
                sorted_indices = np.argsort(lambdas_num)
                lambdas_num = np.array(lambdas_num)[sorted_indices]
                uds_num = np.array(uds_num)[sorted_indices]
                ax.plot(lambdas_num, uds_num, 'o', color='#2563eb', markersize=8, label="Numerical (Ma = 0.0)")
    
    ax.set_xscale('log')
    ax.set_xlabel(r"Viscosity Ratio $\lambda$", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel(r"Droplet Velocity $U_d$", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f"Droplet Velocity $U_d$ Verification vs. $\\lambda$\n($k = {k_target}$, $Pe = {Pe_target}$, Ma = 0.0)", fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=True, facecolor='#ffffff', fontsize=11)
    
    plt.tight_layout()
    filename_ud = f"Ud_vs_lambda_k={k_target}_Ma=0.0_Pe={Pe_target}.png"
    save_path_ud = os.path.join(output_dir, filename_ud)
    plt.savefig(save_path_ud, dpi=250)
    
    # Output kept in project directory (home dir copy removed)
    print(f"Saved locally: {save_path_ud}")
    plt.close(fig)
    
    # ----------------- PLOT 2: Up vs lambda -----------------
    fig, ax = plt.subplots(figsize=(9, 7.5))
    
    # Plot analytical sheet limit
    ax.plot(lambdas_dense, Up_sheet_dense, '--', color='#ef4444', linewidth=2.0, label="Analytical (Sheet Eq 16b)")
    
    # Plot numerical solver results (Ma = 0.0, Pe = 1.0)
    if has_data:
        ma_key = ma_keys[0]
        if Pe_target in parsed_data[ma_key]:
            lambdas_num = []
            ups_num = []
            for lambda_key in parsed_data[ma_key][Pe_target].keys():
                k_matches = [k for k in parsed_data[ma_key][Pe_target][lambda_key].keys() if np.isclose(k, k_target)]
                if k_matches:
                    k_key = k_matches[0]
                    _, Up_val = parsed_data[ma_key][Pe_target][lambda_key][k_key]
                    lambdas_num.append(lambda_key)
                    ups_num.append(Up_val)
                    
            if lambdas_num:
                sorted_indices = np.argsort(lambdas_num)
                lambdas_num = np.array(lambdas_num)[sorted_indices]
                ups_num = np.array(ups_num)[sorted_indices]
                ax.plot(lambdas_num, ups_num, 'o', color='#2563eb', markersize=8, label="Numerical (Ma = 0.0)")
                
    ax.set_xscale('log')
    ax.set_yscale('symlog')
    ax.set_xlabel(r"Viscosity Ratio $\lambda$", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel(r"Particle Velocity $U_p$", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f"Particle Velocity $U_p$ Verification vs. $\\lambda$\n($k = {k_target}$, $Pe = {Pe_target}$, Ma = 0.0)", fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=True, facecolor='#ffffff', fontsize=11)
    
    plt.tight_layout()
    filename_up = f"Up_vs_lambda_k={k_target}_Ma=0.0_Pe={Pe_target}.png"
    save_path_up = os.path.join(output_dir, filename_up)
    plt.savefig(save_path_up, dpi=250)
    
    # Output kept in project directory (home dir copy removed)
    print(f"Saved locally: {save_path_up}")
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


def generate_heatmaps_plot(vel_type='Ud', Ma_target=0.0, Pe_target=1.0):
    # Plot Ud(k, lambda) or Up(k, lambda) heatmaps for solver results and analytical results
    ma_matches = [k for k in parsed_data.keys() if np.isclose(k, Ma_target)]
    if not ma_matches:
        print(f"Warning: Ma={Ma_target} not found for heatmap.")
        return
    ma_key = ma_matches[0]
    
    if Pe_target not in parsed_data[ma_key]:
        print(f"Warning: Pe={Pe_target} not found for Ma={Ma_target} heatmap.")
        return
        
    # Extract unique lambda and k values
    lambdas = sorted(list(parsed_data[ma_key][Pe_target].keys()))
    
    # Collect all unique ks
    ks_set = set()
    for l_key in lambdas:
        ks_set.update(parsed_data[ma_key][Pe_target][l_key].keys())
    ks = sorted(list(ks_set))
    
    if not lambdas or not ks:
        print("Warning: No lambda or k values found for heatmap.")
        return
        
    # Create 2D grid for solver results
    solver_grid = np.zeros((len(ks), len(lambdas)))
    for l_idx, l_val in enumerate(lambdas):
        for k_idx, k_val in enumerate(ks):
            if k_val in parsed_data[ma_key][Pe_target][l_val]:
                U_d, U_p = parsed_data[ma_key][Pe_target][l_val][k_val]
                solver_grid[k_idx, l_idx] = U_d if vel_type == 'Ud' else U_p
                
    # Create dense grid for analytical results
    lambda_dense = np.linspace(min(lambdas), max(lambdas), 100)
    k_dense = np.linspace(min(ks), max(ks), 100)
    
    analytical_grid = np.zeros((len(k_dense), len(lambda_dense)))
    for i, kv in enumerate(k_dense):
        for j, lv in enumerate(lambda_dense):
            if vel_type == 'Ud':
                analytical_grid[i, j] = analytical_U_D(kv, lv)
            else:
                analytical_grid[i, j] = analytical_U_P(kv, lv)
            
    # Set common limits for colorbar comparability
    vmin = min(np.min(solver_grid), np.min(analytical_grid))
    vmax = max(np.max(solver_grid), np.max(analytical_grid))
    
    label_full = r"Droplet Velocity $U_d$" if vel_type == 'Ud' else r"Particle Velocity $U_p$"
    symbol_full = r"U_d" if vel_type == 'Ud' else r"U_p"
    
    # Target folders
    local_folder = os.path.join(output_dir, f"heatmaps_{vel_type}", f"Ma={Ma_target}_Pe={Pe_target}")
    # Output kept in project directory (home dir copy removed)
    # home_folder = os.path.join(home_dir, f"heatmaps_{vel_type}", f"Ma={Ma_target}_Pe={Pe_target}")
    os.makedirs(local_folder, exist_ok=True)
    # os.makedirs(home_folder, exist_ok=True)
    
    # ----------------- PLOT 1: SOLVER HEATMAP -----------------
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
    ax.set_title(f"Solver {label_full} ${symbol_full}(k, \\lambda)$\n(Ma = {Ma_target}, Pe = {Pe_target})", fontsize=13, fontweight='bold', pad=15)
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(label_full, fontsize=11, fontweight='bold', labelpad=10)
    
    plt.tight_layout()
    solver_save_path = os.path.join(local_folder, "solver.png")
    plt.savefig(solver_save_path, dpi=250)
    
    # Output kept in project directory (home dir copy removed)
    print(f"Saved: {solver_save_path}")
        
    plt.close(fig)
    
    # ----------------- PLOT 2: ANALYTICAL HEATMAP -----------------
    fig, ax = plt.subplots(figsize=(9, 7.5))
    
    mesh_l_dense, mesh_k_dense = np.meshgrid(lambda_dense, k_dense)
    im = ax.pcolormesh(mesh_l_dense, mesh_k_dense, analytical_grid, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
    
    ax.set_xlabel(r"Viscosity Ratio $\lambda$", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel(r"Core Size Ratio $k$", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f"Analytical {label_full} ${symbol_full}(k, \\lambda)$\n(Ma = {Ma_target}, Pe = {Pe_target})", fontsize=13, fontweight='bold', pad=15)
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(label_full, fontsize=11, fontweight='bold', labelpad=10)
    
    plt.tight_layout()
    analytical_save_path = os.path.join(local_folder, "analytical.png")
    plt.savefig(analytical_save_path, dpi=250)
    
    # Output kept in project directory (home dir copy removed)
    print(f"Saved: {analytical_save_path}")
        
    plt.close(fig)

    # ----------------- PLOT 3: DIFFERENCE HEATMAP -----------------
    # Compute analytical results on the exact discrete solver coordinates
    analytical_discrete = np.zeros((len(ks), len(lambdas)))
    for l_idx, l_val in enumerate(lambdas):
        for k_idx, k_val in enumerate(ks):
            if vel_type == 'Ud':
                analytical_discrete[k_idx, l_idx] = analytical_U_D(k_val, l_val)
            else:
                analytical_discrete[k_idx, l_idx] = analytical_U_P(k_val, l_val)
            
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
    ax.set_title(f"Difference ${symbol_full}(k, \\lambda)$ (Solver - Analytical)\n(Ma = {Ma_target}, Pe = {Pe_target})", fontsize=13, fontweight='bold', pad=15)
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(fr"Velocity Difference $\Delta {symbol_full}$", fontsize=11, fontweight='bold', labelpad=10)
    
    plt.tight_layout()
    diff_save_path = os.path.join(local_folder, "difference.png")
    plt.savefig(diff_save_path, dpi=250)
    
    # Output kept in project directory (home dir copy removed)
    print(f"Saved: {diff_save_path}")
        
    plt.close(fig)


# Generate verification plots for k-sweep (lambda=2.0, Pe=1.0)
generate_k_sweep_verification_plots(lambda_target=2.0, Pe_target=1.0)

# Generate verification plots for lambda-sweep (k=0.3, Pe=1.0)
generate_lambda_sweep_verification_plots(k_target=0.3, Pe_target=1.0)

# Generate heatmaps for Ma = 0.0 and Pe = 1.0
generate_heatmaps_plot(vel_type='Ud', Ma_target=0.0, Pe_target=1.0)
generate_heatmaps_plot(vel_type='Up', Ma_target=0.0, Pe_target=1.0)

print(f"\nAll particle verification plots, heatmaps, and CSV successfully generated!")
