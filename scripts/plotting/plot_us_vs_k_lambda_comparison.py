import os
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt

# Import the analytical solutions from testing_analytical_squirmer
from squirmer_drop.analytical_squirmer import analytical_U_S
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

squirmer_dir = os.path.join(RUNS_DIR, "plots_squirmer")
output_dir = os.path.join(VERIFICATION_FIGURES_DIR, "squirmer_vs_analytical")
# Output kept in project directory (home dir copy removed)
# home_dir = os.path.expanduser("~")
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
        ma_str = parts[-4].split("=")[1]
        Ma = float(ma_str)

        try:
            n_str = parts[-2].split("=")[1]
            N = int(n_str)
        except Exception:
            N = 5
        
        filename = parts[-1]
        filename_no_ext = os.path.splitext(filename)[0]
        file_parts = filename_no_ext.split("_")
        
        lambda_str = file_parts[1].split("=")[1]
        lambda_val = float(lambda_str)
        
        k_str = file_parts[2].split("=")[1]
        k_val = float(k_str)
    except Exception as e:
        continue
        
    try:
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

# Parameters
Ma_target = 1.0
lambda_targets = [0.1, 10.0]
A_1 = 1.0
B_1 = 1.0

# ----------------- PLOT GENERATION -----------------
plt.close('all')
fig, ax = plt.subplots(figsize=(9, 7.5))

colors_lambda = {
    0.1: '#2563eb',   # Blue
    10.0: '#ef4444'   # Red
}

styles_lambda = {
    0.1: '--',        # Dashed
    10.0: '-.'        # Dash-dot
}

markers_lambda = {
    0.1: 'o',         # Circle
    10.0: 's'         # Square
}

k_dense = np.linspace(0.01, 0.99, 200)
ma_matches = [k for k in parsed_data.keys() if np.isclose(k, Ma_target)]

if ma_matches:
    ma_key = ma_matches[0]
    
    for l_target in lambda_targets:
        l_matches = [k for k in parsed_data[ma_key].keys() if np.isclose(k, l_target)]
        
        if l_matches:
            l_key = l_matches[0]
            k_v_dict = parsed_data[ma_key][l_key]
            ks_num = sorted(list(k_v_dict.keys()))
            us_num = [k_v_dict[k][1] for k in ks_num]
            
            # Evaluate analytical curve
            us_anal = [analytical_U_S(kv, Ma_target, l_target, A_1, B_1) for kv in k_dense]
            
            c = colors_lambda[l_target]
            style = styles_lambda[l_target]
            marker = markers_lambda[l_target]
            
            # Plot analytical curves
            ax.plot(k_dense, us_anal, style, color=c, alpha=0.8, linewidth=2.0, 
                    label=f"Analytical $U_s$ ($\\lambda$ = {l_target})")
            
            # Plot numerical solver results
            ax.plot(ks_num, us_num, marker, color=c, markersize=8, 
                    label=f"Numerical $U_s$ ($\\lambda$ = {l_target})")
        else:
            print(f"Warning: lambda={l_target} data not found for Ma={Ma_target}")
else:
    print(f"Warning: Ma={Ma_target} data not found in parsed_data.")

ax.set_xlabel(r"Core Size Ratio $k$", fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylabel(r"Swimmer Velocity $U_s$", fontsize=12, fontweight='bold', labelpad=10)
ax.set_title(f"Swimmer Velocity $U_s$ vs. $k$ (Ma = {Ma_target}, Pe = 0.1)\nComparison of Viscosity Ratios ($\\lambda = 0.1, 10.0$)", fontsize=13, fontweight='bold', pad=15)
ax.set_xlim(0.0, 1.0)
ax.grid(True, alpha=0.3)
ax.legend(frameon=True, facecolor='#ffffff', fontsize=11)

plt.tight_layout()
filename = "Us_vs_k_Ma=1_lambda_comparison.png"
save_path = os.path.join(output_dir, filename)
plt.savefig(save_path, dpi=250)

# Output kept in project directory (home dir copy removed)
# try:
#     shutil.copy(save_path, os.path.join(home_dir, filename))
#     print(f"Saved & Copied to Home: {filename}")
# except Exception as e:
#     print(f"Saved: {filename} (Error copying to home: {e})")
print(f"Saved: {filename}")

plt.close(fig)
