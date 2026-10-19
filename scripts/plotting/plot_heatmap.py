import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from squirmer_drop.paths import RUNS_DIR, VERIFICATION_FIGURES_DIR

def parse_squirmer_data():
    """
    Parses all CSV files in plots_squirmer/Ma=0.1/Pe=0.1/N=5/
    and returns a list of dictionaries with parameters and U_D.
    """
    base_pattern = os.path.join(RUNS_DIR, "plots_squirmer", "Ma=0.1", "Pe=0.1", "N=5", "data_lambda=*_k=*.csv")
    files = glob.glob(base_pattern)
    print(f"Found {len(files)} files matching pattern.")
    
    data_list = []
    for filepath in files:
        # Extract parameters from filename
        filename = os.path.basename(filepath)
        filename_no_ext = os.path.splitext(filename)[0]
        file_parts = filename_no_ext.split("_")
        
        try:
            # Round parameters to 2 decimal places to prevent floating-point mismatch in grids
            lambda_val = round(float(file_parts[1].split("=")[1]), 2)
            k_val = round(float(file_parts[2].split("=")[1]), 2)
        except Exception as e:
            print(f"Skipping {filename} due to parsing error: {e}")
            continue
            
        # Read file to find U_D
        x_fluid = []
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    row_parts = line.strip().split(",")
                    if len(row_parts) >= 3 and row_parts[2]:
                        x_fluid.append(float(row_parts[2]))
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue
            
        N = 5
        ud_idx = 5 * N + 1
        if len(x_fluid) > ud_idx:
            U_D = x_fluid[ud_idx]
            data_list.append({
                'lambda_val': lambda_val,
                'k_val': k_val,
                'U_D': U_D
            })
            
    return data_list

def plot_heatmaps():
    data = parse_squirmer_data()
    if not data:
        print("No data parsed. Exiting.")
        return
        
    # Get unique sorted values of k and lambda
    lambdas = sorted(list(set(d['lambda_val'] for d in data)))
    ks = sorted(list(set(d['k_val'] for d in data)))
    
    print(f"Unique lambda values ({len(lambdas)}): {lambdas}")
    print(f"Unique k values ({len(ks)}): Min={min(ks)}, Max={max(ks)}, count={len(ks)}")
    
    # Map values to indices
    lambda_map = {val: idx for idx, val in enumerate(lambdas)}
    k_map = {val: idx for idx, val in enumerate(ks)}
    
    # Initialize grids
    Z_solver = np.full((len(lambdas), len(ks)), np.nan)
    Z_anal = np.full((len(lambdas), len(ks)), np.nan)
    
    # Active modes for A_1 = 1.0, B_1 = 1.0
    A_1 = 1.0
    B_1 = 1.0
    Ma = 0.1
    Pe = 0.1
    
    # Fill grids
    for d in data:
        l_val = d['lambda_val']
        k_val = d['k_val']
        l_idx = lambda_map[l_val]
        k_idx = k_map[k_val]
        
        # Solver U_D
        Z_solver[l_idx, k_idx] = d['U_D']
        
        # Analytical U_D
        denom_0 = (6.0 * l_val - 6.0) * (k_val**5) + 9.0 * l_val + 6.0
        U0_D = 10.0 * (k_val**3) * l_val * (A_1 + B_1) / denom_0
        
        denom_1 = ((l_val - 1.0) * (k_val**5) + 1.5 * l_val + 1.0) ** 2
        U1_D_coeff = -5.0 * l_val * (A_1 + B_1) * (k_val**3) * (1.0 - k_val**5) / (6.0 * denom_1)
        
        Z_anal[l_idx, k_idx] = U0_D + Pe * Ma * U1_D_coeff
        
    # Set up plotting mesh (Swapped X and Y)
    L, K = np.meshgrid(lambdas, ks)
    
    artifact_dir = VERIFICATION_FIGURES_DIR
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Determine color limits to ensure same color scale
    all_vals = np.concatenate([Z_solver[~np.isnan(Z_solver)], Z_anal[~np.isnan(Z_anal)]])
    vmin = np.min(all_vals)
    vmax = np.max(all_vals)
    
    # Plot 1: Solver Heatmap
    plt.figure(figsize=(9, 7.5))
    pcm = plt.pcolormesh(L, K, Z_solver.T, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    plt.xscale('log')
    plt.colorbar(pcm, label=r'Drop Velocity ($U_D$)')
    plt.xlabel(r'Viscosity Ratio ($\lambda$)', fontsize=12, fontweight='bold')
    plt.ylabel(r'Core Size Ratio ($k$)', fontsize=12, fontweight='bold')
    plt.title(r'Solver Drop Velocity $U_D(k, \lambda)$' + '\n' + r'(Ma = 0.1, Pe = 0.1, $A_1=B_1=1.0$)', fontsize=13, fontweight='bold', pad=12)
    plt.grid(True, which="both", alpha=0.2, linestyle=':')
    plt.tight_layout()
    solver_path = os.path.join(artifact_dir, "ud_heatmap_solver.png")
    plt.savefig(solver_path, dpi=250)
    print(f"Saved solver heatmap to: {solver_path}")
    plt.close()
    
    # Plot 2: Analytical Heatmap
    plt.figure(figsize=(9, 7.5))
    pcm = plt.pcolormesh(L, K, Z_anal.T, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    plt.xscale('log')
    plt.colorbar(pcm, label=r'Drop Velocity ($U_D$)')
    plt.xlabel(r'Viscosity Ratio ($\lambda$)', fontsize=12, fontweight='bold')
    plt.ylabel(r'Core Size Ratio ($k$)', fontsize=12, fontweight='bold')
    plt.title(r'Analytical Drop Velocity $U_D(k, \lambda)$' + '\n' + r'(Ma = 0.1, Pe = 0.1, $A_1=B_1=1.0$)', fontsize=13, fontweight='bold', pad=12)
    plt.grid(True, which="both", alpha=0.2, linestyle=':')
    plt.tight_layout()
    anal_path = os.path.join(artifact_dir, "ud_heatmap_analytical.png")
    plt.savefig(anal_path, dpi=250)
    print(f"Saved analytical heatmap to: {anal_path}")
    plt.close()

    # Plot 3: Difference Heatmap (Solver - Analytical)
    Z_diff = Z_solver - Z_anal
    max_diff = np.nanmax(np.abs(Z_diff))
    
    plt.figure(figsize=(9, 7.5))
    pcm = plt.pcolormesh(L, K, Z_diff.T, shading='auto', cmap='RdBu_r', vmin=-max_diff, vmax=max_diff)
    plt.xscale('log')
    plt.colorbar(pcm, label=r'Velocity Difference ($U_{D,\text{solver}} - U_{D,\text{analytical}}$)')
    plt.xlabel(r'Viscosity Ratio ($\lambda$)', fontsize=12, fontweight='bold')
    plt.ylabel(r'Core Size Ratio ($k$)', fontsize=12, fontweight='bold')
    plt.title(r'Difference $U_{D,\text{solver}} - U_{D,\text{analytical}}$' + '\n' + r'(Ma = 0.1, Pe = 0.1, $A_1=B_1=1.0$)', fontsize=13, fontweight='bold', pad=12)
    plt.grid(True, which="both", alpha=0.2, linestyle=':')
    plt.tight_layout()
    diff_path = os.path.join(artifact_dir, "ud_heatmap_difference.png")
    plt.savefig(diff_path, dpi=250)
    print(f"Saved difference heatmap to: {diff_path}")
    plt.close()

if __name__ == "__main__":
    plot_heatmaps()
