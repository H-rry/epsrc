import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from squirmer_drop.paths import RUNS_DIR

def parse_data(mode='particle'):
    """
    Dynamically scans the plots_{mode}/ directory, parses all data_lambda=*_k=*.csv files,
    and extracts parameters along with converged surfactant modes c_n and velocities U_D and U_p.
    """
    files = glob.glob(os.path.join(RUNS_DIR, f"plots_{mode}", "Ma=*", "Pe=*", "N=*", "data_lambda=*_k=*.csv"))
    data_list = []
    
    for filepath in files:
        # Standardize path separators for consistent parsing
        parts = filepath.replace("\\", "/").split("/")
        
        try:
            # Extract parameters from path: plots_{mode}/Ma={Ma}/Pe={Pe}/N={N}/data_lambda={lambda}_k={k}.csv
            Ma = float(parts[1].split("=")[1])
            Pe = float(parts[2].split("=")[1])
            N = int(parts[3].split("=")[1])
            
            filename = parts[4]
            filename_no_ext = os.path.splitext(filename)[0]
            file_parts = filename_no_ext.split("_")
            lambda_val = float(file_parts[1].split("=")[1])
            k_val = float(file_parts[2].split("=")[1])
        except Exception as e:
            print(f"Skipping file {filepath} due to parsing error: {e}")
            continue
            
        # Parse CSV file to find c_n and U_D
        c_n = []
        x_fluid = []
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    row_parts = line.strip().split(",")
                    # Column 1 (index 1) is c_n
                    if len(row_parts) >= 2 and row_parts[1]:
                        c_n.append(float(row_parts[1]))
                    # Column 2 (index 2) is x_fluid
                    if len(row_parts) >= 3 and row_parts[2]:
                        x_fluid.append(float(row_parts[2]))
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            continue
        
        # U_D is stored at index 5*N + 1 of x_fluid, delta_U at 5*N + 2
        ud_idx = 5 * N + 1
        delta_u_idx = 5 * N + 2
        if len(x_fluid) > delta_u_idx:
            U_D = x_fluid[ud_idx]
            delta_U = x_fluid[delta_u_idx]
            U_p = delta_U + U_D
            data_list.append({
                'Ma': Ma,
                'Pe': Pe,
                'N': N,
                'lambda_val': lambda_val,
                'k_val': k_val,
                'U_D': U_D,
                'delta_U': delta_U,
                'U_p': U_p,
                'c_n': np.array(c_n)
            })
            
    return data_list

def plot_ud_vs_pe(data_list, mode='particle'):
    """
    Groups data by (N, k, lambda) and plots U_D vs. Pe on a semi-log scale
    with separate lines for each Marangoni number (Ma).
    """
    combinations = set((d['N'], d['k_val'], d['lambda_val']) for d in data_list)
    
    for N, k, lambda_val in combinations:
        subset = [d for d in data_list if d['N'] == N and d['k_val'] == k and d['lambda_val'] == lambda_val]
        mas = sorted(list(set(d['Ma'] for d in subset)))
        
        plt.figure(figsize=(10, 6.5))
        colors = plt.cm.plasma(np.linspace(0.1, 0.8, len(mas)))
        
        for idx, Ma in enumerate(mas):
            ma_subset = [d for d in subset if d['Ma'] == Ma]
            ma_subset = sorted(ma_subset, key=lambda x: x['Pe'])
            
            pes = [d['Pe'] for d in ma_subset]
            uds = [d['U_D'] for d in ma_subset]
            
            plt.plot(pes, uds, 'o-', color=colors[idx], linewidth=2.5, 
                     markersize=8, label=f"Ma = {Ma}")
            
        plt.xscale('log')
        plt.xlabel('Peclet Number ($Pe$)', fontsize=12)
        plt.ylabel('Drop Velocity ($U_D$)', fontsize=12)
        plt.title(f'Drop Velocity $U_D$ vs. $Pe$ ($N={N}$, $\\lambda={lambda_val}$, $k={k}$)', fontsize=14, fontweight='bold', pad=15)
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend(fontsize=11, loc="best")
        plt.tight_layout()
        
        misc_dir = os.path.join(RUNS_DIR, f"plots_{mode}", "misc")
        os.makedirs(misc_dir, exist_ok=True)
        output_path = os.path.join(misc_dir, f"ud_vs_pe_N={N}_lambda={lambda_val}_k={k}.png")
        plt.savefig(output_path, dpi=250)
        print(f"Saved drop velocity comparison plot to: {output_path}")
        plt.close()

def plot_surfactant_profiles(data_list, mode='particle'):
    """
    Plots the surfactant concentration profile Gamma(theta) along the droplet interface
    from theta = 0 to pi for different Pe numbers (at a fixed Ma, e.g., Ma=1.0).
    """
    combinations = set((d['N'], d['k_val'], d['lambda_val']) for d in data_list)
    
    for N, k, lambda_val in combinations:
        subset = [d for d in data_list if d['N'] == N and d['k_val'] == k and d['lambda_val'] == lambda_val]
        mas = list(set(d['Ma'] for d in subset))
        if not mas:
            continue
            
        # Select representative Ma = 1.0 (or fallback to highest Ma)
        Ma_target = 1.0 if 1.0 in mas else max(mas)
        
        ma_subset = [d for d in subset if d['Ma'] == Ma_target]
        ma_subset = sorted(ma_subset, key=lambda x: x['Pe'])
        
        if not ma_subset:
            continue
            
        # Grid of angles: 0 (top pole) to 180 (bottom pole) degrees
        theta_deg = np.linspace(0, 180, 200)
        theta_rad = np.radians(theta_deg)
        mew = np.cos(theta_rad)
        
        # Evaluate Legendre Polynomials P_n(cos theta) for modes n = 1 to N
        V = np.polynomial.legendre.legvander(mew, N)
        P = V[:, 1 : N + 1]  # shape (200, N)
        
        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(ma_subset)))
        
        for idx, d in enumerate(ma_subset):
            c_n_vec = d['c_n']
            # Gamma(theta)/Gamma_0 = 1.0 + sum_{n=1}^N c_n * P_n(cos theta)
            gamma_vals = 1.0 + np.dot(P, c_n_vec)
            
            plt.plot(theta_deg, gamma_vals, color=colors[idx], linewidth=2.5, 
                     label=f"Pe = {d['Pe']}")
            
        plt.xlabel(r"Angle $\theta$ (degrees from top pole)", fontsize=12)
        plt.ylabel(r"Relative Surfactant Concentration $\Gamma(\theta) / \Gamma_0$", fontsize=12)
        plt.title(f"Surfactant Profile along Droplet Interface ($Ma={Ma_target}$, $N={N}$, $\\lambda={lambda_val}$)", fontsize=14, fontweight='bold', pad=15)
        plt.xlim(0, 180)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        misc_dir = os.path.join(RUNS_DIR, f"plots_{mode}", "misc")
        os.makedirs(misc_dir, exist_ok=True)
        output_path = os.path.join(misc_dir, f"gamma_profile_N={N}_lambda={lambda_val}_k={k}.png")
        plt.savefig(output_path, dpi=250)
        print(f"Saved surfactant profile plot to: {output_path}")
        plt.close()

def plot_retardation_ratio(data_list, mode='particle'):
    """
    Plots the retardation ratio U_D / U_D_clean vs Pe for different Ma.
    U_D_clean is taken as the U_D value at Pe = 0.01 (clean limit).
    """
    combinations = set((d['N'], d['k_val'], d['lambda_val']) for d in data_list)
    
    for N, k, lambda_val in combinations:
        subset = [d for d in data_list if d['N'] == N and d['k_val'] == k and d['lambda_val'] == lambda_val]
        mas = sorted(list(set(d['Ma'] for d in subset)))
        
        plt.figure(figsize=(10, 6.5))
        colors = plt.cm.plasma(np.linspace(0.1, 0.8, len(mas)))
        
        for idx, Ma in enumerate(mas):
            ma_subset = [d for d in subset if d['Ma'] == Ma]
            ma_subset = sorted(ma_subset, key=lambda x: x['Pe'])
            
            # Find U_D_clean (lowest Pe, Pe = 0.01)
            clean_point = [d for d in ma_subset if d['Pe'] == 0.01]
            if clean_point:
                U_D_clean = clean_point[0]['U_D']
            else:
                U_D_clean = ma_subset[0]['U_D']
                
            pes = [d['Pe'] for d in ma_subset]
            ratios = [d['U_D'] / U_D_clean for d in ma_subset]
            
            plt.plot(pes, ratios, 'o-', color=colors[idx], linewidth=2.5, 
                     markersize=8, label=f"Ma = {Ma}")
            
        plt.xscale('log')
        plt.xlabel('Peclet Number ($Pe$)', fontsize=12)
        plt.ylabel('Retardation Ratio $U_D / U_{D, \\text{clean}}$', fontsize=12)
        plt.title(f'Velocity Retardation Ratio vs. $Pe$ ($N={N}$, $\\lambda={lambda_val}$, $k={k}$)', fontsize=14, fontweight='bold', pad=15)
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.ylim(0, 1.1)
        plt.legend(fontsize=11, loc="best")
        plt.tight_layout()
        
        misc_dir = os.path.join(RUNS_DIR, f"plots_{mode}", "misc")
        os.makedirs(misc_dir, exist_ok=True)
        output_path = os.path.join(misc_dir, f"retardation_ratio_N={N}_lambda={lambda_val}_k={k}.png")
        plt.savefig(output_path, dpi=250)
        print(f"Saved retardation ratio plot to: {output_path}")
        plt.close()

def plot_gamma_vs_mew(data_list, mode='particle'):
    """
    Plots surfactant concentration Gamma(mew) vs. mew (from -1 to 1)
    for each physical system (N, k, lambda) and Marangoni number (Ma),
    showing the effect of Peclet number (Pe).
    """
    # Group by physical system
    combinations = set((d['N'], d['k_val'], d['lambda_val']) for d in data_list)
    
    for N, k, lambda_val in combinations:
        subset = [d for d in data_list if d['N'] == N and d['k_val'] == k and d['lambda_val'] == lambda_val]
        mas = sorted(list(set(d['Ma'] for d in subset)))
        
        for Ma in mas:
            ma_subset = [d for d in subset if d['Ma'] == Ma]
            ma_subset = sorted(ma_subset, key=lambda x: x['Pe'])
            
            if not ma_subset:
                continue
                
            mew_vals = np.linspace(-1.0, 1.0, 200)
            
            # Evaluate Legendre polynomials at mew_vals
            V = np.polynomial.legendre.legvander(mew_vals, N)
            P = V[:, 1 : N + 1]  # shape (200, N)
            
            plt.figure(figsize=(9, 6))
            colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(ma_subset)))
            
            for idx, d in enumerate(ma_subset):
                c_n_vec = d['c_n']
                # Gamma(mew) = 1.0 + sum_{n=1}^N c_n P_n(mew)
                gamma_vals = 1.0 + np.dot(P, c_n_vec)
                
                plt.plot(mew_vals, gamma_vals, color=colors[idx], linewidth=2.5, 
                         label=f"Pe = {d['Pe']}")
                
            plt.xlabel(r"Coordinate $\mu = \cos(\theta)$", fontsize=12)
            plt.ylabel(r"Relative Surfactant Concentration $\Gamma(\mu) / \Gamma_0$", fontsize=12)
            plt.title(f"Surfactant Profile $\\Gamma(\\mu)$ vs. $\\mu$ ($Ma={Ma}$, $N={N}$, $\\lambda={lambda_val}$, $k={k}$)", fontsize=13, fontweight='bold', pad=15)
            plt.xlim(-1.0, 1.0)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend(fontsize=11)
            plt.tight_layout()
            
            misc_dir = os.path.join(RUNS_DIR, f"plots_{mode}", "misc")
            os.makedirs(misc_dir, exist_ok=True)
            output_path = os.path.join(misc_dir, f"gamma_vs_mew_N={N}_lambda={lambda_val}_k={k}_Ma={Ma}.png")
            plt.savefig(output_path, dpi=250)
            print(f"Saved surfactant vs. mew plot to: {output_path}")
            plt.close()

def plot_up_vs_pe(data_list, mode='particle'):
    """
    Groups data by (N, k, lambda) and plots U_p / U_S vs. Pe on a semi-log scale
    with separate lines for each Marangoni number (Ma).
    """
    combinations = set((d['N'], d['k_val'], d['lambda_val']) for d in data_list)
    
    for N, k, lambda_val in combinations:
        subset = [d for d in data_list if d['N'] == N and d['k_val'] == k and d['lambda_val'] == lambda_val]
        mas = sorted(list(set(d['Ma'] for d in subset)))
        
        plt.figure(figsize=(10, 6.5))
        colors = plt.cm.plasma(np.linspace(0.1, 0.8, len(mas)))
        
        for idx, Ma in enumerate(mas):
            ma_subset = [d for d in subset if d['Ma'] == Ma]
            ma_subset = sorted(ma_subset, key=lambda x: x['Pe'])
            
            # Filter out any points that don't have U_p
            valid_points = [pt for pt in ma_subset if 'U_p' in pt]
            if not valid_points:
                continue
            pes = [pt['Pe'] for pt in valid_points]
            if mode == 'squirmer':
                ups = [pt['U_D'] + pt['delta_U'] for pt in valid_points]
            else:
                ups = [pt['U_p'] for pt in valid_points]
            
            p_label = "Squirmer Velocity ($U_S$)" if mode == 'squirmer' else "Particle Velocity ($U_p$)"
            plt.plot(pes, ups, 'o-', color=colors[idx], linewidth=2.5, 
                     markersize=8, label=f"Ma = {Ma}")
            
        plt.xscale('log')
        plt.xlabel('Peclet Number ($Pe$)', fontsize=12)
        plt.ylabel(p_label, fontsize=12)
        
        title_label = f'Squirmer Velocity $U_S$ vs. $Pe$' if mode == 'squirmer' else f'Particle Velocity $U_p$ vs. $Pe$'
        plt.title(f'{title_label} ($N={N}$, $\\lambda={lambda_val}$, $k={k}$)', fontsize=14, fontweight='bold', pad=15)
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend(fontsize=11, loc="best")
        plt.tight_layout()
        
        misc_dir = os.path.join(RUNS_DIR, f"plots_{mode}", "misc")
        os.makedirs(misc_dir, exist_ok=True)
        output_path = os.path.join(misc_dir, f"up_vs_pe_N={N}_lambda={lambda_val}_k={k}.png")
        plt.savefig(output_path, dpi=250)
        print(f"Saved particle velocity comparison plot to: {output_path}")
        plt.close()

def plot_joint_velocities(data_list, mode='particle'):
    """
    Groups data by (N, k, lambda) and plots U_D (drop velocity, solid lines)
    and U_p / U_S (particle/squirmer velocity, dashed lines) vs. Pe on a semi-log scale
    with separate colors for each Marangoni number (Ma).
    """
    combinations = set((d['N'], d['k_val'], d['lambda_val']) for d in data_list)
    
    for N, k, lambda_val in combinations:
        subset = [d for d in data_list if d['N'] == N and d['k_val'] == k and d['lambda_val'] == lambda_val]
        mas = sorted(list(set(d['Ma'] for d in subset)))
        
        plt.figure(figsize=(10, 6.5))
        colors = plt.cm.plasma(np.linspace(0.1, 0.8, len(mas)))
        
        for idx, Ma in enumerate(mas):
            ma_subset = [d for d in subset if d['Ma'] == Ma]
            ma_subset = sorted(ma_subset, key=lambda x: x['Pe'])
            
            valid_points = [pt for pt in ma_subset if 'U_p' in pt]
            if not valid_points:
                continue
                
            pes = [pt['Pe'] for pt in valid_points]
            uds = [pt['U_D'] for pt in valid_points]
            if mode == 'squirmer':
                ups = [pt['U_D'] + pt['delta_U'] for pt in valid_points]
            else:
                ups = [pt['U_p'] for pt in valid_points]
            
            label_up = f"$U_S$ (Ma = {Ma})" if mode == 'squirmer' else f"$U_p$ (Ma = {Ma})"
            
            # Solid line for U_D
            plt.plot(pes, uds, 'o-', color=colors[idx], linewidth=2.5, 
                     markersize=8, label=f"$U_D$ (Ma = {Ma})")
            # Dashed line for U_p
            plt.plot(pes, ups, '^--', color=colors[idx], linewidth=2.0, 
                     markersize=8, label=label_up)
            
        plt.xscale('log')
        plt.xlabel('Peclet Number ($Pe$)', fontsize=12)
        
        y_label = 'Velocity ($U_D$, $U_S$)' if mode == 'squirmer' else 'Velocity ($U_D$, $U_p$)'
        plt.ylabel(y_label, fontsize=12)
        
        title_label = f'Velocities $U_D$ (Solid) & $U_S$ (Dashed) vs. $Pe$' if mode == 'squirmer' else f'Velocities $U_D$ (Solid) & $U_p$ (Dashed) vs. $Pe$'
        plt.title(f'{title_label}\n($N={N}$, $\\lambda={lambda_val}$, $k={k}$)', fontsize=14, fontweight='bold', pad=15)
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend(fontsize=10, loc="best", ncol=2)
        plt.tight_layout()
        
        misc_dir = os.path.join(RUNS_DIR, f"plots_{mode}", "misc")
        os.makedirs(misc_dir, exist_ok=True)
        output_path = os.path.join(misc_dir, f"joint_velocity_N={N}_lambda={lambda_val}_k={k}.png")
        plt.savefig(output_path, dpi=250)
        print(f"Saved joint velocity comparison plot to: {output_path}")
        plt.close()

if __name__ == "__main__":
    for mode in ['particle', 'squirmer']:
        print(f"\nScanning plots_{mode} directory for CSV files...")
        data = parse_data(mode)
        print(f"Found {len(data)} data points for {mode}.")
        
        if data:
            # plot_ud_vs_pe(data, mode)
            # plot_up_vs_pe(data, mode)
            # plot_joint_velocities(data, mode)
            # plot_gamma_vs_mew(data, mode)
            # plot_surfactant_profiles(data, mode)
            # plot_retardation_ratio(data, mode)
            pass
