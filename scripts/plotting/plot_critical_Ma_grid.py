import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

from squirmer_drop.paths import CRITICAL_MA_GRID_DIR, CRITICAL_MA_FIGURES_DIR, VERIFICATION_DATA_DIR

output_dir = CRITICAL_MA_FIGURES_DIR
os.makedirs(output_dir, exist_ok=True)

# Configure styling
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

def load_data():
    combined_csv = os.path.join(CRITICAL_MA_GRID_DIR, "critical_Ma_grid_combined.csv")
    if not os.path.exists(combined_csv):
        print(f"Error: dataset not found: {combined_csv}")
        return None
    
    rows = []
    with open(combined_csv, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                'lambda': float(r['lambda']),
                'k': float(r['k']),
                'critical_Ma_Us': float(r['critical_Ma_Us']),
                'critical_Ma_Ud': float(r['critical_Ma_Ud'])
            })
    return rows

def generate_plots():
    data = load_data()
    if data is None:
        return
    
    lambdas = sorted(list(set(r['lambda'] for r in data)))
    ks = sorted(list(set(r['k'] for r in data)))
    
    # ---------------- 1. Ma_crit vs k (Parametric in lambda) ----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(lambdas)))
    
    for idx, L in enumerate(lambdas):
        subset = [r for r in data if abs(r['lambda'] - L) < 1e-6]
        subset = sorted(subset, key=lambda x: x['k'])
        k_vals = [r['k'] for r in subset]
        us_vals = [r['critical_Ma_Us'] for r in subset]
        ud_vals = [r['critical_Ma_Ud'] for r in subset]
        
        ax1.plot(k_vals, us_vals, 'o-', color=colors[idx], linewidth=2.0, markersize=7, label=rf"$\lambda = {L}$")
        ax2.plot(k_vals, ud_vals, 's-', color=colors[idx], linewidth=2.0, markersize=7, label=rf"$\lambda = {L}$")
        
    ax1.set_xlabel(r"Core Radius Ratio $k = R_{\text{core}}/R_{\text{drop}}$", fontsize=11, fontweight='bold')
    ax1.set_ylabel(r"Critical Marangoni Number $Ma_{\text{crit}, U_s}$", fontsize=11, fontweight='bold')
    ax1.set_title(r"Swimmer Velocity Validity: $Ma_{\text{crit}, U_s}$ vs. $k$", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(title=r"Viscosity Ratio $\lambda$", fontsize=9, frameon=True, facecolor='#ffffff')
    
    ax2.set_xlabel(r"Core Radius Ratio $k = R_{\text{core}}/R_{\text{drop}}$", fontsize=11, fontweight='bold')
    ax2.set_ylabel(r"Critical Marangoni Number $Ma_{\text{crit}, U_d}$", fontsize=11, fontweight='bold')
    ax2.set_title(r"Drop Velocity Validity: $Ma_{\text{crit}, U_d}$ vs. $k$", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(title=r"Viscosity Ratio $\lambda$", fontsize=9, frameon=True, facecolor='#ffffff')
    
    plt.tight_layout()
    p1 = os.path.join(output_dir, "Ma_crit_vs_k_parametric_lambda.png")
    plt.savefig(p1, dpi=250)
    plt.close(fig)
    print(f"[SAVED] {p1}")
    
    # ---------------- 2. Ma_crit vs lambda (Log-Log, Parametric in k) ----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    k_colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(ks)))
    
    for idx, k in enumerate(ks):
        subset = [r for r in data if abs(r['k'] - k) < 1e-6]
        subset = sorted(subset, key=lambda x: x['lambda'])
        l_vals = [r['lambda'] for r in subset]
        us_vals = [r['critical_Ma_Us'] for r in subset]
        ud_vals = [r['critical_Ma_Ud'] for r in subset]
        
        ax1.plot(l_vals, us_vals, 'o-', color=k_colors[idx], linewidth=2.0, markersize=7, label=rf"$k = {k}$")
        ax2.plot(l_vals, ud_vals, 's-', color=k_colors[idx], linewidth=2.0, markersize=7, label=rf"$k = {k}$")
        
    for ax, title, ylabel in [(ax1, r"Swimmer Velocity: $Ma_{\text{crit}, U_s}$ vs. $\lambda$", r"$Ma_{\text{crit}, U_s}$ (Log Scale)"),
                              (ax2, r"Drop Velocity: $Ma_{\text{crit}, U_d}$ vs. $\lambda$", r"$Ma_{\text{crit}, U_d}$ (Log Scale)")]:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r"Viscosity Ratio $\lambda$ (Log Scale)", fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(title=r"Core Ratio $k$", fontsize=9, frameon=True, facecolor='#ffffff')
        
    plt.tight_layout()
    p2 = os.path.join(output_dir, "Ma_crit_vs_lambda_parametric_k.png")
    plt.savefig(p2, dpi=250)
    plt.close(fig)
    print(f"[SAVED] {p2}")
    
    # ---------------- 3. 2D Heatmaps / Contours in (k, lambda) Space ----------------
    L_grid = np.zeros((len(lambdas), len(ks)))
    K_grid = np.zeros((len(lambdas), len(ks)))
    Z_Us = np.zeros((len(lambdas), len(ks)))
    Z_Ud = np.zeros((len(lambdas), len(ks)))
    
    for l_idx, L in enumerate(lambdas):
        for k_idx, k in enumerate(ks):
            rec = next(r for r in data if abs(r['lambda'] - L) < 1e-6 and abs(r['k'] - k) < 1e-6)
            L_grid[l_idx, k_idx] = L
            K_grid[l_idx, k_idx] = k
            Z_Us[l_idx, k_idx] = rec['critical_Ma_Us']
            Z_Ud[l_idx, k_idx] = rec['critical_Ma_Ud']
            
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    c1 = ax1.pcolormesh(K_grid, L_grid, Z_Us, shading='gouraud', cmap='plasma')
    ax1.set_yscale('log')
    ax1.set_xlabel(r"Core Radius Ratio $k$", fontsize=11, fontweight='bold')
    ax1.set_ylabel(r"Viscosity Ratio $\lambda$ (Log Scale)", fontsize=11, fontweight='bold')
    ax1.set_title(r"2D Map of $Ma_{\text{crit}}$ for Swimmer Velocity $U_s$", fontsize=12, fontweight='bold')
    fig.colorbar(c1, ax=ax1, label=r"$Ma_{\text{crit}, U_s}$")
    
    c2 = ax2.pcolormesh(K_grid, L_grid, Z_Ud, shading='gouraud', cmap='viridis')
    ax2.set_yscale('log')
    ax2.set_xlabel(r"Core Radius Ratio $k$", fontsize=11, fontweight='bold')
    ax2.set_ylabel(r"Viscosity Ratio $\lambda$ (Log Scale)", fontsize=11, fontweight='bold')
    ax2.set_title(r"2D Map of $Ma_{\text{crit}}$ for Droplet Velocity $U_d$", fontsize=12, fontweight='bold')
    fig.colorbar(c2, ax=ax2, label=r"$Ma_{\text{crit}, U_d}$")
    
    plt.tight_layout()
    p3 = os.path.join(output_dir, "Ma_crit_2D_heatmaps.png")
    plt.savefig(p3, dpi=250)
    plt.close(fig)
    print(f"[SAVED] {p3}")
    
    # ---------------- 4. Verification Check Plot ----------------
    verif_csv = os.path.join(VERIFICATION_DATA_DIR, "critical_Ma_verification_table.csv")
    if os.path.exists(verif_csv):
        v_rows = []
        with open(verif_csv, 'r') as f:
            v_reader = csv.DictReader(f)
            for r in v_reader:
                v_rows.append({
                    'lambda': float(r['lambda']),
                    'k': float(r['k']),
                    'factor': float(r['factor']),
                    'err': float(r['relative_error_pct'])
                })
        
        fig, ax = plt.subplots(figsize=(8, 6))
        for L in lambdas:
            subset_L = [r for r in v_rows if abs(r['lambda'] - L) < 1e-6]
            factors = [0.5, 1.0, 1.5]
            avg_errs = [np.mean([r['err'] for r in subset_L if abs(r['factor'] - f) < 1e-4]) for f in factors]
            ax.plot(factors, avg_errs, 'o-', linewidth=2.0, label=rf"$\lambda = {L}$")
            
        ax.axhline(2.0, color='red', linestyle='--', linewidth=1.5, label='2.0% Critical Threshold')
        ax.set_xticks([0.5, 1.0, 1.5])
        ax.set_xticklabels([r'Subcritical ($0.5\,Ma_c$)', r'Critical ($1.0\,Ma_c$)', r'Supercritical ($1.5\,Ma_c$)'])
        ax.set_ylabel(r'Relative Error vs. Analytical Solution (%)', fontsize=11, fontweight='bold')
        ax.set_title('Verification of Critical Threshold Across Regimes', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, facecolor='#ffffff')
        plt.tight_layout()
        p4 = os.path.join(output_dir, "verification_error_crossings.png")
        plt.savefig(p4, dpi=250)
        plt.close(fig)
        print(f"[SAVED] {p4}")

if __name__ == "__main__":
    generate_plots()
