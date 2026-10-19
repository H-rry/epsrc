import os
import sys
import csv
import numpy as np


from squirmer_drop import coupling as ch
from squirmer_drop import analytical_squirmer as tas
from squirmer_drop.paths import CRITICAL_MA_GRID_DIR, VERIFICATION_DATA_DIR

def run_grid_sweep():
    output_dir = CRITICAL_MA_GRID_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    lambdas = [0.1, 0.5, 1.0, 2.0, 10.0]
    ks = [0.2, 0.4, 0.6, 0.8]
    
    A_n = np.zeros(5)
    B_n = np.zeros(5)
    A_n[0] = 0.0
    B_n[0] = 1.0
    
    print("=" * 80)
    print(" 2D PARAMETER GRID SWEEP: CRITICAL MARANGONI NUMBER Ma_crit(lambda, k) ".center(80, "="))
    print("=" * 80)
    print(f"Viscosity ratios (lambda): {lambdas}")
    print(f"Core radius ratios (k)   : {ks}")
    print(f"Target Error Threshold   : 2.0%")
    print(f"Active Squirmer Modes    : A_1 = {A_n[0]}, B_1 = {B_n[0]}")
    print("=" * 80 + "\n")
    
    grid_Us = np.zeros((len(lambdas), len(ks)))
    grid_Ud = np.zeros((len(lambdas), len(ks)))
    combined_records = []
    
    total_points = len(lambdas) * len(ks)
    point_idx = 0
    
    for l_idx, L in enumerate(lambdas):
        for k_idx, k in enumerate(ks):
            point_idx += 1
            print(f"\n>>> [{point_idx}/{total_points}] Processing point: lambda = {L}, k = {k} <<<")
            
            # 1. Critical Ma for Us
            print(f"  Searching for Ma_crit (Swimmer Velocity Us)...")
            Ma_crit_Us, err_Us, Us_sol, Us_an, _ = ch.find_Ma_critical_point(
                lambda_val=L, k_val=k, vel_type='Us', A_n=A_n, B_n=B_n, target_err=2.0, starting_ma=0.1
            )
            grid_Us[l_idx, k_idx] = Ma_crit_Us
            
            # 2. Critical Ma for Ud (seed with Ma_crit_Us for efficiency)
            print(f"  Searching for Ma_crit (Drop Velocity Ud)...")
            Ma_crit_Ud, err_Ud, Ud_sol, Ud_an, _ = ch.find_Ma_critical_point(
                lambda_val=L, k_val=k, vel_type='Ud', A_n=A_n, B_n=B_n, target_err=2.0, starting_ma=Ma_crit_Us
            )
            grid_Ud[l_idx, k_idx] = Ma_crit_Ud
            
            combined_records.append({
                'lambda': L,
                'k': k,
                'critical_Ma_Us': Ma_crit_Us,
                'error_pct_Us': err_Us,
                'critical_Ma_Ud': Ma_crit_Ud,
                'error_pct_Ud': err_Ud
            })
            
    # Save combined tidy CSV
    combined_csv = os.path.join(output_dir, "critical_Ma_grid_combined.csv")
    with open(combined_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['lambda', 'k', 'critical_Ma_Us', 'error_pct_Us', 'critical_Ma_Ud', 'error_pct_Ud'])
        writer.writeheader()
        writer.writerows(combined_records)
    print(f"\n[SAVED] Tidy grid dataset: {combined_csv}")
    
    # Save Matrix CSV for Us
    matrix_csv_Us = os.path.join(output_dir, "critical_Ma_matrix_Us.csv")
    with open(matrix_csv_Us, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['lambda'] + [f"k={k}" for k in ks])
        for l_idx, L in enumerate(lambdas):
            writer.writerow([L] + list(grid_Us[l_idx, :]))
    print(f"[SAVED] Us Matrix grid dataset: {matrix_csv_Us}")
    
    # Save Matrix CSV for Ud
    matrix_csv_Ud = os.path.join(output_dir, "critical_Ma_matrix_Ud.csv")
    with open(matrix_csv_Ud, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['lambda'] + [f"k={k}" for k in ks])
        for l_idx, L in enumerate(lambdas):
            writer.writerow([L] + list(grid_Ud[l_idx, :]))
    print(f"[SAVED] Ud Matrix grid dataset: {matrix_csv_Ud}")
    
    return lambdas, ks, grid_Us, grid_Ud, combined_records

def verify_against_analytical(combined_records):
    print("\n" + "=" * 80)
    print(" VERIFICATION OF PREDICTIONS AGAINST ANALYTICAL SOLUTIONS ".center(80, "="))
    print("=" * 80)
    output_dir = CRITICAL_MA_GRID_DIR
    
    verification_records = []
    
    N = 5
    A_n = np.zeros(N)
    B_n = np.zeros(N)
    A_n[0] = 0.0
    B_n[0] = 1.0
    
    for rec in combined_records:
        L = rec['lambda']
        k = rec['k']
        Ma_c_Us = rec['critical_Ma_Us']
        Ma_c_Ud = rec['critical_Ma_Ud']
        
        # Test Us at 0.5*Ma_c, 1.0*Ma_c, 1.5*Ma_c
        for factor, label in [(0.5, 'subcritical'), (1.0, 'critical'), (1.5, 'supercritical')]:
            test_Ma = factor * Ma_c_Us
            c_n, b_n, x = ch.run_coupling_iteration(
                N=N, k=k, lambda_val=L, Ma=test_Ma, Pe=0.1, max_iter=1500, omega=0.05, mode='squirmer', A_n=A_n, B_n=B_n
            )
            if x is not None:
                U_d = x[5 * N + 1]
                U_s = U_d + x[5 * N + 2]
                U_s_an = tas.analytical_U_S(k, test_Ma, L, A_n[0], B_n[0])
                err_pct = 100.0 * abs(U_s - U_s_an) / abs(U_s) if U_s != 0 else 0.0
                
                verification_records.append({
                    'variable': 'Us',
                    'lambda': L,
                    'k': k,
                    'regime': label,
                    'factor': factor,
                    'Ma': test_Ma,
                    'solver_vel': U_s,
                    'analytical_vel': U_s_an,
                    'relative_error_pct': err_pct,
                    'meets_expected_bound': (err_pct < 2.0 if factor < 1.0 else (err_pct > 2.0 if factor > 1.0 else abs(err_pct - 2.0) < 0.25))
                })
                
    os.makedirs(VERIFICATION_DATA_DIR, exist_ok=True)
    verif_csv = os.path.join(VERIFICATION_DATA_DIR, "critical_Ma_verification_table.csv")
    with open(verif_csv, 'w', newline='') as f:
        fieldnames = ['variable', 'lambda', 'k', 'regime', 'factor', 'Ma', 'solver_vel', 'analytical_vel', 'relative_error_pct', 'meets_expected_bound']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(verification_records)
    print(f"\n[SAVED] Verification table: {verif_csv}")
    print("Verification against analytical solutions completed successfully!\n")

if __name__ == "__main__":
    lambdas, ks, grid_Us, grid_Ud, records = run_grid_sweep()
    verify_against_analytical(records)
