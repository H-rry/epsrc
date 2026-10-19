import numpy as np
import os

from squirmer_drop import stokes as fluid
from squirmer_drop import surfactant
from squirmer_drop.coupling import extract_b_n
from squirmer_drop.paths import VERIFICATION_DATA_DIR

def save_converged_data_custom(c_n, x_fluid, N, k, lambda_val, Ma, Pe, mode, folder_prefix):
    dir_path = os.path.join(VERIFICATION_DATA_DIR, folder_prefix, f"Ma={Ma}", f"Pe={Pe}", f"N={N}")
    os.makedirs(dir_path, exist_ok=True)
    csv_path = os.path.join(dir_path, f"data_lambda={lambda_val}_k={k}.csv")
    
    with open(csv_path, 'w') as f:
        f.write("index,c_n,x_fluid\n")
        max_len = max(len(c_n), len(x_fluid))
        for idx in range(max_len):
            c_val = c_n[idx] if idx < len(c_n) else ""
            x_val = x_fluid[idx] if idx < len(x_fluid) else ""
            f.write(f"{idx},{c_val},{x_val}\n")
    print(f"Saved converged data to: {csv_path}")

def run_coupling_iteration_custom(N=5, k=0.3, lambda_val=0.5, Ma=1.0, Pe=1.0, max_iter=100, tol=1e-6, omega=0.1, mode='particle', folder_prefix=None):
    if Ma == 0.0:
        omega = 1.0
        
    print(f"\n" + "="*60)
    print(f" RUNNING VERIFICATION COUPLING ({mode.upper()}) ".center(60, "="))
    print(f"="*60)
    print(f"Parameters: N={N}, k={k}, lambda={lambda_val}, Ma={Ma}, Pe={Pe}")
    print(f"Relaxation factor (omega): {omega}")
    print(f"Folder Prefix: {folder_prefix}\n")

    if mode == 'squirmer':
        A_n = 0.1 ** np.arange(1, N + 1)
        B_n = 0.1 ** np.arange(1, N + 1)
    else:
        A_n = None
        B_n = None

    c_n = np.zeros(N)
    A = fluid.assemble_A(N, k, lambda_val, shifted=True)
    for iteration in range(1, max_iter + 1):
        b = fluid.assemble_b(N, Ma, c_n=c_n, mode=mode, lambda_val=lambda_val, k=k, A_n=A_n, B_n=B_n)
        x_fluid = np.linalg.solve(A, b)
        
        b_n_new = extract_b_n(x_fluid, N)
        
        try:
            c_n_new = surfactant.solve_c_n(b_n_new, Pe, Ma=Ma, k=k, lambda_val=lambda_val)
        except Exception as e:
            print(f"\n[ERROR] Surfactant solver failed at iteration {iteration}: {e}")
            return None, None, None
        
        diff = np.linalg.norm(c_n_new - c_n)
        if iteration == 1 or iteration % 10 == 0 or diff < tol:
            print(f"  Iteration {iteration:3d} | Difference: {diff:.3e} | U_D: {x_fluid[5*N+1]:.6e}")
        
        if diff < tol:
            b_final = fluid.assemble_b(N, Ma, c_n=c_n_new, mode=mode, lambda_val=lambda_val, k=k, A_n=A_n, B_n=B_n)
            x_fluid = np.linalg.solve(A, b_final)
            
            print(f"\n[CONVERGED] Convergence achieved after {iteration} iterations.")
            
            # Save plots using folder_prefix
            fluid.plot_velocity_field(x_fluid, N, k, lambda_val, Ma=Ma, Pe=Pe, mode=mode, folder_prefix=folder_prefix)
            fluid.plot_stream_function(x_fluid, N, k, lambda_val, Ma=Ma, Pe=Pe, mode=mode, folder_prefix=folder_prefix)
            
            save_converged_data_custom(c_n_new, x_fluid, N, k, lambda_val, Ma, Pe, mode=mode, folder_prefix=folder_prefix)
            
            fluid.run_unit_tests(x_fluid, N, k, lambda_val, Ma=Ma, tolerance=1e-8, c_n=c_n_new, mode=mode, A_n=A_n, B_n=B_n)
            return c_n_new, b_n_new, x_fluid
        
        c_n = omega * c_n_new + (1.0 - omega) * c_n

    print(f"\n[WARNING] Did not converge within {max_iter} iterations.")
    return c_n, None, None

def main():
    # 1. Folders
    folder_particle = "analytical_solution_verification_particle"
    folder_squirmer = "analytical_solution_verification_squirmer"
    
    # 2. Particle Sweeps (Ma = 0, lambda = 2)
    # k values: 0.001, 0.01, 0.02, 0.1, 0.9, 0.95, 0.99, 0.999
    k_particle = [0.001, 0.01, 0.02, 0.1, 0.9, 0.95, 0.99, 0.999]
    for k in k_particle:
        run_coupling_iteration_custom(N=5, k=k, lambda_val=2.0, Ma=0.0, Pe=1.0, max_iter=100, tol=1e-6, omega=0.1, mode='particle', folder_prefix=folder_particle)
        
    # 3. Squirmer Sweeps (Ma = 0.1, k = 0.2, lambda = 2.0)
    # Pe values: 0.1, 0.01, 0.001, 0.0001
    pe_squirmer = [0.1, 0.01, 0.001, 0.0001]
    for Pe in pe_squirmer:
        run_coupling_iteration_custom(N=5, k=0.2, lambda_val=2.0, Ma=0.1, Pe=Pe, max_iter=100, tol=1e-6, omega=0.1, mode='squirmer', folder_prefix=folder_squirmer)

if __name__ == "__main__":
    main()
