import os
import numpy as np
import csv

from squirmer_drop.paths import RUNS_DIR

# Import the solver coupling harness in case we need to generate any missing points
try:
    from squirmer_drop.coupling import run_coupling_iteration
except ImportError:
    run_coupling_iteration = None

def get_data_point(k, lambda_val, Ma, Pe=0.1, N=5):
    """
    Reads U_D and U_p from the corresponding CSV file.
    If the file does not exist, runs the solver to compute and save it.
    """
    # Format paths considering float representation
    l_str = str(int(lambda_val)) if float(lambda_val).is_integer() else str(lambda_val)
    k_str = str(int(k)) if float(k).is_integer() else f"{round(k, 2)}"
    
    # Try different naming formats
    possible_names = [
        f"data_lambda={l_str}_k={k_str}.csv",
        f"data_lambda={float(lambda_val)}_k={float(k)}.csv",
        f"data_lambda={l_str}_k={float(k)}.csv",
        f"data_lambda={float(lambda_val)}_k={k_str}.csv"
    ]
    
    dir_path = os.path.join(RUNS_DIR, "plots_particle", f"Ma={Ma}", f"Pe={Pe}", f"N={N}")
    csv_path = None
    
    for name in possible_names:
        temp_path = os.path.join(dir_path, name)
        if os.path.exists(temp_path):
            csv_path = temp_path
            break
            
    # If the file is missing and solver is available, generate it
    if csv_path is None:
        if run_coupling_iteration is not None:
            print(f"File missing for k={k}, lambda={lambda_val}, Ma={Ma}. Running solver...")
            # Run solver (it automatically saves the CSV)
            run_coupling_iteration(N=N, k=k, lambda_val=lambda_val, Ma=Ma, Pe=Pe, max_iter=500, omega=0.1, mode='particle')
            
            # Search again
            for name in possible_names:
                temp_path = os.path.join(dir_path, name)
                if os.path.exists(temp_path):
                    csv_path = temp_path
                    break
        else:
            print(f"Warning: Data file not found and solver not available for k={k}, lambda={lambda_val}, Ma={Ma}")
            return None, None
            
    if csv_path is None or not os.path.exists(csv_path):
        return None, None
        
    # Read U_D and U_p from CSV
    x_fluid = []
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader) # skip header
            for row in reader:
                if len(row) >= 3 and row[2]:
                    x_fluid.append(float(row[2]))
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None, None
        
    ud_idx = 5 * N + 1
    delta_u_idx = 5 * N + 2
    if len(x_fluid) > delta_u_idx:
        U_D = x_fluid[ud_idx]
        delta_U = x_fluid[delta_u_idx]
        U_p = U_D + delta_U
        return U_D, U_p
        
    return None, None

def main():
    k_list = np.arange(0.1, 0.9, 0.1) # [0.1, 0.2, ..., 0.8]
    Ma_list = [0.1, 1.0, 10.0]
    Pe = 0.1
    Lambda_list = [0.2, 0.5, 1.0, 2.0, 5.0]
    
    # Sort order: increment k first, then lambda, then Ma
    # This means: Ma (outer) -> lambda (middle) -> k (inner, fastest varying)
    output_rows = []
    
    for Ma in Ma_list:
        for lambda_val in Lambda_list:
            for k in k_list:
                U_D, U_p = get_data_point(k, lambda_val, Ma, Pe=Pe)
                if U_D is not None:
                    output_rows.append({
                        'lambda': lambda_val,
                        'k': round(k, 2),
                        'Ma': Ma,
                        'U_D': U_D,
                        'U_p': U_p
                    })
                else:
                    print(f"Failed to retrieve data for k={k}, lambda={lambda_val}, Ma={Ma}")
                    
    # Output path
    misc_dir = os.path.join(RUNS_DIR, "plots_particle", "misc")
    os.makedirs(misc_dir, exist_ok=True)
    out_csv_path = os.path.join(misc_dir, "particle_data.csv")
    
    with open(out_csv_path, 'w', newline='') as f:
        fieldnames = ['lambda', 'k', 'Ma', 'U_D', 'U_p']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)
            
    print(f"\nSuccessfully compiled {len(output_rows)} rows to: {out_csv_path}")

if __name__ == "__main__":
    main()
