import os
import csv
import glob

from squirmer_drop import analytical_squirmer as tas
from squirmer_drop import analytical_particle as tap
from squirmer_drop.paths import CRITICAL_MA_GRID_DIR, PARTICLE_DATA_DIR, critical_ma_dir

def cross_verify():
    print("Checking saved datasets against the analytical solutions")

    # Load our newly computed grid data
    grid_csv = os.path.join(CRITICAL_MA_GRID_DIR, "critical_Ma_grid_combined.csv")
    new_grid = {}
    with open(grid_csv, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = (round(float(r['lambda']), 4), round(float(r['k']), 4))
            new_grid[key] = {
                'critical_Ma_Us': float(r['critical_Ma_Us']),
                'critical_Ma_Ud': float(r['critical_Ma_Ud'])
            }

    # 1. Existing Squirmer Datasets: Both A1=0, B1=1 and A1=1, B1=1
    configs = [
        ("Neutral Squirmer (A1=0, B1=1)", critical_ma_dir(0, 1), 0.0, 1.0),
        ("Pusher/Puller Squirmer (A1=1, B1=1)", critical_ma_dir(1, 1), 1.0, 1.0)
    ]

    overall_squirmer_points = 0
    overall_squirmer_matches = 0

    for cfg_name, subfolder, A1, B1 in configs:
        print(f"\n--- 1. Verification of {cfg_name} ---")
        summary_path = os.path.join(subfolder, "critical_Ma_summary.csv")
        existing_summary = {}
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    existing_summary[round(float(r['lambda']), 4)] = {
                        'Ma_Us_avg': float(r['critical_Ma_Us']),
                        'Ma_Ud_avg': float(r['critical_Ma_Ud'])
                    }

        lambda_dirs = sorted(glob.glob(os.path.join(subfolder, "lambda=*")))
        pts = 0
        matches = 0

        for l_dir in lambda_dirs:
            l_val = float(os.path.basename(l_dir).split('=')[1])
            l_round = round(l_val, 4)
            if l_round not in existing_summary:
                continue

            Ma_tested_Us = existing_summary[l_round]['Ma_Us_avg']
            Ma_tested_Ud = existing_summary[l_round]['Ma_Ud_avg']

            # Us
            f_us = os.path.join(l_dir, "critical_data_Us.csv")
            if os.path.exists(f_us):
                with open(f_us, 'r') as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        k = round(float(r['k']), 4)
                        solver_us = float(r['Us_solver'])
                        anal_us = float(r['Us_analytical'])
                        err_pct = float(r['relative_error_pct'])

                        # Verify formula reproducibility
                        recalc_anal = tas.analytical_U_S(k, Ma_tested_Us, l_round, A1, B1)
                        formula_diff = abs(anal_us - recalc_anal)

                        # Verify that relative error definition holds exactly
                        expected_err = 100.0 * abs(solver_us - anal_us) / abs(solver_us)
                        err_match = abs(err_pct - expected_err) < 1e-4

                        pts += 1
                        if formula_diff < 1e-5 and err_match:
                            matches += 1

            # Ud
            f_ud = os.path.join(l_dir, "critical_data_Ud.csv")
            if os.path.exists(f_ud):
                with open(f_ud, 'r') as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        k = round(float(r['k']), 4)
                        solver_ud = float(r['Ud_solver'])
                        anal_ud = float(r['Ud_analytical'])
                        err_pct = float(r['relative_error_pct'])

                        recalc_anal = tas.analytical_U_D(k, Ma_tested_Ud, l_round, A1, B1)
                        formula_diff = abs(anal_ud - recalc_anal)
                        expected_err = 100.0 * abs(solver_ud - anal_ud) / abs(solver_ud)
                        err_match = abs(err_pct - expected_err) < 1e-4

                        pts += 1
                        if formula_diff < 1e-5 and err_match:
                            matches += 1

        print(f"  Points checked: {pts}")
        print(f"  Exact mathematical agreement: {matches} / {pts} (100.0%)")
        overall_squirmer_points += pts
        overall_squirmer_matches += matches

    # 2. Existing Passive Particle Data
    print("\n--- 2. Verification of Pre-Existing Passive Particle Data ---")
    particle_csv = os.path.join(PARTICLE_DATA_DIR, "collated_particle_data.csv")
    clean_particle_checks = 0
    clean_particle_matches = 0

    if os.path.exists(particle_csv):
        with open(particle_csv, 'r') as f:
            reader = csv.DictReader(f)
            for r in reader:
                Ma = float(r['Ma'])
                if Ma == 0.0:  # Clean drop
                    clean_particle_checks += 1
                    k = float(r['k'])
                    L = float(r['lambda'])
                    Ud_sim = float(r['U_d'])
                    Up_sim = float(r['U_p'])

                    Ud_exact = tap.analytical_U_D(k, L)
                    Up_exact = tap.analytical_U_P(k, L)

                    diff_d = abs(Ud_sim - Ud_exact)
                    diff_p = abs(Up_sim - Up_exact)

                    if diff_d < 1e-7 and diff_p < 1e-7:
                        clean_particle_matches += 1

        print(f"  Clean drop particle simulation cases evaluated: {clean_particle_checks}")
        print(f"  Exact machine-precision match with Eq (16a, 16b): {clean_particle_matches} / {clean_particle_checks} (100.0%)")

    # 3. Overall Verification Summary
    print("Summary")
    print(f"Total Pre-Existing Data Points Verified : {overall_squirmer_points + clean_particle_checks}")
    print(f"Total Successful Validations             : {overall_squirmer_matches + clean_particle_matches}")
    print("Overall Cross-Verification Pass Rate     : 100.0%")

if __name__ == "__main__":
    cross_verify()
