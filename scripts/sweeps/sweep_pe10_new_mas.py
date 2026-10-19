#!/usr/bin/env python3
"""
Adds Ma = 0.01 and 0.05 at Pe = 10 to the ratio sweeps (data/ratio_sweeps/).
"""

import os
import time
import csv
import numpy as np


from squirmer_drop.coupling import solve_clean_drop, run_single_coupling
from squirmer_drop.paths import RATIO_SWEEPS_DIR

def run_pe10_targeted_sweeps():
    data_dir = RATIO_SWEEPS_DIR
    os.makedirs(data_dir, exist_ok=True)
    csv_master = os.path.join(data_dir, "ratio_sweep_master.csv")
    fieldnames = [
        "mode_label", "A_1", "B_1", "Pe", "lambda", "Ma", "k",
        "U_s", "U_d", "U_s_clean", "U_d_clean",
        "ratio_U_s", "ratio_U_d", "U_s_norm", "U_d_norm",
        "iterations", "residual", "converged"
    ]

    existing_keys = set()
    if os.path.exists(csv_master):
        with open(csv_master, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    k_val = round(float(r["k"]), 3)
                    existing_keys.add((r["mode_label"], float(r["Pe"]), float(r["lambda"]), float(r["Ma"]), k_val))
                except Exception:
                    pass

    configs = [
        {"label": "A1=0_B1=1", "A_1": 0.0, "B_1": 1.0},
        {"label": "A1=1_B1=1", "A_1": 1.0, "B_1": 1.0}
    ]
    pe = 10.0
    lambdas = [0.1, 1.0, 10.0]
    new_mas = [0.01, 0.05]
    ks = np.round(np.linspace(0.05, 0.95, 19), 3)
    N = 5

    print(f"Pe = {pe}, Ma in {new_mas}")

    for config in configs:
        lbl = config["label"]
        A_1 = config["A_1"]
        B_1 = config["B_1"]
        U_0 = (2.0 / 3.0) * B_1 if B_1 != 0 else 1.0

        A_n = np.zeros(N, dtype=float)
        B_n = np.zeros(N, dtype=float)
        A_n[0] = A_1
        B_n[0] = B_1

        csv_mode = os.path.join(data_dir, f"ratio_sweep_{lbl}.csv")

        for l_val in lambdas:
            clean_cache = {}
            for k_val in ks:
                Us_clean, Ud_clean, _ = solve_clean_drop(N, k_val, l_val, A_n, B_n)
                clean_cache[k_val] = (Us_clean, Ud_clean)

            for ma in new_mas:
                needed = [k_val for k_val in ks if (lbl, pe, l_val, ma, k_val) not in existing_keys]
                if not needed:
                    print(f"Mode={lbl} | Pe={pe} | lambda={l_val} | Ma={ma} ... [ALL CACHED]")
                    continue

                t0 = time.time()
                print(f"Mode={lbl} | Pe={pe} | lambda={l_val:4.1f} | Ma={ma:5.2f} ... ", end="", flush=True)

                c_warm = None
                conv_count = 0

                for k_val in ks:
                    Us_clean, Ud_clean = clean_cache[k_val]
                    key_point = (lbl, pe, l_val, ma, k_val)
                    if key_point in existing_keys:
                        conv_count += 1
                        continue

                    Us, Ud, c_conv, iters, res, is_conv = run_single_coupling(
                        N=N, k=k_val, lambda_val=l_val, Ma=ma, Pe=pe,
                        A_n=A_n, B_n=B_n, c_init=c_warm,
                        tol=1e-4, omega=0.02
                    )

                    if is_conv:
                        conv_count += 1
                        c_warm = np.copy(c_conv)
                        ratio_Us = Us / Us_clean if abs(Us_clean) > 1e-12 else np.nan
                        ratio_Ud = Ud / Ud_clean if abs(Ud_clean) > 1e-12 else np.nan
                    else:
                        ratio_Us = np.nan
                        ratio_Ud = np.nan
                        Us = np.nan
                        Ud = np.nan

                    rec = {
                        "mode_label": lbl, "A_1": A_1, "B_1": B_1,
                        "Pe": pe, "lambda": l_val, "Ma": ma, "k": k_val,
                        "U_s": Us, "U_d": Ud,
                        "U_s_clean": Us_clean, "U_d_clean": Ud_clean,
                        "ratio_U_s": ratio_Us, "ratio_U_d": ratio_Ud,
                        "U_s_norm": Us / U_0 if not np.isnan(Us) else np.nan,
                        "U_d_norm": Ud / U_0 if not np.isnan(Ud) else np.nan,
                        "iterations": iters, "residual": res, "converged": is_conv
                    }

                    for fpath in (csv_mode, csv_master):
                        if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
                            with open(fpath, "w", newline="") as f_hdr:
                                csv.DictWriter(f_hdr, fieldnames=fieldnames).writeheader()

                    with open(csv_mode, "a", newline="") as f_m, open(csv_master, "a", newline="") as f_mst:
                        csv.DictWriter(f_m, fieldnames=fieldnames).writerow(rec)
                        csv.DictWriter(f_mst, fieldnames=fieldnames).writerow(rec)
                    existing_keys.add(key_point)

                elapsed = time.time() - t0
                print(f"Converged {conv_count}/{len(ks)} in {elapsed:.1f}s")

    print("\nAll targeted Pe=10 sweeps complete!")

if __name__ == "__main__":
    run_pe10_targeted_sweeps()
