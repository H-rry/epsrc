#!/usr/bin/env python3
"""
U_S/U_S,clean and U_D/U_D,clean against k in [0.05, 0.95], as in Shaik et al.
(JFM 2018, 851, 187-230), Fig. 1, for
    Pe     = 0.1, 1, 10
    lambda = 0.1, 1, 10
    Ma     = 0.1, 1, 10 (plus the clean drop, Ma = 0)
and two squirmers: A_1 = 1, B_1 = 1 and A_1 = 0, B_1 = 1 (the case in the paper).
Results go to data/ratio_sweeps/; points already there are skipped, so the run can be
restarted.
"""

import os
import time
import csv
import numpy as np


from squirmer_drop.coupling import solve_clean_drop, run_single_coupling
from squirmer_drop.paths import RATIO_SWEEPS_DIR


def run_full_suite():
    output_dir = RATIO_SWEEPS_DIR
    os.makedirs(output_dir, exist_ok=True)

    csv_master = os.path.join(output_dir, "ratio_sweep_master.csv")
    fieldnames = [
        "mode_label", "A_1", "B_1", "Pe", "lambda", "Ma", "k",
        "U_s", "U_d", "U_s_clean", "U_d_clean",
        "ratio_U_s", "ratio_U_d", "U_s_norm", "U_d_norm",
        "iterations", "residual", "converged"
    ]

    # Load existing records for resume capability
    existing_keys = set()
    if os.path.exists(csv_master) and os.path.getsize(csv_master) > 0:
        with open(csv_master, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    k_val = round(float(r["k"]), 3)
                    existing_keys.add((r["mode_label"], float(r["Pe"]), float(r["lambda"]), float(r["Ma"]), k_val))
                except Exception:
                    pass
        print(f"Found {len(existing_keys)} pre-existing converged records. Resuming...")
    else:
        with open(csv_master, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    configs = [
        {"label": "A1=1_B1=1", "A_1": 1.0, "B_1": 1.0},
        {"label": "A1=0_B1=1", "A_1": 0.0, "B_1": 1.0}
    ]

    pes = [0.1, 1.0, 10.0]
    lambdas = [0.1, 1.0, 10.0]
    mas = [0.1, 1.0, 10.0]
    ks = np.round(np.linspace(0.05, 0.95, 19), 3)

    N = 5
    total_sweeps = len(configs) * len(pes) * len(lambdas) * len(mas)
    current_sweep = 0
    start_time = time.time()

    print("Ratio sweeps")
    print(f"Configs     : {[c['label'] for c in configs]}")
    print(f"Peclet (Pe) : {pes}")
    print(f"Lambdas     : {lambdas}")
    print(f"Marangoni   : {mas} (plus Ma=0.0 clean baseline)")
    print(f"k points    : {len(ks)} points from {ks[0]} to {ks[-1]}")
    print(f"Total series: {total_sweeps} parametric curves")
    print(f"Output dir  : {output_dir}")

    for config in configs:
        lbl = config["label"]
        A_1 = config["A_1"]
        B_1 = config["B_1"]
        U_0 = (2.0 / 3.0) * B_1 if B_1 != 0 else 1.0

        A_n = np.zeros(N, dtype=float)
        B_n = np.zeros(N, dtype=float)
        A_n[0] = A_1
        B_n[0] = B_1

        csv_mode = os.path.join(output_dir, f"ratio_sweep_{lbl}.csv")
        if not os.path.exists(csv_mode) or os.path.getsize(csv_mode) == 0:
            with open(csv_mode, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

        for pe in pes:
            for l_val in lambdas:
                # 1. Clean drop baseline (Ma = 0.0) for this (lambda, k) set
                clean_cache = {}
                for k_val in ks:
                    Us_clean, Ud_clean, x_clean = solve_clean_drop(N, k_val, l_val, A_n, B_n)
                    clean_cache[k_val] = (Us_clean, Ud_clean)

                    # Record clean baseline if not already present
                    key_clean = (lbl, pe, l_val, 0.0, k_val)
                    if key_clean not in existing_keys:
                        rec_clean = {
                            "mode_label": lbl, "A_1": A_1, "B_1": B_1,
                            "Pe": pe, "lambda": l_val, "Ma": 0.0, "k": k_val,
                            "U_s": Us_clean, "U_d": Ud_clean,
                            "U_s_clean": Us_clean, "U_d_clean": Ud_clean,
                            "ratio_U_s": 1.0, "ratio_U_d": 1.0,
                            "U_s_norm": Us_clean / U_0, "U_d_norm": Ud_clean / U_0,
                            "iterations": 1, "residual": 0.0, "converged": True
                        }
                        with open(csv_mode, "a", newline="") as f_mode, open(csv_master, "a", newline="") as f_mst:
                            csv.DictWriter(f_mode, fieldnames=fieldnames).writerow(rec_clean)
                            csv.DictWriter(f_mst, fieldnames=fieldnames).writerow(rec_clean)
                        existing_keys.add(key_clean)

                # 2. Sweeps for each Ma > 0
                for ma in mas:
                    current_sweep += 1

                    # Check if all k for this curve already exist
                    needed_ks = [k_val for k_val in ks if (lbl, pe, l_val, ma, k_val) not in existing_keys]
                    if not needed_ks:
                        print(f"[{current_sweep:2d}/{total_sweeps:2d}] Mode={lbl} | Pe={pe:4.1f} | lambda={l_val:4.1f} | Ma={ma:4.1f} (cached)")
                        continue

                    t0 = time.time()
                    print(f"[{current_sweep:2d}/{total_sweeps:2d}] Running: Mode={lbl} | Pe={pe:4.1f} | lambda={l_val:4.1f} | Ma={ma:4.1f} ...", end="", flush=True)

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
                            A_n=A_n, B_n=B_n, c_init=c_warm
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

                        with open(csv_mode, "a", newline="") as f_mode, open(csv_master, "a", newline="") as f_mst:
                            csv.DictWriter(f_mode, fieldnames=fieldnames).writerow(rec)
                            csv.DictWriter(f_mst, fieldnames=fieldnames).writerow(rec)
                        existing_keys.add(key_point)

                    elapsed = time.time() - t0
                    print(f" Converged {conv_count}/{len(ks)} in {elapsed:.2f}s")

    total_time = time.time() - start_time
    print(f"Finished in {total_time/60:.1f} min")
    print(f" Master dataset saved to: {csv_master}")


if __name__ == "__main__":
    run_full_suite()
