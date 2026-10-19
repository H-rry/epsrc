#!/usr/bin/env python3
"""
1. Critical Ma over (lambda, k) for A_1 = 1, B_1 = 1 at Pe = 0.1, using a 2% error
   threshold against the asymptotic solution:
       lambda = 0.1, 0.5, 1, 2, 10;  k = 0.2, 0.4, 0.6, 0.8
   Output: data/critical_Ma/grid/critical_Ma_grid_A1_B1.csv
2. Critical Ma against Pe for both modes, to check that Ma_c ~ 1/Pe.
   Output: data/critical_Ma/critical_Ma_vs_Pe_scaling.csv
"""

import os
import csv
import time
import numpy as np


from squirmer_drop import coupling as ch
from squirmer_drop.paths import CRITICAL_MA_DIR, CRITICAL_MA_GRID_DIR

def run_pusher_puller_grid():
    output_dir = CRITICAL_MA_GRID_DIR
    os.makedirs(output_dir, exist_ok=True)

    lambdas = [0.1, 0.5, 1.0, 2.0, 10.0]
    ks = [0.2, 0.4, 0.6, 0.8]

    A_n = np.zeros(5)
    B_n = np.zeros(5)
    A_n[0] = 1.0
    B_n[0] = 1.0

    print("Critical Ma grid, A_1=1, B_1=1")

    combined_csv = os.path.join(output_dir, "critical_Ma_grid_A1_B1.csv")
    existing_done = set()
    records = []
    if os.path.exists(combined_csv):
        with open(combined_csv, 'r') as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing_done.add((float(r['lambda']), float(r['k'])))
                records.append(r)

    grid_Us = np.zeros((len(lambdas), len(ks)))
    grid_Ud = np.zeros((len(lambdas), len(ks)))

    total_points = len(lambdas) * len(ks)
    pt_idx = 0

    for l_idx, L in enumerate(lambdas):
        for k_idx, k in enumerate(ks):
            pt_idx += 1
            if (L, k) in existing_done:
                print(f"[{pt_idx}/{total_points}] (lambda={L}, k={k}) (cached)")
                continue

            print(f"\n>>> [{pt_idx}/{total_points}] Computing Ma_crit for A1=1, B1=1 @ lambda = {L}, k = {k} <<<")

            # 1. Critical Ma for Us
            t0 = time.time()
            Ma_crit_Us, err_Us, _, _, _ = ch.find_Ma_critical_point(
                lambda_val=L, k_val=k, vel_type='Us', A_n=A_n, B_n=B_n,
                target_err=2.0, starting_ma=0.5, Pe=0.1
            )
            # 2. Critical Ma for Ud
            Ma_crit_Ud, err_Ud, _, _, _ = ch.find_Ma_critical_point(
                lambda_val=L, k_val=k, vel_type='Ud', A_n=A_n, B_n=B_n,
                target_err=2.0, starting_ma=Ma_crit_Us, Pe=0.1
            )
            elapsed = time.time() - t0

            grid_Us[l_idx, k_idx] = Ma_crit_Us
            grid_Ud[l_idx, k_idx] = Ma_crit_Ud

            rec = {
                'lambda': L,
                'k': k,
                'critical_Ma_Us': Ma_crit_Us,
                'error_pct_Us': err_Us,
                'critical_Ma_Ud': Ma_crit_Ud,
                'error_pct_Ud': err_Ud
            }
            records.append(rec)

            with open(combined_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['lambda', 'k', 'critical_Ma_Us', 'error_pct_Us', 'critical_Ma_Ud', 'error_pct_Ud'])
                writer.writeheader()
                writer.writerows(records)

            print(f"  --> Done in {elapsed:.1f}s | Ma_c(Us)={Ma_crit_Us:.3f}, Ma_c(Ud)={Ma_crit_Ud:.3f}")

    print(f"\nSaved Pusher/Puller Grid: {combined_csv}")


def run_pe_scaling_sweeps():
    output_dir = CRITICAL_MA_DIR
    os.makedirs(output_dir, exist_ok=True)
    scaling_csv = os.path.join(output_dir, "critical_Ma_vs_Pe_scaling.csv")

    fieldnames = [
        "mode_label", "A_1", "B_1", "lambda", "k", "Pe",
        "critical_Ma_Us", "error_pct_Us", "critical_Ma_Ud", "error_pct_Ud",
        "Pe_times_Ma_c_Us", "Pe_times_Ma_c_Ud"
    ]

    existing_done = set()
    records = []
    if os.path.exists(scaling_csv):
        with open(scaling_csv, 'r') as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing_done.add((r['mode_label'], float(r['lambda']), float(r['k']), float(r['Pe'])))
                records.append(r)
    else:
        with open(scaling_csv, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    modes = [
        {"label": "A1=0_B1=1", "A_1": 0.0, "B_1": 1.0},
        {"label": "A1=1_B1=1", "A_1": 1.0, "B_1": 1.0}
    ]

    lambdas = [0.1, 1.0, 10.0]
    pes = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    benchmark_k = 0.4
    N = 5

    print("Critical Ma vs Pe")

    total = len(modes) * len(lambdas) * len(pes)
    idx = 0

    for mode in modes:
        lbl = mode["label"]
        A_1 = mode["A_1"]
        B_1 = mode["B_1"]

        A_n = np.zeros(N)
        B_n = np.zeros(N)
        A_n[0] = A_1
        B_n[0] = B_1

        for L in lambdas:
            for pe in pes:
                idx += 1
                key = (lbl, L, benchmark_k, pe)
                if key in existing_done:
                    print(f"[{idx}/{total}] {lbl} | lam={L} | Pe={pe} (cached)")
                    continue

                t0 = time.time()
                print(f"[{idx}/{total}] {lbl} | lam={L:4.1f} | Pe={pe:4.2f} ... ", end="", flush=True)

                # Estimated starting Ma based on ~0.05/Pe
                start_ma = max(1e-4, 0.05 / pe)

                try:
                    # Ud threshold
                    Ma_c_Ud, err_Ud, _, _, _ = ch.find_Ma_critical_point(
                        lambda_val=L, k_val=benchmark_k, vel_type='Ud',
                        A_n=A_n, B_n=B_n, target_err=2.0, starting_ma=start_ma, Pe=pe
                    )
                    # Us threshold
                    Ma_c_Us, err_Us, _, _, _ = ch.find_Ma_critical_point(
                        lambda_val=L, k_val=benchmark_k, vel_type='Us',
                        A_n=A_n, B_n=B_n, target_err=2.0, starting_ma=Ma_c_Ud, Pe=pe
                    )
                except Exception as e:
                    print(f"failed: {e}")
                    continue

                pe_mac_ud = pe * Ma_c_Ud
                pe_mac_us = pe * Ma_c_Us
                elapsed = time.time() - t0

                rec = {
                    "mode_label": lbl, "A_1": A_1, "B_1": B_1,
                    "lambda": L, "k": benchmark_k, "Pe": pe,
                    "critical_Ma_Us": Ma_c_Us, "error_pct_Us": err_Us,
                    "critical_Ma_Ud": Ma_c_Ud, "error_pct_Ud": err_Ud,
                    "Pe_times_Ma_c_Us": pe_mac_us, "Pe_times_Ma_c_Ud": pe_mac_ud
                }
                records.append(rec)
                with open(scaling_csv, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(rec)

                existing_done.add(key)
                print(f"Done in {elapsed:.1f}s | Ma_c(Ud)={Ma_c_Ud:.4f} (Pe*Ma_c={pe_mac_ud:.4f}), Ma_c(Us)={Ma_c_Us:.4f}")

    print(f"\nSaved Pe scaling dataset: {scaling_csv}")

if __name__ == "__main__":
    run_pusher_puller_grid()
    run_pe_scaling_sweeps()
