"""
Coupling between the Stokes solve and the surfactant distribution.

The Marangoni stress depends on the surfactant coefficients c_n, which in turn depend
on the surface velocity. Both solvers are iterated (Picard, with under-relaxation on
c_n) until c_n stops changing.
"""

import os
import numpy as np

from squirmer_drop import stokes as fluid
from squirmer_drop import surfactant
from squirmer_drop import analytical_squirmer
from squirmer_drop.paths import RUNS_DIR, critical_ma_dir, critical_ma_lambda_figures_dir


def extract_b_n(x, N):
    """Legendre coefficients b_n of the surface velocity u_theta(1, mu), from the outer field."""
    _, (E_o, F_o, G_o, H_o), _, _ = fluid.unpack_coefficients(x, N)

    b_n = np.zeros(N)
    for i in range(N):
        n_val = i + 1
        bracket = (n_val + 3) * E_o[i] + (n_val + 1) * F_o[i] + (2 - n_val) * G_o[i] - n_val * H_o[i]
        b_n[i] = -(2 * n_val + 1) / 2.0 * bracket
    return b_n


def save_converged_data(c_n, x_fluid, N, k, lambda_val, Ma, Pe, mode='particle', A_1=None, B_1=None):
    """Write c_n and the state vector to outputs/runs/<mode folder>/Ma=../Pe=../N=../."""
    if mode == 'squirmer' and A_1 is not None and B_1 is not None:
        folder = f"plots_squirmer_A_1={A_1:.0f}_B_1={B_1:.0f}"
    else:
        folder = f"plots_{mode}"

    dir_path = os.path.join(RUNS_DIR, folder, f"Ma={Ma}", f"Pe={Pe}", f"N={N}")
    os.makedirs(dir_path, exist_ok=True)
    csv_path = os.path.join(dir_path, f"data_lambda={lambda_val}_k={k}.csv")

    with open(csv_path, 'w') as f:
        f.write("index,c_n,x_fluid\n")
        max_len = max(len(c_n), len(x_fluid))
        for idx in range(max_len):
            c_val = c_n[idx] if idx < len(c_n) else ""
            x_val = x_fluid[idx] if idx < len(x_fluid) else ""
            f.write(f"{idx},{c_val},{x_val}\n")
    print(f"Saved {csv_path}")


def run_coupling_iteration(N=5, k=0.3, lambda_val=0.5, Ma=1.0, Pe=1.0, max_iter=100, tol=1e-6, omega=0.1, mode='particle', A_n=None, B_n=None):
    """
    Iterate the flow and surfactant solves until ||c_n_new - c_n|| < tol, with
    under-relaxation factor omega. The converged state is saved and its boundary
    conditions checked. Returns (c_n, b_n, x), or (None, None, None) on failure.
    """
    print(f"Coupled solve ({mode}): N={N}, k={k}, lambda={lambda_val}, Ma={Ma}, Pe={Pe}, omega={omega}")

    c_n = 0.1 ** np.arange(1, N+1, 1)
    A = fluid.assemble_A(N, k, lambda_val)
    for iteration in range(1, max_iter + 1):
        b = fluid.assemble_b(N, Ma, c_n=c_n, mode=mode, lambda_val=lambda_val, k=k, A_n=A_n, B_n=B_n)
        try:
            x_fluid = np.linalg.solve(A, b)
        except np.linalg.LinAlgError as e:
            print(f"Linear solve failed: {e}")
            return None, None, None

        b_n_new = extract_b_n(x_fluid, N)

        try:
            c_n_new = surfactant.solve_c_n(b_n_new, Pe)
        except Exception as e:
            print(f"Surfactant solve failed at iteration {iteration} (k={k}, Ma={Ma}, Pe={Pe}): {e}")
            return None, None, None

        diff = np.linalg.norm(c_n_new - c_n)
        if iteration == 1 or iteration % 100 == 0 or diff < tol:
            print(f"  iter {iteration:4d}  |dc| = {diff:.3e}  U_D = {x_fluid[5*N+1]:.6e}")

        if diff < tol:
            b_final = fluid.assemble_b(N, Ma, c_n=c_n_new, mode=mode, lambda_val=lambda_val, k=k, A_n=A_n, B_n=B_n)
            x_fluid = np.linalg.solve(A, b_final)
            print(f"Converged after {iteration} iterations")

            if mode == 'squirmer' and A_n is not None and B_n is not None:
                save_converged_data(c_n_new, x_fluid, N, k, lambda_val, Ma, Pe, mode=mode, A_1=A_n[0], B_1=B_n[0])
            else:
                save_converged_data(c_n_new, x_fluid, N, k, lambda_val, Ma, Pe, mode=mode)

            fluid.run_unit_tests(x_fluid, N, k, lambda_val, Ma=Ma, tolerance=1e-8, c_n=c_n_new, mode=mode, A_n=A_n, B_n=B_n)
            return c_n_new, b_n_new, x_fluid

        c_n = omega * c_n_new + (1.0 - omega) * c_n

    print(f"Did not converge in {max_iter} iterations (last |dc| = {diff:.2e})")
    return None, None, None

def get_average_error(Ma, lambda_val, k_list, Pe=0.1, N_=5, A_n=None, B_n=None):
    """
    Relative L2 error (%) between the coupled solver and the asymptotic solution over
    k_list, for U_s and U_d. Returns (err_Us, err_Ud, Us_solver, Us_anal, Ud_solver, Ud_anal).
    """
    Us_solver = []
    Us_anal = []
    Ud_solver = []
    Ud_anal = []

    for k in k_list:
        c_n, b_n, x_fluid = run_coupling_iteration(
            N=N_, k=k, lambda_val=lambda_val, Ma=Ma, Pe=Pe,
            max_iter=1500, omega=0.05, mode='squirmer',
            A_n=A_n, B_n=B_n
        )
        if x_fluid is None:
            return 999.0, 999.0, [], [], [], []

        U_d = x_fluid[5*N_ + 1]
        delta_U = x_fluid[5*N_ + 2]
        U_s_solver = U_d + delta_U

        U_s_anal = analytical_squirmer.analytical_U_S(k, Ma, lambda_val, A_n[0], B_n[0])
        U_d_anal = analytical_squirmer.analytical_U_D(k, Ma, lambda_val, A_n[0], B_n[0])

        Us_solver.append(U_s_solver)
        Us_anal.append(U_s_anal)
        Ud_solver.append(U_d)
        Ud_anal.append(U_d_anal)

    Us_solver = np.array(Us_solver)
    Us_anal = np.array(Us_anal)
    Ud_solver = np.array(Ud_solver)
    Ud_anal = np.array(Ud_anal)

    diff_Us = np.linalg.norm(Us_solver - Us_anal)
    norm_Us = np.linalg.norm(Us_solver)
    err_Us = (diff_Us / norm_Us) * 100.0 if norm_Us != 0 else 999.0

    diff_Ud = np.linalg.norm(Ud_solver - Ud_anal)
    norm_Ud = np.linalg.norm(Ud_solver)
    err_Ud = (diff_Ud / norm_Ud) * 100.0 if norm_Ud != 0 else 999.0

    return err_Us, err_Ud, Us_solver, Us_anal, Ud_solver, Ud_anal

def find_Ma_critical_point(lambda_val, k_val, vel_type='Us', A_n=None, B_n=None, target_err=2.0, starting_ma=0.1, max_bisection_steps=8, Pe=0.1, N_=5):
    """
    Ma at which the asymptotic solution differs from the solver by target_err percent, at a
    single (lambda, k). Brackets by doubling/halving Ma, then bisects.
    Returns (Ma_crit, err_crit, solver_crit, anal_crit, history).
    """
    if A_n is None:
        A_n = np.zeros(N_)
        A_n[0] = 0.0
    if B_n is None:
        B_n = np.zeros(N_)
        B_n[0] = 1.0

    def evaluate_point(Ma_val):
        c_n, b_n, x_fluid = run_coupling_iteration(
            N=N_, k=k_val, lambda_val=lambda_val, Ma=Ma_val, Pe=Pe,
            max_iter=1500, omega=0.05, mode='squirmer',
            A_n=A_n, B_n=B_n
        )
        if x_fluid is None:
            return 999.0, 0.0, 0.0

        U_d = x_fluid[5*N_ + 1]
        delta_U = x_fluid[5*N_ + 2]
        U_s = U_d + delta_U

        U_s_anal = analytical_squirmer.analytical_U_S(k_val, Ma_val, lambda_val, A_n[0], B_n[0], Pe=Pe)
        U_d_anal = analytical_squirmer.analytical_U_D(k_val, Ma_val, lambda_val, A_n[0], B_n[0], Pe=Pe)

        if vel_type == 'Us':
            err = 100.0 * abs(U_s - U_s_anal) / abs(U_s) if U_s != 0 else 999.0
            return err, U_s, U_s_anal
        else:
            err = 100.0 * abs(U_d - U_d_anal) / abs(U_d) if U_d != 0 else 999.0
            return err, U_d, U_d_anal

    Ma_low = starting_ma
    err_low, solver_low, anal_low = evaluate_point(Ma_low)
    print(f"[{vel_type} @ (lambda={lambda_val}, k={k_val})] Initial check: Ma={Ma_low:.4e} | Error={err_low:.3f}%")

    history = {Ma_low: (err_low, solver_low, anal_low)}

    if err_low >= target_err:
        Ma_high = Ma_low
        err_high = err_low
        while err_low >= target_err and Ma_low > 1e-8:
            Ma_high = Ma_low
            err_high = err_low
            Ma_low /= 2.0
            err_low, solver_low, anal_low = evaluate_point(Ma_low)
            history[Ma_low] = (err_low, solver_low, anal_low)
            print(f"  Bracketing down: Ma={Ma_low:.4e} | Error={err_low:.3f}%")
        if err_low >= target_err:
            print(f"Warning: error still >= {target_err}% at Ma={Ma_low}")
            return Ma_low, err_low, solver_low, anal_low, history
    else:
        Ma_high = Ma_low * 2.0
        err_high, solver_high, anal_high = evaluate_point(Ma_high)
        history[Ma_high] = (err_high, solver_high, anal_high)
        print(f"  Bracketing up: Ma={Ma_high:.4e} | Error={err_high:.3f}%")
        while err_high < target_err and Ma_high < 200.0:
            Ma_low = Ma_high
            err_low = err_high
            Ma_high *= 2.0
            err_high, solver_high, anal_high = evaluate_point(Ma_high)
            history[Ma_high] = (err_high, solver_high, anal_high)
            print(f"  Bracketing up: Ma={Ma_high:.4e} | Error={err_high:.3f}%")
        if Ma_high >= 200.0 and err_high < target_err:
            print(f"Warning: error still < {target_err}% at Ma={Ma_high}")
            return Ma_high, err_high, solver_high, anal_high, history

    print(f"  Bisection between Ma_low={Ma_low:.4e} and Ma_high={Ma_high:.4e}...")
    for step in range(max_bisection_steps):
        Ma_mid = 0.5 * (Ma_low + Ma_high)
        err_mid, solver_mid, anal_mid = evaluate_point(Ma_mid)
        history[Ma_mid] = (err_mid, solver_mid, anal_mid)
        print(f"    Step {step+1}: Ma={Ma_mid:.4f} | Error={err_mid:.3f}%")
        if err_mid < target_err:
            Ma_low = Ma_mid
        else:
            Ma_high = Ma_mid

    Ma_crit = 0.5 * (Ma_low + Ma_high)
    err_crit, solver_crit, anal_crit = evaluate_point(Ma_crit)
    history[Ma_crit] = (err_crit, solver_crit, anal_crit)
    print(f"[{vel_type} @ (lambda={lambda_val}, k={k_val})] Converged Ma_crit = {Ma_crit:.4f} (Error = {err_crit:.3f}%)\n")
    return Ma_crit, err_crit, solver_crit, anal_crit, history

def find_Ma_critical_squirmer(lambda_val, vel_type='Us', starting_ma=0.001):
    """
    Critical Ma for the neutral squirmer (A_1 = 0, B_1 = 1) at Pe = 0.1, using the error
    averaged over k = 0.1..0.8. Saves the profiles at Ma_crit and a plot of how they
    approach it. Returns (Ma_crit, history).
    """
    import matplotlib.pyplot as plt
    k_list = np.arange(0.1, 0.9, 0.1)
    Pe = 0.1
    N_ = 5
    A_n = np.zeros(N_)
    B_n = np.zeros(N_)
    A_n[0] = 0.0
    B_n[0] = 1.0

    def evaluate_Ma(Ma_val):
        err_Us, err_Ud, Us_s, Us_a, Ud_s, Ud_a = get_average_error(Ma_val, lambda_val, k_list, Pe, N_, A_n, B_n)
        if vel_type == 'Us':
            return err_Us, Us_s, Us_a
        else:
            return err_Ud, Ud_s, Ud_a

    Ma_low = starting_ma
    err_low, solver_low, anal_low = evaluate_Ma(Ma_low)
    print(f"[{vel_type}] Bracketing initial check: Ma={Ma_low:.4e} | Error={err_low:.3f}%")

    history = {}
    history[Ma_low] = (err_low, solver_low, anal_low)

    if err_low >= 2.0:
        print(f"[{vel_type}] Error is already >= 2% at starting Ma={Ma_low}. Bracketing down...")
        Ma_high = Ma_low
        err_high = err_low

        while err_low >= 2.0 and Ma_low > 1e-8:
            Ma_high = Ma_low
            err_high = err_low
            Ma_low /= 2.0
            err_low, solver_low, anal_low = evaluate_Ma(Ma_low)
            print(f"Bracketing down: Ma={Ma_low:.4e} | Error={err_low:.3f}%")
            history[Ma_low] = (err_low, solver_low, anal_low)

        if err_low >= 2.0:
            print(f"Warning: error still >= 2% at Ma={Ma_low}")
            return Ma_low, history
    else:
        print(f"[{vel_type}] Error is < 2% at starting Ma={Ma_low}. Bracketing up...")
        Ma_high = Ma_low * 2.0
        err_high, solver_high, anal_high = evaluate_Ma(Ma_high)
        print(f"Bracketing up: Ma={Ma_high:.4e} | Error={err_high:.3f}%")
        history[Ma_high] = (err_high, solver_high, anal_high)

        while err_high < 2.0 and Ma_high < 100.0:
            Ma_low = Ma_high
            err_low = err_high
            Ma_high *= 2.0
            err_high, solver_high, anal_high = evaluate_Ma(Ma_high)
            print(f"Bracketing up: Ma={Ma_high:.4e} | Error={err_high:.3f}%")
            history[Ma_high] = (err_high, solver_high, anal_high)

        if Ma_high >= 100.0 and err_high < 2.0:
            print(f"Warning: error still < 2% at Ma={Ma_high}")
            return Ma_high, history

    print(f"[{vel_type}] Starting bisection between Ma_low={Ma_low:.4e} and Ma_high={Ma_high:.4e}...")
    for step in range(8):
        Ma_mid = 0.5 * (Ma_low + Ma_high)
        err_mid, solver_mid, anal_mid = evaluate_Ma(Ma_mid)
        print(f"Bisection step {step+1}: Ma={Ma_mid:.4f} | Error={err_mid:.3f}%")
        history[Ma_mid] = (err_mid, solver_mid, anal_mid)

        if err_mid < 2.0:
            Ma_low = Ma_mid
        else:
            Ma_high = Ma_mid

    Ma_crit = 0.5 * (Ma_low + Ma_high)
    err_crit, solver_crit, anal_crit = evaluate_Ma(Ma_crit)
    history[Ma_crit] = (err_crit, solver_crit, anal_crit)
    print(f"[{vel_type}] Converged critical Ma = {Ma_crit:.4f} with error = {err_crit:.3f}%\n")

    # save the profiles at Ma_crit and plot how they approach it
    lambda_dir = os.path.join(critical_ma_dir(A_n[0], B_n[0]), f"lambda={lambda_val}")
    os.makedirs(lambda_dir, exist_ok=True)
    lambda_fig_dir = critical_ma_lambda_figures_dir(A_n[0], B_n[0], lambda_val)
    os.makedirs(lambda_fig_dir, exist_ok=True)

    csv_path = os.path.join(lambda_dir, f"critical_data_{vel_type}.csv")
    with open(csv_path, 'w') as f:
        f.write(f"k,{vel_type}_solver,{vel_type}_analytical,relative_error_pct\n")
        for idx, k in enumerate(k_list):
            err_point = 100.0 * abs(solver_crit[idx] - anal_crit[idx]) / abs(solver_crit[idx]) if solver_crit[idx] != 0 else 0
            f.write(f"{k},{solver_crit[idx]},{anal_crit[idx]},{err_point}\n")
    print(f"Saved {csv_path}")

    plt.close('all')
    fig, ax = plt.subplots(figsize=(9, 7.5))

    sorted_mas = sorted(list(history.keys()))
    valid_mas = [m for m in sorted_mas if history[m][0] < 100.0]

    if len(valid_mas) >= 4:
        indices = [0, len(valid_mas)//3, 2*len(valid_mas)//3, len(valid_mas)-1]
        mas_to_plot = [valid_mas[i] for i in indices]
    else:
        mas_to_plot = valid_mas

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(mas_to_plot)))
    k_dense = np.linspace(0.01, 0.99, 200)

    vel_label = r"Squirmer Velocity $U_s$" if vel_type == 'Us' else r"Drop Velocity $U_d$"

    for idx, Ma_val in enumerate(mas_to_plot):
        err_val, s_vals, a_vals = history[Ma_val]
        c = colors[idx]

        if vel_type == 'Us':
            a_dense = [analytical_squirmer.analytical_U_S(kv, Ma_val, lambda_val, A_n[0], B_n[0]) for kv in k_dense]
        else:
            a_dense = [analytical_squirmer.analytical_U_D(kv, Ma_val, lambda_val, A_n[0], B_n[0]) for kv in k_dense]

        ax.plot(k_dense, a_dense, '--', color=c, alpha=0.7,
                label=f"Analytical (Ma={Ma_val:.3f}, Err={err_val:.2f}%)" if Ma_val != Ma_crit else f"Analytical (Ma_c={Ma_val:.3f}, Err={err_val:.2f}%)")

        ax.plot(k_list, s_vals, 'o', color=c, markersize=7,
                label=f"Solver (Ma={Ma_val:.3f})" if Ma_val != Ma_crit else f"Solver (Ma_c={Ma_val:.3f})")

    ax.set_xlabel(r"Core Size Ratio $k$", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel(vel_label, fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(fr"{vel_label} Profiles Approaching $Ma_{{\text{{crit}}}}$" + "\n" + fr"($\lambda = {lambda_val}$, Pe = 0.1, $A_1={A_n[0]:.0f}, B_1={B_n[0]:.0f}$)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, facecolor='#ffffff', fontsize=10, loc='best')

    plt.tight_layout()
    plot_path = os.path.join(lambda_fig_dir, f"{vel_type}_vs_k_approaching_Mac.png")
    plt.savefig(plot_path, dpi=250)
    plt.close(fig)
    print(f"Saved {plot_path}")

    return Ma_crit, history


def solve_clean_drop(N, k, lambda_val, A_n, B_n):
    """Clean drop (Ma = 0): a single linear solve. Returns (U_s, U_d, x)."""
    A = fluid.assemble_A(N, k, lambda_val)
    b = fluid.assemble_b(N, Ma=0.0, mode='squirmer', lambda_val=lambda_val, k=k, A_n=A_n, B_n=B_n)
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as e:
        print(f"Clean-drop solve failed for k={k}, lambda={lambda_val}: {e}")
        return None, None, None

    U_d_clean = float(x[5 * N + 1])
    delta_U = float(x[5 * N + 2])
    U_s_clean = float(U_d_clean + delta_U)

    return U_s_clean, U_d_clean, x


def run_single_coupling(N, k, lambda_val, Ma, Pe, A_n, B_n, c_init=None, max_iter=1200, tol=None, omega=None):
    """
    Quiet version of run_coupling_iteration used by the sweeps: optional warm start from
    c_init and adaptive relaxation. Nothing is written to disk.
    Returns (U_s, U_d, c_n, iterations, residual, converged).
    """
    if tol is None:
        tol = 1e-4 if Pe >= 10.0 else 1e-5
    if omega is None:
        omega = 0.02 if Pe >= 10.0 else 0.05

    A = fluid.assemble_A(N, k, lambda_val)

    if c_init is not None:
        c_n = np.copy(c_init)
    else:
        c_n = 0.01 ** np.arange(1, N + 1, dtype=float)

    diff = 999.0
    x_fluid = None

    for iteration in range(1, max_iter + 1):
        b = fluid.assemble_b(N, Ma, c_n=c_n, mode='squirmer', lambda_val=lambda_val, k=k, A_n=A_n, B_n=B_n)
        try:
            x_fluid = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None, None, None, iteration, 999.0, False

        b_n_new = extract_b_n(x_fluid, N)

        try:
            c_n_new = surfactant.solve_c_n(b_n_new, Pe)
        except Exception:
            return None, None, None, iteration, 999.0, False

        diff = float(np.linalg.norm(c_n_new - c_n))

        if diff < tol:
            b_final = fluid.assemble_b(N, Ma, c_n=c_n_new, mode='squirmer', lambda_val=lambda_val, k=k, A_n=A_n, B_n=B_n)
            x_fluid = np.linalg.solve(A, b_final)
            U_d = float(x_fluid[5 * N + 1])
            delta_U = float(x_fluid[5 * N + 2])
            U_s = float(U_d + delta_U)
            return U_s, U_d, c_n_new, iteration, diff, True

        # relax harder if the iteration is blowing up, less once it is close
        current_omega = omega
        if diff > 10.0:
            current_omega = min(omega, 0.01)
        elif diff < 1e-2:
            current_omega = min(0.08, omega * 1.5)

        c_n = current_omega * c_n_new + (1.0 - current_omega) * c_n

    # not converged: return the last iterate
    if x_fluid is not None:
        U_d = float(x_fluid[5 * N + 1])
        delta_U = float(x_fluid[5 * N + 2])
        U_s = float(U_d + delta_U)
        return U_s, U_d, c_n, max_iter, diff, False

    return None, None, None, max_iter, diff, False
