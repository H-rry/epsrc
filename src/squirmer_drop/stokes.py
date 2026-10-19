import numpy as np
import matplotlib.pyplot as plt

def get_legendre_and_derivs(mew, N):
    """
    Computes Legendre polynomials P_n(mew) and their derivatives P'_n(mew)
    for n = 1 to N.
    """
    mew = np.atleast_1d(mew)
    V = np.polynomial.legendre.legvander(mew, N)
    P = V[:, 1 : N + 1]  # shape (len(mew), N)
    
    dP_list = []
    for n in range(1, N + 1):
        c = np.zeros(n + 1)
        c[n] = 1.0
        der_c = np.polynomial.legendre.legder(c)
        dp_val = np.polynomial.legendre.legval(mew, der_c)
        dP_list.append(dp_val)
    dP = np.column_stack(dP_list)  # shape (len(mew), N)
    return P, dP

def assemble_A(N, k, lambda_val):
    """
    Assembles the linear system matrix A of size (5N + 3) x (5N + 3).
    Here, the 5th variable of group n, x[5*i + 4] (for i >= 1), represents G^o_n.
    """
    total_rows = 5 * N + 3
    A = np.zeros((total_rows, total_rows))

    # Block N1: Shear stress balance at interface r = 1
    for i in range(N):
        n_val = i + 1
        N_one_coeff = 2.0 * (2.0 * n_val + 1.0) * (1.0 / (n_val * (n_val + 1.0)))
        A[i, 5 * i] = -N_one_coeff * n_val * (n_val + 2.0) * lambda_val
        A[i, 5 * i + 1] = -N_one_coeff * (n_val * n_val - 1.0) * lambda_val
        A[i, 5 * i + 2] = -N_one_coeff * (n_val * n_val - 1.0) * lambda_val
        A[i, 5 * i + 3] = -N_one_coeff * n_val * (n_val + 2.0) * lambda_val
        if i != 0:
            # Enforce G^o_n coupling term
            A[i, 5 * i + 4] = N_one_coeff * ((n_val * n_val - 1.0) - n_val * (n_val + 2.0))
    # n = 1 coupling to H^o_1 (index 5N)
    A[0, 5 * N] = 9.0

    # Block N2: Kinematic BC (impermeability) at droplet interface r = 1
    for i in range(N):
        r_row = N + i
        A[r_row, 5 * i] = 1.0
        A[r_row, 5 * i + 1] = 1.0
        A[r_row, 5 * i + 2] = 1.0
        A[r_row, 5 * i + 3] = 1.0

    # Block N3: Tangential velocity continuity at interface r = 1
    r_row = 2 * N
    # n = 1 mode: coupled to H^o_1 (5N) and U_D (5N+1)
    A[r_row, 0] = -4.0
    A[r_row, 1] = -2.0
    A[r_row, 2] = -1.0
    A[r_row, 3] = 1.0
    A[r_row, 5 * N] = -2.0
    A[r_row, 5 * N + 1] = -1.0 / 3.0  # Reconciled coefficient
    # n >= 2 modes: coupled to G^o_n (5i+4)
    for i in range(1, N):
        n_val = i + 1
        A[r_row + i, 5 * i] = -(n_val + 3.0)
        A[r_row + i, 5 * i + 1] = -(n_val + 1.0)
        A[r_row + i, 5 * i + 2] = -(2.0 - n_val)
        A[r_row + i, 5 * i + 3] = n_val
        A[r_row + i, 5 * i + 4] = 2.0

    # Block N4: Core boundary condition for u_theta at r = k
    r_row = 3 * N
    # n = 1 mode: coupled to relative core velocity delta_U (5N+2)
    A[r_row, 0] = 6.0 * (k ** 2)
    A[r_row, 1] = 3.0
    A[r_row, 2] = 1.5 * (k ** -1)
    A[r_row, 3] = -1.5 * (k ** -3)
    A[r_row, 5 * N + 2] = -1.0
    # n >= 2 modes
    for i in range(1, N):
        n_val = i + 1
        A[r_row + i, 5 * i] = (n_val + 3.0) * (k ** (n_val + 1.0))
        A[r_row + i, 5 * i + 1] = (n_val + 1.0) * (k ** (n_val - 1.0))
        A[r_row + i, 5 * i + 2] = (-n_val + 2.0) * (k ** -n_val)
        A[r_row + i, 5 * i + 3] = -n_val * (k ** (-n_val - 2.0))

    # Block N5: Core boundary condition for u_r at r = k
    r_row = 4 * N
    # n = 1 mode: coupled to relative core velocity delta_U (5N+2)
    A[r_row, 0] = 3.0 * (k ** 2)
    A[r_row, 1] = 3.0
    A[r_row, 2] = 3.0 * (k ** -1)
    A[r_row, 3] = 3.0 * (k ** -3)
    A[r_row, 5 * N + 2] = -1.0
    # n >= 2 modes
    for i in range(1, N):
        n_val = i + 1
        A[r_row + i, 5 * i] = k ** (n_val + 1.0)
        A[r_row + i, 5 * i + 1] = k ** (n_val - 1.0)
        A[r_row + i, 5 * i + 2] = k ** -n_val
        A[r_row + i, 5 * i + 3] = k ** (-n_val - 2.0)

    # Constraint Rows
    A[5 * N, 4] = 1.0
    A[5 * N, 5 * N + 1] = -1.0 / 3.0
    A[5 * N, 5 * N] = 1.0

    A[5 * N + 1, 4] = 1.0
    A[5 * N + 2, 2] = 1.0

    return A

def assemble_b(N, Ma):
    """
    Assembles the right-hand-side vector b of size 5N + 3.
    """
    total_rows = 5 * N + 3
    b = np.zeros(total_rows)
    for i in range(N):
        b[i] = Ma * (0.1 ** (i + 1))
    b[5 * N + 2] = -(1.0 / 12.0) * (1.0 / np.pi)
    return b

def evaluate_fields(x, mew_vals, N, k):
    """
    Computes field velocities at the boundaries (r = 1.0, r = k, r = 10.0)
    for testing.
    """
    P, dP = get_legendre_and_derivs(mew_vals, N)
    n = np.arange(1, N + 1)
    sin_theta = np.sqrt(1.0 - mew_vals**2)
    P_associated = sin_theta[:, np.newaxis] * dP
    factor_theta = -(2 * n + 1) / (n * (n + 1))
    
    # Reconstruct coefficients
    E_i = x[0 : 5 * N : 5]
    F_i = x[1 : 5 * N : 5]
    G_i = x[2 : 5 * N : 5]
    H_i = x[3 : 5 * N : 5]
    
    E_o = np.zeros(N)
    F_o = np.zeros(N)
    G_o = np.zeros(N)
    H_o = np.zeros(N)
    
    U_D = x[5 * N + 1]
    delta_U = x[5 * N + 2]
    
    F_o[0] = -U_D / 3.0
    G_o[0] = x[4]
    H_o[0] = x[5 * N]
    for i in range(1, N):
        G_o[i] = x[5 * i + 4]
        H_o[i] = -G_o[i]
        
    # --- BC 2 & 3: Droplet Interface (r = 1) ---
    r = 1.0
    # Radial Velocity
    bracket_ur_i = E_i * (r ** (n + 1)) + F_i * (r ** (n - 1)) + G_i * (r ** (-n)) + H_i * (r ** (-n - 2))
    ur_i_1 = np.sum((2 * n + 1) * bracket_ur_i * P, axis=1)
    
    bracket_ur_o = E_o * (r ** (n + 1)) + F_o * (r ** (n - 1)) + G_o * (r ** (-n)) + H_o * (r ** (-n - 2))
    ur_o_1 = np.sum((2 * n + 1) * bracket_ur_o * P, axis=1)
    
    # Tangential Velocity
    bracket_ut_i = (n + 3) * E_i * (r ** (n + 1)) + (n + 1) * F_i * (r ** (n - 1)) + (2 - n) * G_i * (r ** (-n)) - n * H_i * (r ** (-n - 2))
    ut_i_1 = np.sum(factor_theta * bracket_ut_i * P_associated, axis=1)
    
    bracket_ut_o = (n + 3) * E_o * (r ** (n + 1)) + (n + 1) * F_o * (r ** (n - 1)) + (2 - n) * G_o * (r ** (-n)) - n * H_o * (r ** (-n - 2))
    ut_o_1 = np.sum(factor_theta * bracket_ut_o * P_associated, axis=1)
    
    # --- BC 5 & 6: Core Boundary (r = k) ---
    r = k
    # Radial Velocity
    bracket_ur_core = E_i * (r ** (n + 1)) + F_i * (r ** (n - 1)) + G_i * (r ** (-n)) + H_i * (r ** (-n - 2))
    ur_i_k = np.sum((2 * n + 1) * bracket_ur_core * P, axis=1)
    ur_rhs_k = delta_U * mew_vals
    
    # Tangential Velocity
    bracket_ut_core = (n + 3) * E_i * (r ** (n + 1)) + (n + 1) * F_i * (r ** (n - 1)) + (2 - n) * G_i * (r ** (-n)) - n * H_i * (r ** (-n - 2))
    ut_i_k = np.sum(factor_theta * bracket_ut_core * P_associated, axis=1)
    ut_rhs_k = -delta_U * sin_theta
    
    # --- BC 1: Far Field limit (evaluated at large r = R_inf) ---
    r_inf = 10.0
    bracket_ur_inf = E_o * (r_inf ** (n + 1)) + F_o * (r_inf ** (n - 1)) + G_o * (r_inf ** (-n)) + H_o * (r_inf ** (-n - 2))
    ur_inf = np.sum((2 * n + 1) * bracket_ur_inf * P, axis=1)
    ur_inf_analytical = -U_D * mew_vals
    
    bracket_ut_inf = (n + 3) * E_o * (r_inf ** (n + 1)) + (n + 1) * F_o * (r_inf ** (n - 1)) + (2 - n) * G_o * (r_inf ** (-n)) - n * H_o * (r_inf ** (-n - 2))
    ut_inf = np.sum(factor_theta * bracket_ut_inf * P_associated, axis=1)
    ut_inf_analytical = U_D * sin_theta
    
    # --- BC 7 & 8: Constraints ---
    g_i_1 = G_i[0]
    g_o_1 = G_o[0]
    
    return {
        "ur_i_1": ur_i_1, "ur_o_1": ur_o_1,
        "ut_i_1": ut_i_1, "ut_o_1": ut_o_1,
        "ur_i_k": ur_i_k, "ur_rhs_k": ur_rhs_k,
        "ut_i_k": ut_i_k, "ut_rhs_k": ut_rhs_k,
        "ur_inf": ur_inf, "ur_inf_analytical": ur_inf_analytical,
        "ut_inf": ut_inf, "ut_inf_analytical": ut_inf_analytical,
        "g_i_1": g_i_1, "g_o_1": g_o_1
    }

def run_unit_tests(x, N, k, tolerance=1e-10):
    """
    Unit test harness that asserts all physical boundary conditions are satisfied
    within the specified tolerance.
    """
    mew_test = np.linspace(-1, 1, 100)
    data = evaluate_fields(x, mew_test, N, k)
    
    # BC 2: Interface radial velocity impermeability
    bc2_err_i = np.max(np.abs(data["ur_i_1"]))
    bc2_err_o = np.max(np.abs(data["ur_o_1"]))
    assert bc2_err_i < tolerance, f"BC 2 (Inner Impermeability) failed: Max error {bc2_err_i:.2e} exceeds tolerance {tolerance:.2e}"
    assert bc2_err_o < tolerance, f"BC 2 (Outer Impermeability) failed: Max error {bc2_err_o:.2e} exceeds tolerance {tolerance:.2e}"
    
    # BC 3: Interface no-slip tangential velocity continuity
    bc3_err = np.max(np.abs(data["ut_i_1"] - data["ut_o_1"]))
    assert bc3_err < tolerance, f"BC 3 (Tangential Continuity) failed: Max error {bc3_err:.2e} exceeds tolerance {tolerance:.2e}"
    
    # BC 5: Core boundary radial kinematic BC
    bc5_err = np.max(np.abs(data["ur_i_k"] - data["ur_rhs_k"]))
    assert bc5_err < tolerance, f"BC 5 (Core Radial Kinematic BC) failed: Max error {bc5_err:.2e} exceeds tolerance {tolerance:.2e}"
    
    # BC 6: Core boundary tangential no-slip BC
    bc6_err = np.max(np.abs(data["ut_i_k"] - data["ut_rhs_k"]))
    assert bc6_err < tolerance, f"BC 6 (Core Tangential BC) failed: Max error {bc6_err:.2e} exceeds tolerance {tolerance:.2e}"
    
    # BC 7: Core Drag Force Balance G_1^i = -1/(12 pi)
    target_g_i_1 = -1.0 / (12.0 * np.pi)
    bc7_err = np.abs(data["g_i_1"] - target_g_i_1)
    assert bc7_err < tolerance, f"BC 7 (Core Drag Constraint) failed: Error {bc7_err:.2e} exceeds tolerance {tolerance:.2e}"
    
    # BC 8: Droplet Force-Free condition G_1^o = 0
    bc8_err = np.abs(data["g_o_1"])
    assert bc8_err < tolerance, f"BC 8 (Droplet Force-Free Condition) failed: Error {bc8_err:.2e} exceeds tolerance {tolerance:.2e}"
    
    print("\n" + "="*50)
    print(" UNIT TEST SUITE: ALL PASSED ".center(50, "="))
    print("="*50)
    print(f"  [PASS] BC 2: Interface Impermeability (ur=0) | err={max(bc2_err_i, bc2_err_o):.1e}")
    print(f"  [PASS] BC 3: Tangential matching (ut_i=ut_o) | err={bc3_err:.1e}")
    print(f"  [PASS] BC 5: Core Radial Kinematic (ur_i=ur_k)| err={bc5_err:.1e}")
    print(f"  [PASS] BC 6: Core Tangential No-Slip (ut_i=ut_k)| err={bc6_err:.1e}")
    print(f"  [PASS] BC 7: Core Drag Force (G_1^i = -1/12pi) | err={bc7_err:.1e}")
    print(f"  [PASS] BC 8: Droplet Force-Free (G_1^o = 0)   | err={bc8_err:.1e}")
    print("="*50 + "\n")

def main():
    # Parameters configuration
    N = 6
    k = 0.3
    lambda_val = 0.5
    Ma = 1.0

    print(f"Solving multi-harmonic Stokes flow for N = {N}, k = {k}, lambda = {lambda_val}, Ma = {Ma}")
    
    # Solve system
    A = assemble_A(N, k, lambda_val)
    b = assemble_b(N, Ma)
    x = np.linalg.solve(A, b)

    print("\nSolved Coefficients Vector x:")
    for idx, val in enumerate(x):
        print(f"  x[{idx:2d}] = {val: .8e}")

    # Run unit testing suite (BCs 2, 3, 5, 6, 7, 8)
    run_unit_tests(x, N, k, tolerance=1e-10)

    # Evaluate boundary conditions for plotting
    mew_vals = np.linspace(-1, 1, 200)
    data = evaluate_fields(x, mew_vals, N, k)

    # Plot results
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))

    # Panel 1: BC 2 Interface Radial
    axs[0, 0].plot(mew_vals, data['ur_i_1'], label=r"$u_{r,i}(r=1)$", color="tab:blue", lw=2)
    axs[0, 0].plot(mew_vals, data['ur_o_1'], label=r"$u_{r,o}(r=1)$", color="tab:orange", linestyle="--", lw=2)
    axs[0, 0].set_ylim(-0.1, 0.1)
    axs[0, 0].set_ylabel(r"Radial Velocity $u_r$")
    axs[0, 0].set_title("BC 2: Interface Impermeability ($r=1.0$)")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # Panel 2: BC 3 Interface Tangential
    axs[0, 1].plot(mew_vals, data['ut_i_1'], label=r"$u_{\theta,i}(r=1)$", color="tab:green", lw=2)
    axs[0, 1].plot(mew_vals, data['ut_o_1'], label=r"$u_{\theta,o}(r=1)$", color="tab:red", linestyle="--", lw=2)
    axs[0, 1].set_ylabel(r"Tangential Velocity $u_\theta$")
    axs[0, 1].set_title("BC 3: Interface No-Slip Continuity ($r=1.0$)")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # Panel 3: BC 5 Core Radial
    axs[1, 0].plot(mew_vals, data['ur_i_k'], label=r"LHS $u_{r,i}(r=k)$", color="tab:blue", lw=2)
    axs[1, 0].plot(mew_vals, data['ur_rhs_k'], label=r"RHS $\mu\Delta U$", color="tab:orange", linestyle="--", lw=2)
    axs[1, 0].set_ylabel(r"Radial Velocity $u_r$")
    axs[1, 0].set_title("BC 5: Core Kinematic Condition ($r=0.3$)")
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    # Panel 4: BC 6 Core Tangential
    axs[1, 1].plot(mew_vals, data['ut_i_k'], label=r"LHS $u_{\theta,i}(r=k)$", color="tab:green", lw=2)
    axs[1, 1].plot(mew_vals, data['ut_rhs_k'], label=r"RHS $-\sin\theta\Delta U$", color="tab:red", linestyle="--", lw=2)
    axs[1, 1].set_ylabel(r"Tangential Velocity $u_\theta$")
    axs[1, 1].set_title("BC 6: Core No-Slip Condition ($r=0.3$)")
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    # Panel 5: BC 1 Far Field Radial (at r = 10.0)
    axs[2, 0].plot(mew_vals, data['ur_inf'], label=r"LHS $u_{r,o}(r=10)$", color="tab:blue", lw=2)
    axs[2, 0].plot(mew_vals, data['ur_inf_analytical'], label=r"RHS $-U_D\cos\theta$", color="tab:orange", linestyle="--", lw=2)
    axs[2, 0].set_xlabel(r"$\mu = \cos(\theta)$")
    axs[2, 0].set_ylabel(r"Radial Velocity $u_r$")
    axs[2, 0].set_title("BC 1: Far-Field Radial Velocity ($r=10.0$)")
    axs[2, 0].legend()
    axs[2, 0].grid(True, alpha=0.3)

    # Panel 6: BC 1 Far Field Tangential (at r = 10.0)
    axs[2, 1].plot(mew_vals, data['ut_inf'], label=r"LHS $u_{\theta,o}(r=10)$", color="tab:green", lw=2)
    axs[2, 1].plot(mew_vals, data['ut_inf_analytical'], label=r"RHS $U_D\sin\theta$", color="tab:red", linestyle="--", lw=2)
    axs[2, 1].set_xlabel(r"$\mu = \cos(\theta)$")
    axs[2, 1].set_ylabel(r"Tangential Velocity $u_\theta$")
    axs[2, 1].set_title("BC 1: Far-Field Tangential Velocity ($r=10.0$)")
    axs[2, 1].legend()
    axs[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = "bc_plot_all.png"
    plt.savefig(plot_path, dpi=200)
    print(f"\nSaved boundary condition verification plot to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    main()
