import os
import numpy as np
import matplotlib.pyplot as plt

from squirmer_drop.paths import RUNS_DIR, OUTPUT_FIGURES_DIR

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

def compute_shear_stress(r, mew_vals, E, F, G, H, N, viscosity):
    """
    Computes the radial-tangential shear stress component sigma_{r,theta}
    for given coefficients and viscosity at radius r.
    """
    P, dP = get_legendre_and_derivs(mew_vals, N)
    n = np.arange(1, N + 1)
    sin_theta = np.sqrt(1.0 - mew_vals**2)
    P_associated = sin_theta[:, np.newaxis] * dP
    factor_sigma = 2.0 * (2 * n + 1) / (n * (n + 1))
    
    bracket_sig = (n * (n + 2) * E * (r**n) + 
                   (n**2 - 1) * F * (r**(n - 2)) + 
                   (n**2 - 1) * G * (r**(-n - 1)) + 
                   n * (n + 2) * H * (r**(-n - 3)))
    
    # Tangential shear stress component
    sig = -viscosity * np.sum(factor_sigma * bracket_sig * P_associated, axis=1)
    return sig

def assemble_A(N, k, lambda_val, shifted=True):
    """
    Assembles the linear system matrix A of size (5N + 3) x (5N + 3).
    Here, the 5th variable of group n, x[5*i + 4] (for i >= 1), represents G^o_n.
    """
    total_rows = 5 * N + 3
    A = np.zeros((total_rows, total_rows))

    # Block N1:
    for i in range(N):
        n_val = i + 1
        N_one_coeff = 2.0 * (2.0 * n_val + 1.0) * (1.0 / (n_val * (n_val + 1.0)))
        A[i, 5 * i] = -N_one_coeff * n_val * (n_val + 2.0) * lambda_val
        A[i, 5 * i + 1] = -N_one_coeff * (n_val * n_val - 1.0) * lambda_val
        A[i, 5 * i + 2] = -N_one_coeff * (n_val * n_val - 1.0) * lambda_val
        A[i, 5 * i + 3] = -N_one_coeff * n_val * (n_val + 2.0) * lambda_val
        if i != 0:
            A[i, 5 * i + 4] = N_one_coeff * ((n_val * n_val - 1.0) - n_val * (n_val + 2.0))
    # n = 1 
    A[0, 5 * N] = 9.0

    # Block N2:
    for i in range(N):
        r_row = N + i
        A[r_row, 5 * i] = 1.0
        A[r_row, 5 * i + 1] = 1.0
        A[r_row, 5 * i + 2] = 1.0
        A[r_row, 5 * i + 3] = 1.0

    # Block N3:
    r_row = 2 * N
    # n = 1 mode:
    A[r_row, 0] = -4.0
    A[r_row, 1] = -2.0
    A[r_row, 2] = -1.0
    A[r_row, 3] = 1.0
    A[r_row, 4] = 1.0
    A[r_row, 5 * N] = -1.0
    A[r_row, 5 * N + 1] = -2.0 / 3.0

    # n >= 2 modes:
    for i in range(1, N):
        n_val = i + 1
        A[r_row + i, 5 * i] = -(n_val + 3.0)
        A[r_row + i, 5 * i + 1] = -(n_val + 1.0)
        A[r_row + i, 5 * i + 2] = -(2.0 - n_val)
        A[r_row + i, 5 * i + 3] = n_val
        A[r_row + i, 5 * i + 4] = 2.0

    # Block N4: 
    r_row = 3 * N
    # n = 1 mode:
    if shifted:
        A[r_row, 0] = 6.0 * (k ** 3)
        A[r_row, 1] = 3.0 * k
        A[r_row, 2] = 1.5
        A[r_row, 3] = -1.5 * (k ** -2)
        A[r_row, 5 * N + 2] = -1.0 * k
    else:
        A[r_row, 0] = 6.0 * (k ** 2)
        A[r_row, 1] = 3.0
        A[r_row, 2] = 1.5 * (k ** -1)
        A[r_row, 3] = -1.5 * (k ** -3)
        A[r_row, 5 * N + 2] = -1.0
    # n >= 2 modes
    for i in range(1, N):
        n_val = i + 1
        if shifted:
            A[r_row + i, 5 * i] = (n_val + 3.0) * (k ** (2.0 * n_val + 1.0))
            A[r_row + i, 5 * i + 1] = (n_val + 1.0) * (k ** (2.0 * n_val - 1.0))
            A[r_row + i, 5 * i + 2] = 2.0 - n_val
            A[r_row + i, 5 * i + 3] = -n_val * (k ** -2)
        else:
            A[r_row + i, 5 * i] = (n_val + 3.0) * (k ** (n_val + 1.0))
            A[r_row + i, 5 * i + 1] = (n_val + 1.0) * (k ** (n_val - 1.0))
            A[r_row + i, 5 * i + 2] = (-n_val + 2.0) * (k ** -n_val)
            A[r_row + i, 5 * i + 3] = -n_val * (k ** (-n_val - 2.0))

    # Block N5: Core boundary condition for u_r at r = k
    r_row = 4 * N
    # n = 1 mode:
    if shifted:
        A[r_row, 0] = 3.0 * (k ** 3)
        A[r_row, 1] = 3.0 * k
        A[r_row, 2] = 3.0
        A[r_row, 3] = 3.0 * (k ** -2)
        A[r_row, 5 * N + 2] = -1.0 * k
    else:
        A[r_row, 0] = 3.0 * (k ** 2)
        A[r_row, 1] = 3.0
        A[r_row, 2] = 3.0 * (k ** -1)
        A[r_row, 3] = 3.0 * (k ** -3)
        A[r_row, 5 * N + 2] = -1.0
    # n >= 2 modes
    for i in range(1, N):
        n_val = i + 1
        if shifted:
            A[r_row + i, 5 * i] = k ** (2.0 * n_val + 1.0)
            A[r_row + i, 5 * i + 1] = k ** (2.0 * n_val - 1.0)
            A[r_row + i, 5 * i + 2] = 1.0
            A[r_row + i, 5 * i + 3] = k ** -2
        else:
            A[r_row + i, 5 * i] = k ** (n_val + 1.0)
            A[r_row + i, 5 * i + 1] = k ** (n_val - 1.0)
            A[r_row + i, 5 * i + 2] = k ** -n_val
            A[r_row + i, 5 * i + 3] = k ** (-n_val - 2.0)

    A[5 * N, 4] = 1.0
    A[5 * N, 5 * N + 1] = -1.0 / 3.0
    A[5 * N, 5 * N] = 1.0

    A[5 * N + 1, 4] = 1.0
    A[5 * N + 2, 2] = 1.0

    return A

def assemble_b(N, Ma, c_n=None, mode='particle', lambda_val=2.0, k=0.4, A_n=None, B_n=None):
    """
    Assembles the right-hand-side vector b of size 5N + 3.
    Supports both 'particle' and 'squirmer' modes.
    """
    total_rows = 5 * N + 3
    b = np.zeros(total_rows)
    for i in range(N):
        val = c_n[i] if c_n is not None else (0.1 ** (i + 1))
        b[i] = Ma * val

    if mode == 'particle':
        b[5 * N + 1] = -1.0 / (12.0 * np.pi)
        b[5 * N + 2] = -1.0 / (12.0 * np.pi * lambda_val)
    elif mode == 'squirmer':
        b[5 * N + 1] = 0.0
        b[5 * N + 2] = 0.0
        
        if A_n is None:
            A_n = 0.1 ** np.arange(1, N + 1)
        else:
            A_n = np.asarray(A_n, dtype=float)
        if B_n is None:
            B_n = 0.1 ** np.arange(1, N + 1)
        else:
            B_n = np.asarray(B_n, dtype=float)
            
        # n = 1 (row 3N is tangential, row 4N is radial, scaled by k^1)
        b[3 * N] = -B_n[0] * k
        b[4 * N] = A_n[0] * k
        
        # n >= 2 (scaled by k^n)
        for i in range(1, N):
            n_val = i + 1
            b[3 * N + i] = -2.0 * B_n[i] * (k ** n_val) / (2.0 * n_val + 1.0)
            b[4 * N + i] = A_n[i] * (k ** n_val) / (2.0 * n_val + 1.0)
            
    return b

def evaluate_fields(x, mew_vals, N, k, lambda_val, Ma=1.0, r_inf=10.0, c_n=None, mode='particle', A_n=None, B_n=None):
    """
    Computes field velocities and shear stresses at the boundaries
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
    
    # Tangential Velocity
    bracket_ut_core = (n + 3) * E_i * (r ** (n + 1)) + (n + 1) * F_i * (r ** (n - 1)) + (2 - n) * G_i * (r ** (-n)) - n * H_i * (r ** (-n - 2))
    ut_i_k = np.sum(factor_theta * bracket_ut_core * P_associated, axis=1)
    
    if mode == 'particle':
        ur_rhs_k = delta_U * mew_vals
        ut_rhs_k = -delta_U * sin_theta
    elif mode == 'squirmer':
        if A_n is None:
            A_n = 0.1 ** np.arange(1, N + 1)
        else:
            A_n = np.asarray(A_n, dtype=float)
        if B_n is None:
            B_n = 0.1 ** np.arange(1, N + 1)
        else:
            B_n = np.asarray(B_n, dtype=float)
            
        ur_rhs_k = delta_U * mew_vals + np.sum(A_n * P, axis=1)
        ut_rhs_k = -delta_U * sin_theta + np.sum((2.0 * B_n / (n * (n + 1.0))) * P_associated, axis=1)
    
    # --- BC 1: Far Field limit (evaluated at large r = r_inf) ---
    bracket_ur_inf = E_o * (r_inf ** (n + 1)) + F_o * (r_inf ** (n - 1)) + G_o * (r_inf ** (-n)) + H_o * (r_inf ** (-n - 2))
    ur_inf = np.sum((2 * n + 1) * bracket_ur_inf * P, axis=1)
    ur_inf_analytical = -U_D * mew_vals + 3.0 * G_o[0] / r_inf * mew_vals
    
    bracket_ut_inf = (n + 3) * E_o * (r_inf ** (n + 1)) + (n + 1) * F_o * (r_inf ** (n - 1)) + (2 - n) * G_o * (r_inf ** (-n)) - n * H_o * (r_inf ** (-n - 2))
    ut_inf = np.sum(factor_theta * bracket_ut_inf * P_associated, axis=1)
    ut_inf_analytical = U_D * sin_theta - 1.5 * G_o[0] / r_inf * sin_theta
    
    # --- BC 4: Shear Stress Balance (r = 1) ---
    sig_i = compute_shear_stress(1.0, mew_vals, E_i, F_i, G_i, H_i, N, lambda_val)
    sig_o = compute_shear_stress(1.0, mew_vals, E_o, F_o, G_o, H_o, N, 1.0)
    sig_diff = sig_i - sig_o
    
    # RHS Stress from surfactant gradient
    if c_n is None:
        c_n = 0.1 ** n
    rhs_stress = Ma * np.sum(c_n * P_associated, axis=1)
    
    # --- BC 7 & 8: Constraints ---
    g_i_1 = G_i[0]
    g_o_1 = G_o[0]
    
    return {
        "ur_i_1": ur_i_1, "ur_o_1": ur_o_1,
        "ut_i_1": ut_i_1, "ut_o_1": ut_o_1,
        "sig_diff": sig_diff, "rhs_stress": rhs_stress,
        "ur_i_k": ur_i_k, "ur_rhs_k": ur_rhs_k,
        "ut_i_k": ut_i_k, "ut_rhs_k": ut_rhs_k,
        "ur_inf": ur_inf, "ur_inf_analytical": ur_inf_analytical,
        "ut_inf": ut_inf, "ut_inf_analytical": ut_inf_analytical,
        "g_i_1": g_i_1, "g_o_1": g_o_1
    }

def run_unit_tests(x, N, k, lambda_val, Ma=1.0, tolerance=1e-10, c_n=None, mode='particle', A_n=None, B_n=None):
    """
    Unit test harness that asserts all physical boundary conditions are satisfied
    within the specified tolerance.
    """
    mew_test = np.linspace(-1, 1, 100)
    # Evaluate at a very large R (10000.0) to test condition 1 (far-field decay to -UD e_z)
    data = evaluate_fields(x, mew_test, N, k, lambda_val, Ma=Ma, r_inf=10000.0, c_n=c_n, mode=mode, A_n=A_n, B_n=B_n)
    
    # BC 1: Far field velocity limit r -> infinity
    bc1_err_r = np.max(np.abs(data["ur_inf"] - data["ur_inf_analytical"]))
    bc1_err_theta = np.max(np.abs(data["ut_inf"] - data["ut_inf_analytical"]))
    assert bc1_err_r < tolerance, f"BC 1 (Far Field Radial) failed: Max error {bc1_err_r:.2e} exceeds tolerance {tolerance:.2e}"
    assert bc1_err_theta < tolerance, f"BC 1 (Far Field Tangential) failed: Max error {bc1_err_theta:.2e} exceeds tolerance {tolerance:.2e}"
    
    # BC 2: Interface radial velocity impermeability
    bc2_err_i = np.max(np.abs(data["ur_i_1"]))
    bc2_err_o = np.max(np.abs(data["ur_o_1"]))
    assert bc2_err_i < tolerance, f"BC 2 (Inner Impermeability) failed: Max error {bc2_err_i:.2e} exceeds tolerance {tolerance:.2e}"
    assert bc2_err_o < tolerance, f"BC 2 (Outer Impermeability) failed: Max error {bc2_err_o:.2e} exceeds tolerance {tolerance:.2e}"
    
    # BC 3: Interface no-slip tangential velocity continuity
    bc3_err = np.max(np.abs(data["ut_i_1"] - data["ut_o_1"]))
    assert bc3_err < tolerance, f"BC 3 (Tangential Continuity) failed: Max error {bc3_err:.2e} exceeds tolerance {tolerance:.2e}"
    
    # BC 4: Interface shear stress balance
    bc4_err = np.max(np.abs(data["sig_diff"] - data["rhs_stress"]))
    assert bc4_err < tolerance, f"BC 4 (Shear Stress Balance) failed: Max error {bc4_err:.2e} exceeds tolerance {tolerance:.2e}"
    
    # BC 5: Core boundary radial kinematic BC
    bc5_err = np.max(np.abs(data["ur_i_k"] - data["ur_rhs_k"]))
    assert bc5_err < tolerance, f"BC 5 (Core Radial Kinematic BC) failed: Max error {bc5_err:.2e} exceeds tolerance {tolerance:.2e}"
    
    # BC 6: Core boundary tangential no-slip BC
    bc6_err = np.max(np.abs(data["ut_i_k"] - data["ut_rhs_k"]))
    assert bc6_err < tolerance, f"BC 6 (Core Tangential BC) failed: Max error {bc6_err:.2e} exceeds tolerance {tolerance:.2e}"
    
    # BC 7: Core Drag Force Balance
    if mode == 'particle':
        target_g_i_1 = -1.0 / (12.0 * np.pi * lambda_val)
        bc7_msg = f"G_1^i = -1/(12 pi lambda)"
    else:
        target_g_i_1 = 0.0
        bc7_msg = "G_1^i = 0"
        
    bc7_err = np.abs(data["g_i_1"] - target_g_i_1)
    assert bc7_err < tolerance, f"BC 7 ({bc7_msg}) failed: Error {bc7_err:.2e} exceeds tolerance {tolerance:.2e}"
    
    # BC 8: Droplet Force condition
    if mode == 'particle':
        target_g_o_1 = -1.0 / (12.0 * np.pi)
        bc8_msg = f"G_1^o = -1/(12 pi)"
    else:
        target_g_o_1 = 0.0
        bc8_msg = "G_1^o = 0"
        
    bc8_err = np.abs(data["g_o_1"] - target_g_o_1)
    assert bc8_err < tolerance, f"BC 8 ({bc8_msg}) failed: Error {bc8_err:.2e} exceeds tolerance {tolerance:.2e}"
    
    print("\n" + "="*50)
    print(" UNIT TEST SUITE: ALL PASSED ".center(50, "="))
    print("="*50)
    print(f"  [PASS] BC 1: Far Field limit (r=10000)       | err={max(bc1_err_r, bc1_err_theta):.1e}")
    print(f"  [PASS] BC 2: Interface Impermeability (ur=0) | err={max(bc2_err_i, bc2_err_o):.1e}")
    print(f"  [PASS] BC 3: Tangential matching (ut_i=ut_o) | err={bc3_err:.1e}")
    print(f"  [PASS] BC 4: Shear Stress Balance            | err={bc4_err:.1e}")
    print(f"  [PASS] BC 5: Core Radial Kinematic (ur_i=ur_k)| err={bc5_err:.1e}")
    print(f"  [PASS] BC 6: Core Tangential No-Slip (ut_i=ut_k)| err={bc6_err:.1e}")
    print(f"  [PASS] BC 7: Core Drag Force ({bc7_msg}) | err={bc7_err:.1e}")
    print(f"  [PASS] BC 8: Droplet Force ({bc8_msg})   | err={bc8_err:.1e}")
    print("="*50 + "\n")

def run_analytical_limit_tests(N, k, lambda_val, Ma=1.0, mode='particle', A_n=None, B_n=None):
    """
    Validates solver results against analytical limiting solutions from Section 5.
    """
    if mode == 'particle' and Ma == 0.0:
        # Eq 16a and 16b checks
        A = assemble_A(N, k, lambda_val, shifted=True)
        b = assemble_b(N, Ma=0.0, mode='particle', lambda_val=lambda_val, k=k)
        
        # NOTE: The analytical clean drop solutions (16a, 16b) were derived assuming
        # G_1^i = -1/(12*pi*lambda) (omitting the core size k in the denominator).
        # We temporarily force this assumption to verify the analytical solution algebra.
        b[5 * N + 2] = -1.0 / (12.0 * np.pi * lambda_val)
        
        x = np.linalg.solve(A, b)
        U_d_solver = x[5 * N + 1]
        U_p_solver = x[5 * N + 1] + x[5 * N + 2]
        
        denom = 12.0 * np.pi * (2.0 * (k**5) * (lambda_val - 1.0) + 3.0 * lambda_val + 2.0)
        U_d_anal = (-4.0 * (k**5) * lambda_val + 4.0 * (k**5) + 5.0 * (k**2) - 6.0 * lambda_val - 9.0) / denom
        
        U_p_num = (-4.0*(k**6)*(lambda_val**2) + 8.0*(k**6)*lambda_val - 4.0*(k**6) 
                   - 9.0*(k**5)*lambda_val + 9.0*(k**5) + 10.0*(k**3)*lambda_val - 10.0*(k**3) 
                   - 6.0*k*(lambda_val**2) - 3.0*k*lambda_val + 9.0*k - 6.0*lambda_val - 4.0)
        U_p_anal = U_p_num / (k * lambda_val * denom)
        
        diff_d = abs(U_d_solver - U_d_anal)
        diff_p = abs(U_p_solver - U_p_anal)
        
        print("\n" + "="*50)
        print(" ANALYTICAL LIMIT TESTS (Ma=0 PARTICLE) ".center(50, "="))
        print("="*50)
        print(f"  [PASS] Clean Drop U_d (Eq 16a) | Solver={U_d_solver:.6f}, Anal={U_d_anal:.6f}, diff={diff_d:.1e}")
        print(f"  [PASS] Clean Drop U_p (Eq 16b) | Solver={U_p_solver:.6f}, Anal={U_p_anal:.6f}, diff={diff_p:.1e}")
        print("="*50 + "\n")
        
        assert diff_d < 1e-10, f"Analytical Clean Drop U_d failed: diff={diff_d:.2e}"
        assert diff_p < 1e-10, f"Analytical Clean Drop U_p failed: diff={diff_p:.2e}"

    elif mode == 'squirmer':
        if A_n is None:
            A_n = 0.1 ** np.arange(1, N + 1)
        if B_n is None:
            B_n = 0.1 ** np.arange(1, N + 1)
            
        A = assemble_A(N, k, lambda_val, shifted=True)
        
        # 1. O(1) Squirmer Check
        b_0 = assemble_b(N, Ma=0.0, mode='squirmer', lambda_val=lambda_val, k=k, A_n=A_n, B_n=B_n)
        x_0 = np.linalg.solve(A, b_0)
        
        U0_D_solver = x_0[5 * N + 1]
        U0_S_solver = x_0[5 * N + 1] + x_0[5 * N + 2]
        
        denom_0 = (6.0 * lambda_val - 6.0) * (k**5) + 9.0 * lambda_val + 6.0
        U0_D_anal = 10.0 * (k**3) * lambda_val * (A_n[0] + B_n[0]) / denom_0
        U0_S_anal = (-12.0 * (lambda_val - 1.0) * (A_n[0] + B_n[0]/2.0) * (k**5) 
                     + 10.0 * (k**3) * (A_n[0] + B_n[0]) * (lambda_val - 1.0) 
                     - 3.0 * (A_n[0] - 2.0*B_n[0]) * (lambda_val + 2.0/3.0)) / denom_0
                     
        diff_u0_d = abs(U0_D_solver - U0_D_anal)
        diff_u0_s = abs(U0_S_solver - U0_S_anal)
        
        # 2. O(Pe_s) Squirmer Check
        c_n = np.zeros(N)
        for i in range(N):
            n_val = i + 1
            c_n[i] = -B_n[i] / (2.0 * np.pi * n_val * (n_val + 1.0))
            
        b_1 = assemble_b(N, Ma=Ma, c_n=c_n, mode='squirmer', lambda_val=lambda_val, k=k, A_n=np.zeros(N), B_n=np.zeros(N))
        x_1 = np.linalg.solve(A, b_1)
        
        U1_D_solver = x_1[5 * N + 1]
        U1_S_solver = x_1[5 * N + 1] + x_1[5 * N + 2]
        
        denom_1 = ((lambda_val - 1.0) * (k**5) + 1.5 * lambda_val + 1.0) ** 2
        U1_D_anal = -5.0 * Ma * lambda_val * (A_n[0] + B_n[0]) * (k**3) * (1.0 - k**5) / (6.0 * denom_1)
        U1_S_anal = -25.0 * Ma * (k**3) * lambda_val * (1.0 - k) * (k + 1.0) * (A_n[0] + B_n[0]) / (12.0 * denom_1)
        
        diff_u1_d = abs(U1_D_solver - U1_D_anal)
        diff_u1_s = abs(U1_S_solver - U1_S_anal)
        
        print("\n" + "="*50)
        print(" ANALYTICAL LIMIT TESTS (SQUIRMER) ".center(50, "="))
        print("="*50)
        print(f"  [PASS] O(1) Squirmer U0_D (Eq 4.4) | Solver={U0_D_solver:.6f}, Anal={U0_D_anal:.6f}, diff={diff_u0_d:.1e}")
        print(f"  [PASS] O(1) Squirmer U0_S (Eq 4.3) | Solver={U0_S_solver:.6f}, Anal={U0_S_anal:.6f}, diff={diff_u0_s:.1e}")
        print(f"  [PASS] O(Pe) Squirmer U1_D (Eq 4.6) | Solver={U1_D_solver:.6f}, Anal={U1_D_anal:.6f}, diff={diff_u1_d:.1e}")
        print(f"  [PASS] O(Pe) Squirmer U1_S (Eq 4.5) | Solver={U1_S_solver:.6f}, Anal={U1_S_anal:.6f}, diff={diff_u1_s:.1e}")
        print("="*50 + "\n")
        
        assert diff_u0_d < 1e-10, f"O(1) Squirmer U0_D failed: diff={diff_u0_d:.2e}"
        assert diff_u0_s < 1e-10, f"O(1) Squirmer U0_S failed: diff={diff_u0_s:.2e}"
        assert diff_u1_d < 1e-3, f"O(Pe) Squirmer U1_D failed: diff={diff_u1_d:.2e}"
        assert diff_u1_s < 1e-3, f"O(Pe) Squirmer U1_S failed: diff={diff_u1_s:.2e}"

def plot_velocity_field(x, N, k, lambda_val, Ma=None, Pe=None, mode='particle', folder_prefix=None):
    """
    Generates a 2D vector arrow (quiver) plot of the velocity field.
    """
    plt.close('all')
    
    # 2D Grid in x-z plane
    grid_size = 35
    x_range = np.linspace(-2.0, 2.0, grid_size)
    z_range = np.linspace(-2.0, 2.0, grid_size)
    X, Z = np.meshgrid(x_range, z_range)
    
    R = np.sqrt(X**2 + Z**2)
    Theta = np.arctan2(X, Z)  # angle from z-axis (axisymmetry)
    mew = np.cos(Theta)
    
    # Reshape to 1D for Legendre processing
    r_flat = R.flatten()
    mew_flat = mew.flatten()
    theta_flat = Theta.flatten()
    
    P, dP = get_legendre_and_derivs(mew_flat, N)
    n = np.arange(1, N + 1)
    sin_theta = np.sqrt(1.0 - mew_flat**2)
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
        
    # Evaluate velocities for all three regions
    ur = np.zeros_like(r_flat)
    ut = np.zeros_like(r_flat)
    
    # Masks
    mask_core = r_flat < k
    mask_inner = (r_flat >= k) & (r_flat <= 1.0)
    mask_outer = r_flat > 1.0
    
    # Region 1: Solid Core (r < k) - set to NaN to prevent plotting arrows inside
    ur[mask_core] = np.nan
    ut[mask_core] = np.nan
    
    # Region 2: Inner Fluid Shell (k <= r <= 1.0)
    if np.any(mask_inner):
        r_sub = r_flat[mask_inner][:, np.newaxis]
        P_sub = P[mask_inner, :]
        P_assoc_sub = P_associated[mask_inner, :]
        
        bracket_ur = E_i * (r_sub ** (n + 1)) + F_i * (r_sub ** (n - 1)) + G_i * (r_sub ** (-n)) + H_i * (r_sub ** (-n - 2))
        ur[mask_inner] = np.sum((2 * n + 1) * bracket_ur * P_sub, axis=1)
        
        bracket_ut = (n + 3) * E_i * (r_sub ** (n + 1)) + (n + 1) * F_i * (r_sub ** (n - 1)) + (2 - n) * G_i * (r_sub ** (-n)) - n * H_i * (r_sub ** (-n - 2))
        ut[mask_inner] = np.sum(factor_theta * bracket_ut * P_assoc_sub, axis=1)
        
    # Region 3: Outer Fluid (r > 1.0)
    if np.any(mask_outer):
        r_sub = r_flat[mask_outer][:, np.newaxis]
        P_sub = P[mask_outer, :]
        P_assoc_sub = P_associated[mask_outer, :]
        
        bracket_ur = E_o * (r_sub ** (n + 1)) + F_o * (r_sub ** (n - 1)) + G_o * (r_sub ** (-n)) + H_o * (r_sub ** (-n - 2))
        ur[mask_outer] = np.sum((2 * n + 1) * bracket_ur * P_sub, axis=1)
        
        bracket_ut = (n + 3) * E_o * (r_sub ** (n + 1)) + (n + 1) * F_o * (r_sub ** (n - 1)) + (2 - n) * G_o * (r_sub ** (-n)) - n * H_o * (r_sub ** (-n - 2))
        ut[mask_outer] = np.sum(factor_theta * bracket_ut * P_assoc_sub, axis=1)
        
    # Transform velocities back to 2D Cartesian (u_x, u_z)
    theta_phys = np.abs(theta_flat)
    U_X_flat = np.sign(X.flatten()) * (ur * np.sin(theta_phys) + ut * np.cos(theta_phys))
    U_Z_flat = ur * np.cos(theta_phys) - ut * np.sin(theta_phys)
    
    U_X = U_X_flat.reshape((grid_size, grid_size))
    U_Z = U_Z_flat.reshape((grid_size, grid_size))
    
    # Compute velocity magnitude (speed) for color coding
    speed_flat = np.sqrt(U_X_flat**2 + U_Z_flat**2)
    speed = speed_flat.reshape((grid_size, grid_size))
    
    # Plotting Vector Field
    fig = plt.figure(figsize=(11, 10))
    
    # Quiver plot (vector arrows colored by speed, with fixed velocity scale and clim)
    q = plt.quiver(X, Z, U_X, U_Z, speed, cmap="plasma", alpha=0.9, width=0.0055, scale=0.1, scale_units='xy', clim=(0.0, 0.08))
    
    # Colorbar
    cbar = plt.colorbar(q, shrink=0.8)
    cbar.set_label(r"Velocity Magnitude $|u|$")
    
    # Quiver key (legend scale)
    plt.quiverkey(q, 0.76, 0.03, 0.05, r"$0.05$ (velocity units)", labelpos="E", coordinates="axes", fontproperties={"size": 10})
    
    # Add faint dashed lines for droplet and core boundaries
    theta_c = np.linspace(0, 2*np.pi, 200)
    # Droplet Boundary r = 1.0
    plt.plot(np.sin(theta_c), np.cos(theta_c), color="black", linestyle="--", linewidth=1.5, alpha=0.45, label="Droplet Interface (r=1.0)")
    # Core Boundary r = k
    plt.plot(k * np.sin(theta_c), k * np.cos(theta_c), color="black", linestyle="--", linewidth=1.2, alpha=0.3, label=f"Solid Core (r={k})")
    
    plt.xlabel("x")
    plt.ylabel("z (Axis of Symmetry)")
    plt.title(f"Velocity Field Map (N = {N}, k = {k}, lambda = {lambda_val}) in Droplet Frame")
    plt.axis("equal")
    plt.xlim(-2.0, 2.0)
    plt.ylim(-2.0, 2.0)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    if Ma is not None and Pe is not None:
        prefix = folder_prefix if folder_prefix is not None else f"plots_{mode}"
        dir_path = os.path.join(RUNS_DIR, prefix, f"Ma={Ma}", f"Pe={Pe}", f"N={N}")
        os.makedirs(dir_path, exist_ok=True)
        save_path = os.path.join(dir_path, f"plot_lambda={lambda_val}_k={k}.png")
    else:
        os.makedirs(OUTPUT_FIGURES_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_FIGURES_DIR, "velocity_field.png")
        
    plt.savefig(save_path, dpi=250)
    print(f"\nSaved velocity field vector plot to: {save_path}")
    plt.close(fig)

def plot_stream_function(x, N, k, lambda_val, Ma=None, Pe=None, mode='particle', folder_prefix=None):
    """
    Plots the stream function psi(r, theta) as a 2D contour plot (color map + lines)
    in the x-z plane.
    """
    plt.close('all')
    
    # 2D Grid in x-z plane (finer for smooth contours)
    grid_size = 200
    x_range = np.linspace(-2.0, 2.0, grid_size)
    z_range = np.linspace(-2.0, 2.0, grid_size)
    X, Z = np.meshgrid(x_range, z_range)
    
    R = np.sqrt(X**2 + Z**2)
    # Avoid division by zero
    R_safe = np.where(R == 0, 1e-15, R)
    mew = Z / R_safe
    
    # Flatten for Legendre evaluations
    r_flat = R.flatten()
    mew_flat = mew.flatten()
    
    P, dP = get_legendre_and_derivs(mew_flat, N)
    n = np.arange(1, N + 1)
    
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
        
    factor = (2 * n + 1) / (n * (n + 1))
    
    # Compute psi in flattened arrays
    psi_inner_flat = np.zeros_like(r_flat)
    psi_outer_flat = np.zeros_like(r_flat)
    
    for idx, n_val in enumerate(n):
        term_i = E_i[idx] * (r_flat ** (n_val + 3)) + F_i[idx] * (r_flat ** (n_val + 1)) + G_i[idx] * (r_flat ** (2 - n_val)) + H_i[idx] * (r_flat ** (-n_val))
        psi_inner_flat += factor[idx] * term_i * (1.0 - mew_flat**2) * dP[:, idx]
        
        term_o = E_o[idx] * (r_flat ** (n_val + 3)) + F_o[idx] * (r_flat ** (n_val + 1)) + G_o[idx] * (r_flat ** (2 - n_val)) + H_o[idx] * (r_flat ** (-n_val))
        psi_outer_flat += factor[idx] * term_o * (1.0 - mew_flat**2) * dP[:, idx]
        
    psi_inner = psi_inner_flat.reshape(X.shape)
    psi_outer = psi_outer_flat.reshape(X.shape)
    
    # Core translation stream function
    psi_core = 0.5 * delta_U * (R**2) * (1.0 - mew**2)
    
    # Combine regions and mask the solid core (r < k) with NaN
    psi = np.zeros_like(X)
    psi = np.where(R < k, np.nan, 
                   np.where(R <= 1.0, psi_inner, psi_outer))
    
    fig = plt.figure(figsize=(11, 10))
    
    # Diverging colormap for positive/negative stream function values in the fluid region
    levels = 40
    plt.contourf(X, Z, psi, levels=levels, cmap="RdBu_r", alpha=0.85)
    cbar = plt.colorbar(shrink=0.8)
    cbar.set_label(r"Stream Function $\psi$")
    
    # Plot streamline contour lines (using np.nanmax since psi contains NaNs)
    max_val = np.nanmax(np.abs(psi))
    if np.isnan(max_val) or np.isinf(max_val) or max_val == 0:
        max_val = 1.0
    line_levels = np.linspace(-max_val, max_val, 21)
    
    cntr = plt.contour(X, Z, psi, levels=line_levels, colors="black", linewidths=0.8, alpha=0.7)
    plt.clabel(cntr, inline=True, fontsize=8, fmt='%.4f')
    
    # Highlight boundary lines (r = 1.0 and r = k) and shade the core
    theta_c = np.linspace(0, 2*np.pi, 250)
    plt.plot(np.sin(theta_c), np.cos(theta_c), color="black", linestyle="-", linewidth=2.0, label="Droplet Interface (r=1.0)")
    plt.plot(k * np.sin(theta_c), k * np.cos(theta_c), color="darkred", linestyle="-", linewidth=2.0, label=f"Solid Core (r={k})")
    plt.fill(k * np.sin(theta_c), k * np.cos(theta_c), color="#e5e7eb", alpha=0.75, edgecolor="darkred", linewidth=1.5)
    
    plt.xlabel("x")
    plt.ylabel("z (Axis of Symmetry)")
    plt.title(f"Stream Function $\\psi$ (N = {N}, k = {k}, lambda = {lambda_val}) in Droplet Frame")
    plt.axis("equal")
    plt.xlim(-2.0, 2.0)
    plt.ylim(-2.0, 2.0)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.15)
    plt.tight_layout()
    
    if Ma is not None and Pe is not None:
        prefix = folder_prefix if folder_prefix is not None else f"plots_{mode}"
        dir_path = os.path.join(RUNS_DIR, prefix, f"Ma={Ma}", f"Pe={Pe}", f"N={N}")
        os.makedirs(dir_path, exist_ok=True)
        save_path = os.path.join(dir_path, f"stream_lambda={lambda_val}_k={k}.png")
    else:
        os.makedirs(OUTPUT_FIGURES_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_FIGURES_DIR, "stream_function.png")
        
    plt.savefig(save_path, dpi=250)
    print(f"Saved stream function contour plot to: {save_path}")
    plt.close(fig)
