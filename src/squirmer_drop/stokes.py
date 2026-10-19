import os
import numpy as np
import matplotlib.pyplot as plt

from squirmer_drop.paths import RUNS_DIR, OUTPUT_FIGURES_DIR

# Layout of the state vector x (length 5N + 3), for mode n = i + 1:
#   x[5i : 5i+4]  E^i_n, F^i_n, G^i_n, H^i_n   inner shell, k < r < 1
#   x[5i + 4]     G^o_n                        outer fluid, r > 1
#   x[5N]         H^o_1
#   x[5N + 1]     U_D       drop velocity
#   x[5N + 2]     delta_U   core velocity relative to the drop


def get_legendre_and_derivs(mew, N):
    """Legendre polynomials P_n(mew) and derivatives P'_n(mew) for n = 1..N."""
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

def unpack_coefficients(x, N):
    """
    Split the state vector into inner (E, F, G, H)^i and outer (E, F, G, H)^o Lamb
    coefficients, plus U_D and delta_U = U_core - U_D.

    The outer field is bounded at infinity, so E^o = 0 and F^o_n = 0 for n >= 2, with
    F^o_1 = -U_D / 3 from the uniform stream. Impermeability at r = 1 gives
    H^o_n = -G^o_n for n >= 2; H^o_1 is stored separately at x[5N].
    """
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

    return (E_i, F_i, G_i, H_i), (E_o, F_o, G_o, H_o), U_D, delta_U

def compute_shear_stress(r, mew_vals, E, F, G, H, N, viscosity):
    """Shear stress sigma_{r theta} at radius r for one set of Lamb coefficients."""
    P, dP = get_legendre_and_derivs(mew_vals, N)
    n = np.arange(1, N + 1)
    sin_theta = np.sqrt(1.0 - mew_vals**2)
    P_associated = sin_theta[:, np.newaxis] * dP
    factor_sigma = 2.0 * (2 * n + 1) / (n * (n + 1))

    bracket_sig = (n * (n + 2) * E * (r**n) +
                   (n**2 - 1) * F * (r**(n - 2)) +
                   (n**2 - 1) * G * (r**(-n - 1)) +
                   n * (n + 2) * H * (r**(-n - 3)))
    sig = -viscosity * np.sum(factor_sigma * bracket_sig * P_associated, axis=1)
    return sig

def assemble_A(N, k, lambda_val):
    """
    Boundary-condition matrix A, size (5N + 3) x (5N + 3). It depends only on N, k and
    lambda, so it can be reused across Picard iterations.

    The core rows (r = k) are multiplied through by powers of k so the entries stay
    O(1) for small k; assemble_b applies the same scaling to the right-hand side.
    """
    total_rows = 5 * N + 3
    A = np.zeros((total_rows, total_rows))

    # N1: tangential stress balance at r = 1 (Marangoni forcing goes in b)
    for i in range(N):
        n_val = i + 1
        N_one_coeff = 2.0 * (2.0 * n_val + 1.0) * (1.0 / (n_val * (n_val + 1.0)))
        A[i, 5 * i] = -N_one_coeff * n_val * (n_val + 2.0) * lambda_val
        A[i, 5 * i + 1] = -N_one_coeff * (n_val * n_val - 1.0) * lambda_val
        A[i, 5 * i + 2] = -N_one_coeff * (n_val * n_val - 1.0) * lambda_val
        A[i, 5 * i + 3] = -N_one_coeff * n_val * (n_val + 2.0) * lambda_val
        if i != 0:
            A[i, 5 * i + 4] = N_one_coeff * ((n_val * n_val - 1.0) - n_val * (n_val + 2.0))
    A[0, 5 * N] = 9.0

    # N2: impermeability of the interface, u_r^i(1) = 0
    for i in range(N):
        r_row = N + i
        A[r_row, 5 * i] = 1.0
        A[r_row, 5 * i + 1] = 1.0
        A[r_row, 5 * i + 2] = 1.0
        A[r_row, 5 * i + 3] = 1.0

    # N3: continuity of u_theta across r = 1
    r_row = 2 * N
    A[r_row, 0] = -4.0
    A[r_row, 1] = -2.0
    A[r_row, 2] = -1.0
    A[r_row, 3] = 1.0
    A[r_row, 4] = 1.0
    A[r_row, 5 * N] = -1.0
    A[r_row, 5 * N + 1] = -2.0 / 3.0

    for i in range(1, N):
        n_val = i + 1
        A[r_row + i, 5 * i] = -(n_val + 3.0)
        A[r_row + i, 5 * i + 1] = -(n_val + 1.0)
        A[r_row + i, 5 * i + 2] = -(2.0 - n_val)
        A[r_row + i, 5 * i + 3] = n_val
        A[r_row + i, 5 * i + 4] = 2.0

    # N4: u_theta at the core surface r = k
    r_row = 3 * N
    A[r_row, 0] = 6.0 * (k ** 3)
    A[r_row, 1] = 3.0 * k
    A[r_row, 2] = 1.5
    A[r_row, 3] = -1.5 * (k ** -2)
    A[r_row, 5 * N + 2] = -1.0 * k
    for i in range(1, N):
        n_val = i + 1
        A[r_row + i, 5 * i] = (n_val + 3.0) * (k ** (2.0 * n_val + 1.0))
        A[r_row + i, 5 * i + 1] = (n_val + 1.0) * (k ** (2.0 * n_val - 1.0))
        A[r_row + i, 5 * i + 2] = 2.0 - n_val
        A[r_row + i, 5 * i + 3] = -n_val * (k ** -2)

    # N5: u_r at the core surface r = k
    r_row = 4 * N
    A[r_row, 0] = 3.0 * (k ** 3)
    A[r_row, 1] = 3.0 * k
    A[r_row, 2] = 3.0
    A[r_row, 3] = 3.0 * (k ** -2)
    A[r_row, 5 * N + 2] = -1.0 * k
    for i in range(1, N):
        n_val = i + 1
        A[r_row + i, 5 * i] = k ** (2.0 * n_val + 1.0)
        A[r_row + i, 5 * i + 1] = k ** (2.0 * n_val - 1.0)
        A[r_row + i, 5 * i + 2] = 1.0
        A[r_row + i, 5 * i + 3] = k ** -2

    # u_r^o(1) = 0 for n = 1, and the two force conditions on G^o_1 and G^i_1
    A[5 * N, 4] = 1.0
    A[5 * N, 5 * N + 1] = -1.0 / 3.0
    A[5 * N, 5 * N] = 1.0

    A[5 * N + 1, 4] = 1.0
    A[5 * N + 2, 2] = 1.0

    return A

def assemble_b(N, Ma, c_n=None, mode='particle', lambda_val=2.0, k=0.4, A_n=None, B_n=None):
    """
    Right-hand side b for mode='particle' (settling under a unit force) or
    mode='squirmer' (force-free, surface modes A_n, B_n). c_n are the Legendre
    coefficients of the surfactant concentration.
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

        # core rows are scaled by k^n, as in assemble_A
        b[3 * N] = -B_n[0] * k
        b[4 * N] = A_n[0] * k

        for i in range(1, N):
            n_val = i + 1
            b[3 * N + i] = -2.0 * B_n[i] * (k ** n_val) / (2.0 * n_val + 1.0)
            b[4 * N + i] = A_n[i] * (k ** n_val) / (2.0 * n_val + 1.0)

    return b

def evaluate_fields(x, mew_vals, N, k, lambda_val, Ma=1.0, r_inf=10.0, c_n=None, mode='particle', A_n=None, B_n=None):
    """Velocities and stresses on each boundary, for checking the solution."""
    P, dP = get_legendre_and_derivs(mew_vals, N)
    n = np.arange(1, N + 1)
    sin_theta = np.sqrt(1.0 - mew_vals**2)
    P_associated = sin_theta[:, np.newaxis] * dP
    factor_theta = -(2 * n + 1) / (n * (n + 1))

    (E_i, F_i, G_i, H_i), (E_o, F_o, G_o, H_o), U_D, delta_U = unpack_coefficients(x, N)

    # Interface r = 1
    r = 1.0
    bracket_ur_i = E_i * (r ** (n + 1)) + F_i * (r ** (n - 1)) + G_i * (r ** (-n)) + H_i * (r ** (-n - 2))
    ur_i_1 = np.sum((2 * n + 1) * bracket_ur_i * P, axis=1)

    bracket_ur_o = E_o * (r ** (n + 1)) + F_o * (r ** (n - 1)) + G_o * (r ** (-n)) + H_o * (r ** (-n - 2))
    ur_o_1 = np.sum((2 * n + 1) * bracket_ur_o * P, axis=1)

    bracket_ut_i = (n + 3) * E_i * (r ** (n + 1)) + (n + 1) * F_i * (r ** (n - 1)) + (2 - n) * G_i * (r ** (-n)) - n * H_i * (r ** (-n - 2))
    ut_i_1 = np.sum(factor_theta * bracket_ut_i * P_associated, axis=1)

    bracket_ut_o = (n + 3) * E_o * (r ** (n + 1)) + (n + 1) * F_o * (r ** (n - 1)) + (2 - n) * G_o * (r ** (-n)) - n * H_o * (r ** (-n - 2))
    ut_o_1 = np.sum(factor_theta * bracket_ut_o * P_associated, axis=1)

    # Core surface r = k
    r = k
    bracket_ur_core = E_i * (r ** (n + 1)) + F_i * (r ** (n - 1)) + G_i * (r ** (-n)) + H_i * (r ** (-n - 2))
    ur_i_k = np.sum((2 * n + 1) * bracket_ur_core * P, axis=1)

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

    # Far field: uniform stream -U_D e_z plus the Stokeslet
    bracket_ur_inf = E_o * (r_inf ** (n + 1)) + F_o * (r_inf ** (n - 1)) + G_o * (r_inf ** (-n)) + H_o * (r_inf ** (-n - 2))
    ur_inf = np.sum((2 * n + 1) * bracket_ur_inf * P, axis=1)
    ur_inf_analytical = -U_D * mew_vals + 3.0 * G_o[0] / r_inf * mew_vals

    bracket_ut_inf = (n + 3) * E_o * (r_inf ** (n + 1)) + (n + 1) * F_o * (r_inf ** (n - 1)) + (2 - n) * G_o * (r_inf ** (-n)) - n * H_o * (r_inf ** (-n - 2))
    ut_inf = np.sum(factor_theta * bracket_ut_inf * P_associated, axis=1)
    ut_inf_analytical = U_D * sin_theta - 1.5 * G_o[0] / r_inf * sin_theta

    # Stress jump at r = 1 must equal the Marangoni stress
    sig_i = compute_shear_stress(1.0, mew_vals, E_i, F_i, G_i, H_i, N, lambda_val)
    sig_o = compute_shear_stress(1.0, mew_vals, E_o, F_o, G_o, H_o, N, 1.0)
    sig_diff = sig_i - sig_o

    if c_n is None:
        c_n = 0.1 ** n
    rhs_stress = Ma * np.sum(c_n * P_associated, axis=1)

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

def run_unit_tests(x, N, k, lambda_val, Ma=1.0, tolerance=1e-10, c_n=None, mode='particle', A_n=None, B_n=None, verbose=True):
    """
    Assert that the solution x satisfies every boundary condition to within `tolerance`.
    Returns the maximum error for each condition.
    """
    mew_test = np.linspace(-1, 1, 100)
    data = evaluate_fields(x, mew_test, N, k, lambda_val, Ma=Ma, r_inf=10000.0, c_n=c_n, mode=mode, A_n=A_n, B_n=B_n)

    if mode == 'particle':
        target_g_i_1 = -1.0 / (12.0 * np.pi * lambda_val)
        target_g_o_1 = -1.0 / (12.0 * np.pi)
    else:
        target_g_i_1 = 0.0
        target_g_o_1 = 0.0

    errors = {
        "far field u_r": np.max(np.abs(data["ur_inf"] - data["ur_inf_analytical"])),
        "far field u_theta": np.max(np.abs(data["ut_inf"] - data["ut_inf_analytical"])),
        "u_r = 0 at r=1 (inner)": np.max(np.abs(data["ur_i_1"])),
        "u_r = 0 at r=1 (outer)": np.max(np.abs(data["ur_o_1"])),
        "u_theta continuous at r=1": np.max(np.abs(data["ut_i_1"] - data["ut_o_1"])),
        "stress balance at r=1": np.max(np.abs(data["sig_diff"] - data["rhs_stress"])),
        "u_r at r=k": np.max(np.abs(data["ur_i_k"] - data["ur_rhs_k"])),
        "u_theta at r=k": np.max(np.abs(data["ut_i_k"] - data["ut_rhs_k"])),
        "force on core (G^i_1)": np.abs(data["g_i_1"] - target_g_i_1),
        "force on drop (G^o_1)": np.abs(data["g_o_1"] - target_g_o_1),
    }
    for name, err in errors.items():
        assert err < tolerance, f"BC check failed: {name}, error {err:.2e} > {tolerance:.0e}"

    if verbose:
        print(f"All boundary conditions satisfied (max error {max(errors.values()):.1e})")
    return errors

def plot_velocity_field(x, N, k, lambda_val, Ma=None, Pe=None, mode='particle', folder_prefix=None):
    """Quiver plot of the velocity field in the x-z plane, in the drop frame."""
    plt.close('all')

    grid_size = 35
    x_range = np.linspace(-2.0, 2.0, grid_size)
    z_range = np.linspace(-2.0, 2.0, grid_size)
    X, Z = np.meshgrid(x_range, z_range)

    R = np.sqrt(X**2 + Z**2)
    Theta = np.arctan2(X, Z)  # polar angle measured from the z axis
    mew = np.cos(Theta)

    r_flat = R.flatten()
    mew_flat = mew.flatten()
    theta_flat = Theta.flatten()

    P, dP = get_legendre_and_derivs(mew_flat, N)
    n = np.arange(1, N + 1)
    sin_theta = np.sqrt(1.0 - mew_flat**2)
    P_associated = sin_theta[:, np.newaxis] * dP
    factor_theta = -(2 * n + 1) / (n * (n + 1))

    (E_i, F_i, G_i, H_i), (E_o, F_o, G_o, H_o), _, _ = unpack_coefficients(x, N)

    ur = np.zeros_like(r_flat)
    ut = np.zeros_like(r_flat)

    mask_core = r_flat < k
    mask_inner = (r_flat >= k) & (r_flat <= 1.0)
    mask_outer = r_flat > 1.0

    # no arrows inside the core
    ur[mask_core] = np.nan
    ut[mask_core] = np.nan

    if np.any(mask_inner):
        r_sub = r_flat[mask_inner][:, np.newaxis]
        P_sub = P[mask_inner, :]
        P_assoc_sub = P_associated[mask_inner, :]

        bracket_ur = E_i * (r_sub ** (n + 1)) + F_i * (r_sub ** (n - 1)) + G_i * (r_sub ** (-n)) + H_i * (r_sub ** (-n - 2))
        ur[mask_inner] = np.sum((2 * n + 1) * bracket_ur * P_sub, axis=1)

        bracket_ut = (n + 3) * E_i * (r_sub ** (n + 1)) + (n + 1) * F_i * (r_sub ** (n - 1)) + (2 - n) * G_i * (r_sub ** (-n)) - n * H_i * (r_sub ** (-n - 2))
        ut[mask_inner] = np.sum(factor_theta * bracket_ut * P_assoc_sub, axis=1)

    if np.any(mask_outer):
        r_sub = r_flat[mask_outer][:, np.newaxis]
        P_sub = P[mask_outer, :]
        P_assoc_sub = P_associated[mask_outer, :]

        bracket_ur = E_o * (r_sub ** (n + 1)) + F_o * (r_sub ** (n - 1)) + G_o * (r_sub ** (-n)) + H_o * (r_sub ** (-n - 2))
        ur[mask_outer] = np.sum((2 * n + 1) * bracket_ur * P_sub, axis=1)

        bracket_ut = (n + 3) * E_o * (r_sub ** (n + 1)) + (n + 1) * F_o * (r_sub ** (n - 1)) + (2 - n) * G_o * (r_sub ** (-n)) - n * H_o * (r_sub ** (-n - 2))
        ut[mask_outer] = np.sum(factor_theta * bracket_ut * P_assoc_sub, axis=1)

    # spherical -> Cartesian components
    theta_phys = np.abs(theta_flat)
    U_X_flat = np.sign(X.flatten()) * (ur * np.sin(theta_phys) + ut * np.cos(theta_phys))
    U_Z_flat = ur * np.cos(theta_phys) - ut * np.sin(theta_phys)

    U_X = U_X_flat.reshape((grid_size, grid_size))
    U_Z = U_Z_flat.reshape((grid_size, grid_size))

    speed_flat = np.sqrt(U_X_flat**2 + U_Z_flat**2)
    speed = speed_flat.reshape((grid_size, grid_size))

    fig = plt.figure(figsize=(11, 10))

    q = plt.quiver(X, Z, U_X, U_Z, speed, cmap="plasma", alpha=0.9, width=0.0055, scale=0.1, scale_units='xy', clim=(0.0, 0.08))

    cbar = plt.colorbar(q, shrink=0.8)
    cbar.set_label(r"Velocity Magnitude $|u|$")

    plt.quiverkey(q, 0.76, 0.03, 0.05, r"$0.05$ (velocity units)", labelpos="E", coordinates="axes", fontproperties={"size": 10})

    theta_c = np.linspace(0, 2*np.pi, 200)
    plt.plot(np.sin(theta_c), np.cos(theta_c), color="black", linestyle="--", linewidth=1.5, alpha=0.45, label="Droplet Interface (r=1.0)")
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
    print(f"Saved {save_path}")
    plt.close(fig)

def plot_stream_function(x, N, k, lambda_val, Ma=None, Pe=None, mode='particle', folder_prefix=None):
    """Filled contour plot of the stream function psi in the x-z plane, in the drop frame."""
    plt.close('all')

    grid_size = 200
    x_range = np.linspace(-2.0, 2.0, grid_size)
    z_range = np.linspace(-2.0, 2.0, grid_size)
    X, Z = np.meshgrid(x_range, z_range)

    R = np.sqrt(X**2 + Z**2)
    R_safe = np.where(R == 0, 1e-15, R)
    mew = Z / R_safe

    r_flat = R.flatten()
    mew_flat = mew.flatten()

    P, dP = get_legendre_and_derivs(mew_flat, N)
    n = np.arange(1, N + 1)

    (E_i, F_i, G_i, H_i), (E_o, F_o, G_o, H_o), _, _ = unpack_coefficients(x, N)

    factor = (2 * n + 1) / (n * (n + 1))

    psi_inner_flat = np.zeros_like(r_flat)
    psi_outer_flat = np.zeros_like(r_flat)

    for idx, n_val in enumerate(n):
        term_i = E_i[idx] * (r_flat ** (n_val + 3)) + F_i[idx] * (r_flat ** (n_val + 1)) + G_i[idx] * (r_flat ** (2 - n_val)) + H_i[idx] * (r_flat ** (-n_val))
        psi_inner_flat += factor[idx] * term_i * (1.0 - mew_flat**2) * dP[:, idx]

        term_o = E_o[idx] * (r_flat ** (n_val + 3)) + F_o[idx] * (r_flat ** (n_val + 1)) + G_o[idx] * (r_flat ** (2 - n_val)) + H_o[idx] * (r_flat ** (-n_val))
        psi_outer_flat += factor[idx] * term_o * (1.0 - mew_flat**2) * dP[:, idx]

    psi_inner = psi_inner_flat.reshape(X.shape)
    psi_outer = psi_outer_flat.reshape(X.shape)

    psi = np.where(R < k, np.nan,
                   np.where(R <= 1.0, psi_inner, psi_outer))

    fig = plt.figure(figsize=(11, 10))

    levels = 40
    plt.contourf(X, Z, psi, levels=levels, cmap="RdBu_r", alpha=0.85)
    cbar = plt.colorbar(shrink=0.8)
    cbar.set_label(r"Stream Function $\psi$")

    max_val = np.nanmax(np.abs(psi))
    if np.isnan(max_val) or np.isinf(max_val) or max_val == 0:
        max_val = 1.0
    line_levels = np.linspace(-max_val, max_val, 21)

    cntr = plt.contour(X, Z, psi, levels=line_levels, colors="black", linewidths=0.8, alpha=0.7)
    plt.clabel(cntr, inline=True, fontsize=8, fmt='%.4f')

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
    print(f"Saved {save_path}")
    plt.close(fig)
