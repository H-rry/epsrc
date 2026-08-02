import numpy as np
import matplotlib.pyplot as plt

def get_legendre_and_derivs(mew, N):
    """
    Computes Legendre polynomials P_n(mew) and their derivatives P'_n(mew)
    for n = 1 to N.
    Returns:
      P: array of shape (len(mew), N)
      dP: array of shape (len(mew), N)
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
            # Enforce H^o_n coupling term
            A[i, 5 * i + 4] = N_one_coeff * ((n_val * n_val - 1.0) - n_val * (n_val + 2.0)) # sign error I think, should be pos
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
    A[r_row, 5 * N + 1] = -2.0 / 3.0
    # n >= 2 modes: coupled to H^o_n (5i+4)
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

    # Block Row 1: Droplet boundary condition relation for G^o_1 + H^o_1 = U_D/3
    A[5 * N, 4] = 1.0
    A[5 * N, 5 * N + 1] = -1.0 / 3.0
    A[5 * N, 5 * N] = 1.0

    # Block Row 2: Force-free droplet condition G^o_1 = 0
    A[5 * N + 1, 4] = 1.0

    # Block Row 3: Drag coupling condition for core G^i_1 = kW
    A[5 * N + 2, 2] = 1.0

    return A

def assemble_b(N, Ma):
    """
    Assembles the right-hand-side vector b of size 5N + 3.
    """
    total_rows = 5 * N + 3
    b = np.zeros(total_rows)
    for i in range(N):
        # b[i] corresponds to the n >= 1 modes of Block N1 (c_n = 0.1^n)
        b[i] = Ma * (0.1 ** (i + 1))
    # Core drag force constraint
    b[5 * N + 2] = -(1.0 / 12.0) * (1.0 / np.pi)
    return b

def u_r(r, mew, x, N):
    """
    Reconstructs radial velocities u_r_o and u_r_i at r.
    """
    mew = np.atleast_1d(mew)
    P, dP = get_legendre_and_derivs(mew, N)
    
    # Reconstruct coefficients from solved vector x
    E_i = x[0 : 5 * N : 5]
    F_i = x[1 : 5 * N : 5]
    G_i = x[2 : 5 * N : 5]
    H_i = x[3 : 5 * N : 5]
    
    E_o = np.zeros(N)
    F_o = np.zeros(N)
    G_o = x[4 : 5 * N : 5]
    H_o = -G_o

    H_o[0] = x[5 * N]    
    U_D = x[5 * N + 1]
    F_o[0] = -U_D / 3.0
        
    n = np.arange(1, N + 1)
    
    # Inner velocity u_r_i
    bracket_i = E_i * (r ** (n + 1)) + F_i * (r ** (n - 1)) + G_i * (r ** (-n)) + H_i * (r ** (-n - 2))
    u_r_i = np.sum((2 * n + 1) * bracket_i * P, axis=1)
    
    # Outer velocity u_r_o
    bracket_o = E_o * (r ** (n + 1)) + F_o * (r ** (n - 1)) + G_o * (r ** (-n)) + H_o * (r ** (-n - 2))
    u_r_o = np.sum((2 * n + 1) * bracket_o * P, axis=1)
    
    return u_r_o, u_r_i

def u_theta(r, mew, x, N):
    """
    Reconstructs angular velocities u_theta_o and u_theta_i at r.
    """
    mew = np.atleast_1d(mew)
    P, dP = get_legendre_and_derivs(mew, N)
    
    # Reconstruct coefficients from solved vector x
    E_i = x[0 : 5 * N : 5]
    F_i = x[1 : 5 * N : 5]
    G_i = x[2 : 5 * N : 5]
    H_i = x[3 : 5 * N : 5]
    
    E_o = np.zeros(N)
    F_o = np.zeros(N)
    G_o = np.zeros(N)
    H_o = np.zeros(N)
    
    U_D = x[5 * N + 1]
    F_o[0] = -U_D / 3.0
    G_o[0] = x[4]
    H_o[0] = x[5 * N]
    for i in range(1, N):
        H_o[i] = x[5 * i + 4]
        G_o[i] = -H_o[i]
        
    n = np.arange(1, N + 1)
    sin_theta = np.sqrt(1.0 - mew**2)
    P_associated = sin_theta[:, np.newaxis] * dP
    factor = -(2 * n + 1) / (n * (n + 1))
    
    # Inner velocity u_theta_i
    bracket_i = (n + 3) * E_i * (r ** (n + 1)) + (n + 1) * F_i * (r ** (n - 1)) + (2 - n) * G_i * (r ** (-n)) - n * H_i * (r ** (-n - 2))
    u_theta_i = np.sum(factor * bracket_i * P_associated, axis=1)
    
    # Outer velocity u_theta_o
    bracket_o = (n + 3) * E_o * (r ** (n + 1)) + (n + 1) * F_o * (r ** (n - 1)) + (2 - n) * G_o * (r ** (-n)) - n * H_o * (r ** (-n - 2))
    u_theta_o = np.sum(factor * bracket_o * P_associated, axis=1)
    
    return u_theta_o, u_theta_i

def main():
    # Parameters configuration
    N = 10
    k = 0.3            # Core radius ratio (Must be < 1.0 to prevent singularity)
    lambda_val = 0.5   # Viscosity ratio of fluids
    Ma = 1.0           # Physical activity parameter

    print(f"Running pure Python solver for N = {N}, k = {k}, lambda = {lambda_val}, Ma = {Ma}")

    # Solve system
    A = assemble_A(N, k, lambda_val)
    b = assemble_b(N, Ma)
    x = np.linalg.solve(A, b)

    print("\nSolved Coefficients Vector x:")
    for idx, val in enumerate(x):
        print(f"  x[{idx:2d}] = {val: .8e}")

    # Evaluate boundary condition velocities at r = 1.0
    mew_vals = np.linspace(-1, 1, 200)
    u_r_o_vals, u_r_i_vals = u_r(r=1.0, mew=mew_vals, x=x, N=N)
    u_theta_o_vals, u_theta_i_vals = u_theta(r=1.0, mew=mew_vals, x=x, N=N)

    # Plot results
    plt.figure(figsize=(10, 8))

    # Radial panel
    plt.subplot(2, 1, 1)
    plt.plot(mew_vals, u_r_i_vals, label=r"$u_{r,i}(r=1)$", color="tab:blue", lw=2)
    plt.plot(mew_vals, u_r_o_vals, label=r"$u_{r,o}(r=1)$", color="tab:orange", linestyle="--", lw=2)
    plt.axhline(0, color="black", linestyle=":", alpha=0.6)
    plt.ylabel(r"Radial Velocity $u_r$")
    plt.title(f"Boundary Condition Verification at Droplet Interface $r = 1$ (N = {N}, k = {k})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Tangential panel
    plt.subplot(2, 1, 2)
    plt.plot(mew_vals, u_theta_i_vals, label=r"$u_{\theta,i}(r=1)$", color="tab:green", lw=2)
    plt.plot(mew_vals, u_theta_o_vals, label=r"$u_{\theta,o}(r=1)$", color="tab:red", linestyle="--", lw=2)
    plt.axhline(0, color="black", linestyle=":", alpha=0.6)
    plt.xlabel(r"$\mu = \cos(\theta)$")
    plt.ylabel(r"Tangential Velocity $u_\theta$")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = "bc_plot.png"
    plt.savefig(plot_path, dpi=200)
    print(f"\nSaved boundary condition verification plot to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    main()
