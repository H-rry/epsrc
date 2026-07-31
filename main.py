import numpy as np
import matplotlib.pyplot as plt
import solver_module

N = 2
x = solver_module.solve_system(N)

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

def u_r(r, mew):    # de-dimensionalised equations. instead of inputing r = R for example, input r' = 1
    mew = np.atleast_1d(mew)
    P, dP = get_legendre_and_derivs(mew, N)
    
    # Reconstruct coefficients from x
    # Inner fluid coefficients:
    E_i = x[0 : 5 * N : 5]
    F_i = x[1 : 5 * N : 5]
    G_i = x[2 : 5 * N : 5]
    H_i = x[3 : 5 * N : 5]
    
    # Outer fluid coefficients:
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
    
    # Inner radial velocity
    bracket_i = E_i * (r ** (n + 1)) + F_i * (r ** (n - 1)) + G_i * (r ** (-n)) + H_i * (r ** (-n - 2))
    u_r_i = np.sum((2 * n + 1) * bracket_i * P, axis=1)
    
    # Outer radial velocity
    bracket_o = E_o * (r ** (n + 1)) + F_o * (r ** (n - 1)) + G_o * (r ** (-n)) + H_o * (r ** (-n - 2))
    u_r_o = np.sum((2 * n + 1) * bracket_o * P, axis=1)
    
    return u_r_o, u_r_i


def u_theta(r, mew):
    mew = np.atleast_1d(mew)
    P, dP = get_legendre_and_derivs(mew, N)
    
    # Reconstruct coefficients from x
    # Inner fluid coefficients:
    E_i = x[0 : 5 * N : 5]
    F_i = x[1 : 5 * N : 5]
    G_i = x[2 : 5 * N : 5]
    H_i = x[3 : 5 * N : 5]
    
    # Outer fluid coefficients:
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
    
    # Inner angular velocity
    bracket_i = (n + 3) * E_i * (r ** (n + 1)) + (n + 1) * F_i * (r ** (n - 1)) + (2 - n) * G_i * (r ** (-n)) - n * H_i * (r ** (-n - 2))
    u_theta_i = np.sum(factor * bracket_i * P_associated, axis=1)
    
    # Outer angular velocity
    bracket_o = (n + 3) * E_o * (r ** (n + 1)) + (n + 1) * F_o * (r ** (n - 1)) + (2 - n) * G_o * (r ** (-n)) - n * H_o * (r ** (-n - 2))
    u_theta_o = np.sum(factor * bracket_o * P_associated, axis=1)
    
    return u_theta_o, u_theta_i


def sigma(r, mew):
    # Stress tensor (not implemented yet)
    return None, None


def boundary_condition_two_three(u, component): # component is a string containing "Radial" or "Theta"
    mew_vals = np.linspace(-1, 1, 200)
    u_o_vals, u_i_vals = u(r=1.0, mew=mew_vals)

    plt.figure(figsize=(8, 5))
    plt.plot(mew_vals, u_i_vals, label=component + r"$_{,i}(r=1)$", color="tab:blue", lw=2)
    plt.plot(mew_vals, u_o_vals, label=component + r"$_{,o}(r=1)$", color="tab:orange", linestyle="--", lw=2)

    plt.axhline(0, color="black", linestyle=":", alpha=0.6)

    plt.xlabel(r"$\mu = \cos(\theta)$")
    plt.ylabel(component + r" Velocity")
    plt.title(f"{component} Boundary Condition Verification at Interface $r = 1$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    boundary_condition_two_three(u_r, "Radial")
    boundary_condition_two_three(u_theta, "Theta")
    return 0

if __name__ == "__main__":
    main()
