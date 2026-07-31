import numpy as np
import matplotlib.pyplot as plt
import solver_module
N = 2
x = solver_module(N)

def u_r(r, mew):    # de-dimensionalised equations. instead of inputing r = R for example, input r' = 1
    n = np.arange(1, N + 1)
    L_vals = np.polynomial.legendre.legvander(mew, N)[0][1: N + 1]
    factor = (2 * n + 1) * L_vals

    u_r_coeff = np.zeros(5 * N + 3)

    u_r_coeff[0 : 5 * N : 5] = factor * r**(n + 1)
    u_r_coeff[1 : 5 * N : 5] = factor * r**(n - 1)
    u_r_coeff[2 : 5 * N : 5] = factor * r**(-n)
    u_r_coeff[3 : 5 * N : 5] = factor * r**(-n - 2)
    u_r_coeff[-3:] = 0 # why am I zero? I need to seriously rehash my understanding of this

    u_r_i = np.dot(u_r_coeff, x)

    u_r_coeff = np.zeros(5 * N + 3)
    u_r_coeff[-2] = -1 * L_vals[1]
    u_r_coeff[4 : 5 * N : 5] = factor *(r**(-n) - r**((-n -2))) 

    u_r_o = np.dot(u_r_coeff, x)

    return u_r_o, u_r_i


def u_theta(r,mew):
    n = np.arange(1, N + 1)
    L_vals = np.polynomial.legendre.legvander(mew, N)[0][1: N + 1] # change
    factor = (-(2 * n + 1)/n * (n + 1)) * L_vals # change to differentiation of it

    u_r_coeff = np.zeros(5 * N + 3)

    u_r_coeff[5 : 5 * N : 5] = factor * (n + 3) * r **(n + 1)
    u_r_coeff[6 : 5 * N : 5] = factor * (n + 1) * r **(n - 1)
    u_r_coeff[7 : 5 * N : 5] = factor * (2 - n) * r **(-n)
    u_r_coeff[8 : 5 * N : 5] = factor * (-n) * r **(-n - 2)
    u_r_coeff[-3:] = 0

    u_r_i = np.dot(u_r_coeff, x)

    u_r_coeff = np.zeros(5 * N + 3)
    u_r_coeff[-2] = -1 * L_vals[1]
    u_r_coeff[4 : 5 * N : 5] = factor *(r**(-n) - r**((-n -2)))

    u_r_o = np.dot(u_r_coeff, x)

    return u_theta_o, u_theta_i


def sigma(r,mew):

    return sigma_o, sigma_i


def boundary_condition_two_three(u, component): # omponent is a string containing "r" or "theta"
    mew_vals = np.linspace(-1, 1, 200)
    
    u_o_vals, u_i_vals = zip(*[u(r=1.0, mew=m) for m in mew_vals])

    plt.figure(figsize=(8, 5))
    plt.plot(mew_vals, u_i_vals, label=r"$u_{r,i}(r=1)$", color="tab:blue", lw=2)
    plt.plot(mew_vals, u_o_vals, label=r"$u_{r,o}(r=1)$", color="tab:orange", linestyle="--", lw=2)

    plt.axhline(0, color="black", linestyle=":", alpha=0.6)

    plt.xlabel(r"$\mu = \cos(\theta)$")
    plt.ylabel(component + r"Velocity $u_r$")
    plt.title(r"Boundary Condition Verification at Interface $r = 1$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    boundary_condition_two_three()

    return 0
