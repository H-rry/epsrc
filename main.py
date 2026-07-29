import numpy as np
import matplotlib.pyplot as plt
import solver_module
N = 2
x = solver_module(N)

def u_r(r, mew):
    n = np.arange(1, N + 1)

    L_vals = np.polynomial.legendre.legvander(mew, N)[0][1, N + 1]

    factor = (2 * n + 1) * L_vals

    u_r_coeff = np.zeros(5 * N + 3)

    u_r_coeff[0 : 5 * N : 5] = factor * r ** (n + 4)  
    u_r_coeff[1 : 5 * N : 5] = factor * r ** (n + 2)
    u_r_coeff[2 : 5 * N : 5] = factor * r ** (1 - n)  
    u_r_coeff[3 : 5 * N : 5] = factor * r ** (-(n + 1))
    u_r_coeff[-3:] = 0

    return np.dot(u_r_coeff, x)
