import numpy as np
import matplotlib.pyplot as plt
import solver_module
N = 2
x = solver_module(N)

def u_r(r, mew):
    i = np.arange(5 * N + 3)
    L_vals = np.polynomial.legendre.legvander(mew, N)[0]
    u_r_coeff = np.zeros(5 * N + 3)
    u_r_coeff[::5] = (2 * (i + 1) + 1) *  L_vals[i // 5] * r**((i + 1)+3)
    u_r_coeff[1::5] = (2 * (i + 1) + 1) *  L_vals[i // 5] * r**((i + 1)+1)
    u_r_coeff[2::5] = (2 * (i + 1) + 1) *  L_vals[i // 5] * r**(2-(i + 1))
    u_r_coeff[3::5] = (2 * (i + 1) + 1) *  L_vals[i // 5] * r**(-(i + 1))
    u_r_coeff[-3:] = 0
    return np.dot(u_r_coeff, x)
