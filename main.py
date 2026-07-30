import numpy as np
import matplotlib.pyplot as plt
import solver_module
N = 2
x = solver_module(N)

def u_r(r, mew):    # de-dimensionalised equations. instead of inputing r = R for example, input r' = 1

    n = np.arange(1, N + 1)
    L_vals = np.polynomial.legendre.legvander(mew, N)[0][1, N + 1]
    factor = (2 * n + 1) * L_vals

    u_r_coeff = np.zeros(5 * N + 3)

    u_r_coeff[0 : 5 * N : 5] = factor * r ** (n + 4)  
    u_r_coeff[1 : 5 * N : 5] = factor * r ** (n + 2)
    u_r_coeff[2 : 5 * N : 5] = factor * r ** (1 - n)  
    u_r_coeff[3 : 5 * N : 5] = factor * r ** (-(n + 1))
    u_r_coeff[-3:] = 0

    u_r_i = np.dot(u_r_coeff, x)

    u_r_coeff = np.zeros(5 * N + 3)
    u_r_coeff[-2] = -1 * L_vals[1]
    u_r_coeff[4 : 5 * N : 5] = factor *(r**(-n) - r**((-n -2))) 

    u_r_o = np.dot(u_r_coeff, x) 

    return u_r_o, u_r_i


def u_theta(r,mew):

    return u_theta_o, u_theta_i


def main():
        
    return 0
