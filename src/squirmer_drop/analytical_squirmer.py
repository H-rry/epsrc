import numpy as np

"""
Analytical benchmark solutions for a squirmer inside a surfactant-covered droplet.
Computes swimmer velocity (U_S) and droplet drift velocity (U_D) as asymptotic
expansions in powers of Pe * Ma.
"""


def analytical_U_S(k, Ma, lambd, A_1, B_1, Pe=0.1):
    """
    Computes analytical swimmer velocity U_S for a spherical squirmer inside a droplet
    with surfactant, up to first order in Pe * Ma.
    """
    num_0_S = (-12.0 * (lambd - 1.0) * (A_1 + 0.5 * B_1) * (k**5) + 
                10.0 * (k**3) * (A_1 + B_1) * (lambd - 1.0) - 
                3.0 * (A_1 - 2.0 * B_1) * (lambd + 2.0 / 3.0))
    denom_0 = (6.0 * lambd - 6.0) * (k**5) + 9.0 * lambd + 6.0
    U0_S = num_0_S / denom_0
    
    denom_1 = ((lambd - 1.0) * (k**5) + 1.5 * lambd + 1.0) ** 2
    U1_S_coeff = -25.0 * lambd * (1.0 - k) * (k + 1.0) * (A_1 + B_1) * (k**3) / (12.0 * denom_1)
    
    return U0_S + Pe * Ma * U1_S_coeff

def analytical_U_D(k, Ma, lambd, A_1, B_1, Pe=0.1):
    """
    Computes analytical droplet velocity U_D for a spherical squirmer inside a droplet
    with surfactant, up to first order in Pe * Ma.
    """
    denom_0 = (6.0 * lambd - 6.0) * (k**5) + 9.0 * lambd + 6.0
    U0_D = 10.0 * (k**3) * lambd * (A_1 + B_1) / denom_0
    
    denom_1 = ((lambd - 1.0) * (k**5) + 1.5 * lambd + 1.0) ** 2
    U1_D_coeff = -5.0 * lambd * (A_1 + B_1) * (k**3) * (1.0 - k**5) / (6.0 * denom_1)
    
    return U0_D + Pe * Ma * U1_D_coeff


if __name__ == "__main__":
    print(f"U_S: {analytical_U_S(0.8, 0.1, 0.5, 0, 1)}")
    print(f"U_D: {analytical_U_D(0.8, 0.1, 0.5, 0, 1)}")

