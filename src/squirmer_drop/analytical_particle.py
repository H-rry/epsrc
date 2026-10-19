import numpy as np

"""
Analytical benchmark solutions for a rigid spherical particle translating
inside a clean spherical droplet (from particle_inside_drop_v2.pdf, Eqs. 16a & 16b).
"""


def analytical_U_D(k, lambd):
    """
    Computes analytical droplet velocity U_D for a rigid particle translating inside a clean droplet (Eq. 16a).
    """
    # Eq 16a from particle_inside_drop_v2.pdf (Sheet assumption)
    denom = 12.0 * np.pi * (2.0 * (k**5) * (lambd - 1.0) + 3.0 * lambd + 2.0)
    num = -4.0 * (k**5) * lambd + 4.0 * (k**5) + 5.0 * (k**2) - 6.0 * lambd - 9.0
    return num / denom


def analytical_U_P(k, lambd):
    """
    Computes analytical particle velocity U_P for a rigid particle translating inside a clean droplet (Eq. 16b).
    """
    if k == 0:
        raise ValueError("k=0 is singular: particle radius is zero")
    # Eq 16b from particle_inside_drop_v2.pdf (Sheet assumption)
    denom = 12.0 * np.pi * (2.0 * (k**5) * (lambd - 1.0) + 3.0 * lambd + 2.0)
    num = (-4.0 * (k**6) * (lambd**2) + 8.0 * (k**6) * lambd - 4.0 * (k**6)
           - 9.0 * (k**5) * lambd + 9.0 * (k**5) + 10.0 * (k**3) * lambd - 10.0 * (k**3)
           - 6.0 * k * (lambd**2) - 3.0 * k * lambd + 9.0 * k - 6.0 * lambd - 4.0)
    return num / (k * lambd * denom)
