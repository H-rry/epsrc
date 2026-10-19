"""
Clean-drop (Ma = 0) velocities of a rigid particle settling at the centre of a drop
(eqs. 16a, 16b of the particle derivation notes), with the same force normalisation
as stokes.assemble_b: G^o_1 = -1/(12 pi) and G^i_1 = -1/(12 pi lambda).
"""

import numpy as np


def analytical_U_D(k, lambd):
    """Drop velocity U_D (eq. 16a)."""
    denom = 12.0 * np.pi * (2.0 * (k**5) * (lambd - 1.0) + 3.0 * lambd + 2.0)
    num = -4.0 * (k**5) * lambd + 4.0 * (k**5) + 5.0 * (k**2) - 6.0 * lambd - 9.0
    return num / denom


def analytical_U_P(k, lambd):
    """Particle velocity U_P (eq. 16b)."""
    if k == 0:
        raise ValueError("k = 0 is singular")
    denom = 12.0 * np.pi * (2.0 * (k**5) * (lambd - 1.0) + 3.0 * lambd + 2.0)
    num = (-4.0 * (k**6) * (lambd**2) + 8.0 * (k**6) * lambd - 4.0 * (k**6)
           - 9.0 * (k**5) * lambd + 9.0 * (k**5) + 10.0 * (k**3) * lambd - 10.0 * (k**3)
           - 6.0 * k * (lambd**2) - 3.0 * k * lambd + 9.0 * k - 6.0 * lambd - 4.0)
    return num / (k * lambd * denom)
