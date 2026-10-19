"""
Small-Pe asymptotic velocities of a squirmer at the centre of a surfactant-laden drop
(Shaik, Vasani & Ardekani, J. Fluid Mech. 851, 2018), to first order in Pe:
U = U_0 + Pe * Ma * U_1. Only the first squirming modes A_1 and B_1 enter.

These are the formulas as given. Shaik et al. scale Gamma by its equilibrium value, so
with the normalisation in surfactant.py the solver's O(Pe Ma) term is C_0 times theirs.
"""


def analytical_U_S(k, Ma, lambd, A_1, B_1, Pe=0.1):
    """Squirmer velocity U_S."""
    num_0_S = (-12.0 * (lambd - 1.0) * (A_1 + 0.5 * B_1) * (k**5) +
                10.0 * (k**3) * (A_1 + B_1) * (lambd - 1.0) -
                3.0 * (A_1 - 2.0 * B_1) * (lambd + 2.0 / 3.0))
    denom_0 = (6.0 * lambd - 6.0) * (k**5) + 9.0 * lambd + 6.0
    U0_S = num_0_S / denom_0

    denom_1 = ((lambd - 1.0) * (k**5) + 1.5 * lambd + 1.0) ** 2
    U1_S_coeff = -25.0 * lambd * (1.0 - k) * (k + 1.0) * (A_1 + B_1) * (k**3) / (12.0 * denom_1)

    return U0_S + Pe * Ma * U1_S_coeff


def analytical_U_D(k, Ma, lambd, A_1, B_1, Pe=0.1):
    """Drop velocity U_D."""
    denom_0 = (6.0 * lambd - 6.0) * (k**5) + 9.0 * lambd + 6.0
    U0_D = 10.0 * (k**3) * lambd * (A_1 + B_1) / denom_0

    denom_1 = ((lambd - 1.0) * (k**5) + 1.5 * lambd + 1.0) ** 2
    U1_D_coeff = -5.0 * lambd * (A_1 + B_1) * (k**3) * (1.0 - k**5) / (6.0 * denom_1)

    return U0_D + Pe * Ma * U1_D_coeff
