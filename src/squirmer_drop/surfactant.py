"""
Steady insoluble surfactant on the drop surface.

Integrating the surface advection-diffusion equation once gives

    Gamma(mu) = Gamma_0 * exp(-Pe * sum_n 2 b_n P_n(mu) / (n (n + 1)))

where b_n are the Legendre coefficients of the surface velocity u_theta(1, mu) and
Gamma_0 fixes the total amount of surfactant. The flow solver needs the Legendre
coefficients c_n of Gamma, which are found here by quadrature.

Gamma is scaled by (total amount of surfactant) / R^2, so the total amount on the drop
is 1: 2 pi int_{-1}^{1} Gamma dmu = 1. The mean concentration, which is also the n = 0
Legendre coefficient, is then c_0 = 1 / (4 pi) whatever the flow.
"""

import numpy as np
import scipy.integrate

C_0 = 1.0 / (4.0 * np.pi)


def gamma_scaled(mew, b_n, gamma_0, mode_n=0, Pe=1.0):
    """Gamma(mew) * P_{mode_n}(mew); mode_n = 0 gives Gamma itself."""
    is_scalar = np.isscalar(mew)
    mew_arr = np.atleast_1d(mew)
    N = len(b_n)
    n = np.arange(1, N + 1)

    V = np.polynomial.legendre.legvander(mew_arr, N)
    P = V[:, 1 : N + 1]
    sum_term = np.sum(2.0 * b_n * P / (n * (n + 1.0)), axis=1)

    P_mode = 1.0 if mode_n == 0 else P[:, mode_n - 1]
    exp_arg = np.clip(-Pe * sum_term, -700.0, 700.0)  # keep exp() finite
    res = gamma_0 * P_mode * np.exp(exp_arg)

    return res[0] if is_scalar else res


def gamma_unscaled(mew, b_n, Pe=1.0):
    """Gamma(mew) / Gamma_0."""
    return gamma_scaled(mew, b_n, 1.0, mode_n=0, Pe=Pe)


def get_c_n(n, b_n, gamma_0, Pe):
    """Legendre coefficient(s) c_n of Gamma for mode n (int or array of ints)."""
    if isinstance(n, (np.ndarray, list)):
        return np.array([get_c_n(ni, b_n, gamma_0, Pe) for ni in n])
    val, _ = scipy.integrate.quad(gamma_scaled, -1.0, 1.0, args=(b_n, gamma_0, n, Pe))
    return ((2*n + 1)/2) * val


def get_gamma_0(b_n, Pe):
    """Gamma_0 such that int Gamma dmu = 2 C_0, i.e. the mean concentration is C_0."""
    result, _ = scipy.integrate.quad(gamma_unscaled, -1.0, 1.0, args=(b_n, Pe))
    return 2.0 * C_0 / result


def solve_c_n(b_n, Pe):
    """Surfactant coefficients c_1..c_N for surface-velocity coefficients b_n."""
    N = len(b_n)
    gamma_0 = get_gamma_0(b_n, Pe)

    n_modes = np.arange(1, N + 1)
    return get_c_n(n_modes, b_n, gamma_0, Pe)
