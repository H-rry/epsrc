import numpy as np
import scipy.integrate


def generate_b_n(N):
    n = np.arange(1, N+1)
    b_n = 0.1**n
    return b_n


def get_legendre_and_derivs(mew, N):
    """
    Computes Legendre polynomials P_n(mew) and their derivatives P'_n(mew)
    for n = 1 to N.
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


def gamma_unscaled(mew, b_n, Pe=1.0):
    """
    Computes unscaled surfactant concentration.
    b_n is passed in rather than generated locally.
    """
    is_scalar = np.isscalar(mew)
    mew_arr = np.atleast_1d(mew)
    N = len(b_n)
    n = np.arange(1, N + 1)
    
    V = np.polynomial.legendre.legvander(mew_arr, N)
    P = V[:, 1 : N + 1]
    sum_term = np.sum(2.0 * b_n * P / (n * (n + 1.0)), axis=1)
    exp_arg = np.clip(-Pe * sum_term, -700.0, 700.0)
    res = np.exp(exp_arg)
    
    return res[0] if is_scalar else res


def gamma_scaled(mew, b_n, gamma_0, mode_n=0, Pe=1.0):
    """
    Computes scaled surfactant concentration Gamma(mew) * P_{mode_n}(mew).
    b_n is passed in rather than generated locally.
    """
    is_scalar = np.isscalar(mew)
    mew_arr = np.atleast_1d(mew)
    N = len(b_n)
    n = np.arange(1, N + 1)
    
    V = np.polynomial.legendre.legvander(mew_arr, N)
    P = V[:, 1 : N + 1]
    sum_term = np.sum(2.0 * b_n * P / (n * (n + 1.0)), axis=1)
    
    P_mode = 1.0 if mode_n == 0 else P[:, mode_n - 1]
    exp_arg = np.clip(-Pe * sum_term, -700.0, 700.0)
    res = gamma_0 * P_mode * np.exp(exp_arg)
    
    return res[0] if is_scalar else res


def get_c_n(n, b_n, gamma_0, Pe):
    """
    Computes Legendre expansion coefficients c_n of the surfactant concentration
    for mode n (or an array of modes) via numerical quadrature.
    """
    if isinstance(n, (np.ndarray, list)):
        return np.array([get_c_n(ni, b_n, gamma_0, Pe) for ni in n])
    val, _ = scipy.integrate.quad(gamma_scaled, -1.0, 1.0, args=(b_n, gamma_0, n, Pe))
    return ((2*n + 1)/2) * val


def solve_c_n(b_n, Pe, Ma=1.0, k=0.3, lambda_val=0.5):
    """
    Solves for surfactant concentration coefficients c_n for modes n = 1 to N
    given flow mode coefficients b_n and Péclet number Pe.
    """
    N = len(b_n)
    result, _ = scipy.integrate.quad(gamma_unscaled, -1.0, 1.0, args=(b_n, Pe))
    gamma_0 = 1.0 / result
    
    n_modes = np.arange(1, N + 1)
    c_n = get_c_n(n_modes, b_n, gamma_0, Pe)
    return c_n
