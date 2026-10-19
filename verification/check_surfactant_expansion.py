"""
Check that the surfactant concentration reconstructed from its Legendre
coefficients c_n matches the exact exponential representation of Gamma(mu).
"""

import os
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

from squirmer_drop.stokes import get_legendre_and_derivs
from squirmer_drop.surfactant import gamma_unscaled, gamma_scaled, get_c_n, solve_c_n
from squirmer_drop.paths import VERIFICATION_FIGURES_DIR


def main():
    N = 5
    Pe = 1.0
    b_n = 0.1 ** np.arange(1, N + 1)  # an arbitrary decaying surface velocity

    c_n = solve_c_n(b_n, Pe)
    print("c_n:", c_n)

    result, _ = scipy.integrate.quad(gamma_unscaled, -1.0, 1.0, args=(b_n, Pe))
    gamma_0 = 1.0 / result
    c_0 = get_c_n(0, b_n, gamma_0, Pe)

    mew_plot = np.linspace(-1, 1, 300)

    # Construct Gamma via the Exponential Representation
    gamma_exp = gamma_scaled(mew_plot, b_n, gamma_0, mode_n=0, Pe=Pe)

    # Construct Gamma via the Summation Representation
    P_plot, _ = get_legendre_and_derivs(mew_plot, N)
    gamma_sum = c_0 + np.sum(c_n * P_plot, axis=1)

    # Graph both forms
    plt.close('all')
    fig = plt.figure(figsize=(9, 6))

    plt.plot(mew_plot, gamma_exp, label="Exponential Representation", color="tab:blue", lw=2.5)
    plt.plot(mew_plot, gamma_sum, label="Legendre Summation Representation", color="tab:orange", linestyle="--", lw=2.5)

    plt.xlabel(r"$\mu = \cos(\theta)$")
    plt.ylabel(r"Surfactant Concentration $\Gamma$")
    plt.title(f"Comparison of Exponential and Legendre Summation of $\Gamma$ (N = {N})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(VERIFICATION_FIGURES_DIR, exist_ok=True)
    plot_path = os.path.join(VERIFICATION_FIGURES_DIR, "gamma_comparison.png")
    plt.savefig(plot_path, dpi=200)
    print(f"Saved Gamma comparison plot to: {plot_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
