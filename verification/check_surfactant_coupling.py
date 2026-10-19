"""
Checks of the coupled flow-surfactant problem:

1. the mean surfactant concentration stays at c_0 = C_0 in converged coupled states,
   including Pe = 10 where Gamma is far from uniform;
2. the O(Pe Ma) part of the coupled solution is C_0 times the small-Pe asymptotics of
   Shaik et al. (2018), who scale Gamma by its equilibrium value (c_0 = 1). A change in
   the normalisation of Gamma shows up as a different constant in this ratio.
"""

import numpy as np
import scipy.integrate

from squirmer_drop import coupling, stokes, surfactant
from squirmer_drop import analytical_squirmer as an

N = 5


def modes(A_1, B_1=1.0):
    A_n = np.zeros(N)
    B_n = np.zeros(N)
    A_n[0], B_n[0] = A_1, B_1
    return A_n, B_n


def check_conservation():
    for (k, lam, Ma, Pe, A_1) in [(0.3, 1.0, 50.0, 0.1, 0.0), (0.6, 0.1, 50.0, 1.0, 1.0), (0.5, 10.0, 20.0, 10.0, 0.0)]:
        A_n, B_n = modes(A_1)
        U_s, U_d, c_n, iters, res, conv = coupling.run_single_coupling(N, k, lam, Ma, Pe, A_n, B_n)
        assert conv, f"no convergence at k={k}, lambda={lam}, Ma={Ma}, Pe={Pe}"
        A = stokes.assemble_A(N, k, lam)
        x = np.linalg.solve(A, stokes.assemble_b(N, Ma, c_n=c_n, mode='squirmer', lambda_val=lam, k=k, A_n=A_n, B_n=B_n))
        b_n = coupling.extract_b_n(x, N)
        gamma_0 = surfactant.get_gamma_0(b_n, Pe)
        mean = 0.5 * scipy.integrate.quad(surfactant.gamma_scaled, -1.0, 1.0, args=(b_n, gamma_0, 0, Pe))[0]
        assert abs(mean - surfactant.C_0) < 1e-10, f"mean surfactant concentration {mean} != {surfactant.C_0}"
        print(f"Surfactant conserved (k={k}, lambda={lam}, Ma={Ma}, Pe={Pe}): mean concentration = {mean:.12f}")


def check_small_pe_limit(Pe=1e-3, tolerance=1e-2):
    Ma = 1.0
    worst = 0.0
    for k in (0.2, 0.5, 0.8):
        for lam in (0.1, 1.0, 10.0):
            for A_1 in (0.0, 1.0):
                A_n, B_n = modes(A_1)
                U_s0, U_d0, _ = coupling.solve_clean_drop(N, k, lam, A_n, B_n)
                U_s, U_d, *_ = coupling.run_single_coupling(N, k, lam, Ma, Pe, A_n, B_n, tol=1e-15, omega=0.5, max_iter=20000)
                ratio_s = (U_s - U_s0) / (an.analytical_U_S(k, Ma, lam, A_1, 1.0, Pe=Pe) - U_s0)
                ratio_d = (U_d - U_d0) / (an.analytical_U_D(k, Ma, lam, A_1, 1.0, Pe=Pe) - U_d0)
                for name, r in (("U_S", ratio_s), ("U_D", ratio_d)):
                    rel = abs(r / surfactant.C_0 - 1.0)
                    assert rel < tolerance, f"O(Pe Ma) {name}: ratio to asymptotics {r:.5f}, expected C_0 = {surfactant.C_0:.5f} (k={k}, lambda={lam}, A_1={A_1})"
                    worst = max(worst, rel)
    print(f"O(Pe Ma) correction = C_0 x asymptotics at Pe = {Pe:g} (worst relative difference {worst:.1e})")


def main():
    check_conservation()
    check_small_pe_limit()


if __name__ == "__main__":
    main()
