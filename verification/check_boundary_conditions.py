"""
Boundary-condition and analytical-limit check for the Stokes solver.

Solves the particle problem for several viscosity ratios, asserts every boundary
condition is satisfied to 1e-10, and plots LHS vs RHS of each condition.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from squirmer_drop.stokes import assemble_A, assemble_b, evaluate_fields, run_unit_tests, plot_velocity_field
from squirmer_drop.paths import VERIFICATION_FIGURES_DIR

def main():
    # Parameters configuration
    N = 6
    k = 0.3
    Ma = 1.0

    print(f"Solving multi-harmonic Stokes flow for N = {N}, k = {k}, Ma = {Ma}")
    
    # Evaluate boundary conditions for plotting
    mew_vals = np.linspace(-1, 1, 200)
    lambda_vals = [0.2, 0.5, 2.0, 5.0]
    colors = {0.2: "tab:blue", 0.5: "tab:green", 2.0: "tab:orange", 5.0: "tab:red"}

    # Plot boundary conditions (BC 2 to 6 in a 3x2 grid with 6th axis turned off)
    plt.close('all')
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))

    x_05 = None
    for l_val in lambda_vals:
        # Solve system
        A = assemble_A(N, k, l_val)
        b = assemble_b(N, Ma, lambda_val=l_val, k=k)
        x = np.linalg.solve(A, b)

        # Run unit testing suite (BCs 2, 3, 5, 6, 7, 8) to verify accuracy for each lambda
        run_unit_tests(x, N, k, l_val, Ma, tolerance=1e-10)

        # Save solved variables for standard lambda = 0.5 to plot at the end
        if l_val == 0.5:
            x_05 = x

        # Evaluate boundary conditions
        data = evaluate_fields(x, mew_vals, N, k, l_val, Ma=Ma)
        
        color = colors[l_val]
        lbl = f"$\\lambda={l_val}$"

        # Panel 1: BC 2 Interface Radial
        axs[0, 0].plot(mew_vals, data['ur_i_1'], label=f"{lbl} LHS", color=color, lw=1.5)
        axs[0, 0].plot(mew_vals, data['ur_o_1'], label=f"{lbl} RHS", color=color, linestyle="--", lw=1.5)

        # Panel 2: BC 3 Interface Tangential
        axs[0, 1].plot(mew_vals, data['ut_i_1'], label=f"{lbl} LHS", color=color, lw=1.5)
        axs[0, 1].plot(mew_vals, data['ut_o_1'], label=f"{lbl} RHS", color=color, linestyle="--", lw=1.5)

        # Panel 3: BC 4 Shear Stress Balance
        axs[1, 0].plot(mew_vals, data['sig_diff'], label=f"{lbl} LHS", color=color, lw=1.5)
        axs[1, 0].plot(mew_vals, data['rhs_stress'], label=f"{lbl} RHS", color=color, linestyle="--", lw=1.5)

        # Panel 4: BC 5 Core Radial
        axs[1, 1].plot(mew_vals, data['ur_i_k'], label=f"{lbl} LHS", color=color, lw=1.5)
        axs[1, 1].plot(mew_vals, data['ur_rhs_k'], label=f"{lbl} RHS", color=color, linestyle="--", lw=1.5)

        # Panel 5: BC 6 Core Tangential
        axs[2, 0].plot(mew_vals, data['ut_i_k'], label=f"{lbl} LHS", color=color, lw=1.5)
        axs[2, 0].plot(mew_vals, data['ut_rhs_k'], label=f"{lbl} RHS", color=color, linestyle="--", lw=1.5)

    # Styling and labeling after loop completes
    axs[0, 0].set_ylim(-0.1, 0.1)
    axs[0, 0].set_ylabel(r"Radial Velocity $u_r$")
    axs[0, 0].set_title("BC 2: Interface Impermeability ($r=1.0$)")
    axs[0, 0].legend(ncol=2, fontsize=8)
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].set_ylabel(r"Tangential Velocity $u_\theta$")
    axs[0, 1].set_title("BC 3: Interface No-Slip Continuity ($r=1.0$)")
    axs[0, 1].legend(ncol=2, fontsize=8)
    axs[0, 1].grid(True, alpha=0.3)

    axs[1, 0].set_ylabel(r"Shear Stress $\sigma_{r\theta}$")
    axs[1, 0].set_title("BC 4: Interface Shear Stress Balance ($r=1.0$)")
    axs[1, 0].legend(ncol=2, fontsize=8)
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].set_xlabel(r"$\mu = \cos(\theta)$")
    axs[1, 1].set_ylabel(r"Radial Velocity $u_r$")
    axs[1, 1].set_title(f"BC 5: Core Kinematic Condition ($r={k}$)")
    axs[1, 1].legend(ncol=2, fontsize=8)
    axs[1, 1].grid(True, alpha=0.3)

    axs[2, 0].set_xlabel(r"$\mu = \cos(\theta)$")
    axs[2, 0].set_ylabel(r"Tangential Velocity $u_\theta$")
    axs[2, 0].set_title(f"BC 6: Core No-Slip Condition ($r={k}$)")
    axs[2, 0].legend(ncol=2, fontsize=8)
    axs[2, 0].grid(True, alpha=0.3)

    # Panel 6: Empty/Off
    axs[2, 1].axis('off')

    plt.tight_layout()
    plot_path = os.path.join(VERIFICATION_FIGURES_DIR, "bc_plot_all.png")
    plt.savefig(plot_path, dpi=200)
    print(f"Saved boundary condition verification plot to: {plot_path}")
    plt.close(fig)

    # Now safely generate the 2D velocity vector plot for lambda = 0.5 without affecting subplots
    if x_05 is not None:
        plot_velocity_field(x_05, N, k, 0.5)

if __name__ == "__main__":
    main()
