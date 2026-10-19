"""
Stokes-flow solver for a squirmer or settling particle inside a surfactant-laden drop.

Modules:
    stokes               Lamb-solution linear system for the flow (assemble_A, assemble_b, field evaluation)
    surfactant           Steady surfactant distribution and its Legendre coefficients c_n
    coupling             Picard iteration coupling flow and surfactant; critical-Ma search
    analytical_squirmer  Small-Pe asymptotic squirmer and drop velocities
    analytical_particle  Clean-drop settling particle velocities
    paths                Repository data / figure / output locations
"""
