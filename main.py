import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

length = 1
time = 2

dx = 0.01
dt = 0.05

N = int(length // dx)

A = np.empty((N,N))
c_n = np.zeros(N)
c_m = np.zeros(N)

c_n[N - 1] = 1


# Accumulators for simulation results
sim_data = {
    "implicit": {},
    "explicit": {}
}

def manage_plot(mode, x=None, history=None, t_history=None, Pe=None, type="implicit"):
    """Handles storing simulation runs and generating animations containing all Pe lines."""
    if mode == "store":
        if x is None or history is None or t_history is None or Pe is None:
            return
        sim_data[type][Pe] = {
            "x": x.copy(),
            "history": [h.copy() for h in history],
            "t_history": list(t_history)
        }
    elif mode == "animate_all":
        # Generate combined animations for both schemes
        for scheme in ["implicit", "explicit"]:
            runs = sim_data[scheme]
            if not runs:
                continue
            
            # Get the grid and time info from the first run
            first_pe = list(runs.keys())[0]
            x_grid = runs[first_pe]["x"]
            t_history = runs[first_pe]["t_history"]
            num_frames = len(t_history)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_xlim(0, length)
            ax.set_ylim(-0.1, 1.1)
            ax.set_xlabel("Spatial Position (x)")
            ax.set_ylabel("Concentration (c)")
            ax.grid(True)
            
            # Setup lines for each Pe
            lines = {}
            # Generate distinct colors for each Peclet number
            color_map = plt.colormaps.get_cmap("viridis")
            pe_list = sorted(list(runs.keys()))
            colors = {pe: color_map(i / (len(pe_list) - 1)) for i, pe in enumerate(pe_list)} if len(pe_list) > 1 else {pe_list[0]: color_map(0.0)}
            
            for pe_val in pe_list:
                color = colors[pe_val]
                if abs(pe_val) < 1e-9:
                    c_analytical = x_grid / length
                else:
                    c_analytical = (np.exp(pe_val * x_grid / length) - 1.0) / (np.exp(pe_val) - 1.0)
                
                # Plot Analytical as a dashed line
                ax.plot(x_grid, c_analytical, '--', color=color, lw=1.2, alpha=0.6,
                        label=f"Analytical Pe={pe_val}")
                
                # Setup Numerical line (empty at start)
                line, = ax.plot([], [], lw=2, color=color, label=f"Numerical Pe={pe_val}")
                lines[pe_val] = line
            
            title = ax.set_title("")
            # Place legend nicely outside the main plot area to prevent overlap
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
            fig.tight_layout()
            
            def init():
                for line in lines.values():
                    line.set_data([], [])
                title.set_text("")
                return list(lines.values()) + [title]
            
            def update(frame):
                for pe_val in pe_list:
                    lines[pe_val].set_data(x_grid, runs[pe_val]["history"][frame])
                title.set_text(f"Concentration Profile ({scheme.capitalize()}) | t = {t_history[frame]:.2f}")
                return list(lines.values()) + [title]
            
            anim = FuncAnimation(
                fig, update, frames=num_frames, init_func=init, blit=True, interval=100
            )
            
            gif_filename = f"diffusion-all-Pe-dx-{dx}-dt-{dt}-{scheme}.gif"
            print(f"Saving animation to {gif_filename}...")
            anim.save(gif_filename, writer="pillow")
            print("Save complete!")
            plt.close(fig)

def fill_implicit_matrix(A, Pe):

    a = dt/(2*dx)
    b = dt/((dx**2)*Pe)

    beta = 1 + 2*b
    alpha = a + b
    gamma = b - a


    A.fill(0.0)
    
    for i in range(1, N - 1):
        A[i, i] = beta          
        A[i, i - 1] = -alpha
        A[i, i + 1] = -gamma
    
    A[0, 0] = 1.0
    A[N-1, N-1] = 1.0
    
    return A

def implicit(A, c_n, c_m, Pe):
    A = fill_implicit_matrix(A, Pe)
    t = 0.0

    step = 0
    plot_interval = 5


    history = []
    t_history = []


    x = np.linspace(0, length, N)
    pass


    while t <= time:

        history.append(c_n.copy())
        t_history.append(t)
    
        c_m = np.linalg.solve(A,c_n)
        
        c_n = c_m.copy()
        t += dt
        step += 1

    manage_plot("store", x=x, history=history, t_history=t_history, Pe=Pe, type="implicit")


def fill_explicit_matrix(A, Pe):

    a = dt/(2*dx)
    b = dt/((dx**2)*Pe)

    beta = 1 - 2*b
    alpha = a + b
    gamma = b - a


    A.fill(0.0)
    
    for i in range(1, N - 1):
        A[i, i] = beta          
        A[i, i - 1] = alpha
        A[i, i + 1] = gamma
    
    A[0, 0] = 1.0
    A[N-1, N-1] = 1.0
    
    return A

def explicit(A, c_n, c_m, Pe):
    A = fill_explicit_matrix(A, Pe)
    t = 0.0

    step = 0
    plot_interval = 5


    history = []
    t_history = []


    x = np.linspace(0, length, N)
    pass


    while t <= time:

        history.append(c_n.copy())
        t_history.append(t)
    
        c_m = np.dot(A,c_n)
        
        c_n = c_m.copy()
        t += dt
        step += 1

    manage_plot("store", x=x, history=history, t_history=t_history, Pe=Pe, type="explicit")



def main():
    for Pe in [0.1, 0.2, 0.5, 1, 2, 5, 10]:
        implicit(A, c_n, c_m, Pe)
        explicit(A, c_n, c_m, Pe)
    manage_plot("animate_all")



if __name__ == "__main__":
    main()