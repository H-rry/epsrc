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


def manage_plot(mode, x=None, history=None, t_history=None, Pe=None, type=None):
    """Handles generating and saving the simulation as an animated GIF."""
    if mode == "animate":
        if x is None or history is None or t_history is None:
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.set_xlim(0, length)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel("Spatial Position (x)")
        ax.set_ylabel("Concentration (c)")
        ax.grid(True)
        
        if abs(Pe) < 1e-9:
            c_analytical = x / length
        else:
            c_analytical = (np.exp(Pe * x / length) - 1.0) / (np.exp(Pe) - 1.0)
            
        ax.plot(x, c_analytical, 'r--', lw=1.5, label="Analytical (Steady State)")
        
        line, = ax.plot([], [], lw=2, color="blue", label="Numerical (Transient)")
        title = ax.set_title("")
        ax.legend(loc="upper left")

        def init():
            line.set_data([], [])
            title.set_text("")
            return line, title

        def update(frame):
            line.set_data(x, history[frame])
            title.set_text(f"Concentration Profile (Pe = {Pe}) | t = {t_history[frame]:.2f}")
            return line, title

        anim = FuncAnimation(
            fig, update, frames=len(history), init_func=init, blit=True, interval=100
        )
        
        if type == "explicit":
            gif_filename = f"diffusion-Pe-{Pe}-dx-{dx}-dt-{dt}-explicit.gif"
        else:
            gif_filename = f"diffusion-Pe-{Pe}-dx-{dx}-dt-{dt}-implicit.gif"
            
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
    manage_plot("setup")


    while t <= time:

        history.append(c_n.copy())
        t_history.append(t)
    
        c_m = np.linalg.solve(A,c_n)
        
        c_n = c_m.copy()
        t += dt
        step += 1

    manage_plot("animate", x=x, history=history, t_history=t_history, Pe=Pe)


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
    manage_plot("setup")


    while t <= time:

        history.append(c_n.copy())
        t_history.append(t)
    
        c_m = np.dot(A,c_n)
        
        c_n = c_m.copy()
        t += dt
        step += 1

    manage_plot("animate", x=x, history=history, t_history=t_history, Pe=Pe, type='explicit')



def main():
    for Pe in [0.1, 0.2, 0.5, 1, 2, 5, 10]:
        implicit(A, c_n, c_m, Pe)
        explicit(A, c_n, c_m, Pe)



if __name__ == "__main__":
    main()