import tkinter as tk
from tkinter import ttk
import os

from squirmer_drop.paths import RUNS_DIR

class StokesDashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stokes Flow Interactive Desktop Explorer ($N = 5$)")
        self.geometry("1200x820")
        self.configure(bg="#0b0f19")

        # Set window icon if possible, else skip
        self.resizable(True, True)

        # Import and apply ttk styling
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("Horizontal.TScale", background="#111827", troughcolor="#1e293b", sliderthickness=15)

        # State variables
        self.state_mode = tk.StringVar(value="particle")
        self.state_view = tk.StringVar(value="stream")
        self.state_k = tk.StringVar(value="0.4")
        self.state_Ma = tk.StringVar(value="1.0")
        self.state_Pe = tk.StringVar(value="1.0")

        # Lists of available parameter values
        self.k_values = ["0.2", "0.4", "0.6", "0.8"]
        self.ma_values = ["0.1", "1.0", "10.0"]
        self.pe_values = ["0.01", "0.1", "1.0", "10.0", "100.0", "1000.0"]

        # Track previous indices to avoid redundant reloads
        self.prev_k_idx = 1
        self.prev_ma_idx = 1
        self.prev_pe_idx = 2

        # Lock to prevent infinite recursion during slider snapping
        self.lock_slide = False

        self.setup_ui()
        self.update_image()

    def setup_ui(self):
        # Left Panel (Controls)
        control_panel = tk.Frame(self, bg="#111827", bd=1, relief="flat", width=340)
        control_panel.pack(side="left", fill="y", padx=15, pady=15)
        control_panel.pack_propagate(False)

        # Styling constants
        bg_dark = "#111827"
        fg_light = "#f1f5f9"
        fg_gray = "#94a3b8"

        # Title
        title_label = tk.Label(control_panel, text="Stokes Flow Solver", font=("Inter", 16, "bold"), bg=bg_dark, fg="#60a5fa")
        title_label.pack(anchor="w", padx=20, pady=(20, 5))

        subtitle_label = tk.Label(control_panel, text="Desktop Explorer (N = 5)", font=("Inter", 10), bg=bg_dark, fg=fg_gray)
        subtitle_label.pack(anchor="w", padx=20, pady=(0, 20))

        # Separator line
        tk.Frame(control_panel, height=1, bg="#374151").pack(fill="x", padx=15, pady=10)

        # 1. Physical Configuration Mode Group (Particle vs. Squirmer)
        group_mode = tk.LabelFrame(control_panel, text=" PHYSICAL CONFIGURATION ", font=("Inter", 9, "bold"), bg=bg_dark, fg="#3b82f6", bd=1, relief="groove")
        group_mode.pack(fill="x", padx=15, pady=8)

        btn_particle = tk.Radiobutton(group_mode, text="particle", variable=self.state_mode, value="particle",
                                      bg=bg_dark, fg=fg_light, activebackground=bg_dark, activeforeground=fg_light,
                                      selectcolor="#1f2937", font=("Inter", 10), command=self.update_image)
        btn_particle.pack(anchor="w", padx=15, pady=(8, 4))

        btn_squirmer = tk.Radiobutton(group_mode, text="squirmer", variable=self.state_mode, value="squirmer",
                                      bg=bg_dark, fg=fg_light, activebackground=bg_dark, activeforeground=fg_light,
                                      selectcolor="#1f2937", font=("Inter", 10), command=self.update_image)
        btn_squirmer.pack(anchor="w", padx=15, pady=(4, 8))

        # 2. View Mode Group
        group_view = tk.LabelFrame(control_panel, text=" VISUALIZATION MODE ", font=("Inter", 9, "bold"), bg=bg_dark, fg="#3b82f6", bd=1, relief="groove")
        group_view.pack(fill="x", padx=15, pady=8)

        btn_stream = tk.Radiobutton(group_view, text="Streamlines (2D Contours)", variable=self.state_view, value="stream",
                                    bg=bg_dark, fg=fg_light, activebackground=bg_dark, activeforeground=fg_light,
                                    selectcolor="#1f2937", font=("Inter", 10), command=self.update_image)
        btn_stream.pack(anchor="w", padx=15, pady=4)

        btn_vector = tk.Radiobutton(group_view, text="Velocity Fields (Vectors)", variable=self.state_view, value="plot",
                                    bg=bg_dark, fg=fg_light, activebackground=bg_dark, activeforeground=fg_light,
                                    selectcolor="#1f2937", font=("Inter", 10), command=self.update_image)
        btn_vector.pack(anchor="w", padx=15, pady=4)

        btn_ud = tk.Radiobutton(group_view, text="U_D vs Pe Curves (all Ma)", variable=self.state_view, value="ud_vs_pe",
                                bg=bg_dark, fg=fg_light, activebackground=bg_dark, activeforeground=fg_light,
                                selectcolor="#1f2937", font=("Inter", 10), command=self.update_image)
        btn_ud.pack(anchor="w", padx=15, pady=4)

        btn_up = tk.Radiobutton(group_view, text="U_p / U_S vs Pe Curves", variable=self.state_view, value="up_vs_pe",
                                bg=bg_dark, fg=fg_light, activebackground=bg_dark, activeforeground=fg_light,
                                selectcolor="#1f2937", font=("Inter", 10), command=self.update_image)
        btn_up.pack(anchor="w", padx=15, pady=4)

        btn_joint = tk.Radiobutton(group_view, text="Joint U_D & U_p/U_S vs Pe", variable=self.state_view, value="joint_velocity",
                                   bg=bg_dark, fg=fg_light, activebackground=bg_dark, activeforeground=fg_light,
                                   selectcolor="#1f2937", font=("Inter", 10), command=self.update_image)
        btn_joint.pack(anchor="w", padx=15, pady=(4, 8))

        # 3. Core Size Group (k)
        group_k = tk.LabelFrame(control_panel, text=" CORE SIZE RATIO (k) ", font=("Inter", 9, "bold"), bg=bg_dark, fg="#3b82f6", bd=1, relief="groove")
        group_k.pack(fill="x", padx=15, pady=8)

        self.k_label = tk.Label(group_k, text="k = 0.4", font=("Inter", 11, "bold"), bg=bg_dark, fg=fg_light)
        self.k_label.pack(anchor="w", padx=15, pady=(8, 2))

        self.k_slider = ttk.Scale(group_k, from_=0, to=len(self.k_values)-1, value=1, orient="horizontal", command=self.on_k_slide)
        self.k_slider.pack(fill="x", padx=15, pady=(0, 10))

        # 4. Marangoni Group (Ma)
        self.group_ma = tk.LabelFrame(control_panel, text=" MARANGONI NUMBER (Ma) ", font=("Inter", 9, "bold"), bg=bg_dark, fg="#3b82f6", bd=1, relief="groove")
        self.group_ma.pack(fill="x", padx=15, pady=8)

        self.ma_label = tk.Label(self.group_ma, text="Ma = 1.0", font=("Inter", 11, "bold"), bg=bg_dark, fg=fg_light)
        self.ma_label.pack(anchor="w", padx=15, pady=(8, 2))

        self.ma_slider = ttk.Scale(self.group_ma, from_=0, to=len(self.ma_values)-1, value=1, orient="horizontal", command=self.on_ma_slide)
        self.ma_slider.pack(fill="x", padx=15, pady=(0, 10))

        # 5. Peclet Group (Pe)
        self.group_pe = tk.LabelFrame(control_panel, text=" PECLET NUMBER (Pe) ", font=("Inter", 9, "bold"), bg=bg_dark, fg="#3b82f6", bd=1, relief="groove")
        self.group_pe.pack(fill="x", padx=15, pady=8)

        self.pe_label = tk.Label(self.group_pe, text="Pe = 1.0", font=("Inter", 11, "bold"), bg=bg_dark, fg=fg_light)
        self.pe_label.pack(anchor="w", padx=15, pady=(8, 2))

        self.pe_slider = ttk.Scale(self.group_pe, from_=0, to=len(self.pe_values)-1, value=2, orient="horizontal", command=self.on_pe_slide)
        self.pe_slider.pack(fill="x", padx=15, pady=(0, 10))

        # 6. Solutions Info Group
        self.group_sol = tk.LabelFrame(control_panel, text=" VELOCITY SOLUTIONS ", font=("Inter", 9, "bold"), bg=bg_dark, fg="#3b82f6", bd=1, relief="groove")
        self.group_sol.pack(fill="x", padx=15, pady=8)

        self.lbl_ud = tk.Label(self.group_sol, text="U_D = N/A", font=("Consolas", 10, "bold"), bg=bg_dark, fg=fg_light)
        self.lbl_ud.pack(anchor="w", padx=15, pady=(8, 4))

        self.lbl_up = tk.Label(self.group_sol, text="U_p = N/A", font=("Consolas", 10, "bold"), bg=bg_dark, fg=fg_light)
        self.lbl_up.pack(anchor="w", padx=15, pady=(4, 8))

        # Status Label
        self.lbl_status = tk.Label(control_panel, text="System Ready", font=("Inter", 9), bg=bg_dark, fg="#10b981")
        self.lbl_status.pack(side="bottom", pady=10)

        # Right Panel (Viewport)
        self.view_panel = tk.Frame(self, bg="#020617", bd=1, relief="flat")
        self.view_panel.pack(side="right", fill="both", expand=True, padx=(0, 15), pady=15)

        self.image_label = tk.Label(self.view_panel, bg="#020617")
        self.image_label.pack(expand=True)

        self.error_label = tk.Label(self.view_panel, text="No visualization found for these parameters.\nSweep data might have skipped this run.",
                                    font=("Inter", 12, "bold"), bg="#020617", fg="#f87171")

    def update_image(self):
        mode = self.state_mode.get()
        view = self.state_view.get()
        k = self.state_k.get()
        Ma = self.state_Ma.get()
        Pe = self.state_Pe.get()

        # Update frame headers/labels based on particle vs squirmer
        p_header = " SWIMMING MODE " if mode == 'squirmer' else " CORE DRAG MODE "
        self.group_sol.configure(text=f" VELOCITY SOLUTIONS ({mode.upper()}) ")

        if view in ["ud_vs_pe", "up_vs_pe", "joint_velocity"]:
            # Disable Ma and Pe sliders
            self.ma_slider.configure(state="disabled")
            self.pe_slider.configure(state="disabled")
            self.group_ma.configure(fg="#4b5563")  # Dim border/header
            self.group_pe.configure(fg="#4b5563")
            self.ma_label.configure(fg="#4b5563")  # Dim labels
            self.pe_label.configure(fg="#4b5563")

            # Swept curves are in plots_{mode}/misc/
            filename_1 = f"{view}_N=5_lambda=2.0_k={k}.png"
            filename_2 = f"{view}_N=5_lambda=2_k={k}.png"
            base_dir = os.path.join(RUNS_DIR, f"plots_{mode}", "misc")
            path_1 = os.path.join(base_dir, filename_1)
            path_2 = os.path.join(base_dir, filename_2)

            target_path = None
            if os.path.exists(path_1):
                target_path = path_1
            elif os.path.exists(path_2):
                target_path = path_2
        else:
            # Enable Ma and Pe sliders
            self.ma_slider.configure(state="normal")
            self.pe_slider.configure(state="normal")
            self.group_ma.configure(fg="#3b82f6")  # Restore blue border/header
            self.group_pe.configure(fg="#3b82f6")
            self.ma_label.configure(fg="#f1f5f9")  # Restore labels
            self.pe_label.configure(fg="#f1f5f9")

            # 2D Field images are in plots_{mode}/Ma={Ma}/Pe={Pe}/N=5/
            filename_1 = f"{view}_lambda=2_k={k}.png"
            filename_2 = f"{view}_lambda=2.0_k={k}.png"
            base_dir = os.path.join(RUNS_DIR, f"plots_{mode}", f"Ma={Ma}", f"Pe={Pe}", "N=5")
            path_1 = os.path.join(base_dir, filename_1)
            path_2 = os.path.join(base_dir, filename_2)

            target_path = None
            if os.path.exists(path_1):
                target_path = path_1
            elif os.path.exists(path_2):
                target_path = path_2

        if target_path and os.path.isfile(target_path):
            try:
                # Load image using native PhotoImage
                photo = tk.PhotoImage(file=target_path)

                width = photo.width()
                if width > 1800:
                    scaled_photo = photo.subsample(3, 3)
                elif width > 1000:
                    scaled_photo = photo.subsample(2, 2)
                else:
                    scaled_photo = photo

                self.image_label.config(image=scaled_photo)
                self.image_label.image = scaled_photo # Keep reference

                self.error_label.pack_forget()
                self.image_label.pack(expand=True)
                self.lbl_status.config(text=f"Loaded: {os.path.basename(target_path)}", fg="#10b981")
            except Exception as e:
                self.show_error(f"Error loading image: {str(e)}")
        else:
            fallback_filename = filename_1 if 'filename_1' in locals() else f"{view}_k={k}.png"
            self.show_error(f"Missing file: {fallback_filename}")

        # Dynamic loading and displaying of solved velocities U_D and U_p / U_S
        if view not in ["ud_vs_pe", "up_vs_pe", "joint_velocity"]:
            vels = self.read_velocities(k, Ma, Pe, mode)
            if vels:
                U_D, U_p = vels
                p_label = "U_S" if mode == 'squirmer' else "U_p"
                self.lbl_ud.config(text=f"U_D = {U_D:.6e}")
                self.lbl_up.config(text=f"{p_label} = {U_p:.6e}")
            else:
                self.lbl_ud.config(text="U_D = N/A")
                self.lbl_up.config(text="U_p = N/A")
        else:
            p_label = "U_S" if mode == 'squirmer' else "U_p"
            self.lbl_ud.config(text="U_D = Curves Shown")
            self.lbl_up.config(text=f"{p_label} = Curves Shown")

    def read_velocities(self, k, Ma, Pe, mode):
        filename_1 = f"data_lambda=2_k={k}.csv"
        filename_2 = f"data_lambda=2.0_k={k}.csv"

        base_dir = os.path.join(RUNS_DIR, f"plots_{mode}", f"Ma={Ma}", f"Pe={Pe}", "N=5")
        path_1 = os.path.join(base_dir, filename_1)
        path_2 = os.path.join(base_dir, filename_2)

        target_path = None
        if os.path.exists(path_1):
            target_path = path_1
        elif os.path.exists(path_2):
            target_path = path_2

        if not target_path:
            return None

        try:
            x_fluid = []
            with open(target_path, 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    parts = line.strip().split(",")
                    if len(parts) >= 3 and parts[2]:
                        x_fluid.append(float(parts[2]))

            # For N=5: U_D is at index 5N+1 = 26, delta_U is at index 5N+2 = 27
            ud_idx = 26
            dup_idx = 27
            if len(x_fluid) > dup_idx:
                U_D = x_fluid[ud_idx]
                delta_U = x_fluid[dup_idx]
                U_p = delta_U + U_D
                return U_D, U_p
        except Exception as e:
            print(f"Error reading velocities from CSV: {e}")
        return None

    def show_error(self, message):
        self.image_label.pack_forget()
        self.error_label.config(text=f"No Visualization Available\n\n({message})")
        self.error_label.pack(expand=True)
        self.lbl_status.config(text="Parameter Set Unresolved", fg="#f87171")

    def on_k_slide(self, val):
        if self.lock_slide:
            return
        idx = int(round(float(val)))

        if self.k_slider.get() != idx:
            self.lock_slide = True
            self.k_slider.set(idx)
            self.lock_slide = False

        if idx != self.prev_k_idx:
            self.prev_k_idx = idx
            k_val = self.k_values[idx]
            self.state_k.set(k_val)
            self.k_label.config(text=f"k = {k_val}")
            self.update_image()

    def on_ma_slide(self, val):
        if self.lock_slide:
            return
        idx = int(round(float(val)))

        if self.ma_slider.get() != idx:
            self.lock_slide = True
            self.ma_slider.set(idx)
            self.lock_slide = False

        if idx != self.prev_ma_idx:
            self.prev_ma_idx = idx
            ma_val = self.ma_values[idx]
            self.state_Ma.set(ma_val)
            self.ma_label.config(text=f"Ma = {ma_val}")
            self.update_image()

    def on_pe_slide(self, val):
        if self.lock_slide:
            return
        idx = int(round(float(val)))

        if self.pe_slider.get() != idx:
            self.lock_slide = True
            self.pe_slider.set(idx)
            self.lock_slide = False

        if idx != self.prev_pe_idx:
            self.prev_pe_idx = idx
            pe_val = self.pe_values[idx]
            self.state_Pe.set(pe_val)
            self.pe_label.config(text=f"Pe = {pe_val}")
            self.update_image()

if __name__ == "__main__":
    app = StokesDashboard()
    app.mainloop()
