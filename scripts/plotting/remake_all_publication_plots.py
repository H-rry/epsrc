#!/usr/bin/env python3
"""
Ratio and velocity figures in the style of Shaik et al. (2018), Fig. 1, from
data/ratio_sweeps/ratio_sweep_master.csv. Curves are labelled directly on the plot.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from squirmer_drop.paths import RATIO_SWEEPS_DIR, RATIO_FIGURES_DIR

data_dir = RATIO_SWEEPS_DIR
plots_dir = RATIO_FIGURES_DIR
os.makedirs(plots_dir, exist_ok=True)

# Styling configuration
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12.5,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'mathtext.fontset': 'cm',
    'lines.linewidth': 1.6,
    'lines.markersize': 5.5,
})

# High-contrast color palette matching Shaik et al.
LAMBDA_CFG = {
    10.0: {'color': '#0044cc', 'marker': 'o', 'label': r'$\lambda = 10$'},    # Blue
    1.0:  {'color': '#cc0000', 'marker': 's', 'label': r'$\lambda = 1$'},     # Red
    0.1:  {'color': '#771199', 'marker': '^', 'label': r'$\lambda = 0.1$'},   # Purple
}

MAC_ANNOTATION_TEXT = (
    r"$\mathbf{Critical\; Marangoni\; Ma_c}$" + "\n"
    r"($2\%$ dev. from perturbation theory, $k=0.4$):" + "\n"
    r"  $\lambda=0.1:\; Ma_c \approx 0.22$" + "\n"
    r"  $\lambda=1.0:\; Ma_c \approx 0.48$" + "\n"
    r"  $\lambda=10.0:\; Ma_c \approx 3.08$"
)


def load_dataset():
    csv_master = os.path.join(data_dir, "ratio_sweep_master.csv")
    if not os.path.exists(csv_master):
        raise FileNotFoundError(f"Dataset not found: {csv_master}")
    records = []
    with open(csv_master, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                records.append({
                    'mode_label': r['mode_label'],
                    'A_1': float(r['A_1']),
                    'B_1': float(r['B_1']),
                    'Pe': float(r['Pe']),
                    'lambda': float(r['lambda']),
                    'Ma': float(r['Ma']),
                    'k': float(r['k']),
                    'ratio_U_s': float(r['ratio_U_s']) if r['ratio_U_s'] not in ['', 'nan'] else np.nan,
                    'ratio_U_d': float(r['ratio_U_d']) if r['ratio_U_d'] not in ['', 'nan'] else np.nan,
                    'U_s_norm': float(r['U_s_norm']) if r['U_s_norm'] not in ['', 'nan'] else np.nan,
                    'U_d_norm': float(r['U_d_norm']) if r['U_d_norm'] not in ['', 'nan'] else np.nan,
                    'U_s_clean': float(r['U_s_clean']) if r['U_s_clean'] not in ['', 'nan'] else np.nan,
                    'U_d_clean': float(r['U_d_clean']) if r['U_d_clean'] not in ['', 'nan'] else np.nan,
                    'converged': r['converged'].lower() == 'true'
                })
            except Exception:
                continue
    return records


def filter_recs(records, mode_lbl, pe, lamb, ma):
    res = [r for r in records if r['mode_label'] == mode_lbl and np.isclose(r['Pe'], pe) and np.isclose(r['lambda'], lamb) and np.isclose(r['Ma'], ma)]
    res.sort(key=lambda x: x['k'])
    return res


def get_ma_suite(pe_val, optimized=True):
    if pe_val >= 10.0 and optimized:
        return [0.01, 0.1, 1.0]
    return [0.1, 1.0, 10.0]


def format_axes(ax, xlim=(0.0, 1.0), ylim=(0.0, 1.2)):
    ax.tick_params(direction='in', which='both', top=True, right=True, length=5, width=1.0)
    ax.tick_params(direction='in', which='minor', top=True, right=True, length=3, width=0.7)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, linestyle=':', alpha=0.35)


def annotate_curve(ax, text, xy, xytext, color, fontsize=10.5, lw=1.3):
    ax.annotate(text, xy=xy, xytext=xytext,
                arrowprops=dict(arrowstyle="->", color=color, lw=lw),
                fontsize=fontsize, fontweight='bold', color=color,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.88),
                zorder=10)


# ==============================================================================
# remake single-pe ratio plots with direct labels
# ==============================================================================
def plot_single_pe_ratios(records, mode_lbl, pe_val, optimized=True):
    fig, (ax_s, ax_d) = plt.subplots(1, 2, figsize=(13.5, 5.6), dpi=300)

    has_pole = (mode_lbl == 'A1=1_B1=1')
    format_axes(ax_s, xlim=(0.0, 1.0), ylim=(-2.5, 4.0) if has_pole else (0.0, 1.4))
    format_axes(ax_d, xlim=(0.0, 1.0), ylim=(0.0, 1.2))

    ax_s.axhline(1.0, color='black', linestyle='-', linewidth=1.0, alpha=0.5, zorder=1)
    ax_d.axhline(1.0, color='black', linestyle='-', linewidth=1.0, alpha=0.5, zorder=1)

    if has_pole:
        ax_s.axvline(0.565, color='#771199', linestyle=':', alpha=0.7, linewidth=1.3)
        ax_s.text(0.575, 3.4, r'Asymptote ($U_{S,\mathrm{clean}}=0$)', color='#771199', fontsize=8.5, style='italic')

    mas = get_ma_suite(pe_val, optimized=optimized)

    # Plot curves
    for l_val, cfg in LAMBDA_CFG.items():
        col = cfg['color']
        mkr = cfg['marker']

        for ma_idx, ma in enumerate(mas):
            recs = filter_recs(records, mode_lbl, pe_val, l_val, ma)
            if not recs: continue

            k_vals = np.array([r['k'] for r in recs])
            rs = np.array([r['ratio_U_s'] for r in recs])
            rd = np.array([r['ratio_U_d'] for r in recs])
            usc = np.array([r['U_s_clean'] for r in recs])

            if has_pole and l_val == 0.1:
                mask = np.abs(usc) < 0.02
                rs[mask] = np.nan
                cross = np.where(np.diff(np.sign(usc)) != 0)[0]
                if len(cross) > 0:
                    rs[cross[0]] = np.nan
                    if cross[0]+1 < len(rs): rs[cross[0]+1] = np.nan

            if ma_idx == 0:
                ls, mfc = '--', 'none'
            elif ma_idx == 1:
                ls, mfc = '-.', 'white'
            else:
                ls, mfc = ':', col

            ax_s.plot(k_vals, rs, linestyle=ls, color=col, marker=mkr,
                      markerfacecolor=mfc, markeredgecolor=col, markeredgewidth=1.2)
            ax_d.plot(k_vals, rd, linestyle=ls, color=col, marker=mkr,
                      markerfacecolor=mfc, markeredgecolor=col, markeredgewidth=1.2)

    # Subplot titles
    ax_s.text(0.03, 0.93, r"$(a)$", transform=ax_s.transAxes, fontsize=13, fontweight='bold')
    ax_s.set_ylabel(r"$U_S / U_{S,\mathrm{clean}}$")
    ax_s.set_xlabel(r"$\chi \equiv k = R_{\mathrm{core}} / R_{\mathrm{drop}}$")

    ax_d.text(0.03, 0.93, r"$(b)$", transform=ax_d.transAxes, fontsize=13, fontweight='bold')
    ax_d.set_ylabel(r"$U_D / U_{D,\mathrm{clean}}$")
    ax_d.set_xlabel(r"$\chi \equiv k = R_{\mathrm{core}} / R_{\mathrm{drop}}$")

    # DIRECT IN-PLOT LABELS & ARROWS (EXPLICIT COLORS & SHAPES)
    if mode_lbl == 'A1=0_B1=1':
        # Panel (a): Swimmer Velocity Ratio
        if pe_val == 0.1:
            annotate_curve(ax_s, r"$\lambda = 10$ (Blue Circle)", xy=(0.70, 0.98), xytext=(0.60, 1.18), color="#0044cc")
            annotate_curve(ax_s, r"$\lambda = 1$ (Red Square)", xy=(0.65, 0.88), xytext=(0.68, 0.70), color="#cc0000")
            annotate_curve(ax_s, r"$\lambda = 0.1$ (Purple Triangle)", xy=(0.85, 0.68), xytext=(0.72, 0.45), color="#771199")
        elif pe_val == 1.0:
            annotate_curve(ax_s, r"$\lambda = 10$ (Blue Circle)", xy=(0.70, 0.98), xytext=(0.60, 1.18), color="#0044cc")
            annotate_curve(ax_s, r"$\lambda = 1$ (Red Square)", xy=(0.65, 0.62), xytext=(0.68, 0.42), color="#cc0000")
            annotate_curve(ax_s, r"$\lambda = 0.1$ (Purple Triangle)", xy=(0.80, 0.72), xytext=(0.78, 0.92), color="#771199")
        else: # Pe = 10.0
            annotate_curve(ax_s, r"$\lambda = 10$ (Blue Circle)", xy=(0.60, 0.95), xytext=(0.55, 1.15), color="#0044cc")
            annotate_curve(ax_s, r"$\lambda = 1$ (Red Square)", xy=(0.65, 0.62), xytext=(0.68, 0.42), color="#cc0000")
            annotate_curve(ax_s, r"$\lambda = 0.1$ (Purple Triangle)", xy=(0.75, 0.88), xytext=(0.75, 1.10), color="#771199")

        # Panel (b): Droplet Velocity Ratio
        if pe_val == 0.1:
            annotate_curve(ax_d, r"$\lambda = 10$ (Blue Circle)", xy=(0.80, 0.96), xytext=(0.65, 1.08), color="#0044cc")
            annotate_curve(ax_d, r"$\lambda = 1$ (Red Square)", xy=(0.50, 0.62), xytext=(0.56, 0.75), color="#cc0000")
            annotate_curve(ax_d, r"$\lambda = 0.1$ (Purple Triangle)", xy=(0.60, 0.425), xytext=(0.42, 0.52), color="#771199")
        elif pe_val == 1.0:
            annotate_curve(ax_d, r"$\lambda = 10$ (Blue Circle)", xy=(0.65, 0.88), xytext=(0.55, 1.06), color="#0044cc")
            annotate_curve(ax_d, r"$\lambda = 1$ (Red Square)", xy=(0.50, 0.62), xytext=(0.56, 0.75), color="#cc0000")
            annotate_curve(ax_d, r"$\lambda = 0.1$ (Purple Triangle)", xy=(0.60, 0.425), xytext=(0.42, 0.52), color="#771199")
        else: # Pe = 10.0
            annotate_curve(ax_d, r"$\lambda = 10$ (Blue Circle)", xy=(0.60, 0.88), xytext=(0.52, 1.05), color="#0044cc")
            annotate_curve(ax_d, r"$\lambda = 1$ (Red Square)", xy=(0.40, 0.61), xytext=(0.46, 0.74), color="#cc0000")
            annotate_curve(ax_d, r"$\lambda = 0.1$ (Purple Triangle)", xy=(0.60, 0.425), xytext=(0.42, 0.52), color="#771199")

        # Marangoni Progression Indicator in Panel (b)
        ax_d.annotate("", xy=(0.28, 0.62), xytext=(0.28, 0.95),
                      arrowprops=dict(arrowstyle="->", color="#222222", lw=1.3))
        ax_d.text(0.29, 0.78, r"$\mathrm{increasing}\; Ma$" + "\n" + r"(open $\to$ half $\to$ filled)",
                  fontsize=8.5, color="#222222", rotation=-90, va='center')

    else: # Mode A1=1_B1=1 (Pusher/Puller)
        # Panel (a): Swimmer Velocity Ratio
        annotate_curve(ax_s, r"$\lambda = 0.1$ (Purple Triangle)", xy=(0.75, 1.8), xytext=(0.70, 2.5), color="#771199")
        annotate_curve(ax_s, r"$\lambda = 10$ (Blue Circle)", xy=(0.65, 0.45), xytext=(0.70, 0.65), color="#0044cc")
        annotate_curve(ax_s, r"$\lambda = 1$ (Red Square)", xy=(0.65, -0.4), xytext=(0.70, -0.8), color="#cc0000")

        # Panel (b): Droplet Velocity Ratio
        annotate_curve(ax_d, r"$\lambda = 10$ (Blue Circle)", xy=(0.60, 0.88), xytext=(0.52, 1.05), color="#0044cc")
        annotate_curve(ax_d, r"$\lambda = 1$ (Red Square)", xy=(0.45, 0.61), xytext=(0.52, 0.74), color="#cc0000")
        annotate_curve(ax_d, r"$\lambda = 0.1$ (Purple Triangle)", xy=(0.60, 0.425), xytext=(0.42, 0.52), color="#771199")

        ax_d.annotate("", xy=(0.28, 0.62), xytext=(0.28, 0.95),
                      arrowprops=dict(arrowstyle="->", color="#222222", lw=1.3))
        ax_d.text(0.29, 0.78, r"$\mathrm{increasing}\; Ma$" + "\n" + r"(open $\to$ half $\to$ filled)",
                  fontsize=8.5, color="#222222", rotation=-90, va='center')

    # Comprehensive Legend Boxes in Panel (a)
    h_lam = [
        mlines.Line2D([], [], color='#0044cc', marker='o', markerfacecolor='#0044cc', markeredgecolor='#0044cc',
                      linestyle='-', lw=1.6, label=r'$\lambda = 10$ (Blue Circle)'),
        mlines.Line2D([], [], color='#cc0000', marker='s', markerfacecolor='#cc0000', markeredgecolor='#cc0000',
                      linestyle='-', lw=1.6, label=r'$\lambda = 1$ (Red Square)'),
        mlines.Line2D([], [], color='#771199', marker='^', markerfacecolor='#771199', markeredgecolor='#771199',
                      linestyle='-', lw=1.6, label=r'$\lambda = 0.1$ (Purple Triangle)'),
    ]
    h_ma = [
        mlines.Line2D([], [], color='black', linestyle='-', lw=1.2, label=r'Clean ($Ma=0$, ratio = 1)'),
        mlines.Line2D([], [], color='black', linestyle='--', marker='o', markerfacecolor='none', markeredgecolor='black',
                      markeredgewidth=1.2, label=f'$Ma = {mas[0]}$ (Open, dashed)'),
        mlines.Line2D([], [], color='black', linestyle='-.', marker='o', markerfacecolor='white', markeredgecolor='black',
                      markeredgewidth=1.2, label=f'$Ma = {mas[1]}$ (Half, dash-dot)'),
        mlines.Line2D([], [], color='black', linestyle=':', marker='o', markerfacecolor='black', markeredgecolor='black',
                      markeredgewidth=1.2, label=f'$Ma = {mas[2]}$ (Filled, dotted)'),
    ]

    if has_pole:
        leg_lam = ax_s.legend(handles=h_lam, title=r"$\mathbf{Viscosity\; Ratio\; \lambda}$",
                              loc='lower left', bbox_to_anchor=(0.03, 0.65),
                              fontsize=8.0, title_fontsize=8.5,
                              framealpha=0.95, facecolor='white', edgecolor='black', borderpad=0.5)
        ax_s.add_artist(leg_lam)

        ax_s.legend(handles=h_ma, title=r"$\mathbf{Marangoni\; Number\; Ma}$",
                    loc='lower left', bbox_to_anchor=(0.03, 0.05),
                    fontsize=8.0, title_fontsize=8.5,
                    framealpha=0.95, facecolor='white', edgecolor='black', borderpad=0.5)
    else:
        leg_lam = ax_s.legend(handles=h_lam, title=r"$\mathbf{Viscosity\; Ratio\; \lambda}$",
                              loc='lower left', bbox_to_anchor=(0.03, 0.04),
                              fontsize=8.0, title_fontsize=8.5,
                              framealpha=0.95, facecolor='white', edgecolor='black', borderpad=0.5)
        ax_s.add_artist(leg_lam)

        ax_s.legend(handles=h_ma, title=r"$\mathbf{Marangoni\; Number\; Ma}$",
                    loc='lower left', bbox_to_anchor=(0.35, 0.04),
                    fontsize=8.0, title_fontsize=8.5,
                    framealpha=0.95, facecolor='white', edgecolor='black', borderpad=0.5)

    # Critical Ma Box in Panel (b) lower right
    ax_d.text(0.96, 0.05, MAC_ANNOTATION_TEXT, transform=ax_d.transAxes, fontsize=8.2,
              horizontalalignment='right', verticalalignment='bottom',
              bbox=dict(boxstyle='square,pad=0.4', facecolor='white', edgecolor='black', lw=1.0))

    mode_title = r"$A_1 = 0, B_1 = 1$ (Neutral Squirmer)" if mode_lbl == 'A1=0_B1=1' else r"$A_1 = 1, B_1 = 1$ (Pusher/Puller Squirmer)"
    opt_sub = " (Optimized Marangoni Numbers)" if (pe_val == 10.0 and optimized) else ""
    fig.suptitle(f"Locomotion Velocity Ratios inside Surfactant-Laden Droplet | $Pe = {pe_val}${opt_sub} | {mode_title}",
                 fontsize=13, fontweight='semibold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(plots_dir, f"shaik_ratio_Pe={pe_val}_{mode_lbl}.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[REMADE] Ratio figure: {save_path}")
    return save_path


# ==============================================================================
# REMAKE DIRECT VELOCITY PLOTS (1-to-1 REPLICA OF SHAIK ET AL. FIG 1)
# ==============================================================================
def plot_direct_velocities(records, mode_lbl, pe_val, optimized=True):
    fig, (ax_s, ax_d) = plt.subplots(1, 2, figsize=(13.5, 5.6), dpi=300)

    format_axes(ax_s, xlim=(0.0, 1.0), ylim=(0.2, 1.4) if mode_lbl == 'A1=0_B1=1' else (-0.4, 1.2))
    format_axes(ax_d, xlim=(0.0, 1.0), ylim=(0.0, 1.0) if mode_lbl == 'A1=0_B1=1' else (0.0, 1.0))

    mas = get_ma_suite(pe_val, optimized=optimized)

    for l_val, cfg in LAMBDA_CFG.items():
        col = cfg['color']
        mkr = cfg['marker']

        # Clean baseline curve (line)
        curve_clean = filter_recs(records, mode_lbl, pe_val, l_val, 0.0)
        if curve_clean:
            kc = np.array([r['k'] for r in curve_clean])
            usc = np.array([r['U_s_norm'] for r in curve_clean])
            udc = np.array([r['U_d_norm'] for r in curve_clean])
            ax_s.plot(kc, usc, linestyle='-', color=col, linewidth=1.5, alpha=0.9)
            ax_d.plot(kc, udc, linestyle='-', color=col, linewidth=1.5, alpha=0.9)

        # Ma curves
        for ma_idx, ma in enumerate(mas):
            recs = filter_recs(records, mode_lbl, pe_val, l_val, ma)
            if not recs: continue

            k_vals = np.array([r['k'] for r in recs])
            usn = np.array([r['U_s_norm'] for r in recs])
            udn = np.array([r['U_d_norm'] for r in recs])

            if ma_idx == 0:
                ls, mfc = '--', 'none'
            elif ma_idx == 1:
                ls, mfc = '-.', 'white'
            else:
                ls, mfc = ':', col

            ax_s.plot(k_vals, usn, linestyle=ls, color=col, marker=mkr,
                      markerfacecolor=mfc, markeredgecolor=col, markeredgewidth=1.2)
            ax_d.plot(k_vals, udn, linestyle=ls, color=col, marker=mkr,
                      markerfacecolor=mfc, markeredgecolor=col, markeredgewidth=1.2)

    ax_s.text(0.03, 0.93, r"$(a)$", transform=ax_s.transAxes, fontsize=13, fontweight='bold')
    ax_s.set_ylabel(r"$U_S / U_0$")
    ax_s.set_xlabel(r"$\chi \equiv k = R_{\mathrm{core}} / R_{\mathrm{drop}}$")

    ax_d.text(0.03, 0.93, r"$(b)$", transform=ax_d.transAxes, fontsize=13, fontweight='bold')
    ax_d.set_ylabel(r"$U_D / U_0$")
    ax_d.set_xlabel(r"$\chi \equiv k = R_{\mathrm{core}} / R_{\mathrm{drop}}$")

    # Direct labels with explicit colors and shapes
    if mode_lbl == 'A1=0_B1=1':
        # Panel (a):
        annotate_curve(ax_s, r"$\lambda = 10$ (Blue Circle)", xy=(0.68, 1.21), xytext=(0.45, 1.30), color="#0044cc")
        annotate_curve(ax_s, r"$\lambda = 1$ (Red Square)", xy=(0.85, 0.98), xytext=(0.76, 1.10), color="#cc0000")
        annotate_curve(ax_s, r"$\lambda = 0.1$ (Purple Triangle)", xy=(0.75, 0.55), xytext=(0.72, 0.70), color="#771199")

        # Panel (b):
        annotate_curve(ax_d, r"$\lambda = 10$ (Blue Circle)", xy=(0.78, 0.65), xytext=(0.65, 0.85), color="#0044cc")
        annotate_curve(ax_d, r"$\lambda = 1$ (Red Square)", xy=(0.78, 0.48), xytext=(0.80, 0.60), color="#cc0000")
        annotate_curve(ax_d, r"$\lambda = 0.1$ (Purple Triangle)", xy=(0.88, 0.35), xytext=(0.66, 0.38), color="#771199")

        # Marangoni Progression Indicator in Panel (b)
        ax_d.annotate("", xy=(0.28, 0.62), xytext=(0.28, 0.95),
                      arrowprops=dict(arrowstyle="->", color="#222222", lw=1.3))
        ax_d.text(0.29, 0.78, r"$\mathrm{increasing}\; Ma$" + "\n" + r"(open $\to$ half $\to$ filled)",
                  fontsize=8.5, color="#222222", rotation=-90, va='center')
    else:
        # Pusher/Puller labels
        annotate_curve(ax_s, r"$\lambda = 10$ (Blue Circle)", xy=(0.70, 0.95), xytext=(0.55, 1.08), color="#0044cc")
        annotate_curve(ax_s, r"$\lambda = 1$ (Red Square)", xy=(0.60, 0.50), xytext=(0.65, 0.65), color="#cc0000")
        annotate_curve(ax_s, r"$\lambda = 0.1$ (Purple Triangle)", xy=(0.60, -0.28), xytext=(0.68, -0.08), color="#771199")

        annotate_curve(ax_d, r"$\lambda = 10$ (Blue Circle)", xy=(0.70, 0.90), xytext=(0.55, 0.95), color="#0044cc")
        annotate_curve(ax_d, r"$\lambda = 1$ (Red Square)", xy=(0.70, 0.65), xytext=(0.76, 0.75), color="#cc0000")
        annotate_curve(ax_d, r"$\lambda = 0.1$ (Purple Triangle)", xy=(0.88, 0.35), xytext=(0.66, 0.38), color="#771199")

        ax_d.annotate("", xy=(0.28, 0.62), xytext=(0.28, 0.95),
                      arrowprops=dict(arrowstyle="->", color="#222222", lw=1.3))
        ax_d.text(0.29, 0.78, r"$\mathrm{increasing}\; Ma$" + "\n" + r"(open $\to$ half $\to$ filled)",
                  fontsize=8.5, color="#222222", rotation=-90, va='center')

    # Comprehensive Side-by-Side Legend Boxes in Panel (a)
    h_lam = [
        mlines.Line2D([], [], color='#0044cc', marker='o', markerfacecolor='#0044cc', markeredgecolor='#0044cc',
                      linestyle='-', lw=1.6, label=r'$\lambda = 10$ (Blue Circle)'),
        mlines.Line2D([], [], color='#cc0000', marker='s', markerfacecolor='#cc0000', markeredgecolor='#cc0000',
                      linestyle='-', lw=1.6, label=r'$\lambda = 1$ (Red Square)'),
        mlines.Line2D([], [], color='#771199', marker='^', markerfacecolor='#771199', markeredgecolor='#771199',
                      linestyle='-', lw=1.6, label=r'$\lambda = 0.1$ (Purple Triangle)'),
    ]
    h_ma = [
        mlines.Line2D([], [], color='black', linestyle='-', lw=1.2, label=r'Lines: Clean drop ($Ma=0$)'),
        mlines.Line2D([], [], color='black', linestyle='--', marker='o', markerfacecolor='none', markeredgecolor='black',
                      markeredgewidth=1.2, label=f'Open: $Ma = {mas[0]}$ (dashed)'),
        mlines.Line2D([], [], color='black', linestyle='-.', marker='o', markerfacecolor='white', markeredgecolor='black',
                      markeredgewidth=1.2, label=f'Half: $Ma = {mas[1]}$ (dash-dot)'),
        mlines.Line2D([], [], color='black', linestyle=':', marker='o', markerfacecolor='black', markeredgecolor='black',
                      markeredgewidth=1.2, label=f'Filled: $Ma = {mas[2]}$ (dotted)'),
    ]

    if mode_lbl == 'A1=1_B1=1':
        leg_lam = ax_s.legend(handles=h_lam, title=r"$\mathbf{Viscosity\; Ratio\; \lambda}$",
                              loc='lower left', bbox_to_anchor=(0.03, 0.24),
                              fontsize=7.8, title_fontsize=8.2,
                              framealpha=0.95, facecolor='white', edgecolor='black', borderpad=0.4)
        ax_s.add_artist(leg_lam)

        ax_s.legend(handles=h_ma, title=r"$\mathbf{Marangoni\; Number\; Ma}$",
                    loc='lower left', bbox_to_anchor=(0.03, 0.03),
                    fontsize=7.8, title_fontsize=8.2,
                    framealpha=0.95, facecolor='white', edgecolor='black', borderpad=0.4)
    else:
        leg_lam = ax_s.legend(handles=h_lam, title=r"$\mathbf{Viscosity\; Ratio\; \lambda}$",
                              loc='lower left', bbox_to_anchor=(0.03, 0.04),
                              fontsize=8.0, title_fontsize=8.5,
                              framealpha=0.95, facecolor='white', edgecolor='black', borderpad=0.5)
        ax_s.add_artist(leg_lam)

        ax_s.legend(handles=h_ma, title=r"$\mathbf{Marangoni\; Number\; Ma}$",
                    loc='lower left', bbox_to_anchor=(0.35, 0.04),
                    fontsize=8.0, title_fontsize=8.5,
                    framealpha=0.95, facecolor='white', edgecolor='black', borderpad=0.5)

    # Critical Ma Box in Panel (b) lower right
    ax_d.text(0.96, 0.05, MAC_ANNOTATION_TEXT, transform=ax_d.transAxes, fontsize=8.2,
              horizontalalignment='right', verticalalignment='bottom',
              bbox=dict(boxstyle='square,pad=0.4', facecolor='white', edgecolor='black', lw=1.0))

    mode_title = r"$A_1 = 0, B_1 = 1$ (Neutral Squirmer)" if mode_lbl == 'A1=0_B1=1' else r"$A_1 = 1, B_1 = 1$ (Pusher/Puller Squirmer)"
    fig.suptitle(f"Normalized Velocities $U / U_0$ Replication of Shaik et al. (2018) | $Pe = {pe_val}$ | {mode_title}",
                 fontsize=13, fontweight='semibold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(plots_dir, f"shaik_direct_vel_Pe={pe_val}_{mode_lbl}.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[REMADE] Direct velocity figure: {save_path}")
    return save_path


# ==============================================================================
# REMAKE MASTER 3x2 GRID FIGURES WITH IN-PLOT LABELS
# ==============================================================================
def plot_master_grid_ratios(records, mode_lbl):
    pes = [0.1, 1.0, 10.0]
    fig, axs = plt.subplots(3, 2, figsize=(14, 15), dpi=300)
    has_pole = (mode_lbl == "A1=1_B1=1")

    for row_idx, pe_val in enumerate(pes):
        ax_s = axs[row_idx, 0]
        ax_d = axs[row_idx, 1]

        format_axes(ax_s, xlim=(0.0, 1.0), ylim=(-2.5, 4.0) if has_pole else (0.0, 1.4))
        format_axes(ax_d, xlim=(0.0, 1.0), ylim=(0.0, 1.2))

        ax_s.axhline(1.0, color='black', linestyle='-', linewidth=1.0, alpha=0.5, zorder=1)
        ax_d.axhline(1.0, color='black', linestyle='-', linewidth=1.0, alpha=0.5, zorder=1)

        if has_pole:
            ax_s.axvline(0.565, color='#771199', linestyle=':', alpha=0.6, linewidth=1.2)

        mas = get_ma_suite(pe_val, optimized=True)

        for l_val, cfg in LAMBDA_CFG.items():
            col = cfg['color']
            mkr = cfg['marker']

            for ma_idx, ma in enumerate(mas):
                recs = filter_recs(records, mode_lbl, pe_val, l_val, ma)
                if not recs: continue

                k_vals = np.array([r['k'] for r in recs])
                rs = np.array([r['ratio_U_s'] for r in recs])
                rd = np.array([r['ratio_U_d'] for r in recs])
                usc = np.array([r['U_s_clean'] for r in recs])

                if has_pole and l_val == 0.1:
                    mask = np.abs(usc) < 0.02
                    rs[mask] = np.nan
                    cross = np.where(np.diff(np.sign(usc)) != 0)[0]
                    if len(cross) > 0:
                        rs[cross[0]] = np.nan
                        if cross[0]+1 < len(rs): rs[cross[0]+1] = np.nan

                if ma_idx == 0:
                    ls, mfc = '--', 'none'
                elif ma_idx == 1:
                    ls, mfc = '-.', 'white'
                else:
                    ls, mfc = ':', col

                ax_s.plot(k_vals, rs, linestyle=ls, color=col, marker=mkr,
                          markerfacecolor=mfc, markeredgecolor=col, markeredgewidth=1.1)
                ax_d.plot(k_vals, rd, linestyle=ls, color=col, marker=mkr,
                          markerfacecolor=mfc, markeredgecolor=col, markeredgewidth=1.1)

        # In-plot labels on panel (a) and (b) for each row
        if mode_lbl == 'A1=0_B1=1':
            annotate_curve(ax_s, r"$\lambda = 10$ (Blue Circle)", xy=(0.60, 0.98), xytext=(0.50, 1.15), color="#0044cc", fontsize=9.2)
            annotate_curve(ax_s, r"$\lambda = 1$ (Red Square)", xy=(0.65, 0.62 if pe_val>=1 else 0.88), xytext=(0.68, 0.42 if pe_val>=1 else 0.70), color="#cc0000", fontsize=9.2)
            annotate_curve(ax_s, r"$\lambda = 0.1$ (Purple Triangle)", xy=(0.80, 0.72 if pe_val>=1 else 0.68), xytext=(0.72, 0.90 if pe_val>=1 else 0.45), color="#771199", fontsize=9.2)
        else: # Pusher/puller
            annotate_curve(ax_s, r"$\lambda = 0.1$ (Purple Triangle)", xy=(0.75, 1.8), xytext=(0.70, 2.5), color="#771199", fontsize=9.2)
            annotate_curve(ax_s, r"$\lambda = 10$ (Blue Circle)", xy=(0.65, 0.45), xytext=(0.70, 0.65), color="#0044cc", fontsize=9.2)
            annotate_curve(ax_s, r"$\lambda = 1$ (Red Square)", xy=(0.65, -0.4 if pe_val>=1 else 0.35), xytext=(0.70, -0.8 if pe_val>=1 else 0.15), color="#cc0000", fontsize=9.2)

        if pe_val < 10:
            annotate_curve(ax_d, r"$\lambda = 10$ (Blue Circle)", xy=(0.80, 0.95), xytext=(0.65, 1.05), color="#0044cc", fontsize=9.2)
            annotate_curve(ax_d, r"$\lambda = 1$ (Red Square)", xy=(0.50, 0.62), xytext=(0.56, 0.74), color="#cc0000", fontsize=9.2)
            annotate_curve(ax_d, r"$\lambda = 0.1$ (Purple Triangle)", xy=(0.60, 0.425), xytext=(0.42, 0.52), color="#771199", fontsize=9.2)
        else:
            annotate_curve(ax_d, r"$\lambda = 10$ (Blue Circle)", xy=(0.75, 0.75), xytext=(0.65, 0.95), color="#0044cc", fontsize=9.2)
            annotate_curve(ax_d, r"$\lambda = 1$ (Red Square)", xy=(0.40, 0.61), xytext=(0.46, 0.74), color="#cc0000", fontsize=9.2)
            annotate_curve(ax_d, r"$\lambda = 0.1$ (Purple Triangle)", xy=(0.60, 0.425), xytext=(0.42, 0.52), color="#771199", fontsize=9.2)

        # Marangoni Progression Indicator in Panel (b)
        ax_d.annotate("", xy=(0.28, 0.62), xytext=(0.28, 0.95),
                      arrowprops=dict(arrowstyle="->", color="#222222", lw=1.2))
        ax_d.text(0.29, 0.78, r"$\mathrm{increasing}\; Ma$", fontsize=8.0, color="#222222", rotation=-90, va='center')

        pe_desc = f"$Pe = {pe_val}$"
        ma_desc = f"$Ma \\in \\{{{mas[0]}, {mas[1]}, {mas[2]}\\}}$"
        ax_s.set_ylabel(f"{pe_desc}\n({ma_desc})\n" + r"$U_S / U_{S,\mathrm{clean}}$")
        ax_d.set_ylabel(f"{pe_desc}\n({ma_desc})\n" + r"$U_D / U_{D,\mathrm{clean}}$")

        if row_idx == 0:
            ax_s.set_title(r"$(a)\;\; \text{Swimmer Velocity Ratio}\; U_S / U_{S,\mathrm{clean}}$", fontsize=12)
            ax_d.set_title(r"$(b)\;\; \text{Droplet Velocity Ratio}\; U_D / U_{D,\mathrm{clean}}$", fontsize=12)

        if row_idx == 2:
            ax_s.set_xlabel(r"$\chi \equiv k = R_{\mathrm{core}} / R_{\mathrm{drop}}$")
            ax_d.set_xlabel(r"$\chi \equiv k = R_{\mathrm{core}} / R_{\mathrm{drop}}$")

        # Inset critical Ma box in middle row
        if row_idx == 1:
            ax_d.text(0.96, 0.05, MAC_ANNOTATION_TEXT, transform=ax_d.transAxes,
                      fontsize=7.8, horizontalalignment='right', verticalalignment='bottom',
                      bbox=dict(boxstyle='square,pad=0.4', facecolor='white', edgecolor='black', lw=1.0))

    # Unified Header Legend with explicit colors and shapes
    legend_handles = [
        mlines.Line2D([], [], color=LAMBDA_CFG[10.0]['color'], marker='o', markerfacecolor=LAMBDA_CFG[10.0]['color'], linestyle='-', lw=1.6, label=r'$\lambda = 10$ (Blue Circle)'),
        mlines.Line2D([], [], color=LAMBDA_CFG[1.0]['color'], marker='s', markerfacecolor=LAMBDA_CFG[1.0]['color'], linestyle='-', lw=1.6, label=r'$\lambda = 1$ (Red Square)'),
        mlines.Line2D([], [], color=LAMBDA_CFG[0.1]['color'], marker='^', markerfacecolor=LAMBDA_CFG[0.1]['color'], linestyle='-', lw=1.6, label=r'$\lambda = 0.1$ (Purple Triangle)'),
        mlines.Line2D([], [], color='black', linestyle='-', lw=1.3, label=r'Clean drop ($Ma=0$, solid)'),
        mlines.Line2D([], [], color='black', linestyle='--', marker='o', markerfacecolor='none', markeredgecolor='black', label=r'Subcritical ($Ma=0.1/0.01$, open, dashed)'),
        mlines.Line2D([], [], color='black', linestyle='-.', marker='o', markerfacecolor='white', markeredgecolor='black', label=r'Transitional ($Ma=1.0/0.1$, half, dash-dot)'),
        mlines.Line2D([], [], color='black', linestyle=':', marker='o', markerfacecolor='black', markeredgecolor='black', label=r'Supercritical ($Ma=10/1.0$, filled, dotted)')
    ]
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=4, fontsize=8.8, frameon=True)

    mode_text = r"$A_1=1, B_1=1$ (Pusher/Puller Squirmer)" if mode_lbl == "A1=1_B1=1" else r"$A_1=0, B_1=1$ (Neutral Squirmer)"
    fig.suptitle(f"Master Parametric Grid: Locomotion Velocity Ratios across Péclet Regimes | {mode_text}", y=1.02, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    save_path = os.path.join(plots_dir, f"shaik_ratio_master_grid_{mode_lbl}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[REMADE] Master grid figure: {save_path}")
    return save_path


# ==============================================================================
# remake pe=10 diagnostic comparison figures
# ==============================================================================
def plot_pe10_comparison(records, mode_lbl):
    fig, axs = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    has_pole = (mode_lbl == "A1=1_B1=1")

    configs = [
        (False, [0.1, 1.0, 10.0], "Old Set ($Ma \\in \\{0.1, 1.0, 10.0\\}$)"),
        (True, [0.01, 0.1, 1.0], "New Optimized Set ($Ma \\in \\{0.01, 0.1, 1.0\\}$)")
    ]

    for row_idx, (is_opt, mas, row_title) in enumerate(configs):
        ax_s = axs[row_idx, 0]
        ax_d = axs[row_idx, 1]

        format_axes(ax_s, xlim=(0.0, 1.0), ylim=(-2.5, 4.0) if has_pole else (0.0, 1.4))
        format_axes(ax_d, xlim=(0.0, 1.0), ylim=(0.0, 1.2))

        ax_s.axhline(1.0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)
        ax_d.axhline(1.0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)

        if has_pole:
            ax_s.axvline(0.565, color='#771199', linestyle=':', alpha=0.5)

        for l_val, cfg in LAMBDA_CFG.items():
            col = cfg['color']
            mkr = cfg['marker']

            for ma_idx, ma in enumerate(mas):
                recs = filter_recs(records, mode_lbl, 10.0, l_val, ma)
                if not recs: continue

                k_vals = np.array([r['k'] for r in recs])
                rs = np.array([r['ratio_U_s'] for r in recs])
                rd = np.array([r['ratio_U_d'] for r in recs])
                usc = np.array([r['U_s_clean'] for r in recs])

                if has_pole and l_val == 0.1:
                    mask = np.abs(usc) < 0.02
                    rs[mask] = np.nan
                    cross = np.where(np.diff(np.sign(usc)) != 0)[0]
                    if len(cross) > 0:
                        rs[cross[0]] = np.nan
                        if cross[0]+1 < len(rs): rs[cross[0]+1] = np.nan

                if ma_idx == 0:
                    ls, mfc = '--', 'none'
                elif ma_idx == 1:
                    ls, mfc = '-.', 'white'
                else:
                    ls, mfc = ':', col

                ax_s.plot(k_vals, rs, linestyle=ls, color=col, marker=mkr,
                          markerfacecolor=mfc, markeredgecolor=col, markeredgewidth=1.1)
                ax_d.plot(k_vals, rd, linestyle=ls, color=col, marker=mkr,
                          markerfacecolor=mfc, markeredgecolor=col, markeredgewidth=1.1)

        ax_s.set_title(f"(a{row_idx+1}) Swimmer Ratio: {row_title}", fontsize=11, pad=6)
        ax_d.set_title(f"(b{row_idx+1}) Drop Ratio: {row_title}", fontsize=11, pad=6)

        ax_s.set_ylabel(r"$U_S / U_{S,\mathrm{clean}}$")
        ax_d.set_ylabel(r"$U_D / U_{D,\mathrm{clean}}$")

        if row_idx == 1:
            ax_s.set_xlabel(r"$\chi \equiv k = R_{\mathrm{core}} / R_{\mathrm{drop}}$")
            ax_d.set_xlabel(r"$\chi \equiv k = R_{\mathrm{core}} / R_{\mathrm{drop}}$")

        # In-plot labels on panel (a) and (b)
        if mode_lbl == 'A1=0_B1=1':
            annotate_curve(ax_s, r"$\lambda = 10$ (Blue Circle)", xy=(0.60, 0.95), xytext=(0.50, 1.12), color="#0044cc", fontsize=9.2)
            annotate_curve(ax_s, r"$\lambda = 1$ (Red Square)", xy=(0.65, 0.62), xytext=(0.68, 0.42), color="#cc0000", fontsize=9.2)
            annotate_curve(ax_s, r"$\lambda = 0.1$ (Purple Triangle)", xy=(0.75, 0.88), xytext=(0.75, 1.08), color="#771199", fontsize=9.2)
        else:
            annotate_curve(ax_s, r"$\lambda = 0.1$ (Purple Triangle)", xy=(0.75, 1.8), xytext=(0.70, 2.5), color="#771199", fontsize=9.2)
            annotate_curve(ax_s, r"$\lambda = 10$ (Blue Circle)", xy=(0.65, 0.45), xytext=(0.70, 0.65), color="#0044cc", fontsize=9.2)
            annotate_curve(ax_s, r"$\lambda = 1$ (Red Square)", xy=(0.65, -0.4), xytext=(0.70, -0.8), color="#cc0000", fontsize=9.2)

        annotate_curve(ax_d, r"$\lambda = 10$ (Blue Circle)", xy=(0.80, 0.90 if is_opt else 0.58),
                       xytext=(0.65, 1.05 if is_opt else 0.70), color="#0044cc", fontsize=9.2)
        annotate_curve(ax_d, r"$\lambda = 1$ (Red Square)", xy=(0.40, 0.61),
                       xytext=(0.46, 0.74), color="#cc0000", fontsize=9.2)
        annotate_curve(ax_d, r"$\lambda = 0.1$ (Purple Triangle)", xy=(0.60, 0.425),
                       xytext=(0.42, 0.52), color="#771199", fontsize=9.2)

        # Inset critical Ma
        ax_d.text(0.96, 0.05, MAC_ANNOTATION_TEXT, transform=ax_d.transAxes,
                  fontsize=7.5, horizontalalignment='right', verticalalignment='bottom',
                  bbox=dict(boxstyle='square,pad=0.4', facecolor='white', edgecolor='black', lw=1.0))

    legend_handles = [
        mlines.Line2D([], [], color=LAMBDA_CFG[10.0]['color'], marker='o', markerfacecolor=LAMBDA_CFG[10.0]['color'], linestyle='-', lw=1.6, label=r'$\lambda = 10$ (Blue Circle)'),
        mlines.Line2D([], [], color=LAMBDA_CFG[1.0]['color'], marker='s', markerfacecolor=LAMBDA_CFG[1.0]['color'], linestyle='-', lw=1.6, label=r'$\lambda = 1$ (Red Square)'),
        mlines.Line2D([], [], color=LAMBDA_CFG[0.1]['color'], marker='^', markerfacecolor=LAMBDA_CFG[0.1]['color'], linestyle='-', lw=1.6, label=r'$\lambda = 0.1$ (Purple Triangle)'),
        mlines.Line2D([], [], color='black', linestyle='-', lw=1.3, label=r'Clean drop ($Ma=0$, solid)'),
        mlines.Line2D([], [], color='black', linestyle='--', marker='o', markerfacecolor='none', markeredgecolor='black', label=r'Subcritical ($Pe\cdot Ma=0.1$, open, dashed)'),
        mlines.Line2D([], [], color='black', linestyle='-.', marker='o', markerfacecolor='white', markeredgecolor='black', label=r'Transitional ($Pe\cdot Ma=1.0$, half, dash-dot)'),
        mlines.Line2D([], [], color='black', linestyle=':', marker='o', markerfacecolor='black', markeredgecolor='black', label=r'Supercritical ($Pe\cdot Ma=10$, filled, dotted)')
    ]
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=4, fontsize=8.5, frameon=True)

    mode_text = r"$A_1=1, B_1=1$ (Pusher/Puller Squirmer)" if mode_lbl == "A1=1_B1=1" else r"$A_1=0, B_1=1$ (Neutral Squirmer)"
    fig.suptitle(f"Pe = 10.0 Marangoni Number Selection Comparison | {mode_text}", y=1.025, fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = os.path.join(plots_dir, f"shaik_ratio_Pe=10.0_comparison_{mode_lbl}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[REMADE] Pe=10 comparison figure: {save_path}")
    return save_path


# ==============================================================================
# MAIN RUNNER
# ==============================================================================
def remake_all():
    records = load_dataset()
    print(f"Loaded master dataset: {len(records)} records.")

    modes = ['A1=0_B1=1', 'A1=1_B1=1']
    pes = [0.1, 1.0, 10.0]

    all_generated = []

    for mode_lbl in modes:
        print(f"\n--- Remaking figures for Mode: {mode_lbl} ---")
        for pe in pes:
            f1 = plot_single_pe_ratios(records, mode_lbl, pe, optimized=True)
            f2 = plot_direct_velocities(records, mode_lbl, pe, optimized=True)
            all_generated.extend([f1, f2])

        f_grid = plot_master_grid_ratios(records, mode_lbl)
        f_comp = plot_pe10_comparison(records, mode_lbl)
        all_generated.extend([f_grid, f_comp])

    print(f"Remade {len(all_generated)} figures")

if __name__ == "__main__":
    remake_all()
