#!/usr/bin/env python3
"""
Generate publication-quality 3x2 grid figures showing velocity ratio
U/U_clean vs core radius ratio k, for both squirmer modes.

Layout:
  - 3 rows: Pe = 0.1, 1.0, 10.0
  - 2 columns: U_s/U_s_clean (left), U_d/U_d_clean (right)
  - Colours identify lambda (viscosity ratio)
  - Marker shapes identify Ma (Marangoni number)
  - Ma values are adapted per Pe to show meaningful variation
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import AutoMinorLocator

# ──────────────────── CONFIGURATION ────────────────────

from squirmer_drop.paths import RATIO_SWEEPS_DIR, RATIO_FIGURES_DIR

MASTER_CSV = os.path.join(RATIO_SWEEPS_DIR, "ratio_sweep_master.csv")
OUTPUT_DIR = RATIO_FIGURES_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

PE_VALUES = [0.1, 1.0, 10.0]

# Adaptive Ma values per Pe
MA_PER_PE = {
    0.1:  [0.1, 1.0, 10.0],
    1.0:  [0.1, 1.0, 5.0],
    10.0: [0.01, 0.1, 1.0],
}

LAMBDA_VALUES = [0.1, 1.0, 10.0]

# Colour scheme: high contrast, colourblind-friendly
LAMBDA_COLOURS = {
    0.1:  '#7B2D8E',  # Deep purple
    1.0:  '#D62728',  # Red
    10.0: '#1F77B4',  # Blue
}

# Marker encoding for Ma levels (0 = lowest, 1 = mid, 2 = highest)
MA_MARKERS = ['o', 's', 'D']
MA_SIZES = [5, 5, 4.5]

MODE_LABELS = {
    'A1=0_B1=1': r'Squirmer ($A_1 = 0,\ B_1 = 1$)',
    'A1=1_B1=1': r'Squirmer ($A_1 = 1,\ B_1 = 1$)',
}


def load_master_data():
    """Load the master ratio sweep CSV."""
    data = {}
    with open(MASTER_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mode = row['mode_label']
            pe = float(row['Pe'])
            lam = float(row['lambda'])
            ma = float(row['Ma'])
            k = float(row['k'])
            
            key = (mode, pe, lam, ma)
            if key not in data:
                data[key] = {'k': [], 'ratio_Us': [], 'ratio_Ud': [],
                             'converged': []}
            
            data[key]['k'].append(k)
            data[key]['ratio_Us'].append(float(row['ratio_U_s']))
            data[key]['ratio_Ud'].append(float(row['ratio_U_d']))
            data[key]['converged'].append(row['converged'] == 'True')
    
    for key in data:
        idx = np.argsort(data[key]['k'])
        for field in data[key]:
            data[key][field] = np.array(data[key][field])[idx]
    
    return data


def format_ma(ma):
    """Format Ma value for labels."""
    if ma >= 1:
        return f'{ma:g}'
    else:
        return f'{ma}'


def generate_grid_figure(mode_label, data):
    """Generate the 3x2 grid figure for one squirmer mode."""
    
    fig, axes = plt.subplots(3, 2, figsize=(13, 15))
    
    plt.rcParams.update({
        'mathtext.fontset': 'cm',
        'font.family': 'serif',
        'font.size': 11,
    })
    
    for row_idx, pe in enumerate(PE_VALUES):
        ax_s = axes[row_idx, 0]
        ax_d = axes[row_idx, 1]
        ma_list = MA_PER_PE[pe]
        
        for lam in LAMBDA_VALUES:
            colour = LAMBDA_COLOURS[lam]
            
            for ma_idx, ma in enumerate(ma_list):
                marker = MA_MARKERS[ma_idx]
                ms = MA_SIZES[ma_idx]
                
                key = (mode_label, pe, lam, ma)
                if key not in data:
                    print(f"  WARNING: Missing {key}")
                    continue
                
                d = data[key]
                mask = np.isfinite(d['ratio_Us']) & np.isfinite(d['ratio_Ud']) & d['converged']
                if not np.any(mask):
                    continue
                
                k_plot = d['k'][mask]
                ratio_Us = d['ratio_Us'][mask]
                ratio_Ud = d['ratio_Ud'][mask]
                
                if ma_idx == 0:
                    # Ma_low: solid line only, no markers
                    plot_kwargs = dict(
                        color=colour, linestyle='-', linewidth=1.8,
                        marker='None', zorder=3,
                    )
                elif ma_idx == 1:
                    # Ma_mid: open square + dashed line
                    plot_kwargs = dict(
                        color=colour, linestyle='--', linewidth=1.2,
                        marker='s', markersize=5.5,
                        markerfacecolor='white', markeredgecolor=colour,
                        markeredgewidth=1.2, zorder=3,
                    )
                else:
                    # Ma_high: filled square + dashed line
                    plot_kwargs = dict(
                        color=colour, linestyle='--', linewidth=1.2,
                        marker='s', markersize=5.5,
                        markerfacecolor=colour, markeredgecolor=colour,
                        markeredgewidth=0.8, zorder=3,
                    )
                
                ax_s.plot(k_plot, ratio_Us, **plot_kwargs)
                ax_d.plot(k_plot, ratio_Ud, **plot_kwargs)
        
        # Reference line at ratio = 1
        for ax in [ax_s, ax_d]:
            ax.axhline(y=1.0, color='k', linewidth=0.7, alpha=0.35, zorder=1)
        
        # Axis formatting
        for ax in [ax_s, ax_d]:
            ax.set_xlim(-0.02, 1.02)
            ax.tick_params(axis='both', which='both', direction='in',
                          top=True, right=True, labelsize=10)
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.grid(True, alpha=0.12, linewidth=0.4)
        
        # Y-labels
        ax_s.set_ylabel(r'$U_s / U_{s,\mathrm{clean}}$', fontsize=13)
        ax_d.set_ylabel(r'$U_d / U_{d,\mathrm{clean}}$', fontsize=13)
        
        # Pe and Ma annotation box — compact, top-right of each panel
        ma_str = ', '.join([format_ma(m) for m in ma_list])
        info_text = f'$Pe_s = {pe:g}$\n$Ma = {ma_str}$'
        
        for ax in [ax_s, ax_d]:
            ax.text(0.03, 0.04, info_text, transform=ax.transAxes,
                   fontsize=9.5, va='bottom', ha='left',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                            edgecolor='#999999', alpha=0.92),
                   zorder=10)
        
        # Bottom row x-labels
        if row_idx == 2:
            ax_s.set_xlabel(r'$k$', fontsize=13)
            ax_d.set_xlabel(r'$k$', fontsize=13)
    
    # ──────────── Column titles ────────────
    axes[0, 0].set_title(r'Squirmer Velocity Ratio $U_s / U_{s,\mathrm{clean}}$',
                         fontsize=12, fontweight='bold', pad=10)
    axes[0, 1].set_title(r'Drop Velocity Ratio $U_d / U_{d,\mathrm{clean}}$',
                         fontsize=12, fontweight='bold', pad=10)
    
    # ──────────── BUILD CLEAN LEGEND ────────────
    # Lambda legend handles
    lambda_handles = []
    for lam in LAMBDA_VALUES:
        h = mlines.Line2D([], [], color=LAMBDA_COLOURS[lam], marker='o',
                         markersize=7, markerfacecolor=LAMBDA_COLOURS[lam],
                         linestyle='-', linewidth=2.0,
                         label=fr'$\lambda = {lam:g}$')
        lambda_handles.append(h)
    
    # Separator
    sep = mlines.Line2D([], [], color='none', label='')
    
    # Ma shape handles
    ma_handles = [
        mlines.Line2D([], [], color='#444444', marker='None',
                     linestyle='-', linewidth=2.0,
                     label=r'$Ma_\mathrm{low}$ (line)'),
        mlines.Line2D([], [], color='#444444', marker='s', markersize=7,
                     markerfacecolor='white', markeredgecolor='#444444',
                     markeredgewidth=1.2, linestyle='--', linewidth=1.2,
                     label=r'$Ma_\mathrm{mid}$ (open)'),
        mlines.Line2D([], [], color='#444444', marker='s', markersize=7,
                     markerfacecolor='#444444', markeredgecolor='#444444',
                     linestyle='--', linewidth=1.2,
                     label=r'$Ma_\mathrm{high}$ (filled)'),
    ]
    
    all_handles = lambda_handles + [sep] + ma_handles
    
    fig.legend(handles=all_handles, loc='upper center',
              ncol=7, fontsize=9.5, frameon=True,
              fancybox=True, shadow=False,
              bbox_to_anchor=(0.5, 1.01),
              edgecolor='#bbbbbb', columnspacing=1.2,
              handletextpad=0.5)
    
    # Suptitle
    mode_title = MODE_LABELS[mode_label]
    fig.suptitle(mode_title, fontsize=14, fontweight='bold', y=1.05)
    
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    
    return fig


def main():
    print("Loading master dataset...")
    data = load_master_data()
    print(f"  Loaded {len(data)} parameter combinations")
    
    for mode in ['A1=0_B1=1', 'A1=1_B1=1']:
        print(f"\nGenerating grid figure for {mode}...")
        fig = generate_grid_figure(mode, data)
        
        fname = f"velocity_ratio_grid_{mode}.png"
        fpath = os.path.join(OUTPUT_DIR, fname)
        fig.savefig(fpath, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"  Saved: {fpath}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
