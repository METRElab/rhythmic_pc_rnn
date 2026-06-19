"""
Shared matplotlib styling for publication-ready paper figures.

Single source of truth for all visual constants: fonts, colors, line styles.
Call apply_paper_style() once before generating any figures.
"""

import matplotlib.pyplot as plt


PAPER_RCPARAMS = {
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'figure.titlesize': 20,
    'lines.linewidth': 2,
    'figure.dpi': 300,
    'savefig.dpi': 300,
}

COLORS = {
    'actual': '#2C3E50',       # Dark blue-gray for ground truth / original signals
    'vest_pred': '#E74C3C',    # Muted red for vestibular predictions
    'beat_input': 'b',         # Blue for auditory input
    'beat_pred': 'g',          # Green for auditory predictions
    'vline': 'gray',           # Vertical beat alignment lines
    'transition': 'black',     # Transition marker in continuation figure
    'input_region': 'blue',    # Shaded input period
    'continuation_region': 'red',   # Shaded continuation period
    'no_input_region': 'gray',      # Shaded no-input period
}

STYLES = {
    'ground_truth_linewidth': 1.5,
    'ground_truth_alpha': 0.8,
    'prediction_linewidth': 1.5,
    'prediction_alpha': 0.9,
    'vline_linestyle': ':',
    'vline_alpha': 0.4,
    'vline_linewidth': 1,
    'grid_alpha': 0.3,
}


def apply_paper_style():
    """Apply publication-ready matplotlib style. Call once at script start."""
    plt.style.use('default')
    plt.rcParams.update(PAPER_RCPARAMS)


def remove_chart_junk(ax):
    """Remove top and right spines from an axes."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
