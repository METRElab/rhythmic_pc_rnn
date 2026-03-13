"""
Publication-ready plotting functions for paper figures.

Each function takes numpy arrays (not file paths) and returns a matplotlib
Figure object. The caller decides whether to show, save, or compose figures.

Usage:
    from visualization.paper_figures import plot_learning_curve, plot_before_after_2x2
    from visualization.style import apply_paper_style

    apply_paper_style()
    fig = plot_learning_curve(steps, errors)
    fig.savefig('figure_1a.png', dpi=300, bbox_inches='tight', facecolor='white')
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.ndimage import uniform_filter1d

from visualization.style import COLORS, STYLES, remove_chart_junk


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_ylim(arrays: List[np.ndarray], margin_frac: float = 0.1) -> List[float]:
    """Compute y-axis limits with margin from a list of arrays."""
    all_min = min(a.min() for a in arrays)
    all_max = max(a.max() for a in arrays)
    margin = (all_max - all_min) * margin_frac
    return [all_min - margin, all_max + margin]


def _detect_cycles(
    beat_seq: np.ndarray, threshold: float = 0.5
) -> Tuple[np.ndarray, int]:
    """
    Detect cycle boundaries from a beat sequence.

    Returns:
        (beat_peaks, cycle_length) where beat_peaks are indices where beats occur
    """
    beat_peaks = np.where(beat_seq > threshold)[0]
    if len(beat_peaks) > 1:
        cycle_length = int(beat_peaks[1] - beat_peaks[0])
    else:
        cycle_length = 100
    return beat_peaks, cycle_length


def _add_beat_vlines(
    ax: plt.Axes,
    time_steps: np.ndarray,
    beat_seq: np.ndarray,
    threshold: float = 0.5,
) -> None:
    """Add vertical gray dotted lines at beat times."""
    beat_times = time_steps[beat_seq > threshold]
    for bt in beat_times:
        ax.axvline(
            x=bt,
            color=COLORS['vline'],
            linestyle=STYLES['vline_linestyle'],
            alpha=STYLES['vline_alpha'],
            linewidth=STYLES['vline_linewidth'],
        )


def _slice_to_cycles(
    data: Dict[str, np.ndarray],
    start_cycle: int,
    end_cycle: int,
    cycle_length: int,
) -> Dict[str, np.ndarray]:
    """Slice all arrays in data dict to the specified cycle range."""
    start_idx = start_cycle * cycle_length
    end_idx = end_cycle * cycle_length
    return {key: arr[start_idx:end_idx] for key, arr in data.items()}


# ---------------------------------------------------------------------------
# Figure 1a / 4a: Learning curve
# ---------------------------------------------------------------------------

def _smooth_curve(
    steps: np.ndarray,
    errors: np.ndarray,
    moving_average_window: int = 1,
    smoothing_window: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply moving average then uniform_filter1d, returning (steps, smoothed)."""
    if moving_average_window > 1:
        n = len(errors)
        w = moving_average_window
        errors = np.array([errors[i:i + w].mean() for i in range(n - w + 1)])
        steps = steps[:len(errors)]

    if smoothing_window > 1:
        errors = uniform_filter1d(errors, size=smoothing_window, mode='nearest')

    return steps, errors


def plot_learning_curve(
    steps: np.ndarray,
    errors: np.ndarray,
    smoothing_window: int = 30,
    moving_average_window: int = 1,
    max_step: Optional[int] = None,
    title: str = 'Sensory Prediction Error',
    figsize: Tuple[float, float] = (13, 6),
    label: Optional[str] = None,
    overlay_steps: Optional[np.ndarray] = None,
    overlay_errors: Optional[np.ndarray] = None,
    overlay_label: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a learning curve with log-scale y-axis and optional smoothing.

    Used for Figure 1a (sensorimotor) and Figure 4a (double auditory).

    Two levels of smoothing are available and applied in order:
    1. moving_average_window: simple moving average (reduces length by window-1)
    2. smoothing_window: uniform_filter1d (preserves length)

    Args:
        steps: Training step numbers
        errors: Error values corresponding to each step
        smoothing_window: uniform_filter1d window size (1 = no smoothing)
        moving_average_window: Simple moving average window (1 = no moving average)
        max_step: If set, truncate to show only up to this step index
        title: Plot title
        figsize: Figure size
        label: Legend label for the main curve (None = no legend)
        overlay_steps: Training step numbers for a second experiment overlay
        overlay_errors: Error values for the overlay curve
        overlay_label: Legend label for the overlay curve (None = no label)

    Returns:
        matplotlib Figure
    """
    if max_step is not None:
        mask = steps <= max_step
        steps = steps[mask]
        errors = errors[mask]

    steps, errors_smooth = _smooth_curve(
        steps.copy(), errors.copy(), moving_average_window, smoothing_window,
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Overlay (behind main curve, faded)
    if overlay_steps is not None and overlay_errors is not None:
        if max_step is not None:
            o_mask = overlay_steps <= max_step
            overlay_steps = overlay_steps[o_mask]
            overlay_errors = overlay_errors[o_mask]
        o_steps, o_smooth = _smooth_curve(
            overlay_steps.copy(), overlay_errors.copy(),
            moving_average_window, smoothing_window,
        )
        ax.plot(o_steps, o_smooth, '--', linewidth=1.5, alpha=0.3,
                color='tab:blue', label=overlay_label)

    ax.plot(steps, errors_smooth, '-', linewidth=1.5, alpha=0.9, label=label)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Sensory Prediction Error')
    ax.set_title(title)
    ax.grid(True, alpha=STYLES['grid_alpha'], linestyle='-', linewidth=0.5)
    ax.set_yscale('log')
    remove_chart_junk(ax)

    # Only show legend when at least one label is provided
    if label is not None or overlay_label is not None:
        ax.legend(loc='upper right')

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 1b / 2: Before/after 2x2 comparison
# ---------------------------------------------------------------------------

def plot_before_after_2x2(
    data: Dict[str, np.ndarray],
    max_cycles: int = 6,
    start_cycle: int = 0,
    end_cycle: Optional[int] = None,
    vest_ground_truth_style: str = '--',
    vest_ground_truth_alpha: float = 0.8,
    show_before_preds: bool = False,
    title_before: str = 'TRAINING INPUT',
    title_after: str = 'AFTER TRAINING',
    vest_label: str = 'Original Vestibular Signal',
    figsize: Tuple[float, float] = (14, 10),
) -> plt.Figure:
    """
    2x2 subplot: vestibular (top) / auditory (bottom) x before (left) / after (right).

    Used for:
    - Figure 1b: Standard before/after (left column shows raw input only)
    - Figure 2: Auditory-only (vest_ground_truth_style=':', vest_ground_truth_alpha=0.5,
                 vest_label='Original Vestibular Signal (not provided as input)')

    Args:
        data: Dict with keys vest_pred_before, vest_pred_after, vest_seq_before,
              vest_seq_after, beat_pred_before, beat_pred_after, beat_seq_before,
              beat_seq_after
        max_cycles: Max cycles to display (used when end_cycle is None)
        start_cycle: Starting cycle (0-based)
        end_cycle: Ending cycle. If None, uses max_cycles from the start.
        vest_ground_truth_style: Linestyle for vestibular ground truth ('--' or ':')
        vest_ground_truth_alpha: Alpha for vestibular ground truth line
        show_before_preds: If True, overlay prediction lines in before-training panels
        title_before: Column header for left panels
        title_after: Column header for right panels
        vest_label: Label for the vestibular ground truth line
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    actual_color = COLORS['actual']
    pred_color = COLORS['vest_pred']

    # Detect cycles and slice data
    _, cycle_length = _detect_cycles(data['beat_seq_after'])

    if end_cycle is not None:
        sliced = _slice_to_cycles(data, start_cycle, end_cycle, cycle_length)
    else:
        # Use max_cycles from the start
        min_length = min(len(v) for v in data.values())
        max_points = min(max_cycles * cycle_length, min_length)
        sliced = {key: arr[:max_points] for key, arr in data.items()}

    time_steps = np.arange(len(sliced['vest_pred_before']))

    # Compute y-axis limits
    vest_ylim = _compute_ylim([
        sliced['vest_seq_before'], sliced['vest_seq_after'],
        sliced['vest_pred_before'], sliced['vest_pred_after'],
    ])
    beat_ylim = _compute_ylim([
        sliced['beat_seq_before'], sliced['beat_seq_after'],
        sliced['beat_pred_before'], sliced['beat_pred_after'],
    ])

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize, sharex=True)

    # --- Top Left: Before - Vestibular ---
    if show_before_preds:
        # Show ground truth + prediction overlay
        ax1.plot(
            time_steps, sliced['vest_seq_before'],
            color=actual_color, linestyle=vest_ground_truth_style,
            linewidth=STYLES['ground_truth_linewidth'],
            alpha=vest_ground_truth_alpha, label=vest_label,
        )
        ax1.plot(
            time_steps, sliced['vest_pred_before'],
            color=pred_color, linestyle='-',
            linewidth=STYLES['prediction_linewidth'],
            alpha=STYLES['prediction_alpha'], label='Predicted Vestibular',
        )
    else:
        # Show input signal only (solid line)
        ax1.plot(
            time_steps, sliced['vest_seq_before'],
            color=actual_color, linestyle='-',
            linewidth=1.5, alpha=1.0, label='Vestibular Input',
        )
    ax1.set_title('Vestibular', fontweight='bold')
    ax1.set_ylabel('Vestibular Signal')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=STYLES['grid_alpha'])
    ax1.set_ylim(vest_ylim)

    # --- Top Right: After - Vestibular ---
    ax2.plot(
        time_steps, sliced['vest_seq_after'],
        color=actual_color, linestyle=vest_ground_truth_style,
        linewidth=STYLES['ground_truth_linewidth'],
        alpha=vest_ground_truth_alpha, label=vest_label,
    )
    ax2.plot(
        time_steps, sliced['vest_pred_after'],
        color=pred_color, linestyle='-',
        linewidth=STYLES['prediction_linewidth'],
        alpha=STYLES['prediction_alpha'], label='Predicted Vestibular',
    )
    ax2.set_title('Vestibular', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=STYLES['grid_alpha'])
    ax2.set_ylim(vest_ylim)

    # --- Bottom Left: Before - Auditory ---
    if show_before_preds:
        ax3.plot(
            time_steps, sliced['beat_seq_before'],
            color=actual_color, linestyle='--',
            linewidth=STYLES['ground_truth_linewidth'],
            alpha=STYLES['ground_truth_alpha'], label='Original Auditory Signal',
        )
        ax3.plot(
            time_steps, sliced['beat_pred_before'],
            color=pred_color, linestyle='-',
            linewidth=STYLES['prediction_linewidth'],
            alpha=STYLES['prediction_alpha'], label='Predicted Auditory Signal',
        )
    else:
        ax3.plot(
            time_steps, sliced['beat_seq_before'],
            color=actual_color, linestyle='-',
            linewidth=1.5, alpha=1.0, label='Auditory Input',
        )
    ax3.set_title('Auditory', fontweight='bold')
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Auditory Signal')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=STYLES['grid_alpha'])
    ax3.set_ylim(beat_ylim)

    # --- Bottom Right: After - Auditory ---
    ax4.plot(
        time_steps, sliced['beat_seq_after'],
        color=actual_color, linestyle='--',
        linewidth=STYLES['ground_truth_linewidth'],
        alpha=STYLES['ground_truth_alpha'], label='Original Auditory Signal',
    )
    ax4.plot(
        time_steps, sliced['beat_pred_after'],
        color=pred_color, linestyle='-',
        linewidth=STYLES['prediction_linewidth'],
        alpha=STYLES['prediction_alpha'], label='Predicted Auditory Signal',
    )
    ax4.set_title('Auditory', fontweight='bold')
    ax4.set_xlabel('Time Steps')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=STYLES['grid_alpha'])
    ax4.set_ylim(beat_ylim)

    # Consistent x-axis limits
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(0, len(time_steps) - 1)

    # Vertical beat alignment lines
    _add_beat_vlines(ax1, time_steps, sliced['beat_seq_before'])
    _add_beat_vlines(ax3, time_steps, sliced['beat_seq_before'])
    _add_beat_vlines(ax2, time_steps, sliced['beat_seq_after'])
    _add_beat_vlines(ax4, time_steps, sliced['beat_seq_after'])

    # Remove chart junk
    for ax in [ax1, ax2, ax3, ax4]:
        remove_chart_junk(ax)

    # Column headers
    fig.text(0.25, 0.95, title_before, ha='center', fontsize=16, fontweight='bold')
    fig.text(0.75, 0.95, title_after, ha='center', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    return fig


# ---------------------------------------------------------------------------
# Figure 3: Continuation (zero-input)
# ---------------------------------------------------------------------------

def plot_continuation(
    data: Dict[str, np.ndarray],
    transition_point: Optional[int] = None,
    max_points: Optional[int] = None,
    figsize: Tuple[float, float] = (12, 10),
) -> plt.Figure:
    """
    Plot the continuation/zero-input condition.

    Three subplots:
    - Top: Auditory input (shows zeros after transition)
    - Middle: Vestibular prediction (continuous through transition)
    - Bottom: Auditory prediction (continuous through transition)

    Args:
        data: Dict with keys vest_pred, beat_pred, vest_seq, beat_seq
        transition_point: Timestep where input is set to zero. If None, uses midpoint.
        max_points: Truncate arrays to this length. If None, use full length.
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    pred_vest = data['vest_pred'].copy()
    pred_beat = data['beat_pred'].copy()
    beat_input = data['beat_seq'].copy()
    vest_input = data['vest_seq'].copy()

    if max_points is not None:
        pred_vest = pred_vest[:max_points]
        pred_beat = pred_beat[:max_points]
        beat_input = beat_input[:max_points]
        vest_input = vest_input[:max_points]

    if transition_point is None:
        transition_point = len(pred_vest) // 2

    time = np.arange(len(pred_vest))

    # Detect beat baseline from non-pulse values (negative for zero-mean mode, 0 otherwise)
    non_pulse_mask = beat_input[:transition_point] < 0.5
    if non_pulse_mask.any():
        beat_baseline = float(np.median(beat_input[:transition_point][non_pulse_mask]))
    else:
        beat_baseline = 0.0

    # Modify beat input to show baseline after transition
    beat_input_modified = beat_input.copy()
    beat_input_modified[transition_point:] = beat_baseline

    # Compute y-axis limits: include both input and prediction ranges so
    # nothing is cropped, while keeping the input scale visible for comparison.
    vest_min = min(vest_input.min(), pred_vest.min())
    vest_max = max(vest_input.max(), pred_vest.max())
    beat_min = min(beat_input.min(), pred_beat.min())
    beat_max = max(beat_input.max(), pred_beat.max())
    vest_padding = (vest_max - vest_min) * 0.1 if vest_max > vest_min else 0.1
    beat_padding = (beat_max - beat_min) * 0.1 if beat_max > beat_min else 0.1
    vest_ylim = (vest_min - vest_padding, vest_max + vest_padding)
    beat_ylim = (beat_min - beat_padding, beat_max + beat_padding)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Build title suffix depending on whether baseline is zero or negative
    if abs(beat_baseline) < 1e-9:
        baseline_desc = 'Set to Zero'
    else:
        baseline_desc = f'Set to Baseline ({beat_baseline:.2f})'

    # --- Top: Auditory input ---
    ax1.plot(time, beat_input_modified, 'b-', linewidth=1.5, label='Auditory Input')
    ax1.set_ylabel('Auditory Input')
    ax1.set_title(f'Auditory Input ({baseline_desc} After Transition)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=STYLES['grid_alpha'])
    ax1.set_ylim(beat_ylim)

    # --- Middle: Vestibular prediction ---
    ax2.plot(time, pred_vest, color=COLORS['vest_pred'], linewidth=3,
             label='Predicted Vestibular Movement')
    ax2.set_ylabel('Vestibular Signal')
    ax2.set_title(f'Vestibular Continuation After Auditory Input {baseline_desc}')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=STYLES['grid_alpha'])
    ax2.set_ylim(vest_ylim)

    # --- Bottom: Auditory prediction ---
    ax3.plot(time, pred_beat, color=COLORS['beat_pred'], linewidth=2,
             label='Predicted Auditory')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Auditory Signal')
    ax3.set_title(f'Auditory Continuation After Input {baseline_desc}')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=STYLES['grid_alpha'])
    ax3.set_ylim(beat_ylim)

    # Transition marker
    transition_label = 'Input Set to Baseline' if abs(beat_baseline) > 1e-9 else 'Input Set to Zero'
    for ax in [ax1, ax2, ax3]:
        ax.axvline(
            x=transition_point, color=COLORS['transition'],
            linestyle='--', linewidth=2, alpha=0.8, label=transition_label,
        )

    # Shaded regions
    no_input_label = 'Baseline Input Period' if abs(beat_baseline) > 1e-9 else 'Zero Input Period'
    ax1.axvspan(0, transition_point, alpha=0.1, color=COLORS['input_region'],
                label='Input Period')
    ax1.axvspan(transition_point, len(time), alpha=0.1, color=COLORS['no_input_region'],
                label=no_input_label)

    for ax in [ax2, ax3]:
        ax.axvspan(0, transition_point, alpha=0.1, color=COLORS['input_region'],
                    label='With Input')
        ax.axvspan(transition_point, len(time), alpha=0.1,
                    color=COLORS['continuation_region'], label='Internal Continuation')

    # Beat alignment lines
    beat_peaks = time[beat_input > 0.5]
    for peak_time in beat_peaks:
        if peak_time < transition_point:
            for ax in [ax1, ax2, ax3]:
                ax.axvline(x=peak_time, color='gray', linestyle=':', alpha=0.6, linewidth=1)
        else:
            for ax in [ax2, ax3]:
                ax.axvline(x=peak_time, color='orange', linestyle=':', alpha=0.4, linewidth=1)

    # Update legends
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    ax3.legend(loc='upper right')

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4: Double auditory (no vestibular) 1x2
# ---------------------------------------------------------------------------

def plot_double_auditory_1x2(
    data: Dict[str, np.ndarray],
    start_cycle: int = 0,
    end_cycle: int = 6,
    figsize: Tuple[float, float] = (14, 6),
) -> plt.Figure:
    """
    Plot the double auditory condition (1x2 layout, beat only).

    Shows that without vestibular scaffolding, the network fails to learn.

    Args:
        data: Dict with keys beat_pred_before, beat_pred_after,
              beat_seq_before, beat_seq_after
        start_cycle: Starting cycle
        end_cycle: Ending cycle
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    _, cycle_length = _detect_cycles(data['beat_seq_after'])
    sliced = _slice_to_cycles(data, start_cycle, end_cycle, cycle_length)
    time_steps = np.arange(len(sliced['beat_pred_before']))

    beat_ylim = _compute_ylim([
        sliced['beat_seq_before'], sliced['beat_seq_after'],
        sliced['beat_pred_before'], sliced['beat_pred_after'],
    ])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # --- Before Training ---
    ax1.plot(
        time_steps, sliced['beat_seq_before'], 'b--',
        linewidth=STYLES['ground_truth_linewidth'],
        label='Original Auditory Signal', alpha=STYLES['ground_truth_alpha'],
    )
    ax1.plot(
        time_steps, sliced['beat_pred_before'], 'g-',
        linewidth=STYLES['prediction_linewidth'],
        label='Predicted Auditory Signal', alpha=STYLES['prediction_alpha'],
    )
    ax1.set_title('Before Training', fontweight='bold')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Auditory Signal')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=STYLES['grid_alpha'])
    ax1.set_ylim(beat_ylim)

    # --- After Training ---
    ax2.plot(
        time_steps, sliced['beat_seq_after'], 'b--',
        linewidth=STYLES['ground_truth_linewidth'],
        label='Original Auditory Signal', alpha=STYLES['ground_truth_alpha'],
    )
    ax2.plot(
        time_steps, sliced['beat_pred_after'], 'g-',
        linewidth=STYLES['prediction_linewidth'],
        label='Predicted Auditory Signal', alpha=STYLES['prediction_alpha'],
    )
    ax2.set_title('After Training', fontweight='bold')
    ax2.set_xlabel('Time Steps')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=STYLES['grid_alpha'])
    ax2.set_ylim(beat_ylim)

    # x-axis limits
    for ax in [ax1, ax2]:
        ax.set_xlim(0, len(time_steps) - 1)

    # Beat vlines
    _add_beat_vlines(ax1, time_steps, sliced['beat_seq_before'])
    _add_beat_vlines(ax2, time_steps, sliced['beat_seq_after'])

    # Clean up
    for ax in [ax1, ax2]:
        remove_chart_junk(ax)

    fig.text(0.5, 0.95, 'Pulse Training', ha='center', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    return fig


# ---------------------------------------------------------------------------
# Combined multi-panel figure (2x3)
# ---------------------------------------------------------------------------

def plot_combined_multi_panel(
    data_sensorimotor: Dict[str, np.ndarray],
    data_auditory_only: Dict[str, np.ndarray],
    max_cycles: int = 6,
    start_cycle_c: int = 2,
    end_cycle_c: int = 8,
    figsize: Tuple[float, float] = (21, 10),
) -> plt.Figure:
    """
    Combined 2x3 figure: sensorimotor before/after + auditory-only after training.

    Layout:
        Col 1: Training input (before, sensorimotor data)
        Col 2: After training (sensorimotor data)
        Col 3: After training - auditory only (auditory-only data)
        Row 1: Vestibular
        Row 2: Auditory

    Args:
        data_sensorimotor: Before/after data for sensorimotor condition
        data_auditory_only: Before/after data for auditory-only condition
        max_cycles: Cycles to show for sensorimotor columns
        start_cycle_c: Starting cycle for auditory-only column
        end_cycle_c: Ending cycle for auditory-only column
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    actual_color = COLORS['actual']
    pred_color = COLORS['vest_pred']

    # Process sensorimotor data (columns 1 & 2)
    _, cycle_length_ab = _detect_cycles(data_sensorimotor['beat_seq_after'])
    min_length_ab = min(len(v) for v in data_sensorimotor.values())
    max_points_ab = min(max_cycles * cycle_length_ab, min_length_ab)
    d_ab = {k: v[:max_points_ab] for k, v in data_sensorimotor.items()}
    time_ab = np.arange(max_points_ab)

    # Process auditory-only data (column 3)
    _, cycle_length_c = _detect_cycles(data_auditory_only['beat_seq_after'])
    d_c = _slice_to_cycles(data_auditory_only, start_cycle_c, end_cycle_c, cycle_length_c)
    time_c = np.arange(len(d_c['vest_pred_before']))

    # Y-axis limits
    vest_ylim_ab = _compute_ylim([
        d_ab['vest_seq_before'], d_ab['vest_seq_after'],
        d_ab['vest_pred_before'], d_ab['vest_pred_after'],
    ])
    beat_ylim_ab = _compute_ylim([
        d_ab['beat_seq_before'], d_ab['beat_seq_after'],
        d_ab['beat_pred_before'], d_ab['beat_pred_after'],
    ])
    vest_ylim_c = _compute_ylim([
        d_c['vest_seq_before'], d_c['vest_seq_after'],
        d_c['vest_pred_before'], d_c['vest_pred_after'],
    ])
    beat_ylim_c = _compute_ylim([
        d_c['beat_seq_before'], d_c['beat_seq_after'],
        d_c['beat_pred_before'], d_c['beat_pred_after'],
    ])

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    ax1, ax2, ax3 = axes[0]  # Top row: vestibular
    ax4, ax5, ax6 = axes[1]  # Bottom row: auditory

    # --- Col 1: Training input (before) ---
    ax1.plot(time_ab, d_ab['vest_seq_before'], color=actual_color, linestyle='-',
             linewidth=1.5, alpha=1, label='Vestibular Input')
    ax1.set_title('Vestibular', fontweight='bold')
    ax1.set_ylabel('Vestibular Signal')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=STYLES['grid_alpha'])
    ax1.set_ylim(vest_ylim_ab)

    ax4.plot(time_ab, d_ab['beat_seq_before'], color=actual_color, linestyle='-',
             linewidth=1.5, alpha=1, label='Auditory Input')
    ax4.set_title('Auditory', fontweight='bold')
    ax4.set_xlabel('Time Steps')
    ax4.set_ylabel('Auditory Signal')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=STYLES['grid_alpha'])
    ax4.set_ylim(beat_ylim_ab)

    # --- Col 2: After training (sensorimotor) ---
    ax2.plot(time_ab, d_ab['vest_seq_after'], color=actual_color, linestyle='--',
             linewidth=1.5, alpha=0.8, label='Vestibular Input')
    ax2.plot(time_ab, d_ab['vest_pred_after'], color=pred_color, linestyle='-',
             linewidth=1.5, alpha=0.9, label='Predicted Vestibular Signal')
    ax2.set_title('Vestibular', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=STYLES['grid_alpha'])
    ax2.set_ylim(vest_ylim_ab)

    ax5.plot(time_ab, d_ab['beat_seq_after'], color=actual_color, linestyle='--',
             linewidth=1.5, alpha=0.8, label='Auditory Input')
    ax5.plot(time_ab, d_ab['beat_pred_after'], color=pred_color, linestyle='-',
             linewidth=1.5, alpha=0.9, label='Predicted Auditory Signal')
    ax5.set_title('Auditory', fontweight='bold')
    ax5.set_xlabel('Time Steps')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=STYLES['grid_alpha'])
    ax5.set_ylim(beat_ylim_ab)

    # --- Col 3: After training - auditory only ---
    ax3.plot(time_c, d_c['vest_seq_after'], color=actual_color, linestyle=':',
             linewidth=1.5, alpha=0.5,
             label='Vestibular Training Signal (not provided as input)')
    ax3.plot(time_c, d_c['vest_pred_after'], color=pred_color, linestyle='-',
             linewidth=1.5, alpha=1, label='Predicted Vestibular Signal')
    ax3.set_title('Vestibular', fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=STYLES['grid_alpha'])
    ax3.set_ylim(vest_ylim_c)

    ax6.plot(time_c, d_c['beat_seq_after'], color=actual_color, linestyle='--',
             linewidth=1.5, alpha=1, label='Auditory Input')
    ax6.plot(time_c, d_c['beat_pred_after'], color=pred_color, linestyle='-',
             linewidth=1.5, alpha=1, label='Predicted Auditory Signal')
    ax6.set_title('Auditory', fontweight='bold')
    ax6.set_xlabel('Time Steps')
    ax6.legend(loc='upper right')
    ax6.grid(True, alpha=STYLES['grid_alpha'])
    ax6.set_ylim(beat_ylim_c)

    # X-axis limits
    for ax in [ax1, ax2, ax4, ax5]:
        ax.set_xlim(0, max_points_ab - 1)
    for ax in [ax3, ax6]:
        ax.set_xlim(0, len(time_c) - 1)

    # Beat vlines — sensorimotor columns
    _add_beat_vlines(ax1, time_ab, d_ab['beat_seq_before'])
    _add_beat_vlines(ax4, time_ab, d_ab['beat_seq_before'])
    _add_beat_vlines(ax2, time_ab, d_ab['beat_seq_after'])
    _add_beat_vlines(ax5, time_ab, d_ab['beat_seq_after'])

    # Beat vlines — auditory-only column
    _add_beat_vlines(ax3, time_c, d_c['beat_seq_after'])
    _add_beat_vlines(ax6, time_c, d_c['beat_seq_after'])

    # Remove chart junk
    for ax in axes.flat:
        remove_chart_junk(ax)

    # Column headers
    fig.text(0.17, 0.95, 'TRAINING INPUT', ha='center', fontsize=16, fontweight='bold')
    fig.text(0.5, 0.95, 'AFTER TRAINING', ha='center', fontsize=16, fontweight='bold')
    fig.text(
        0.83, 0.95, 'AFTER TRAINING \u2014 AUDITORY-ONLY',
        ha='center', fontsize=16, fontweight='bold',
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    return fig


# ---------------------------------------------------------------------------
# Summary figure: Learning curve + after-training panels
# ---------------------------------------------------------------------------

def plot_summary(
    lc_steps: np.ndarray,
    lc_errors: np.ndarray,
    data_sensorimotor: Dict[str, np.ndarray],
    data_auditory_only: Dict[str, np.ndarray],
    max_cycles: int = 6,
    start_cycle_ao: int = 2,
    end_cycle_ao: int = 8,
    smoothing_window: int = 30,
    moving_average_window: int = 1,
    figsize: Tuple[float, float] = (14, 14),
    lc_label: Optional[str] = None,
    overlay_steps: Optional[np.ndarray] = None,
    overlay_errors: Optional[np.ndarray] = None,
    overlay_label: Optional[str] = None,
) -> plt.Figure:
    """
    Summary figure combining the learning curve with after-training panels.

    Layout (using GridSpec):
        Top row (spanning full width): Learning curve
        Bottom 2x2:
            Col 1: After training (sensorimotor)
            Col 2: After training (auditory-only)
            Row 1: Vestibular
            Row 2: Auditory

    Args:
        lc_steps: Training step numbers for learning curve
        lc_errors: Total error values for learning curve
        data_sensorimotor: Before/after data for sensorimotor condition
        data_auditory_only: Before/after data for auditory-only condition
        max_cycles: Cycles to show for sensorimotor panels
        start_cycle_ao: Starting cycle for auditory-only panels
        end_cycle_ao: Ending cycle for auditory-only panels
        smoothing_window: uniform_filter1d window for learning curve
        moving_average_window: Simple moving average window for learning curve
        figsize: Figure size
        lc_label: Legend label for main learning curve (None = no legend)
        overlay_steps: Training step numbers for overlay experiment
        overlay_errors: Error values for overlay experiment
        overlay_label: Legend label for overlay curve (None = no label)

    Returns:
        matplotlib Figure
    """
    actual_color = COLORS['actual']
    pred_color = COLORS['vest_pred']

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1], hspace=0.35, wspace=0.3)

    # --- Top row: Learning curve (spans both columns) ---
    ax_lc = fig.add_subplot(gs[0, :])

    lc_steps_plot, lc_errors_plot = _smooth_curve(
        lc_steps.copy(), lc_errors.copy(), moving_average_window, smoothing_window,
    )

    # Overlay (behind main curve, faded)
    if overlay_steps is not None and overlay_errors is not None:
        o_steps, o_smooth = _smooth_curve(
            overlay_steps.copy(), overlay_errors.copy(),
            moving_average_window, smoothing_window,
        )
        ax_lc.plot(o_steps, o_smooth, '--', linewidth=1.5, alpha=0.3,
                   color='tab:blue', label=overlay_label)

    ax_lc.plot(lc_steps_plot, lc_errors_plot, '-', linewidth=1.5, alpha=0.9,
               label=lc_label)
    ax_lc.set_xlabel('Training Step')
    ax_lc.set_ylabel('Sensory Prediction Error')
    ax_lc.set_title('Learning Curve')
    ax_lc.grid(True, alpha=STYLES['grid_alpha'], linestyle='-', linewidth=0.5)
    ax_lc.set_yscale('log')
    remove_chart_junk(ax_lc)

    # Only show legend when at least one label is provided
    if lc_label is not None or overlay_label is not None:
        ax_lc.legend(loc='upper right')

    # --- Process sensorimotor data (column 1) ---
    _, cycle_length_sm = _detect_cycles(data_sensorimotor['beat_seq_after'])
    min_length_sm = min(len(v) for v in data_sensorimotor.values())
    max_points_sm = min(max_cycles * cycle_length_sm, min_length_sm)
    d_sm = {k: v[:max_points_sm] for k, v in data_sensorimotor.items()}
    time_sm = np.arange(max_points_sm)

    # --- Process auditory-only data (column 2) ---
    _, cycle_length_ao = _detect_cycles(data_auditory_only['beat_seq_after'])
    d_ao = _slice_to_cycles(data_auditory_only, start_cycle_ao, end_cycle_ao, cycle_length_ao)
    time_ao = np.arange(len(d_ao['vest_pred_after']))

    # Y-axis limits
    vest_ylim_sm = _compute_ylim([
        d_sm['vest_seq_after'], d_sm['vest_pred_after'],
    ])
    beat_ylim_sm = _compute_ylim([
        d_sm['beat_seq_after'], d_sm['beat_pred_after'],
    ])
    vest_ylim_ao = _compute_ylim([
        d_ao['vest_seq_after'], d_ao['vest_pred_after'],
    ])
    beat_ylim_ao = _compute_ylim([
        d_ao['beat_seq_after'], d_ao['beat_pred_after'],
    ])

    # --- Row 2, Col 1: Sensorimotor vest after ---
    ax_sv = fig.add_subplot(gs[1, 0])
    ax_sv.plot(time_sm, d_sm['vest_seq_after'], color=actual_color, linestyle='--',
               linewidth=1.5, alpha=0.8, label='Vestibular Input')
    ax_sv.plot(time_sm, d_sm['vest_pred_after'], color=pred_color, linestyle='-',
               linewidth=1.5, alpha=0.9, label='Predicted Vestibular')
    ax_sv.set_ylabel('Vestibular Signal')
    ax_sv.legend(loc='upper right')
    ax_sv.grid(True, alpha=STYLES['grid_alpha'])
    ax_sv.set_ylim(vest_ylim_sm)
    ax_sv.set_xlim(0, max_points_sm - 1)
    _add_beat_vlines(ax_sv, time_sm, d_sm['beat_seq_after'])
    remove_chart_junk(ax_sv)

    # --- Row 3, Col 1: Sensorimotor aud after ---
    ax_sa = fig.add_subplot(gs[2, 0], sharex=ax_sv)
    ax_sa.plot(time_sm, d_sm['beat_seq_after'], color=actual_color, linestyle='--',
               linewidth=1.5, alpha=0.8, label='Auditory Input')
    ax_sa.plot(time_sm, d_sm['beat_pred_after'], color=pred_color, linestyle='-',
               linewidth=1.5, alpha=0.9, label='Predicted Auditory')
    ax_sa.set_xlabel('Time Steps')
    ax_sa.set_ylabel('Auditory Signal')
    ax_sa.legend(loc='upper right')
    ax_sa.grid(True, alpha=STYLES['grid_alpha'])
    ax_sa.set_ylim(beat_ylim_sm)
    _add_beat_vlines(ax_sa, time_sm, d_sm['beat_seq_after'])
    remove_chart_junk(ax_sa)

    # --- Row 2, Col 2: Auditory-only vest after ---
    ax_av = fig.add_subplot(gs[1, 1])
    ax_av.plot(time_ao, d_ao['vest_seq_after'], color=actual_color, linestyle=':',
               linewidth=1.5, alpha=0.5,
               label='Vestibular Training Signal (not provided)')
    ax_av.plot(time_ao, d_ao['vest_pred_after'], color=pred_color, linestyle='-',
               linewidth=1.5, alpha=1, label='Predicted Vestibular')
    ax_av.legend(loc='upper right')
    ax_av.grid(True, alpha=STYLES['grid_alpha'])
    ax_av.set_ylim(vest_ylim_ao)
    ax_av.set_xlim(0, len(time_ao) - 1)
    _add_beat_vlines(ax_av, time_ao, d_ao['beat_seq_after'])
    remove_chart_junk(ax_av)

    # --- Row 3, Col 2: Auditory-only aud after ---
    ax_aa = fig.add_subplot(gs[2, 1], sharex=ax_av)
    ax_aa.plot(time_ao, d_ao['beat_seq_after'], color=actual_color, linestyle='--',
               linewidth=1.5, alpha=1, label='Auditory Input')
    ax_aa.plot(time_ao, d_ao['beat_pred_after'], color=pred_color, linestyle='-',
               linewidth=1.5, alpha=1, label='Predicted Auditory')
    ax_aa.set_xlabel('Time Steps')
    ax_aa.legend(loc='upper right')
    ax_aa.grid(True, alpha=STYLES['grid_alpha'])
    ax_aa.set_ylim(beat_ylim_ao)
    _add_beat_vlines(ax_aa, time_ao, d_ao['beat_seq_after'])
    remove_chart_junk(ax_aa)

    # Column headers for the 2x2 panel area
    fig.text(0.3, 0.64, 'AFTER TRAINING', ha='center', fontsize=14, fontweight='bold')
    fig.text(
        0.74, 0.64, 'AFTER TRAINING \u2014 AUDITORY-ONLY',
        ha='center', fontsize=14, fontweight='bold',
    )

    return fig
