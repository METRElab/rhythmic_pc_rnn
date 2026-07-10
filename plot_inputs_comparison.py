"""
Compare training inputs from up to 3 experiment configs side by side.

Produces a 2xN publication-ready figure (same style as the combined figure)
with vestibular (top row) and auditory (bottom row) for each config.

Usage:
    python plot_inputs_comparison.py \
        --configs experiments/sensorimotor/.../config.yaml \
                  experiments/uncorrelated/.../config.yaml \
                  experiments/doublebeat/.../config.yaml \
        --labels "Sensorimotor" "Uncorrelated" "Double-beat" \
        --tempo 0.5 \
        --output inputs_comparison.png

    # Use different cycle ranges
    python plot_inputs_comparison.py \
        --configs config1.yaml config2.yaml config3.yaml \
        --start-cycle 0 --end-cycle 8
"""

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import yaml

from utils import generate_input_sequences
from visualization.paper_figures import (
    _add_beat_vlines,
    _compute_ylim,
    _detect_cycles,
)
from visualization.style import COLORS, STYLES, apply_paper_style, remove_chart_junk

import matplotlib.pyplot as plt


def _generate_inputs(
    config_path: Path,
    tempo: float,
) -> dict:
    """Generate vestibular + auditory input arrays for one config."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = config['experiment'].get('random_seed', 42)
    mode = config['experiment']['mode']

    rng = np.random.default_rng(seed)
    result = generate_input_sequences(config=config, tempo=tempo, rng=rng)

    if mode == 'beat':
        beat_seq = result
        vest_seq = torch.zeros_like(beat_seq)
    elif mode == 'doublebeat':
        vest_seq, beat_seq = result
    else:
        vest_seq, beat_seq = result

    return {
        'vest': vest_seq.cpu().numpy(),
        'beat': beat_seq.cpu().numpy(),
        'mode': mode,
    }


def plot_inputs_comparison(
    configs: List[Path],
    labels: List[str],
    tempo: float,
    start_cycle: int = 0,
    end_cycle: int = 6,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """Build a 2xN comparison figure of training inputs."""
    n = len(configs)
    if figsize is None:
        figsize = (7 * n, 10)

    # Generate inputs for each config
    all_data = []
    for cfg in configs:
        all_data.append(_generate_inputs(cfg, tempo))

    # Detect cycle length from first config's beat sequence
    _, cycle_length = _detect_cycles(all_data[0]['beat'])
    start_idx = start_cycle * cycle_length
    end_idx = end_cycle * cycle_length

    # Slice all arrays
    sliced = []
    for d in all_data:
        sliced.append({
            'vest': d['vest'][start_idx:end_idx],
            'beat': d['beat'][start_idx:end_idx],
            'mode': d['mode'],
        })

    time_steps = np.arange(len(sliced[0]['vest']))

    # Compute per-column y-axis limits so each panel fits its own data
    vest_ylims = [_compute_ylim([s['vest']]) for s in sliced]
    beat_ylims = [_compute_ylim([s['beat']]) for s in sliced]

    actual_color = COLORS['actual']

    fig, axes = plt.subplots(2, n, figsize=figsize, squeeze=False)

    for col, (s, label) in enumerate(zip(sliced, labels)):
        ax_v = axes[0, col]
        ax_b = axes[1, col]

        # Vestibular
        ax_v.plot(time_steps, s['vest'], color=actual_color, linestyle='-',
                  linewidth=1.5, alpha=1, label='Vestibular Input')
        ax_v.set_title('Vestibular', fontweight='bold')
        if col == 0:
            ax_v.set_ylabel('Vestibular Signal')
        ax_v.legend(loc='upper right')
        ax_v.grid(True, alpha=STYLES['grid_alpha'])
        ax_v.set_ylim(vest_ylims[col])
        ax_v.set_xlim(0, len(time_steps) - 1)

        # Auditory
        ax_b.plot(time_steps, s['beat'], color=actual_color, linestyle='-',
                  linewidth=1.5, alpha=1, label='Auditory Input')
        ax_b.set_title('Auditory', fontweight='bold')
        ax_b.set_xlabel('Time Steps')
        if col == 0:
            ax_b.set_ylabel('Auditory Signal')
        ax_b.legend(loc='upper right')
        ax_b.grid(True, alpha=STYLES['grid_alpha'])
        ax_b.set_ylim(beat_ylims[col])
        ax_b.set_xlim(0, len(time_steps) - 1)

        # Beat vlines
        _add_beat_vlines(ax_v, time_steps, s['beat'])
        _add_beat_vlines(ax_b, time_steps, s['beat'])

        # Remove chart junk
        remove_chart_junk(ax_v)
        remove_chart_junk(ax_b)

        # Column header
        x_pos = (col + 0.5) / n
        fig.text(x_pos, 0.95, label, ha='center',
                 fontsize=20, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Compare training inputs from multiple experiment configs'
    )
    parser.add_argument(
        '--configs', type=str, nargs='+', required=True,
        help='Paths to experiment config.yaml files (up to 3)',
    )
    parser.add_argument(
        '--labels', type=str, nargs='+', default=None,
        help='Column labels for each config (default: config filenames)',
    )
    parser.add_argument(
        '--tempo', type=float, default=0.5,
        help='Tempo in seconds (default: 0.5)',
    )
    parser.add_argument(
        '--start-cycle', type=int, default=0,
        help='Starting cycle to display (default: 0)',
    )
    parser.add_argument(
        '--end-cycle', type=int, default=6,
        help='Ending cycle to display (default: 6)',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output file path (e.g. comparison.png). If not set, shows plot.',
    )
    args = parser.parse_args()

    # Canonical display names (case-insensitive lookup)
    _LABEL_MAP = {
        'sensorimotor': 'Correlated',
        'doublebeat': 'Pulse Training',
        'double-beat': 'Pulse Training',
        'double beat': 'Pulse Training',
        'uncorrelated': 'Random Auditory',
        'beat': 'Beat Only',
    }

    def _normalize_label(label: str) -> str:
        return _LABEL_MAP.get(label.lower(), label)

    configs = [Path(c) for c in args.configs]
    labels = args.labels
    if labels is None:
        # Derive labels from each config's experiment mode
        labels = []
        for c in configs:
            with open(c) as f:
                cfg = yaml.safe_load(f)
            mode = cfg['experiment']['mode']
            labels.append(_normalize_label(mode))
    else:
        labels = [_normalize_label(l) for l in labels]

    if len(labels) != len(configs):
        print(f"Error: {len(configs)} configs but {len(labels)} labels")
        return

    apply_paper_style()
    fig = plot_inputs_comparison(
        configs=configs,
        labels=labels,
        tempo=args.tempo,
        start_cycle=args.start_cycle,
        end_cycle=args.end_cycle,
    )

    if args.output:
        fig.savefig(args.output, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == '__main__':
    main()


# python plot_inputs_comparison.py \
#     --configs experiments/paper_revision/final_ones/sensorimotor_mean_vestibular_included/config.yaml experiments/paper_revision/final_ones/doublebeat_mean_vestibular_included/config.yaml experiments/paper_revision/final_ones/uncorrelated_mean_vestibular_included/config.yaml \
#     --labels "Correlated" "Pulse Training" "Random Auditory" \
#     --tempo 0.5 \
#     --start-cycle 0 --end-cycle 6 \
#     --output inputs_comparison_2.png
# #
# python plot_inputs_comparison.py --configs experiments/paper_revision/final_ones/doublebeat_mean_vestibular_included/config.yaml experiments/paper_revision/final_ones/sensorimotor_mean_vestibular_included/config.yaml experiments/paper_revision/final_ones/uncorrelated_mean_vestibular_included/config.yaml --labels "Sensorimotor" "Uncorrelated" "Double-beat" --tempo 0.5 --start-cycle 0 --end-cycle 6 --output inputs_comparison.png