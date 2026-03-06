"""
Plot sample training inputs for a given config file.

If the config has random behavior (tempo range or uncorrelated mode),
plots multiple samples to show the variability. Otherwise plots a single sample.

Usage:
    python plot_inputs.py --config configs/config_sensorimotor.yaml
    python plot_inputs.py --config configs/config_sensorimotor.yaml --n_samples 5
    python plot_inputs.py --config configs/config_sensorimotor.yaml --output inputs.png
"""

import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt

from utils import generate_input_sequences, get_tempo_values, sample_tempo


def _get_tempos(config):
    """Get tempo list, handling both old (float) and new (dict) tempo formats."""
    tempo_cfg = config['experiment']['tempo']
    if isinstance(tempo_cfg, (int, float)):
        return [float(tempo_cfg)]
    return get_tempo_values(config)


def _sample_tempo(config):
    """Sample a tempo, handling both old (float) and new (dict) tempo formats."""
    tempo_cfg = config['experiment']['tempo']
    if isinstance(tempo_cfg, (int, float)):
        return float(tempo_cfg)
    return sample_tempo(config)


def has_randomness(config):
    """Check whether the config produces variable inputs across rounds."""
    exp = config['experiment']
    mode = exp['mode']

    # Multiple tempos means each round can get a different tempo
    tempos = _get_tempos(config)
    if len(tempos) > 1:
        return True

    # Uncorrelated mode generates random beat sequences
    if mode == 'uncorrelated':
        return True

    return False


def generate_sample(config, rng=None):
    """Generate one sample of inputs, returning (time, vest, beat, tempo) arrays."""
    tempo = _sample_tempo(config)
    result = generate_input_sequences(config=config, tempo=tempo, rng=rng)

    mode = config['experiment']['mode']
    dt = config['experiment']['dt']

    if mode == 'beat':
        beat = result.numpy()
        vest = None
    else:
        vest, beat = result
        vest = vest.numpy()
        beat = beat.numpy()

    n_steps = len(beat)
    time = np.arange(n_steps) * dt

    return time, vest, beat, tempo


def plot_samples(config, n_samples=1, output_path=None):
    """Plot input samples for the given config."""
    mode = config['experiment']['mode']
    is_random = has_randomness(config)

    if not is_random:
        n_samples = 1

    # Generate samples
    samples = []
    for i in range(n_samples):
        rng = np.random.default_rng(seed=i)
        samples.append(generate_sample(config, rng=rng))

    has_vest = samples[0][1] is not None

    if has_vest:
        fig, (ax_vest, ax_beat) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    else:
        fig, ax_beat = plt.subplots(1, 1, figsize=(14, 4))
        ax_vest = None

    # Color map for multiple samples
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_samples, 10)))

    for i, (time, vest, beat, tempo) in enumerate(samples):
        label = f'tempo={tempo:.3f}' if n_samples > 1 else None
        color = colors[i % len(colors)]
        alpha = 0.8 if n_samples <= 3 else 0.5

        if ax_vest is not None and vest is not None:
            ax_vest.plot(time, vest, color=color, alpha=alpha, linewidth=1.2,
                         label=label)

        ax_beat.plot(time, beat, color=color, alpha=alpha, linewidth=1.2,
                     label=label)

    # Labels and titles
    title_parts = [f'Mode: {mode}']
    zero_mean = config['experiment'].get('zero_mean_beat', False)
    if zero_mean:
        title_parts.append('zero_mean_beat=True')
    if is_random:
        title_parts.append(f'{n_samples} sample{"s" if n_samples > 1 else ""}')
    else:
        title_parts.append('deterministic (single sample)')
    fig.suptitle(' | '.join(title_parts), fontsize=14, fontweight='bold')

    if ax_vest is not None:
        ax_vest.set_ylabel('Vestibular')
        ax_vest.grid(True, alpha=0.3)
        if n_samples > 1:
            ax_vest.legend(loc='upper right', fontsize=8)

    ax_beat.set_xlabel('Time (s)')
    ax_beat.set_ylabel('Auditory')
    ax_beat.grid(True, alpha=0.3)
    if n_samples > 1:
        ax_beat.legend(loc='upper right', fontsize=8)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved to {output_path}")
    else:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Plot sample training inputs for a config file'
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to experiment config.yaml',
    )
    parser.add_argument(
        '--n_samples', type=int, default=5,
        help='Number of samples to plot (ignored if inputs are deterministic)',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output file path (e.g. inputs.png). If not set, shows interactive plot.',
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    plot_samples(config, n_samples=args.n_samples, output_path=args.output)


if __name__ == '__main__':
    main()
