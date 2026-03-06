"""
Data extraction pipeline for paper figures.

Replaces the manual workflow of running inference in notebooks and pickling
arrays by hand. Takes experiment directories + model steps and produces
standardized .npz data files.

Usage:
    from visualization.generate_figure_data import (
        load_model, run_condition_inference, generate_before_after_data,
        extract_tensorboard_learning_curve, save_figure_data, load_figure_data,
        generate_all_figure_data,
    )
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

# Add project root to path so we can import project modules
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from network import SensorimotorPCRNN
from test_and_plot import run_inference
from utils import generate_input_sequences, get_tempo_values


def load_model(
    config_path: Path, model_step: int
) -> Tuple[SensorimotorPCRNN, Dict[str, Any]]:
    """
    Load a trained model from an experiment directory.

    Args:
        config_path: Path to the experiment's config.yaml
        model_step: Which checkpoint step to load

    Returns:
        (network, config) tuple
    """
    config_path = Path(config_path)
    exp_dir = config_path.parent

    with open(config_path) as f:
        config = yaml.safe_load(f)

    net_config = config['network']

    network = SensorimotorPCRNN(
        higher_size=net_config['higher_size'],
        associative_size=net_config['associative_size'],
        vestibular_size=net_config['vestibular_size'],
        alpha_H=net_config['alpha_H'],
        alpha_x=net_config['alpha_x'],
        inference_learning_rate_H=net_config['inference_learning_rate_H'],
        inference_learning_rate_x=net_config['inference_learning_rate_x'],
        weight_learning_rate_H=net_config['weight_learning_rate_H'],
        weight_learning_rate_x=net_config['weight_learning_rate_x'],
        n_inference_steps=net_config['n_inference_steps'],
        random_seed=config['experiment']['random_seed'],
    )

    checkpoint_path = exp_dir / f'model_step_{model_step}.pt'
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # Remap old checkpoint key names to current ones
    _KEY_MAP = {
        'Wrec': 'W_rec',
        'W_vestibular': 'W_v',
        'W_beat': 'W_b',
    }
    state_dict = checkpoint['model_state_dict']
    state_dict = {_KEY_MAP.get(k, k): v for k, v in state_dict.items()}

    network.load_state_dict(state_dict, strict=False)

    return network, config


def run_condition_inference(
    network: SensorimotorPCRNN,
    config: Dict[str, Any],
    tempo: float,
    continuation: bool = False,
    prediction_timing: str = "after",
    auditory_only: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Run inference for a single condition and return raw arrays.

    Generates input sequences from config, runs the network, and returns
    both the input sequences and the network's predictions.

    Args:
        network: Trained SensorimotorPCRNN
        config: Experiment config dict
        tempo: Tempo in seconds
        continuation: If True, zero out all input after 50% of sequence
        prediction_timing: "before" or "after" inference optimization
        auditory_only: If True, vestibular is NOT given to inference loop

    Returns:
        Dict with keys: vest_pred, beat_pred, vest_seq, beat_seq,
        e_v, e_b, e_x, and optionally e_H
    """
    result = generate_input_sequences(config=config, tempo=tempo)

    mode = config['experiment']['mode']
    dt = config['experiment']['dt']
    if mode == 'beat':
        beat_seq = result
        # Create zero vestibular for beat-only mode
        vestibular_seq = torch.zeros_like(beat_seq)
    else:
        vestibular_seq, beat_seq = result

    # Compute beat baseline for zero-mean mode
    zero_mean_beat = config['experiment'].get('zero_mean_beat', False)
    if zero_mean_beat:
        steps_per_period = round(tempo / dt)
        beat_baseline = -1.0 / steps_per_period
    else:
        beat_baseline = 0.0

    inference_results = run_inference(
        network=network,
        vestibular_seq=vestibular_seq,
        beat_seq=beat_seq,
        continuation=continuation,
        prediction_timing=prediction_timing,
        auditory_only=auditory_only,
        beat_baseline=beat_baseline,
    )

    # Combine input sequences with inference results
    output = {
        'vest_pred': inference_results['vest_pred'],
        'beat_pred': inference_results['beat_pred'],
        'vest_seq': vestibular_seq.cpu().numpy(),
        'beat_seq': beat_seq.cpu().numpy(),
        'e_v': inference_results['e_v'],
        'e_b': inference_results['e_b'],
        'e_x': inference_results['e_x'],
    }

    if 'e_H' in inference_results:
        output['e_H'] = inference_results['e_H']

    return output


def generate_before_after_data(
    config_path: Path,
    before_step: int,
    after_step: int,
    tempo: float,
    continuation: bool = False,
    prediction_timing: str = "after",
    auditory_only: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Generate before/after training comparison data.

    Loads the model at two different steps (before and after training),
    runs inference on both, and returns the combined arrays.

    Args:
        config_path: Path to experiment config.yaml
        before_step: Model step for "before training" (typically 0)
        after_step: Model step for "after training" (final checkpoint)
        tempo: Tempo in seconds
        continuation: If True, use continuation mode
        prediction_timing: "before" or "after" inference optimization
        auditory_only: If True, vestibular is NOT given to inference loop

    Returns:
        Dict with keys like vest_pred_before, vest_pred_after,
        vest_seq_before, vest_seq_after, beat_pred_before, etc.
    """
    # Run inference at "before" step
    network_before, config = load_model(config_path, before_step)
    before_data = run_condition_inference(
        network=network_before,
        config=config,
        tempo=tempo,
        continuation=continuation,
        prediction_timing=prediction_timing,
        auditory_only=auditory_only,
    )

    # Run inference at "after" step
    network_after, _ = load_model(config_path, after_step)
    after_data = run_condition_inference(
        network=network_after,
        config=config,
        tempo=tempo,
        continuation=continuation,
        prediction_timing=prediction_timing,
        auditory_only=auditory_only,
    )

    # Combine with before/after suffixes
    combined = {}
    for key, value in before_data.items():
        combined[f'{key}_before'] = value
    for key, value in after_data.items():
        combined[f'{key}_after'] = value

    return combined


def extract_tensorboard_learning_curve(
    exp_dir: Path,
    metrics: Optional[List[str]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract learning curve data from TensorBoard event files.

    Auto-discovers the event file in exp_dir/logs/.

    Args:
        exp_dir: Experiment directory containing logs/ subfolder
        metrics: List of metric names to extract (default: all available)

    Returns:
        Dict mapping metric names to {'steps': array, 'values': array}
    """
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    exp_dir = Path(exp_dir)
    log_dir = exp_dir / 'logs'

    # Find event file(s)
    event_files = list(log_dir.glob('events.out.tfevents.*'))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {log_dir}")

    # Use the most recent event file
    event_file = sorted(event_files)[-1]

    event_acc = EventAccumulator(str(event_file))
    event_acc.Reload()

    available_tags = event_acc.Tags().get('scalars', [])
    if metrics is None:
        metrics = available_tags

    data = {}
    for metric in metrics:
        if metric not in available_tags:
            print(f"Warning: metric '{metric}' not found. Available: {available_tags}")
            continue

        scalar_events = event_acc.Scalars(metric)
        steps = np.array([e.step for e in scalar_events])
        values = np.array([e.value for e in scalar_events])
        data[metric] = {'steps': steps, 'values': values}

    return data


def save_figure_data(
    data: Dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Save figure data as compressed .npz file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), **data)


def load_figure_data(path: Path) -> Dict[str, np.ndarray]:
    """Load previously saved .npz figure data."""
    loaded = np.load(str(path))
    return dict(loaded)


def _extract_learning_curve_data(exp_dir: Path) -> Dict[str, np.ndarray]:
    """
    Extract learning curve data, trying inference error metrics first,
    then falling back to legacy metric names.

    Returns dict with keys: beat_error_steps, beat_error_values,
    vest_error_steps, vest_error_values.
    """
    # Try newer metric names first
    lc = extract_tensorboard_learning_curve(
        exp_dir, metrics=['beat_inference_error', 'vest_inference_error']
    )
    if 'beat_inference_error' in lc and 'vest_inference_error' in lc:
        beat_key = 'beat_inference_error'
        vest_key = 'vest_inference_error'
    else:
        # Fall back to legacy metric names
        lc = extract_tensorboard_learning_curve(
            exp_dir, metrics=['beat_error', 'vest_error']
        )
        beat_key = 'beat_error'
        vest_key = 'vest_error'

    return {
        'beat_error_steps': lc[beat_key]['steps'],
        'beat_error_values': lc[beat_key]['values'],
        'vest_error_steps': lc[vest_key]['steps'],
        'vest_error_values': lc[vest_key]['values'],
    }


def generate_all_figure_data(
    sensorimotor_config: Path,
    before_step: int,
    after_step_sensorimotor: int,
    tempo: float,
    output_dir: Path,
    doublebeat_config: Optional[Path] = None,
    after_step_doublebeat: Optional[int] = None,
    prediction_timing: str = "after",
) -> None:
    """
    Generate all data needed for all paper figures.

    Produces .npz files in output_dir/data/ for each figure condition,
    plus a metadata.yaml recording the parameters used.

    Args:
        sensorimotor_config: Path to sensorimotor experiment config.yaml
        before_step: Model step for "before training" (typically 0)
        after_step_sensorimotor: Final model step for sensorimotor experiment
        tempo: Tempo in seconds
        output_dir: Where to save output files
        doublebeat_config: Path to doublebeat experiment config.yaml (optional)
        after_step_doublebeat: Final model step for doublebeat experiment (optional)
        prediction_timing: "before" or "after" inference optimization
    """
    output_dir = Path(output_dir)
    data_dir = output_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    has_doublebeat = doublebeat_config is not None and after_step_doublebeat is not None
    total_steps = 7 if has_doublebeat else 5

    print("=== Generating paper figure data ===")

    # Figure 1b: Sensorimotor before/after
    print(f"\n[1/{total_steps}] Sensorimotor before/after (Figure 1b)...")
    sm_data = generate_before_after_data(
        config_path=sensorimotor_config,
        before_step=before_step,
        after_step=after_step_sensorimotor,
        tempo=tempo,
        prediction_timing=prediction_timing,
    )
    save_figure_data(sm_data, data_dir / 'sensorimotor_before_after.npz')

    # Figure 2: Auditory-only before/after (same sensorimotor model,
    # but vestibular is NOT given to the inference loop)
    print(f"[2/{total_steps}] Auditory-only before/after (Figure 2)...")
    ao_data = generate_before_after_data(
        config_path=sensorimotor_config,
        before_step=before_step,
        after_step=after_step_sensorimotor,
        tempo=tempo,
        prediction_timing=prediction_timing,
        auditory_only=True,
    )
    save_figure_data(ao_data, data_dir / 'auditory_only_before_after.npz')

    # Figure 3: Continuation (sensorimotor model, auditory + continuation)
    print(f"[3/{total_steps}] Continuation condition (Figure 3)...")
    network_after, config = load_model(sensorimotor_config, after_step_sensorimotor)
    continuation_data = run_condition_inference(
        network=network_after,
        config=config,
        tempo=tempo,
        continuation=True,
        prediction_timing=prediction_timing,
    )
    save_figure_data(continuation_data, data_dir / 'continuation.npz')

    # Figure 3b: Continuation auditory-only (same model, vestibular withheld)
    print(f"[4/{total_steps}] Continuation auditory-only (Figure 3b)...")
    continuation_ao_data = run_condition_inference(
        network=network_after,
        config=config,
        tempo=tempo,
        continuation=True,
        prediction_timing=prediction_timing,
        auditory_only=True,
    )
    save_figure_data(continuation_ao_data, data_dir / 'continuation_auditory_only.npz')

    # Figure 1a: Learning curve (sensorimotor)
    print(f"[5/{total_steps}] Learning curve - sensorimotor (Figure 1a)...")
    sm_exp_dir = Path(sensorimotor_config).parent
    try:
        sm_lc = _extract_learning_curve_data(sm_exp_dir)
        save_figure_data(sm_lc, data_dir / 'learning_curve_sensorimotor.npz')
    except (FileNotFoundError, ImportError) as e:
        print(f"  Warning: Could not extract TensorBoard data: {e}")

    # Doublebeat figures (only if config provided)
    if has_doublebeat:
        # Figure 4: Double auditory before/after
        print(f"[6/{total_steps}] Double auditory before/after (Figure 4)...")
        db_data = generate_before_after_data(
            config_path=doublebeat_config,
            before_step=before_step,
            after_step=after_step_doublebeat,
            tempo=tempo,
            prediction_timing=prediction_timing,
        )
        save_figure_data(db_data, data_dir / 'double_auditory_before_after.npz')

        # Figure 4a: Learning curve (doublebeat)
        print(f"[7/{total_steps}] Learning curve - doublebeat (Figure 4a)...")
        db_exp_dir = Path(doublebeat_config).parent
        try:
            db_lc = _extract_learning_curve_data(db_exp_dir)
            save_figure_data(db_lc, data_dir / 'learning_curve_doublebeat.npz')
        except (FileNotFoundError, ImportError) as e:
            print(f"  Warning: Could not extract TensorBoard data: {e}")
    else:
        print("\nSkipping doublebeat figures (no doublebeat config provided)")

    # Save metadata
    metadata = {
        'sensorimotor_config': str(sensorimotor_config),
        'doublebeat_config': str(doublebeat_config) if doublebeat_config else None,
        'before_step': before_step,
        'after_step_sensorimotor': after_step_sensorimotor,
        'after_step_doublebeat': after_step_doublebeat,
        'tempo': tempo,
        'prediction_timing': prediction_timing,
    }
    with open(data_dir / 'metadata.yaml', 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)

    print(f"\nAll data saved to {data_dir}")
