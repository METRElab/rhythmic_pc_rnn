"""
Testing and visualization script for trained hierarchical sensorimotor networks.

Loads a trained model and generates inference plots showing predictions
versus actual inputs for both vestibular and beat signals.
"""

import argparse
import yaml
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from network import SensorimotorPCRNN
from utils import generate_input_sequences, get_tempo_values
from logger import create_logger, ExperimentLogger


def generate_plot_filename(
    model_step: int,
    tempo: float,
    prediction_timing: str,
    continuation: bool,
    auditory_only: bool = False,
    extension: str = "html"
) -> str:
    """
    Generate a compact plot filename based on test parameters.

    Format: s{step}_t{tempo}_{timing}[_cont][_ao].{ext}

    Args:
        model_step: Step of the loaded model
        tempo: Tempo value used in testing
        prediction_timing: "before" or "after" inference optimization
        continuation: Whether continuation mode was used
        auditory_only: Whether auditory-only mode was used
        extension: File extension (html or png)

    Returns:
        Formatted filename string
    """
    parts = [f"s{model_step}", f"t{tempo}", prediction_timing]
    if continuation:
        parts.append("cont")
    if auditory_only:
        parts.append("ao")
    return "_".join(parts) + f".{extension}"


def plot_inference_sequence_sensorimotor(
    vestibular_seq: torch.Tensor,
    beat_seq: torch.Tensor,
    vestibular_predicted_seq: np.ndarray,
    beat_predicted_seq: np.ndarray,
    e_v_seq: np.ndarray,
    e_b_seq: np.ndarray,
    save_path_html: Path,
    save_path_png: Optional[Path] = None,
    annotation_text: Optional[str] = None,
) -> None:
    """
    Plot inference sequences with Plotly in four subplots.

    Creates an interactive visualization showing actual vs predicted
    signals for both vestibular and beat modalities.

    Args:
        vestibular_seq: Actual vestibular input tensor of shape [n_steps] or [n_steps, 2]
        beat_seq: Actual beat input tensor of shape [n_steps]
        vestibular_predicted_seq: Predicted vestibular array of shape [n_steps] or [n_steps, 2]
        beat_predicted_seq: Predicted beat array of shape [n_steps]
        e_v_seq: Vestibular prediction error array (unused, kept for API compatibility)
        e_b_seq: Beat prediction error array (unused, kept for API compatibility)
        save_path_html: Path to save interactive HTML plot
        save_path_png: Optional path to save static PNG plot
        annotation_text: Optional text to display at the bottom of the figure
            (e.g. input parameters, notes)
    """
    from scipy.signal import find_peaks

    # Convert to NumPy if still torch.Tensor
    if hasattr(vestibular_predicted_seq, "numpy"):
        vestibular_predicted_seq = vestibular_predicted_seq.numpy()

    if hasattr(beat_predicted_seq, "numpy"):
        beat_predicted_seq = beat_predicted_seq.numpy()

    # Time axis
    n_steps = len(beat_seq)
    t = list(range(n_steps))

    # Create subplots: 4 rows, 1 column (removed error plots)
    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=[
            "Beat Sequence",
            "Predicted Beat",
            "Actual Vestibular Movement",
            "Predicted Vestibular Movement",
        ],
        shared_xaxes=True,
        vertical_spacing=0.08,
    )

    # --- Row 1: Beat Sequence ---
    fig.add_trace(
        go.Scatter(
            x=t,
            y=beat_seq.cpu().numpy(),
            mode="lines+markers",
            name="Beat",
            marker=dict(color="red"),
        ),
        row=1,
        col=1,
    )

    # --- Row 2: Predicted Beat Sequence ---
    fig.add_trace(
        go.Scatter(
            x=t,
            y=beat_predicted_seq,
            mode="lines+markers",
            name="Predicted Beat",
            marker=dict(color="purple"),
        ),
        row=2,
        col=1,
    )

    # Find vestibular peaks for vertical lines
    vestibular_np = vestibular_seq.cpu().numpy()
    if vestibular_np.ndim == 2:
        # Use first channel for peak detection
        vestibular_for_peaks = vestibular_np[:, 0]
    else:
        vestibular_for_peaks = vestibular_np

    # Find peaks in the vestibular signal
    vestibular_peaks, _ = find_peaks(vestibular_for_peaks, height=0.1)

    # Handle 1D or 2D vestibular sequences
    if vestibular_seq.ndim == 2:
        # --- Row 3: Actual Vestibular Movement (2 channels) ---
        fig.add_trace(
            go.Scatter(
                x=t,
                y=vestibular_seq[:, 0].cpu().numpy(),
                mode="lines",
                name="Vestibular ch0",
                line=dict(color="blue"),
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=vestibular_seq[:, 1].cpu().numpy(),
                mode="lines",
                name="Vestibular ch1",
                line=dict(color="cyan"),
            ),
            row=3,
            col=1,
        )

        # --- Row 4: Predicted Vestibular Movement (2 channels) ---
        fig.add_trace(
            go.Scatter(
                x=t,
                y=vestibular_predicted_seq[:, 0],
                mode="lines",
                name="Predicted ch0",
                line=dict(color="green"),
            ),
            row=4,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=vestibular_predicted_seq[:, 1],
                mode="lines",
                name="Predicted ch1",
                line=dict(color="magenta"),
            ),
            row=4,
            col=1,
        )

    else:
        # --- Row 3: Actual Vestibular Movement (1 channel) ---
        fig.add_trace(
            go.Scatter(
                x=t,
                y=vestibular_seq.cpu().numpy(),
                mode="lines",
                name="Vestibular",
                line=dict(color="blue"),
            ),
            row=3,
            col=1,
        )

        # --- Row 4: Predicted Vestibular Movement (1 channel) ---
        fig.add_trace(
            go.Scatter(
                x=t,
                y=vestibular_predicted_seq,
                mode="lines",
                name="Predicted Vestibular",
                line=dict(color="magenta"),
            ),
            row=4,
            col=1,
        )

    # Add vertical lines at beat times on row 2 (predicted beat)
    beat_times = [i for i, beat in enumerate(beat_seq) if beat.item() > 0]
    for b in beat_times:
        fig.add_vline(
            x=b, line_width=1, line_dash="dash", line_color="red", row=2, col=1
        )

    # Add vertical lines at vestibular peaks on rows 3-4
    for peak in vestibular_peaks:
        for row in [3, 4]:
            fig.add_vline(
                x=peak,
                line_width=1,
                line_dash="dash",
                line_color="blue",
                row=row,
                col=1,
            )

    # Layout settings
    layout_kwargs = dict(
        title="Inference Results", height=1000, width=3000, showlegend=True
    )

    # Add bottom annotation with parameters / notes
    if annotation_text:
        layout_kwargs["height"] = 1100  # extra space for annotation
        layout_kwargs["margin"] = dict(b=120)
        layout_kwargs["annotations"] = [
            dict(
                text=annotation_text,
                xref="paper", yref="paper",
                x=0.0, y=-0.08,
                showarrow=False,
                font=dict(size=12, color="gray"),
                align="left",
                xanchor="left", yanchor="top",
            )
        ]

    fig.update_layout(**layout_kwargs)

    # Update y-axis labels
    y_labels = [
        "Beat",
        "Pred Beat",
        "Vestibular",
        "Pred Vestibular",
    ]
    for i, label in enumerate(y_labels, start=1):
        fig.update_yaxes(title_text=label, row=i, col=1, automargin=True)

    # Save interactive HTML
    fig.write_html(str(save_path_html), auto_open=False)

    # Optionally save high-res static PNG
    if save_path_png is not None:
        fig.write_image(str(save_path_png), scale=3)


def run_inference(
    network: SensorimotorPCRNN,
    vestibular_seq: torch.Tensor,
    beat_seq: torch.Tensor,
    continuation: bool,
    continuation_start_fraction: float = 0.5,
    prediction_timing: str = "after",
    auditory_only: bool = True,
    beat_baseline: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Run inference on a sequence and collect predictions.

    Args:
        network: Trained SensorimotorPCRNN network
        vestibular_seq: Vestibular input sequence
        beat_seq: Beat input sequence
        continuation: Whether to enable continuation mode
        continuation_start_fraction: Fraction of sequence after which continuation starts
        prediction_timing: "before" or "after" inference optimization
        auditory_only: If True, vestibular is NOT given to the inference loop
        beat_baseline: Baseline value for auditory input during continuation
            (0.0 for standard mode, negative flat value for zero_mean_beat mode)

    Returns:
        Dictionary containing:
            - vest_pred: Vestibular predictions
            - beat_pred: Beat predictions
            - e_v: Vestibular errors
            - e_b: Beat errors
            - e_x: Associative layer errors
            - e_H: Higher layer errors (if hierarchy enabled)
    """
    network.reset_states()

    vest_pred_list = []
    beat_pred_list = []
    e_v_list = []
    e_b_list = []
    e_x_list = []
    e_H_list = []

    n_steps = len(vestibular_seq)
    continuation_start = int(continuation_start_fraction * n_steps)
    use_hierarchy = network.use_hierarchy

    print(f"auditory only = {auditory_only}")
    for step, (vestibular, beat) in enumerate(zip(vestibular_seq, beat_seq)):
        continuation_flag = continuation and (step > continuation_start)

        result = network.timestep_inference(
            vestibular_input=vestibular,
            beat_input=beat,
            continuation=continuation_flag,
            prediction_timing=prediction_timing,
            auditory_only=auditory_only,
            beat_baseline=beat_baseline,
        )

        vest_pred_list.append(result['vest_pred'].squeeze().tolist())
        beat_pred_list.append(result['beat_pred'].squeeze().tolist())
        e_v_list.append(result['e_v'].squeeze().tolist())
        e_b_list.append(result['e_b'].squeeze().tolist())
        e_x_list.append(result['e_x'].squeeze().tolist())

        if use_hierarchy:
            e_H_list.append(result['e_H'].squeeze().tolist())

    output = {
        'vest_pred': np.array(vest_pred_list),
        'beat_pred': np.array(beat_pred_list),
        'e_v': np.array(e_v_list),
        'e_b': np.array(e_b_list),
        'e_x': np.array(e_x_list)
    }

    if use_hierarchy:
        output['e_H'] = np.array(e_H_list)

    return output


def calculate_test_errors(
    vestibular_seq: torch.Tensor,
    beat_seq: torch.Tensor,
    inference_results: Dict[str, np.ndarray],
    use_hierarchy: bool
) -> Dict[str, float]:
    """
    Calculate average prediction errors for test sequence.

    Args:
        vestibular_seq: Actual vestibular input
        beat_seq: Actual beat input
        inference_results: Dictionary from run_inference
        use_hierarchy: Whether hierarchy is enabled

    Returns:
        Dictionary of average errors
    """
    vestibular_np = vestibular_seq.cpu().numpy()
    beat_np = beat_seq.cpu().numpy()

    errors = {
        'vest_error': float(np.mean((vestibular_np - inference_results['vest_pred']) ** 2)),
        'beat_error': float(np.mean((beat_np - inference_results['beat_pred']) ** 2)),
        'x_error': float(np.mean(inference_results['e_x'] ** 2))
    }

    if use_hierarchy:
        errors['H_error'] = float(np.mean(inference_results['e_H'] ** 2))

    return errors


def test_sensorimotor_model(
    config_path: Path,
    model_step: int,
    tempo: float,
    continuation: bool,
    prediction_timing: str,
    logger: ExperimentLogger,
    n_beats: Optional[int] = None,
    wait_time: float = 0.0,
    note: Optional[str] = None,
) -> None:
    """
    Test a trained sensorimotor model and generate plots.

    Args:
        config_path: Path to experiment config file
        model_step: Which model checkpoint to load
        tempo: Tempo to use for testing
        continuation: Whether to enable continuation mode
        prediction_timing: "before" or "after" inference optimization
        logger: ExperimentLogger for logging test results
        n_beats: If set, zero out beat input after this many beats
        wait_time: Seconds of silence to prepend before input starts
        note: Optional free-text note to display on the plot
    """
    exp_dir = config_path.parent

    with open(config_path) as f:
        config = yaml.safe_load(f)

    net_config = config['network']
    use_hierarchy = net_config['higher_size'] > 0

    # Create plots directory
    plots_dir = exp_dir / 'inference_plots'
    plots_dir.mkdir(exist_ok=True)

    # Initialize network
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
        random_seed=config['experiment']['random_seed']
    )

    # Load saved model (with backward-compatible key remapping)
    checkpoint = torch.load(exp_dir / f'model_step_{model_step}.pt', weights_only=False)
    _KEY_MAP = {
        'Wrec': 'W_rec',
        'W_vestibular': 'W_v',
        'W_beat': 'W_b',
    }
    state_dict = checkpoint['model_state_dict']
    state_dict = {_KEY_MAP.get(k, k): v for k, v in state_dict.items()}
    network.load_state_dict(state_dict, strict=False)

    # Log testing start
    logger.log_testing_start(
        model_step=model_step,
        tempo=tempo,
        continuation=continuation,
        prediction_timing=prediction_timing
    )

    # Generate test sequences (use a seeded rng for reproducibility)
    test_rng = np.random.default_rng(config['experiment'].get('random_seed', 42))
    result = generate_input_sequences(
        tempo=tempo,
        config=config,
        rng=test_rng,
    )

    mode = config['experiment']['mode']
    if mode == 'beat':
        beat_seq = result
        vestibular_seq = torch.zeros_like(beat_seq)
    else:
        vestibular_seq, beat_seq = result

    dt = config['experiment']['dt']

    # Compute beat baseline for zero-mean mode
    zero_mean_beat = config['experiment'].get('zero_mean_beat', False)
    if zero_mean_beat:
        steps_per_period = round(tempo / dt)
        beat_baseline = -1.0 / steps_per_period
    else:
        beat_baseline = 0.0

    # Prepend silent wait time at the beginning
    if wait_time > 0:
        n_wait = int(wait_time / dt)
        wait_pad_v = torch.zeros(n_wait)
        wait_pad_b = torch.full((n_wait,), beat_baseline)
        vestibular_seq = torch.cat([wait_pad_v, vestibular_seq])
        beat_seq = torch.cat([wait_pad_b, beat_seq])

    # Set beat input to baseline after n_beats
    if n_beats is not None:
        beat_indices = torch.where(beat_seq > 0)[0]
        if len(beat_indices) >= n_beats:
            cutoff = beat_indices[n_beats - 1].item() + 1
            beat_seq[cutoff:] = beat_baseline

    # If continuation mode, prepare the beat sequence for plotting
    if continuation:
        continuation_start = int(0.5 * len(beat_seq))
        beat_seq_plot = beat_seq.clone()
        beat_seq_plot[continuation_start + 1:] = beat_baseline
    else:
        beat_seq_plot = beat_seq

    # Common annotation parts (auditory_only label added per condition)
    mode = config['experiment']['mode']
    base_annotation_parts = [
        f"config: {config_path.name}",
        f"mode: {mode}",
        f"model_step: {model_step}",
        f"tempo: {tempo}",
        f"prediction_timing: {prediction_timing}",
        f"continuation: {continuation}",
    ]
    if n_beats is not None:
        base_annotation_parts.append(f"n_beats: {n_beats}")
    if wait_time > 0:
        base_annotation_parts.append(f"wait_time: {wait_time}s")
    if zero_mean_beat:
        base_annotation_parts.append(f"zero_mean_beat: True (baseline={beat_baseline:.4f})")
    if note:
        base_annotation_parts.append(f"note: {note}")

    # Run inference and plot for both auditory_only conditions
    for auditory_only in [False, True]:
        ao_label = "auditory_only" if auditory_only else "sensorimotor"
        print(f"  Running inference ({ao_label})...")

        inference_results = run_inference(
            network=network,
            vestibular_seq=vestibular_seq,
            beat_seq=beat_seq,
            continuation=continuation,
            prediction_timing=prediction_timing,
            auditory_only=auditory_only,
            beat_baseline=beat_baseline,
        )

        errors = calculate_test_errors(
            vestibular_seq=vestibular_seq,
            beat_seq=beat_seq,
            inference_results=inference_results,
            use_hierarchy=use_hierarchy
        )

        # Generate filenames
        html_filename = generate_plot_filename(
            model_step=model_step, tempo=tempo,
            prediction_timing=prediction_timing,
            continuation=continuation,
            auditory_only=auditory_only,
            extension="html",
        )
        png_filename = generate_plot_filename(
            model_step=model_step, tempo=tempo,
            prediction_timing=prediction_timing,
            continuation=continuation,
            auditory_only=auditory_only,
            extension="png",
        )

        plot_path_html = plots_dir / html_filename
        plot_path_png = plots_dir / png_filename

        # Build annotation text
        annotation_parts = base_annotation_parts + [f"auditory_only: {auditory_only}"]
        annotation_text = "  |  ".join(annotation_parts)

        # Plot and save results
        plot_inference_sequence_sensorimotor(
            vestibular_seq=vestibular_seq,
            beat_seq=beat_seq_plot,
            vestibular_predicted_seq=inference_results['vest_pred'],
            beat_predicted_seq=inference_results['beat_pred'],
            e_v_seq=inference_results['e_v'],
            e_b_seq=inference_results['e_b'],
            save_path_html=plot_path_html,
            save_path_png=plot_path_png,
            annotation_text=annotation_text,
        )

        # Log completion
        logger.log_testing_complete(
            vest_error=errors['vest_error'],
            beat_error=errors['beat_error'],
            x_error=errors['x_error'],
            H_error=errors.get('H_error'),
            plot_path=plot_path_html
        )


def test_and_plot() -> None:
    """
    Main testing entry point.

    Parses command line arguments and runs testing for all configured tempos.
    """
    parser = argparse.ArgumentParser(
        description="Test trained hierarchical sensorimotor network and generate plots"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to saved config file in experiment directory'
    )
    parser.add_argument(
        '--model_step',
        type=int,
        required=True,
        help='Which saved model step to load'
    )
    parser.add_argument(
        '--continuation',
        action='store_true',
        help='Enable continuation mode (no auditory input after 50%% of sequence)'
    )
    parser.add_argument(
        '--prediction_timing',
        type=str,
        choices=['before', 'after'],
        default='after',
        help='When to capture predictions: before or after inference optimization'
    )
    parser.add_argument(
        '--n_beats',
        type=int,
        default=None,
        help='Number of beats to provide; beat input becomes 0 after this many beats'
    )
    parser.add_argument(
        '--wait_time',
        type=float,
        default=0.0,
        help='Silent wait time (seconds) at the beginning before input starts'
    )
    parser.add_argument(
        '--note',
        type=str,
        default=None,
        help='Free-text note to display at the bottom of the plot'
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    exp_dir = config_path.parent

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize logger for testing
    log_level = config.get('logging', {}).get('level', 'INFO')
    logger = create_logger(
        exp_dir=exp_dir,
        log_filename="testing.log",
        level=log_level
    )

    use_hierarchy = config['network']['higher_size'] > 0

    logger.info("=" * 60)
    logger.info("TESTING STARTED")
    logger.info(f"Model step: {args.model_step}")
    logger.info(f"Continuation: {args.continuation}")
    logger.info(f"Prediction timing: {args.prediction_timing}")
    logger.info(f"Hierarchy enabled: {use_hierarchy}")
    if args.n_beats is not None:
        logger.info(f"N beats: {args.n_beats}")
    if args.wait_time > 0:
        logger.info(f"Wait time: {args.wait_time}s")
    logger.info("=" * 60)

    # Get tempos to test
    tempo_values = get_tempo_values(config)
    logger.info(f"Testing tempos: {tempo_values}")

    # Run tests for each tempo
    mode = config['experiment']['mode']

    if mode in {'sensorimotor', 'doublebeat', 'uncorrelated', 'beat'}:
        for tempo in tempo_values:
            test_sensorimotor_model(
                config_path=config_path,
                model_step=args.model_step,
                tempo=tempo,
                continuation=args.continuation,
                prediction_timing=args.prediction_timing,
                logger=logger,
                n_beats=args.n_beats,
                wait_time=args.wait_time,
                note=args.note,
            )
    else:
        raise ValueError(f"Unknown experiment mode: {mode}")

    logger.info("=" * 60)
    logger.info("TESTING COMPLETED")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_and_plot()
