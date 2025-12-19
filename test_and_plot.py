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
    extension: str = "html"
) -> str:
    """
    Generate a standardized plot filename based on test parameters.

    Args:
        model_step: Step of the loaded model
        tempo: Tempo value used in testing
        prediction_timing: "before" or "after" inference optimization
        continuation: Whether continuation mode was used
        extension: File extension (html or png)

    Returns:
        Formatted filename string
    """
    continuation_suffix = "with_continuation" if continuation else "no_continuation"
    filename = (
        f"inference_plot_step_{model_step}_"
        f"tempo_{tempo}_"
        f"{prediction_timing}_inference_"
        f"{continuation_suffix}.{extension}"
    )
    return filename


def plot_inference_sequence_sensorimotor(
    vestibular_seq: torch.Tensor,
    beat_seq: torch.Tensor,
    vestibular_predicted_seq: np.ndarray,
    beat_predicted_seq: np.ndarray,
    e_v_seq: np.ndarray,
    e_b_seq: np.ndarray,
    save_path_html: Path,
    save_path_png: Optional[Path] = None
) -> None:
    """
    Plot inference sequences with Plotly in six subplots.

    Creates an interactive visualization showing actual vs predicted
    signals for both vestibular and beat modalities, plus prediction errors.

    Args:
        vestibular_seq: Actual vestibular input tensor of shape [n_steps] or [n_steps, 2]
        beat_seq: Actual beat input tensor of shape [n_steps]
        vestibular_predicted_seq: Predicted vestibular array of shape [n_steps] or [n_steps, 2]
        beat_predicted_seq: Predicted beat array of shape [n_steps]
        e_v_seq: Vestibular prediction error array
        e_b_seq: Beat prediction error array
        save_path_html: Path to save interactive HTML plot
        save_path_png: Optional path to save static PNG plot
    """
    # Convert to NumPy if still torch.Tensor
    if hasattr(vestibular_predicted_seq, 'numpy'):
        vestibular_predicted_seq = vestibular_predicted_seq.numpy()

    if hasattr(beat_predicted_seq, 'numpy'):
        beat_predicted_seq = beat_predicted_seq.numpy()

    # Time axis
    n_steps = len(beat_seq)
    t = list(range(n_steps))

    # Create subplots: 6 rows, 1 column
    fig = make_subplots(
        rows=6, cols=1,
        subplot_titles=[
            "Beat Sequence",
            "Predicted Beat",
            "Error of Predicted Beat",
            "Actual Vestibular Movement",
            "Predicted Vestibular Movement",
            "Error of Vestibular Movement"
        ],
        shared_xaxes=True,
        vertical_spacing=0.05
    )

    # --- Row 1: Beat Sequence ---
    fig.add_trace(
        go.Scatter(
            x=t,
            y=beat_seq.cpu().numpy(),
            mode='lines+markers',
            name='Beat',
            marker=dict(color='red')
        ),
        row=1, col=1
    )

    # --- Row 2: Predicted Beat Sequence ---
    fig.add_trace(
        go.Scatter(
            x=t,
            y=beat_predicted_seq,
            mode='lines+markers',
            name='Predicted Beat',
            marker=dict(color='purple')
        ),
        row=2, col=1
    )

    # --- Row 3: Error of Predicted Beat ---
    fig.add_trace(
        go.Scatter(
            x=t,
            y=e_b_seq,
            mode='lines+markers',
            name='Error of Predicted Beat',
            line=dict(dash='dash', color='purple'),
            marker=dict(color='purple')
        ),
        row=3, col=1
    )

    # Handle 1D or 2D vestibular sequences
    if vestibular_seq.ndim == 2:
        # --- Row 4: Actual Vestibular Movement (2 channels) ---
        fig.add_trace(
            go.Scatter(
                x=t,
                y=vestibular_seq[:, 0].cpu().numpy(),
                mode='lines',
                name='Vestibular ch0',
                line=dict(color='blue')
            ),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=vestibular_seq[:, 1].cpu().numpy(),
                mode='lines',
                name='Vestibular ch1',
                line=dict(color='cyan')
            ),
            row=4, col=1
        )

        # --- Row 5: Predicted Vestibular Movement (2 channels) ---
        fig.add_trace(
            go.Scatter(
                x=t,
                y=vestibular_predicted_seq[:, 0],
                mode='lines',
                name='Predicted ch0',
                line=dict(color='green')
            ),
            row=5, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=vestibular_predicted_seq[:, 1],
                mode='lines',
                name='Predicted ch1',
                line=dict(color='magenta')
            ),
            row=5, col=1
        )

        # --- Row 6: Error of Predicted Vestibular Movement (2 channels) ---
        fig.add_trace(
            go.Scatter(
                x=t,
                y=e_v_seq[:, 0],
                mode='lines',
                name='Error ch0',
                line=dict(dash='dash', color='green'),
            ),
            row=6, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=e_v_seq[:, 1],
                mode='lines',
                name='Error ch1',
                line=dict(dash='dash', color='magenta')
            ),
            row=6, col=1
        )

    else:
        # --- Row 4: Actual Vestibular Movement (1 channel) ---
        fig.add_trace(
            go.Scatter(
                x=t,
                y=vestibular_seq.cpu().numpy(),
                mode='lines',
                name='Vestibular',
                line=dict(color='blue')
            ),
            row=4, col=1
        )

        # --- Row 5: Predicted Vestibular Movement (1 channel) ---
        fig.add_trace(
            go.Scatter(
                x=t,
                y=vestibular_predicted_seq,
                mode='lines',
                name='Predicted Vestibular',
                line=dict(color='magenta')
            ),
            row=5, col=1
        )

        # --- Row 6: Error of Predicted Vestibular Movement (1 channel) ---
        fig.add_trace(
            go.Scatter(
                x=t,
                y=e_v_seq,
                mode='lines',
                name='Error Vestibular',
                line=dict(dash='dash', color='magenta'),
            ),
            row=6, col=1
        )

    # Add vertical lines at beat times on rows 2-6
    beat_times = [i for i, beat in enumerate(beat_seq) if beat.item() > 0]
    for b in beat_times:
        for row in range(2, 7):
            fig.add_vline(
                x=b,
                line_width=1,
                line_dash='dash',
                line_color='red',
                row=row,
                col=1
            )

    # Layout settings
    fig.update_layout(
        title='Inference Results',
        height=1500,
        width=3000,
        showlegend=True
    )

    # Update y-axis labels
    y_labels = [
        "Beat",
        "Pred Beat",
        "Beat Error",
        "Vestibular",
        "Pred Vestibular",
        "Vest Error"
    ]
    for i, label in enumerate(y_labels, start=1):
        fig.update_yaxes(
            title_text=label,
            row=i, col=1,
            automargin=True
        )

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
    continuation_start_fraction: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Run inference on a sequence and collect predictions.

    Args:
        network: Trained SensorimotorPCRNN network
        vestibular_seq: Vestibular input sequence
        beat_seq: Beat input sequence
        continuation: Whether to enable continuation mode
        continuation_start_fraction: Fraction of sequence after which continuation starts

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

    for step, (vestibular, beat) in enumerate(zip(vestibular_seq, beat_seq)):
        continuation_flag = continuation and (step > continuation_start)

        result = network.timestep_inference(
            vestibular_input=vestibular,
            beat_input=beat,
            continuation=continuation_flag
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
    logger: ExperimentLogger
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

    # Load saved model
    checkpoint = torch.load(exp_dir / f'model_step_{model_step}.pt')
    network.load_state_dict(checkpoint['model_state_dict'])

    # Log testing start
    logger.log_testing_start(
        model_step=model_step,
        tempo=tempo,
        continuation=continuation,
        prediction_timing=prediction_timing
    )

    # Generate test sequences
    vestibular_seq, beat_seq = generate_input_sequences(
        tempo=tempo,
        dt=config['experiment']['dt'],
        duration=config['testing']['test_duration'],
        vestibular_size=net_config['vestibular_size'],
        mode=config["experiment"]["mode"],
    )

    # Run inference
    inference_results = run_inference(
        network=network,
        vestibular_seq=vestibular_seq,
        beat_seq=beat_seq,
        continuation=continuation
    )

    # Calculate errors
    errors = calculate_test_errors(
        vestibular_seq=vestibular_seq,
        beat_seq=beat_seq,
        inference_results=inference_results,
        use_hierarchy=use_hierarchy
    )

    # Generate filenames
    html_filename = generate_plot_filename(
        model_step=model_step,
        tempo=tempo,
        prediction_timing=prediction_timing,
        continuation=continuation,
        extension="html"
    )
    png_filename = generate_plot_filename(
        model_step=model_step,
        tempo=tempo,
        prediction_timing=prediction_timing,
        continuation=continuation,
        extension="png"
    )

    plot_path_html = plots_dir / html_filename
    plot_path_png = plots_dir / png_filename

    # Plot and save results
    plot_inference_sequence_sensorimotor(
        vestibular_seq=vestibular_seq,
        beat_seq=beat_seq,
        vestibular_predicted_seq=inference_results['vest_pred'],
        beat_predicted_seq=inference_results['beat_pred'],
        e_v_seq=inference_results['e_v'],
        e_b_seq=inference_results['e_b'],
        save_path_html=plot_path_html,
        save_path_png=plot_path_png,
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
    logger.info("=" * 60)

    # Get tempos to test
    tempo_values = get_tempo_values(config)
    logger.info(f"Testing tempos: {tempo_values}")

    # Run tests for each tempo
    mode = config['experiment']['mode']

    if mode in {'sensorimotor', 'doublebeat'}:
        for tempo in tempo_values:
            test_sensorimotor_model(
                config_path=config_path,
                model_step=args.model_step,
                tempo=tempo,
                continuation=args.continuation,
                prediction_timing=args.prediction_timing,
                logger=logger
            )
    else:
        raise ValueError(f"Unknown experiment mode: {mode}")

    logger.info("=" * 60)
    logger.info("TESTING COMPLETED")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_and_plot()
