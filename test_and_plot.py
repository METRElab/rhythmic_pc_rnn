import argparse
import yaml
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import numpy as np

from network import SensorimotorPCRNN, BeatPCRNN
from utils import generate_input_sequences


def plot_inference_sequence_sensorimotor(
        vestibular_seq, beat_seq,
        vestibular_predicted_seq, beat_predicted_seq,
        e_v_seq, e_b_seq,
        save_path_html, save_path_png=None
):
    """
    Plot sequences with Plotly in three subplots and
    add vertical lines at beat times for clarity.

    Parameters
    ----------
    beat_seq : torch.Tensor of shape [n_steps]
    vestibular_seq : torch.Tensor of shape [n_steps, 2]
                    (sine + cosine)
    vestibular_predicted_seq : np.array or torch.Tensor of shape [n_steps, 2]
                    (predicted sine + cosine)
    beat_predicted_seq : np.array or torch.Tensor of shape [n_steps]
    e_v_seq : error of vestibular
    e_b_seq : error of beat
    save_path_html : Path or str
        Where to save the interactive HTML plot.
    save_path_png : Path or str, optional
        Where to save a high-res static PNG plot (requires kaleido).
    """

    # Convert predicted_seq to NumPy if it's still a torch.Tensor
    if hasattr(vestibular_predicted_seq, 'numpy'):
        vestibular_predicted_seq = vestibular_predicted_seq.numpy()  # shape [n_steps, 2]

    if hasattr(beat_predicted_seq, 'numpy'):
        beat_predicted_seq = beat_predicted_seq.numpy()  # shape [n_steps]

    # Time axis
    n_steps = len(beat_seq)
    t = list(range(n_steps))

    # Create subplots: 6 rows, 1 column with larger row heights
    fig = make_subplots(
        rows=6, cols=1,
        subplot_titles=["Beat Sequence", "Predicted Beat", "Error of Predicted Beat",
                        "Actual Vestibular Movement", "Predicted Vestibular Movement", "Error of Vestibular Movement"],
        shared_xaxes=True,  # so x-zoom is shared
        vertical_spacing=0.05  # Reduced spacing to allow for larger plots
    )

    # --- Row 1: Beat Sequence ---
    fig.add_trace(
        go.Scatter(
            x=t,
            y=beat_seq.cpu().numpy(),  # use .cpu() if on GPU
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
            y=beat_predicted_seq,  # use .cpu() if on GPU
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
            y=e_b_seq,  # use .cpu() if on GPU
            mode='lines+markers',
            name='Error of Predicted Beat',
            line=dict(dash='dash', color='purple'),
            marker=dict(color='purple')
        ),
        row=3, col=1
    )

    if vestibular_seq.ndim == 2:
        # --- Row 4: Actual Vestibular Movement (2 channels) ---
        fig.add_trace(
            go.Scatter(
                x=t,
                y=vestibular_seq[:, 0].cpu().numpy(),
                mode='lines',
                name='Vestibular sin',
                line=dict(color='blue')
            ),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=vestibular_seq[:, 1].cpu().numpy(),
                mode='lines',
                name='Vestibular cos',
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
                name='Predicted sin',
                line=dict(color='green')
            ),
            row=5, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=vestibular_predicted_seq[:, 1],
                mode='lines',
                name='Predicted cos',
                line=dict(color='magenta')
            ),
            row=5, col=1
        )

        # --- Row 6: Error of Predicted Vestibular Movement (2 channels) ---
        fig.add_trace(
            go.Scatter(
                x=t,
                y=e_v_seq[:, 0].cpu().numpy(),
                mode='lines',
                name='Error of Predicted sin',
                line=dict(dash='dash', color='green'),
            ),
            row=6, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=e_v_seq[:, 1].cpu().numpy(),
                mode='--',
                name='Error of Predicted cos',
                line=dict(dash='dash', color='magenta')
            ),
            row=6, col=1
        )

    if vestibular_seq.ndim == 1:
        # --- Row 4: Actual Vestibular Movement (1 channel) ---
        fig.add_trace(
            go.Scatter(
                x=t,
                y=vestibular_seq.cpu().numpy(),
                mode='lines',
                name='Vestibular sin',
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
                name='Predicted sin',
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
                name='Error of Predicted sin',
                line=dict(dash='dash', color='magenta'),
            ),
            row=6, col=1
        )

    # Add vertical lines at beat times on rows 2, 3, and 4
    beat_times = [i for i, beat in enumerate(beat_seq) if beat.item() > 0]
    for b in beat_times:
        # add_vline row argument is the *domain* row index, not the subplot index
        # but Plotly 5+ allows `row` and `col` in add_vline:
        fig.add_vline(x=b, line_width=1, line_dash='dash', line_color='red', row=2, col=1)
        fig.add_vline(x=b, line_width=1, line_dash='dash', line_color='red', row=3, col=1)
        fig.add_vline(x=b, line_width=1, line_dash='dash', line_color='red', row=4, col=1)
        fig.add_vline(x=b, line_width=1, line_dash='dash', line_color='red', row=5, col=1)
        fig.add_vline(x=b, line_width=1, line_dash='dash', line_color='red', row=6, col=1)

    # Layout settings with increased height
    fig.update_layout(
        title='Inference Results',
        height=1500,  # Increased from 900 to make plots larger
        showlegend=True
    )

    # Additional subplot adjustments for larger row heights
    for i in range(1, 7):
        fig.update_yaxes(
            title_text=f"Row {i}",
            row=i, col=1,
            automargin=True  # Give more margin for better visibility
        )

    # Save interactive HTML (dynamic)
    fig.write_html(str(save_path_html), auto_open=False)
    print(f"Interactive Plotly figure saved to: {save_path_html}")

    # Optionally, save a high-res static PNG (requires kaleido).
    if save_path_png is not None:
        # scale=2 or higher for increased resolution
        fig.write_image(str(save_path_png), scale=3)
        print(f"High-resolution PNG saved to: {save_path_png}")


def plot_inference_sequence_beat(
        beat_seq,
        beat_predicted_seq,
        e_b_seq,
        save_path_html,
        save_path_png=None
):
    """
    Plot beat sequences with Plotly in three subplots and
    add vertical lines at beat times for clarity.

    Parameters
    ----------
    beat_seq : torch.Tensor of shape [n_steps]
    beat_predicted_seq : np.array or torch.Tensor of shape [n_steps]
    e_b_seq : error of beat
    save_path_html : Path or str
        Where to save the interactive HTML plot.
    save_path_png : Path or str, optional
        Where to save a high-res static PNG plot (requires kaleido).
    """

    # Convert predicted_seq to NumPy if it's still a torch.Tensor
    if hasattr(beat_predicted_seq, 'numpy'):
        beat_predicted_seq = beat_predicted_seq.numpy()  # shape [n_steps]

    # Time axis
    n_steps = len(beat_seq)
    t = list(range(n_steps))

    # Create subplots: 3 rows, 1 column
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=["Beat Sequence", "Predicted Beat", "Error of Predicted Beat"],
        shared_xaxes=True,  # so x-zoom is shared
        vertical_spacing=0.05  # Reduced spacing to allow for larger plots
    )

    # --- Row 1: Beat Sequence ---
    fig.add_trace(
        go.Scatter(
            x=t,
            y=beat_seq.cpu().numpy(),  # use .cpu() if on GPU
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
            y=beat_predicted_seq,  # use .cpu() if on GPU
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
            y=e_b_seq,  # use .cpu() if on GPU
            mode='lines+markers',
            name='Error of Predicted Beat',
            line=dict(dash='dash', color='purple'),
            marker=dict(color='purple')
        ),
        row=3, col=1
    )

    # Add vertical lines at beat times on all rows
    beat_times = [i for i, beat in enumerate(beat_seq) if beat.item() > 0]
    for b in beat_times:
        fig.add_vline(x=b, line_width=1, line_dash='dash', line_color='red', row=1, col=1)
        fig.add_vline(x=b, line_width=1, line_dash='dash', line_color='red', row=2, col=1)
        fig.add_vline(x=b, line_width=1, line_dash='dash', line_color='red', row=3, col=1)

    # Layout settings with increased height
    fig.update_layout(
        title='Beat Inference Results',
        height=800,
        showlegend=True
    )

    # Additional subplot adjustments for larger row heights
    for i in range(1, 4):
        fig.update_yaxes(
            title_text=f"Row {i}",
            row=i, col=1,
            automargin=True  # Give more margin for better visibility
        )

    # Save interactive HTML (dynamic)
    fig.write_html(str(save_path_html), auto_open=False)
    print(f"Interactive Plotly figure saved to: {save_path_html}")

    # Optionally, save a high-res static PNG (requires kaleido).
    if save_path_png is not None:
        # scale=2 or higher for increased resolution
        fig.write_image(str(save_path_png), scale=3)
        print(f"High-resolution PNG saved to: {save_path_png}")


def test_sensorimotor_model(config_path: Path, args):

    exp_dir = config_path.parent

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create plots directory
    plots_dir = exp_dir / 'inference_plots'
    plots_dir.mkdir(exist_ok=True)

    # Initialize network
    network = SensorimotorPCRNN(
        associative_size=config['network']['associative_size'],
        vestibular_size=config['network']['vestibular_size'],
        inference_learning_rate=config['network']['inference_learning_rate'],
        weight_learning_rate=config['network']['weight_learning_rate'],
        n_inference_steps=config['network']['n_inference_steps'],
        random_seed=config['experiment']['random_seed']
    )

    # Load saved model
    checkpoint = torch.load(exp_dir / f'model_step_{args.model_step}.pt')
    network.load_state_dict(checkpoint['model_state_dict'])

    # Generate test sequences
    vestibular_seq, beat_seq = generate_input_sequences(
        tempo=config['experiment']['tempo'],
        dt=config['experiment']['dt'],
        duration=config['testing']['test_duration'],
        vestibular_size=config['network']['vestibular_size'],
        mode=config["experiment"]["mode"],
    )

    # Run inference
    network.reset_states()
    vestibular_predicted_seq = []
    beat_predicted_seq = []
    e_v_seq = []
    e_b_seq = []

    for vestibular, beat in zip(vestibular_seq, beat_seq):
        vestibular_pred, beat_pred, e_v, e_b = network.timestep_inference(
            vestibular_input=vestibular,
            beat_input=beat
        )
        vestibular_predicted_seq.append(vestibular_pred.squeeze().tolist())
        beat_predicted_seq.append(beat_pred.squeeze().tolist())
        e_v_seq.append(e_v.squeeze().tolist())
        e_b_seq.append(e_b.squeeze().tolist())

        # predicted_seq.append(vestibular_pred.item())
    vestibular_predicted_seq = np.array(vestibular_predicted_seq)
    beat_predicted_seq = np.array(beat_predicted_seq)
    e_v_seq = np.array(e_v_seq)
    e_b_seq = np.array(e_b_seq)

    # Plot and save results
    # plot_path = plots_dir / f'inference_plot_step_{args.model_step}.png'
    plot_path_html = plots_dir / f'inference_plot_step_{args.model_step}.html'
    plot_path_png = plots_dir / f'inference_plot_step_{args.model_step}.png'
    plot_inference_sequence_sensorimotor(
        vestibular_seq,
        beat_seq,
        vestibular_predicted_seq,
        beat_predicted_seq,
        e_v_seq,
        e_b_seq,
        plot_path_html,
        plot_path_png,
    )

    print(f"Generated plot saved at: {plots_dir}")


def test_beat_model(config_path: Path, args):

    exp_dir = config_path.parent

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create plots directory
    plots_dir = exp_dir / 'inference_plots'
    plots_dir.mkdir(exist_ok=True)

    # Initialize network
    network = BeatPCRNN(
        associative_size=config['network']['associative_size'],
        inference_learning_rate=config['network']['inference_learning_rate'],
        weight_learning_rate=config['network']['weight_learning_rate'],
        n_inference_steps=config['network']['n_inference_steps'],
        random_seed=config['experiment']['random_seed']
    )

    # Load saved model
    checkpoint = torch.load(exp_dir / f'model_step_{args.model_step}.pt')
    network.load_state_dict(checkpoint['model_state_dict'])

    # Generate test sequence
    beat_seq = generate_input_sequences(
        tempo=config['experiment']['tempo'],
        dt=config['experiment']['dt'],
        duration=config['testing']['test_duration'],
        mode=config['experiment']['mode'],
    )

    # Run inference
    network.reset_states()
    beat_predicted_seq = []
    e_b_seq = []

    for i in range(len(beat_seq)):
        beat = beat_seq[i]
        beat_pred, e_b = network.timestep_inference(beat_input=beat)
        beat_predicted_seq.append(beat_pred.squeeze().tolist())
        e_b_seq.append(e_b.squeeze().tolist())

    beat_predicted_seq = np.array(beat_predicted_seq)
    e_b_seq = np.array(e_b_seq)

    # Plot and save results
    plot_path_html = plots_dir / f'beat_inference_plot_step_{args.model_step}.html'
    plot_path_png = plots_dir / f'beat_inference_plot_step_{args.model_step}.png'

    plot_inference_sequence_beat(beat_seq, beat_predicted_seq, e_b_seq, plot_path_html, plot_path_png)

    print(f"Generated plot saved at: {plots_dir}")


def test_and_plot():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to saved config file')
    parser.add_argument('--model_step', type=int, required=True, help='Which saved model step to load')
    args = parser.parse_args()

    print('here')
    # Load config
    config_path = Path(args.config)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if config['experiment']['mode'] == 'beat':
        test_beat_model(config_path, args)
    elif config['experiment']['mode'] == 'sensorimotor' or  config['experiment']['mode'] == 'doublebeat':
        test_sensorimotor_model(config_path, args)


if __name__ == "__main__":
    test_and_plot()
