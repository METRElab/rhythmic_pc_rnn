import argparse
import yaml
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from network import SensorimotorPredictiveNetworkRNN
from utils import generate_input_sequences
import numpy as np


def plot_inference_sequence(beat_seq, vestibular_seq, predicted_seq, save_path_html, save_path_png=None):
    """
    Plot sequences with Plotly in three subplots and
    add vertical lines at beat times for clarity.

    Parameters
    ----------
    beat_seq : torch.Tensor of shape [n_steps]
    vestibular_seq : torch.Tensor of shape [n_steps, 2]
                    (sine + cosine)
    predicted_seq : np.array or torch.Tensor of shape [n_steps, 2]
                    (predicted sine + cosine)
    save_path_html : Path or str
        Where to save the interactive HTML plot.
    save_path_png : Path or str, optional
        Where to save a high-res static PNG plot (requires kaleido).
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Convert predicted_seq to NumPy if it's still a torch.Tensor
    if hasattr(predicted_seq, 'numpy'):
        predicted_seq = predicted_seq.numpy()  # shape [n_steps, 2]

    # Time axis
    n_steps = len(beat_seq)
    t = list(range(n_steps))

    # Create subplots: 3 rows, 1 column
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=["Beat Sequence", "Actual Vestibular Movement", "Predicted Vestibular Movement"],
        shared_xaxes=True,  # so x-zoom is shared
        vertical_spacing=0.1
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

    if vestibular_seq.ndim == 2:
        # --- Row 2: Actual Vestibular Movement (2 channels) ---
        fig.add_trace(
            go.Scatter(
                x=t,
                y=vestibular_seq[:, 0].cpu().numpy(),
                mode='lines',
                name='Vestibular sin',
                line=dict(color='blue')
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=vestibular_seq[:, 1].cpu().numpy(),
                mode='lines',
                name='Vestibular cos',
                line=dict(color='cyan')
            ),
            row=2, col=1
        )

        # --- Row 3: Predicted Vestibular Movement (2 channels) ---
        fig.add_trace(
            go.Scatter(
                x=t,
                y=predicted_seq[:, 0],
                mode='lines',
                name='Predicted sin',
                line=dict(color='green')
            ),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=predicted_seq[:, 1],
                mode='lines',
                name='Predicted cos',
                line=dict(color='magenta')
            ),
            row=3, col=1
        )

    if vestibular_seq.ndim == 1:
        # --- Row 2: Actual Vestibular Movement (1 channel) ---
        fig.add_trace(
            go.Scatter(
                x=t,
                y=vestibular_seq.cpu().numpy(),
                mode='lines',
                name='Vestibular sin',
                line=dict(color='blue')
            ),
            row=2, col=1
        )

        # --- Row 3: Predicted Vestibular Movement (1 channel) ---
        fig.add_trace(
            go.Scatter(
                x=t,
                y=predicted_seq,
                mode='lines',
                name='Predicted sin',
                line=dict(color='green')
            ),
            row=3, col=1
        )

    # Add vertical lines at beat times on rows 2 and 3
    beat_times = [i for i, beat in enumerate(beat_seq) if beat.item() > 0]
    for b in beat_times:
        # add_vline row argument is the *domain* row index, not the subplot index
        # but Plotly 5+ allows `row` and `col` in add_vline:
        fig.add_vline(x=b, line_width=1, line_dash='dash', line_color='red', row=2, col=1)
        fig.add_vline(x=b, line_width=1, line_dash='dash', line_color='red', row=3, col=1)

    # Layout settings
    fig.update_layout(
        title='Inference Results',
        height=900,
        showlegend=True
    )

    # Save interactive HTML (dynamic)
    fig.write_html(str(save_path_html), auto_open=False)
    print(f"Interactive Plotly figure saved to: {save_path_html}")

    # Optionally, save a high-res static PNG (requires kaleido).
    if save_path_png is not None:
        # scale=2 or higher for increased resolution
        fig.write_image(str(save_path_png), scale=3)
        print(f"High-resolution PNG saved to: {save_path_png}")


def test_saved_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to saved config file')
    parser.add_argument('--model_step', type=int, required=True, help='Which saved model step to load')
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    exp_dir = config_path.parent

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create plots directory
    plots_dir = exp_dir / 'inference_plots'
    plots_dir.mkdir(exist_ok=True)

    # Initialize network
    network = SensorimotorPredictiveNetworkRNN(
        associative_size=config['network']['associative_size'],
        vestibular_size=config['network']['vestibular_size'],
        inference_learning_rate=config['network']['inference_learning_rate'],
        weight_learning_rate=config['network']['weight_learning_rate'],
        n_inference_steps=config['network']['n_inference_steps']
    )

    # Load saved model
    checkpoint = torch.load(exp_dir / f'model_step_{args.model_step}.pt')
    network.load_state_dict(checkpoint['model_state_dict'])

    # Generate test sequences
    beat_seq, vestibular_seq = generate_input_sequences(
        tempo=config['experiment']['tempo'],
        dt=config['experiment']['dt'],
        duration=config['testing']['test_duration'],
        vestibular_size=config['network']['vestibular_size']
    )

    # Run inference
    network.reset_states()
    predicted_seq = []

    for beat in beat_seq:
        vestibular_pred = network.timestep_inference(beat)
        predicted_seq.append(vestibular_pred.squeeze().tolist())

        # predicted_seq.append(vestibular_pred.item())
    predicted_seq = np.array(predicted_seq)

    # Plot and save results
    # plot_path = plots_dir / f'inference_plot_step_{args.model_step}.png'
    plot_path_html = plots_dir / f'inference_plot_step_{args.model_step}.html'
    plot_path_png = plots_dir / f'inference_plot_step_{args.model_step}.png'
    plot_inference_sequence(beat_seq, vestibular_seq, predicted_seq, plot_path_html, plot_path_png)

    print(f"Generated plot saved at: {plots_dir}")


if __name__ == "__main__":
    test_saved_model()
