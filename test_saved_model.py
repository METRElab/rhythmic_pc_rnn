import argparse
import yaml
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from network import SensorimotorPredictiveNetworkRNN
from utils import generate_input_sequences


def plot_inference_sequence(beat_seq, vestibular_seq, predicted_seq, save_path):
    """
    Plot sequences with three subplots and vertical beat indicators on prediction plot.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

    # Time points
    t = range(len(beat_seq))

    # Plot beats
    ax1.stem(t, beat_seq.numpy(), label='Beats', linefmt='r-', markerfmt='ro')
    ax1.set_ylabel('Beat Amplitude')
    ax1.set_title('Input Beat Sequence')
    ax1.grid(True)

    # Plot actual vestibular movement
    ax2.plot(t, vestibular_seq.numpy(), 'b-', linewidth=2)
    ax2.set_ylabel('Movement Amplitude')
    ax2.set_title('Actual Movement (Target)')
    ax2.grid(True)

    # Plot predicted movement with beat indicators
    ax3.plot(t, predicted_seq, 'g-', linewidth=2)

    # Add vertical lines at beat times
    beat_times = [i for i, beat in enumerate(beat_seq) if beat.item() > 0]
    for beat_time in beat_times:
        ax2.axvline(x=beat_time, color='r', linestyle='--', alpha=0.3)
        ax3.axvline(x=beat_time, color='r', linestyle='--', alpha=0.3)

    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Movement Amplitude')
    ax3.set_title('Predicted Movement (with beat indicators)')
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


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
        duration=config['testing']['test_duration']
    )

    # Run inference
    network.reset_states()
    predicted_seq = []

    for beat in beat_seq:
        vestibular_pred = network.timestep_inference(beat)
        predicted_seq.append(vestibular_pred.item())

    # Plot and save results
    plot_path = plots_dir / f'inference_plot_step_{args.model_step}.png'
    plot_inference_sequence(beat_seq, vestibular_seq, predicted_seq, plot_path)

    print(f"Generated plot saved at: {plot_path}")


if __name__ == "__main__":
    test_saved_model()
