import numpy as np
import torch
from pathlib import Path
import yaml
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def generate_input_sequences(tempo, dt, duration):
    """
    Generate synchronized beat and vestibular sequences.

    Parameters:
    -----------
    tempo : float
        Time between beats in seconds
    dt : float
        Timestep size in seconds
    duration : float
        Total sequence duration in seconds
    """
    # # Calculate number of timesteps
    # n_steps = int(duration / dt)
    #
    # # Generate time points
    # t = np.arange(0, duration, dt)
    #
    # # Generate beat sequence (1 at beat times, 0 elsewhere)
    # beat_times = np.arange(0, duration, tempo)
    # beat_indices = (beat_times / dt).astype(int)
    # beat_sequence = np.zeros(n_steps)
    # beat_sequence[beat_indices] = 1
    #
    # # Generate vestibular sequence (sinusoid matching beat frequency)
    # frequency = 1 / tempo
    # # vestibular_sequence = np.sin(2 * np.pi * frequency * t)
    # vestibular_sequence = np.cos(2 * np.pi * frequency * t)
    #
    # # Convert to torch tensors
    # beat_sequence = torch.FloatTensor(beat_sequence)
    # vestibular_sequence = torch.FloatTensor(vestibular_sequence)
    #
    # return beat_sequence, vestibular_sequence

    n_steps = int(duration / dt)
    t = np.arange(0, duration, dt)

    # Beat sequence (unchanged)
    beat_times = np.arange(0, duration, tempo)
    beat_indices = (beat_times / dt).astype(int)
    beat_sequence = np.zeros(n_steps)
    beat_sequence[beat_indices] = 1

    # Frequency
    frequency = 1 / tempo

    # Generate *both* sine and cosine:
    vestibular_sin = np.sin(2 * np.pi * frequency * t)
    vestibular_cos = np.cos(2 * np.pi * frequency * t)

    # Combine into a single array, shape [n_steps, 2]
    vestibular_sequence = np.stack([vestibular_sin, vestibular_cos], axis=1)

    # Convert to torch tensors
    beat_sequence = torch.FloatTensor(beat_sequence)               # shape [n_steps]
    vestibular_sequence = torch.FloatTensor(vestibular_sequence)   # shape [n_steps, 2]

    return beat_sequence, vestibular_sequence


class ExperimentManager:
    def __init__(self, config_path):

        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Add timestamp to experiment name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config['experiment']['name'] = f"{self.config['experiment']['name']}_{timestamp}"

        # Create experiment directory
        self.exp_dir = Path(self.config['saving']['base_dir']) / self.config['experiment']['name']
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Save config to experiment directory
        with open(self.exp_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)

        # Initialize tensorboard writer
        self.writer = SummaryWriter(self.exp_dir / 'logs')

        # Add print header
        print("\n" + "="*50)
        print(f"Starting experiment: {self.config['experiment']['name']}")
        print("="*50 + "\n")

    def save_model(self, model, step):
        """Save model state."""
        save_path = self.exp_dir / f'model_step_{step}.pt'
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'config': self.config
        }, save_path)
        print(f"\nModel saved at step {step}")

    def log_metrics(self, metrics, step):
        """Log metrics to tensorboard."""
        if step % (1 * self.config['saving']['log_every']) == 0:
            print("\nStep    ", end='')
            for name in metrics.keys():
                print(f"{name:<20}", end='')
            print("\n" + "-"*80)

        # Print metrics
        print(f"{step:<8}", end='')
        for name, value in metrics.items():
            # Log to tensorboard
            self.writer.add_scalar(name, value, step)
            # Print to terminal
            print(f"{value:<20.5f}", end='')
        print()  # New line
