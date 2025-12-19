"""
Utility functions for experiment management and input generation.

Provides input sequence generation, tempo handling, and experiment management.
"""

import numpy as np
import torch
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Union, Optional
from torch.utils.tensorboard import SummaryWriter

from logger import ExperimentLogger, create_logger


def get_tempo_values(config: Dict[str, Any]) -> List[float]:
    """
    Get list of all tempo values based on configuration.

    For single mode, returns a list with one value.
    For range mode, returns discrete values from min to max with given step.

    Args:
        config: Experiment configuration dictionary

    Returns:
        List of tempo values (e.g., [0.4, 0.5, 0.6, 0.7, 0.8])

    Raises:
        ValueError: If tempo mode is not 'single' or 'range'
    """
    tempo_config = config['experiment']['tempo']
    mode = tempo_config['mode']

    if mode == 'single':
        return [tempo_config['value']]
    elif mode == 'range':
        min_tempo = tempo_config['min']
        max_tempo = tempo_config['max']
        step = tempo_config['step']

        # Generate discrete tempo values
        n_steps = int(round((max_tempo - min_tempo) / step)) + 1
        tempos = [round(min_tempo + i * step, 4) for i in range(n_steps)]
        return tempos
    else:
        raise ValueError(f"Unknown tempo mode: {mode}. Must be 'single' or 'range'.")


def sample_tempo(config: Dict[str, Any]) -> float:
    """
    Sample a single tempo value based on configuration.

    For single mode, returns the configured value.
    For range mode, randomly samples from the discrete tempo set.

    Args:
        config: Experiment configuration dictionary

    Returns:
        A single tempo value
    """
    tempos = get_tempo_values(config)

    if len(tempos) == 1:
        return tempos[0]
    else:
        return float(np.random.choice(tempos))


def generate_input_sequences(
    tempo: float,
    dt: float,
    duration: float,
    vestibular_size: int = 1,
    mode: str = "sensorimotor"
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate synchronized beat and vestibular sequences.

    Creates input sequences for training/testing the predictive coding network.
    Beat sequence is a binary pulse train, vestibular sequence is a triangular wave.

    Args:
        tempo: Time between beats in seconds
        dt: Timestep size in seconds
        duration: Total sequence duration in seconds
        vestibular_size: Dimension of vestibular signal (default 1)
        mode: Input mode - 'sensorimotor', 'doublebeat', or 'beat'

    Returns:
        For 'beat' mode: beat_sequence tensor of shape [n_steps]
        For 'doublebeat' mode: tuple of (beat_sequence, beat_sequence)
        For 'sensorimotor' mode: tuple of (vestibular_sequence, beat_sequence)

    Raises:
        ValueError: If mode is not recognized
    """
    n_steps = int(duration / dt)
    t = np.arange(0, duration, dt)

    # Beat sequence: binary pulse train
    beat_times = np.arange(0, duration, tempo)
    beat_indices = (beat_times / dt).astype(int)
    beat_indices = beat_indices[beat_indices < n_steps]
    beat_sequence = np.zeros(n_steps)
    beat_sequence[beat_indices] = 1

    # Convert to torch tensor
    beat_sequence = torch.FloatTensor(beat_sequence)

    if mode == "beat":
        return beat_sequence

    elif mode == "doublebeat":
        return beat_sequence, beat_sequence

    elif mode == "sensorimotor":
        frequency = 1 / tempo
        sawtooth = 2 * (t * frequency - np.floor(0.5 + t * frequency))
        vestibular_tri = 1 - 2 * np.abs(sawtooth)
        vestibular_sequence = torch.FloatTensor(vestibular_tri)
        return vestibular_sequence, beat_sequence

    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'sensorimotor', 'doublebeat', or 'beat'.")


class ExperimentManager:
    """
    Manages experiment setup, logging, and model saving.

    Handles directory creation, configuration saving, TensorBoard logging,
    file logging, and model checkpointing.

    Attributes:
        config: Experiment configuration dictionary
        exp_dir: Path to experiment directory
        writer: TensorBoard SummaryWriter
        logger: ExperimentLogger for file/console logging
        use_hierarchy: Whether the network uses hierarchical layers
    """

    def __init__(self, config_path: str) -> None:
        """
        Initialize experiment manager from a config file.

        Creates experiment directory, saves config copy, initializes
        TensorBoard writer and file logger.

        Args:
            config_path: Path to YAML configuration file
        """
        # Load config
        with open(config_path) as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        # Check if hierarchy is enabled
        self.use_hierarchy = self.config['network']['higher_size'] > 0

        # Add timestamp to experiment name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config['experiment']['name'] = f"{self.config['experiment']['name']}_{timestamp}"

        # Create experiment directory
        self.exp_dir: Path = (
            Path(self.config['saving']['base_dir']) /
            self.config["experiment"]["mode"] /
            self.config['experiment']['name']
        )
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Save config to experiment directory
        with open(self.exp_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)

        # Initialize TensorBoard writer
        self.writer: SummaryWriter = SummaryWriter(self.exp_dir / 'logs')

        # Initialize file logger
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        self.logger: ExperimentLogger = create_logger(
            exp_dir=self.exp_dir,
            log_filename="training.log",
            level=log_level
        )

        # Log experiment start
        self.logger.log_experiment_start(self.config)

    def save_model(self, model: torch.nn.Module, step: int) -> Path:
        """
        Save model state to checkpoint file.

        Args:
            model: PyTorch model to save
            step: Current training step

        Returns:
            Path to saved model file
        """
        save_path = self.exp_dir / f'model_step_{step}.pt'
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'config': self.config
        }, save_path)

        self.logger.log_model_saved(step, save_path)
        return save_path

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int
    ) -> None:
        """
        Log metrics to TensorBoard and file logger.

        Args:
            metrics: Dictionary of metric names to values
            step: Current training step
        """
        # Log to TensorBoard
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)

        # Prepare optional H metrics
        H_error = metrics.get('H_error') if self.use_hierarchy else None
        H_inference_error = metrics.get('H_inference_error') if self.use_hierarchy else None

        # Log to file logger
        self.logger.log_training_step(
            step=step,
            vest_error=metrics.get('vest_error', 0.0),
            beat_error=metrics.get('beat_error', 0.0),
            x_error=metrics.get('x_error', 0.0),
            vest_inference_error=metrics.get('vest_inference_error', 0.0),
            beat_inference_error=metrics.get('beat_inference_error', 0.0),
            x_inference_error=metrics.get('x_inference_error', 0.0),
            H_error=H_error,
            H_inference_error=H_inference_error
        )

    def update_best_metric(
        self,
        metric_name: str,
        value: float,
        step: int
    ) -> bool:
        """
        Update best metric tracking.

        Args:
            metric_name: Name of the metric
            value: Current metric value
            step: Current training step

        Returns:
            True if this is a new best value
        """
        return self.logger.update_best(metric_name, value, step)

    def get_best_metric(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the best recorded value for a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Dictionary with 'value' and 'step' keys, or None if not tracked
        """
        return self.logger.get_best(metric_name)

    def finish(self) -> None:
        """
        Finalize experiment logging.

        Logs experiment completion and closes TensorBoard writer.
        """
        self.logger.log_experiment_end()
        self.writer.close()
