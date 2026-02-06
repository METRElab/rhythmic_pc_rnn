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


def generate_random_pulses(
    dt: float,
    duration: float,
    distribution: str = 'uniform',
    ipi_min: Optional[float] = None,
    ipi_max: Optional[float] = None,
    rate: Optional[float] = None,
    probability: Optional[float] = None,
    rng: Optional[np.random.Generator] = None
) -> torch.Tensor:
    """
    Generate a random pulse train with inter-pulse intervals drawn from a specified distribution.

    This creates auditory input that is temporally uncorrelated with any regular
    periodic structure, useful for control experiments testing whether cross-modal
    association (rather than mere co-presence) is necessary for learning.

    Args:
        dt: Timestep size in seconds
        duration: Total sequence duration in seconds
        distribution: Distribution type ('uniform', 'poisson', or 'white_noise')
        ipi_min: Minimum inter-pulse interval in seconds (required for 'uniform')
        ipi_max: Maximum inter-pulse interval in seconds (required for 'uniform')
        rate: Rate parameter (λ) for Poisson process in Hz (required for 'poisson').
              The mean inter-pulse interval will be 1/rate seconds.
        probability: Probability of pulse at each timestep (required for 'white_noise').
              Should be in range [0, 1].
        rng: NumPy random generator (optional, for reproducibility)

    Returns:
        Tensor of shape [n_steps] with 1s at pulse times, 0s elsewhere

    Raises:
        ValueError: If distribution is unknown or required parameters are missing

    Notes:
        - 'uniform': Inter-pulse intervals drawn uniformly from [ipi_min, ipi_max]
        - 'poisson': Inter-pulse intervals follow exponential distribution with mean 1/rate
        - 'white_noise': Each timestep independently has probability p of being a pulse
    """
    if rng is None:
        rng = np.random.default_rng()

    n_steps = int(duration / dt)
    pulse_sequence = np.zeros(n_steps)

    if distribution == 'white_noise':
        if probability is None:
            raise ValueError(
                "For 'white_noise' distribution, 'probability' must be specified."
            )
        if not 0 <= probability <= 1:
            raise ValueError(f"Probability must be in [0, 1], got {probability}")

        # Each timestep independently has probability p of being a pulse
        pulse_sequence = rng.random(n_steps) < probability
        return torch.FloatTensor(pulse_sequence.astype(float))

    # For uniform and poisson, generate pulse times
    pulse_times = []

    if distribution == 'uniform':
        if ipi_min is None or ipi_max is None:
            raise ValueError(
                "For 'uniform' distribution, both 'ipi_min' and 'ipi_max' must be specified."
            )
        current_time = rng.uniform(0, ipi_max)  # Random start offset
        while current_time < duration:
            pulse_times.append(current_time)
            ipi = rng.uniform(ipi_min, ipi_max)
            current_time += ipi

    elif distribution == 'poisson':
        if rate is None:
            raise ValueError(
                "For 'poisson' distribution, 'rate' (λ in Hz) must be specified."
            )
        if rate <= 0:
            raise ValueError(f"Poisson rate must be positive, got {rate}")

        # Mean inter-pulse interval is 1/rate
        mean_ipi = 1.0 / rate
        current_time = rng.exponential(mean_ipi)  # Random start offset
        while current_time < duration:
            pulse_times.append(current_time)
            ipi = rng.exponential(mean_ipi)
            current_time += ipi

    else:
        raise ValueError(
            f"Unknown distribution: {distribution}. Must be 'uniform', 'poisson', or 'white_noise'."
        )

    # Convert times to indices and set pulses
    if pulse_times:
        pulse_indices = np.round(np.array(pulse_times) / dt).astype(int)
        pulse_indices = pulse_indices[pulse_indices < n_steps]
        pulse_sequence[pulse_indices] = 1

    return torch.FloatTensor(pulse_sequence)


def generate_input_sequences(
    config: Dict[str, Any],
    tempo: float,
    rng: Optional[np.random.Generator] = None
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate input sequences based on experiment configuration.

    Creates input sequences for training/testing the predictive coding network.
    Retrieves all necessary parameters (dt, duration, mode, random_audio settings)
    from the config dictionary.

    Args:
        config: Experiment configuration dictionary containing:
            - experiment.dt: Timestep size in seconds
            - experiment.duration: Total sequence duration in seconds
            - experiment.mode: Input mode ('sensorimotor', 'doublebeat', 'beat', 'uncorrelated')
            - experiment.random_audio (optional): Config for random pulses
        tempo: Time between beats in seconds
        rng: NumPy random generator for reproducibility (used in 'uncorrelated' mode)

    Returns:
        For 'beat' mode: beat_sequence tensor of shape [n_steps]
        For other modes: tuple of (vestibular_or_beat_sequence, beat_sequence)

    Raises:
        ValueError: If mode is not recognized or required config is missing

    Modes:
        'sensorimotor': Regular vestibular triangular wave + correlated auditory pulses
        'doublebeat': Auditory pulses on both channels
        'beat': Auditory pulses only
        'uncorrelated': Regular vestibular + RANDOM auditory pulses

    Notes:
        The 'uncorrelated' mode supports two distributions for random pulses:
        - 'uniform': Inter-pulse intervals drawn uniformly from [ipi_min, ipi_max]
        - 'poisson': Inter-pulse intervals drawn from exponential distribution with given rate
    """
    # Extract parameters from config
    exp_config = config['experiment']
    dt = exp_config['dt']
    duration = exp_config['duration']
    mode = exp_config['mode']

    n_steps = int(duration / dt)
    t = np.arange(0, duration, dt)

    # Beat sequence: binary pulse train (regular, correlated with vestibular)
    beat_times = np.arange(0, duration, tempo)
    beat_indices = np.round(beat_times / dt).astype(int)
    beat_indices = beat_indices[beat_indices < n_steps]
    beat_sequence = np.zeros(n_steps)
    beat_sequence[beat_indices] = 1
    beat_sequence = torch.FloatTensor(beat_sequence)

    if mode == "beat":
        return beat_sequence

    elif mode == "doublebeat":
        return beat_sequence, beat_sequence

    elif mode == "sensorimotor":
        # Regular vestibular triangular wave, synchronized with beats
        frequency = 1 / tempo
        sawtooth = 2 * (t * frequency - np.floor(0.5 + t * frequency))
        vestibular_tri = 1 - 2 * np.abs(sawtooth)
        vestibular_sequence = torch.FloatTensor(vestibular_tri)
        return vestibular_sequence, beat_sequence

    elif mode == "uncorrelated":
        # Regular vestibular triangular wave (same as sensorimotor)
        frequency = 1 / tempo
        sawtooth = 2 * (t * frequency - np.floor(0.5 + t * frequency))
        vestibular_tri = 1 - 2 * np.abs(sawtooth)
        vestibular_sequence = torch.FloatTensor(vestibular_tri)

        # Random auditory pulses (uncorrelated with vestibular)
        random_audio_config = exp_config.get('random_audio')

        if random_audio_config is None:
            raise ValueError(
                "experiment.random_audio is required for 'uncorrelated' mode."
            )

        # Get distribution type (default to 'uniform' for backward compatibility)
        distribution = random_audio_config.get('distribution', 'uniform')

        if distribution == 'uniform':
            ipi_min = random_audio_config.get('ipi_min')
            ipi_max = random_audio_config.get('ipi_max')

            if ipi_min is None or ipi_max is None:
                raise ValueError(
                    "For 'uniform' distribution, experiment.random_audio must contain "
                    f"'ipi_min' and 'ipi_max' keys. Got: {random_audio_config}"
                )

            random_beat_sequence = generate_random_pulses(
                dt=dt,
                duration=duration,
                distribution='uniform',
                ipi_min=ipi_min,
                ipi_max=ipi_max,
                rng=rng
            )

        elif distribution == 'poisson':
            rate = random_audio_config.get('rate')

            if rate is None:
                raise ValueError(
                    "For 'poisson' distribution, experiment.random_audio must contain "
                    f"'rate' key (λ in Hz). Got: {random_audio_config}"
                )

            random_beat_sequence = generate_random_pulses(
                dt=dt,
                duration=duration,
                distribution='poisson',
                rate=rate,
                rng=rng
            )

        elif distribution == 'white_noise':
            probability = random_audio_config.get('probability')

            if probability is None:
                raise ValueError(
                    "For 'white_noise' distribution, experiment.random_audio must contain "
                    f"'probability' key (pulse probability per timestep). Got: {random_audio_config}"
                )

            random_beat_sequence = generate_random_pulses(
                dt=dt,
                duration=duration,
                distribution='white_noise',
                probability=probability,
                rng=rng
            )

        else:
            raise ValueError(
                f"Unknown distribution: {distribution}. Must be 'uniform', 'poisson', or 'white_noise'."
            )

        return vestibular_sequence, random_beat_sequence

    else:
        raise ValueError(
            f"Unknown mode: {mode}. Must be 'sensorimotor', 'doublebeat', 'beat', or 'uncorrelated'."
        )


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
        rng: NumPy random generator for reproducible random input generation
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

        # Initialize random generator for reproducible random input generation
        seed = self.config['experiment'].get('random_seed')
        self.rng: np.random.Generator = np.random.default_rng(seed)

        # Log experiment start
        self.logger.log_experiment_start(self.config)

    def generate_training_input(
        self,
        tempo: float
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate input sequences for training using experiment configuration.

        This is a convenience method that uses the stored config and rng.

        Args:
            tempo: Time between beats in seconds

        Returns:
            Input sequences appropriate for the configured mode
        """
        return generate_input_sequences(
            config=self.config,
            tempo=tempo,
            rng=self.rng
        )

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
