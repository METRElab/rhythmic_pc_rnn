"""
Training script for the sensorimotor predictive coding network.

Trains the network on paired auditory-vestibular input to learn
rhythm prediction and cross-modal associations.
"""

import argparse
from typing import Dict, Any, Tuple
import random

import numpy as np
import torch

from network import SensorimotorPCRNN
from utils import (
    ExperimentManager,
    generate_input_sequences,
    sample_tempo,
    get_tempo_values,
)


def calc_inference_error_sensorimotor(
    network: SensorimotorPCRNN, config: Dict[str, Any], n_inference_rounds: int = 1
) -> Tuple[float, float]:
    """
    Calculate inference error over multiple test rounds.

    Runs inference with different tempos (if range mode) and computes
    average prediction errors for both vestibular and beat signals.

    Args:
        network: Trained SensorimotorPCRNN network
        config: Experiment configuration dictionary
        n_inference_rounds: Number of inference rounds to average over

    Returns:
        Tuple of (avg_vest_inference_error, avg_beat_inference_error)
    """
    total_vest_error = 0.0
    total_beat_error = 0.0
    total_steps = 0

    tempos = get_tempo_values(config)
    for tempo in tempos:
        for _ in range(n_inference_rounds):
            network.reset_states()

            # Sample tempo for this inference round
            # tempo = sample_tempo(config)

            # Generate test sequences
            test_vestibular_seq, test_beat_seq = generate_input_sequences(
                tempo=tempo,
                dt=config["experiment"]["dt"],
                duration=config["testing"]["test_duration"],
                vestibular_size=config["network"]["vestibular_size"],
                mode=config["experiment"]["mode"],
            )

            # Run inference for each timestep
            for vestibular, beat in zip(test_vestibular_seq, test_beat_seq):
                vestibular_pred, beat_pred, _, _ = network.timestep_inference(
                    vestibular_input=vestibular, beat_input=beat
                )

                # Accumulate error
                total_vest_error += ((vestibular - vestibular_pred) ** 2).sum().item()
                total_beat_error += ((beat - beat_pred) ** 2).sum().item()
                total_steps += 1

    # Calculate averages
    avg_vest_inference_error = total_vest_error / total_steps
    avg_beat_inference_error = total_beat_error / total_steps

    return avg_vest_inference_error, avg_beat_inference_error


def train_sensorimotor(exp_manager: ExperimentManager) -> None:
    """
    Train the sensorimotor predictive coding network.

    Trains on paired vestibular-auditory input, logging metrics and
    saving model checkpoints when inference error improves.

    Args:
        exp_manager: ExperimentManager instance for logging and saving
    """
    config = exp_manager.config

    # Setting random seed for all libraries
    seed = config["experiment"]["random_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create network
    network = SensorimotorPCRNN(
        associative_size=config["network"]["associative_size"],
        vestibular_size=config["network"]["vestibular_size"],
        inference_learning_rate=config["network"]["inference_learning_rate"],
        weight_learning_rate=config["network"]["weight_learning_rate"],
        n_inference_steps=config["network"]["n_inference_steps"],
        random_seed=config["experiment"]["random_seed"],
    )

    if config["experiment"]["retrain"]:
        checkpoint = torch.load(config["experiment"]["checkpoint_path"])
        network.load_state_dict(checkpoint["model_state_dict"])

    # Log tempo configuration
    tempo_values = get_tempo_values(config)
    exp_manager.logger.info(f"Training with tempos: {tempo_values}")

    # Calculate steps per round (using first tempo for reference)
    reference_tempo = tempo_values[0]
    _, reference_beat_seq = generate_input_sequences(
        tempo=reference_tempo,
        dt=config["experiment"]["dt"],
        duration=config["experiment"]["duration"],
        vestibular_size=config["network"]["vestibular_size"],
        mode=config["experiment"]["mode"],
    )
    n_steps_per_round = len(reference_beat_seq)

    # Initialize error accumulators
    accumulated_vest_error = 0.0
    accumulated_beat_error = 0.0
    steps_since_last_log = 0
    min_avg_vest_inference_error = np.inf

    # Training loop
    for round_idx in range(config["experiment"]["n_training_rounds"]):
        network.reset_states()

        # Sample tempo for this training round
        tempo = sample_tempo(config)

        # Generate input sequences for this round
        vestibular_seq, beat_seq = generate_input_sequences(
            tempo=tempo,
            dt=config["experiment"]["dt"],
            duration=config["experiment"]["duration"],
            vestibular_size=config["network"]["vestibular_size"],
            mode=config["experiment"]["mode"],
        )

        n_steps = len(beat_seq)

        for step in range(n_steps):
            global_step = round_idx * n_steps_per_round + step

            # Get current inputs
            beat = beat_seq[step]
            vestibular = vestibular_seq[step]

            # Training step
            vestibular_pred, beat_pred = network.timestep_train(vestibular, beat)

            # Accumulate errors
            accumulated_vest_error += ((vestibular - vestibular_pred) ** 2).sum().item()
            accumulated_beat_error += (beat - beat_pred).item() ** 2
            steps_since_last_log += 1

            # Log metrics periodically
            if global_step % config["saving"]["log_every"] == 0:
                # Calculate average training errors
                avg_vest_error = accumulated_vest_error / steps_since_last_log
                avg_beat_error = accumulated_beat_error / steps_since_last_log

                # Calculate inference errors
                avg_vest_inference_error, avg_beat_inference_error = (
                    calc_inference_error_sensorimotor(network, config)
                )

                # Log metrics
                metrics = {
                    "vest_error": avg_vest_error,
                    "beat_error": avg_beat_error,
                    "vest_inference_error": avg_vest_inference_error,
                    "beat_inference_error": avg_beat_inference_error,
                }
                exp_manager.log_metrics(metrics, global_step)

                # Reset accumulators
                accumulated_vest_error = 0.0
                accumulated_beat_error = 0.0
                steps_since_last_log = 0

                # Save model if inference error improved
                if avg_vest_inference_error < min_avg_vest_inference_error:
                    min_avg_vest_inference_error = avg_vest_inference_error
                    exp_manager.save_model(network, global_step)

    # Finish experiment
    exp_manager.finish()


def train() -> None:
    """
    Main training entry point.

    Parses command line arguments and starts training based on
    experiment mode specified in config.
    """
    parser = argparse.ArgumentParser(
        description="Train sensorimotor predictive coding network"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    # Initialize experiment
    exp_manager = ExperimentManager(args.config)

    # Start training based on mode
    mode = exp_manager.config["experiment"]["mode"]

    if mode in {"sensorimotor", "doublebeat"}:
        train_sensorimotor(exp_manager)
    else:
        raise ValueError(
            f"Unknown experiment mode: {mode}. Must be 'sensorimotor' or 'doublebeat'."
        )


if __name__ == "__main__":
    train()
