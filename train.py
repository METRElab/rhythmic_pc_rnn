"""
Training script for the hierarchical sensorimotor predictive coding network.

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
    get_tempo_values
)


def calc_inference_error_sensorimotor(
    network: SensorimotorPCRNN,
    config: Dict[str, Any],
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Calculate inference error over multiple test rounds.

    Runs inference with different tempos (if range mode) and computes
    average prediction errors for all layers.

    Args:
        network: Trained SensorimotorPCRNN network
        config: Experiment configuration dictionary
        n_inference_rounds: Number of inference rounds to average over

    Returns:
        Dictionary containing average inference errors:
            - vest_inference_error
            - beat_inference_error
            - x_inference_error
            - H_inference_error (if hierarchy enabled)
    """
    total_vest_error = 0.0
    total_beat_error = 0.0
    total_x_error = 0.0
    total_H_error = 0.0
    total_steps = 0

    use_hierarchy = network.use_hierarchy

    # Use a separate rng for evaluation so it doesn't perturb training rng
    # eval_seed = config['experiment'].get('random_seed', 42)
    # eval_rng = np.random.default_rng(eval_seed)

    tempos = get_tempo_values(config)
    for tempo in tempos:
        for _ in range(config["testing"]["n_inference_rounds"]):
            network.reset_states()

            # Generate test sequences
            test_vestibular_seq, test_beat_seq = generate_input_sequences(
                tempo=tempo,
                config=config,
                rng=rng,
            )

            # Run inference for each timestep
            for vestibular, beat in zip(test_vestibular_seq, test_beat_seq):
                result = network.timestep_inference(
                    vestibular_input=vestibular,
                    beat_input=beat,
                    auditory_only=False,
                    prediction_timing="before",
                )

                # Accumulate errors
                total_vest_error += ((vestibular - result['vest_pred'].squeeze()) ** 2).sum().item()
                total_beat_error += ((beat - result['beat_pred'].squeeze()) ** 2).sum().item()
                total_x_error += (result['e_x'] ** 2).sum().item()

                if use_hierarchy:
                    total_H_error += (result['e_H'] ** 2).sum().item()

                total_steps += 1

    # Calculate averages
    errors = {
        'vest_inference_error': total_vest_error / total_steps,
        'beat_inference_error': total_beat_error / total_steps,
        'x_inference_error': total_x_error / total_steps
    }

    if use_hierarchy:
        errors['H_inference_error'] = total_H_error / total_steps

    return errors


def train_sensorimotor(exp_manager: ExperimentManager) -> None:
    """
    Train the hierarchical sensorimotor predictive coding network.

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

    net_config = config['network']

    # Create network
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
        random_seed=config["experiment"]["random_seed"]
    )

    use_hierarchy = network.use_hierarchy

    # Log tempo and network configuration
    tempo_values = get_tempo_values(config)
    exp_manager.logger.info(f"Training with tempos: {tempo_values}")
    exp_manager.logger.info(f"Hierarchy enabled: {use_hierarchy}")

    # Calculate steps per round (using first tempo for reference)
    # Use a throwaway rng so we don't advance the training rng
    # reference_tempo = tempo_values[0]
    # ref_rng = np.random.default_rng(seed)
    # _, reference_beat_seq = generate_input_sequences(
    #     tempo=reference_tempo,
    #     config=config,
    #     rng=ref_rng,
    # )
    # n_steps_per_round = len(reference_beat_seq)
    n_steps_per_round = int(config["experiment"]["duration"] / config["experiment"]["dt"])

    # Use a specific inference rng so we don't advance the training rng while doing inference
    inference_rng = np.random.default_rng(seed)

    # Initialize error accumulators
    accumulated_vest_error = 0.0
    accumulated_beat_error = 0.0
    accumulated_x_error = 0.0
    accumulated_H_error = 0.0
    steps_since_last_log = 0
    min_avg_vest_inference_error = np.inf

    # Training loop
    for round_idx in range(config['experiment']['n_training_rounds']):
        network.reset_states()

        # Sample tempo for this training round
        tempo = sample_tempo(config, rng=exp_manager.rng)

        # Generate input sequences for this round
        vestibular_seq, beat_seq = generate_input_sequences(
            tempo=tempo,
            config=config,
            rng=exp_manager.rng,
        )

        n_steps = len(beat_seq)

        for step in range(n_steps):
            global_step = round_idx * n_steps_per_round + step

            # Get current inputs
            beat = beat_seq[step]
            vestibular = vestibular_seq[step]

            # Training step
            result = network.timestep_train(vestibular, beat)

            # Accumulate errors
            accumulated_vest_error += ((vestibular - result['vest_pred'].squeeze()) ** 2).sum().item()
            accumulated_beat_error += ((beat - result['beat_pred'].squeeze()) ** 2).sum().item()

            # We need to compute x_error and H_error from current state
            # Run a quick error computation
            predictions = network.compute_predictions()
            if use_hierarchy:
                # Approximate x and H errors from predictions
                # These are computed after inference, so they should be small
                x_err = (network.x - predictions['mu_x']).pow(2).sum().item()
                H_err = (network.H - predictions['mu_H']).pow(2).sum().item()
                accumulated_x_error += x_err
                accumulated_H_error += H_err
            else:
                x_err = (network.x - predictions['mu_x']).pow(2).sum().item()
                accumulated_x_error += x_err

            # Log metrics periodically
            # if global_step % config['saving']['log_every'] == 0:
            if global_step % config['saving']['log_every'] == 0 and steps_since_last_log > 0:
                # Calculate average training errors
                avg_vest_error = accumulated_vest_error / steps_since_last_log
                avg_beat_error = accumulated_beat_error / steps_since_last_log
                avg_x_error = accumulated_x_error / steps_since_last_log

                # Calculate inference errors
                inference_errors = calc_inference_error_sensorimotor(network, config, inference_rng)

                # Build metrics dict
                metrics = {
                    'vest_error': avg_vest_error,
                    'beat_error': avg_beat_error,
                    'x_error': avg_x_error,
                    'vest_inference_error': inference_errors['vest_inference_error'],
                    'beat_inference_error': inference_errors['beat_inference_error'],
                    'x_inference_error': inference_errors['x_inference_error']
                }

                if use_hierarchy:
                    avg_H_error = accumulated_H_error / steps_since_last_log
                    metrics['H_error'] = avg_H_error
                    metrics['H_inference_error'] = inference_errors['H_inference_error']

                # Log metrics
                exp_manager.log_metrics(metrics, global_step)

                # Reset accumulators
                accumulated_vest_error = 0.0
                accumulated_beat_error = 0.0
                accumulated_x_error = 0.0
                accumulated_H_error = 0.0
                steps_since_last_log = 0

                # Save model if inference error improved
                avg_vest_inference_error = inference_errors['vest_inference_error']
                if avg_vest_inference_error < min_avg_vest_inference_error:
                    min_avg_vest_inference_error = avg_vest_inference_error
                    exp_manager.save_model(network, global_step)

            steps_since_last_log += 1

    # Finish experiment
    exp_manager.finish()


def train() -> None:
    """
    Main training entry point.

    Parses command line arguments and starts training based on
    experiment mode specified in config.
    """
    parser = argparse.ArgumentParser(
        description="Train hierarchical sensorimotor predictive coding network"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )
    args = parser.parse_args()

    # Initialize experiment
    exp_manager = ExperimentManager(args.config)

    # Start training based on mode
    mode = exp_manager.config['experiment']['mode']

    if mode in {"sensorimotor", "doublebeat", "uncorrelated"}:
        train_sensorimotor(exp_manager)
    else:
        raise ValueError(f"Unknown experiment mode: {mode}. Must be 'sensorimotor' or 'doublebeat'.")


if __name__ == "__main__":
    train()
