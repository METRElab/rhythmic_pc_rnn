"""
Hyperparameter tuning script using Optuna.

Optimizes network hyperparameters to minimize inference error
on the hierarchical sensorimotor prediction task.
"""

import argparse
import os
import yaml
import optuna
from optuna.samplers import TPESampler
import copy
from pathlib import Path
import time
from typing import Dict, Any
import random

import numpy as np
import torch

from network import SensorimotorPCRNN
from utils import ExperimentManager, generate_input_sequences, sample_tempo
from train import calc_inference_error_sensorimotor


def objective(trial: optuna.Trial, base_config: Dict[str, Any]) -> float:
    """
    Optuna objective function to minimize inference error.

    Creates a network with trial hyperparameters, trains it for a reduced
    number of rounds, and returns the best inference error achieved.

    Args:
        trial: Optuna trial object for suggesting hyperparameters
        base_config: Base configuration dictionary to modify

    Returns:
        Best vestibular inference error achieved during training

    Raises:
        optuna.exceptions.TrialPruned: If trial should be pruned early
    """
    # Create a deep copy of the base config to modify
    config = copy.deepcopy(base_config)

    # Define the hyperparameters to tune
    config['experiment']['random_seed'] = trial.suggest_int('random_seed', 1, 1000)

    # Layer sizes
    config['network']['higher_size'] = trial.suggest_int('higher_size', 0, 4)
    config['network']['associative_size'] = trial.suggest_int('associative_size', 16, 64, step=4)

    # Timescales
    config['network']['alpha_H'] = trial.suggest_float('alpha_H', 0.01, 0.5, log=True)
    config['network']['alpha_x'] = trial.suggest_float('alpha_x', 0.1, 1.0, step=0.1)

    # Inference learning rates
    config['network']['inference_learning_rate_H'] = trial.suggest_float(
        'inference_learning_rate_H', 0.01, 0.5, log=True
    )
    config['network']['inference_learning_rate_x'] = trial.suggest_float(
        'inference_learning_rate_x', 0.001, 0.5, log=True
    )

    # Weight learning rates
    config['network']['weight_learning_rate_H'] = trial.suggest_float(
        'weight_learning_rate_H', 0.001, 0.5, log=True
    )
    config['network']['weight_learning_rate_x'] = trial.suggest_float(
        'weight_learning_rate_x', 0.001, 0.5, log=True
    )

    # Inference steps
    config['network']['n_inference_steps'] = trial.suggest_int('n_inference_steps', 5, 50, step=5)

    # Reduce the number of training rounds for faster tuning
    config['experiment']['n_training_rounds'] = 100

    # Create a temporary experiment name for this trial
    config['experiment']['name'] = f"optuna_trial_{trial.number}"

    # Initialize experiment manager with the modified config
    temp_config_path = f"temp_config_trial_{trial.number}.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)

    # Setting random seed for all libraries
    seed = config["experiment"]["random_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    try:
        exp_manager = ExperimentManager(temp_config_path)
        net_config = config['network']

        # Create network with trial parameters
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

        # Generate initial input sequences for step counting
        initial_tempo = sample_tempo(config)
        _, beat_seq = generate_input_sequences(
            tempo=initial_tempo,
            dt=config['experiment']['dt'],
            duration=config['experiment']['duration'],
            vestibular_size=net_config['vestibular_size'],
            mode=config['experiment']['mode']
        )
        n_steps_per_round = len(beat_seq)

        # Initialize error tracking
        accumulated_vest_error = 0.0
        accumulated_beat_error = 0.0
        steps_since_last_log = 0
        min_inference_error = np.inf

        # Training loop
        for round_idx in range(config['experiment']['n_training_rounds']):
            network.reset_states()

            # Sample tempo for this round
            tempo = sample_tempo(config)

            # Generate input sequences
            vestibular_seq, beat_seq = generate_input_sequences(
                tempo=tempo,
                dt=config['experiment']['dt'],
                duration=config['experiment']['duration'],
                vestibular_size=net_config['vestibular_size'],
                mode=config['experiment']['mode']
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
                steps_since_last_log += 1

                # Report intermediate values for pruning
                if global_step % 200 == 0 and steps_since_last_log > 0:
                    inference_errors = calc_inference_error_sensorimotor(network, config)
                    current_inference_error = inference_errors['vest_inference_error']

                    if current_inference_error < min_inference_error:
                        min_inference_error = current_inference_error

                    avg_vest_error = accumulated_vest_error / steps_since_last_log
                    trial.report(avg_vest_error, global_step)

                    # Reset accumulators
                    accumulated_vest_error = 0.0
                    accumulated_beat_error = 0.0
                    steps_since_last_log = 0

                    # Enable early stopping if the trial is not promising
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

        return min_inference_error

    finally:
        # Clean up temporary config file
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


def save_best_config(
    study: optuna.Study,
    base_config: Dict[str, Any],
    results_dir: Path,
    timestamp: str
) -> Path:
    """
    Save the best hyperparameters to a config file.

    Args:
        study: Completed Optuna study
        base_config: Base configuration dictionary
        results_dir: Directory to save results
        timestamp: Timestamp string for filename

    Returns:
        Path to saved config file
    """
    best_params = study.best_params
    best_config = copy.deepcopy(base_config)

    # Update config with best parameters
    best_config['experiment']['random_seed'] = best_params['random_seed']

    # Layer sizes
    best_config['network']['higher_size'] = best_params['higher_size']
    best_config['network']['associative_size'] = best_params['associative_size']

    # Timescales
    best_config['network']['alpha_H'] = best_params['alpha_H']
    best_config['network']['alpha_x'] = best_params['alpha_x']

    # Inference learning rates
    best_config['network']['inference_learning_rate_H'] = best_params['inference_learning_rate_H']
    best_config['network']['inference_learning_rate_x'] = best_params['inference_learning_rate_x']

    # Weight learning rates
    best_config['network']['weight_learning_rate_H'] = best_params['weight_learning_rate_H']
    best_config['network']['weight_learning_rate_x'] = best_params['weight_learning_rate_x']

    # Inference steps
    best_config['network']['n_inference_steps'] = best_params['n_inference_steps']

    # Add timestamp and best trial info to experiment name
    best_config['experiment']['name'] = (
        f"best_params_{timestamp}_trial_{study.best_trial.number}"
    )

    # Save best config
    best_config_path = results_dir / f"best_config_{timestamp}.yaml"
    with open(best_config_path, 'w') as f:
        yaml.dump(best_config, f)

    return best_config_path


def save_visualizations(
    study: optuna.Study,
    results_dir: Path,
    timestamp: str
) -> None:
    """
    Save Optuna visualization plots.

    Args:
        study: Completed Optuna study
        results_dir: Directory to save plots
        timestamp: Timestamp string for filenames
    """
    try:
        # Optimization history
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(str(results_dir / f"optimization_history_{timestamp}.html"))

        # Parameter importances
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(str(results_dir / f"param_importances_{timestamp}.html"))

        # Contour plot
        fig = optuna.visualization.plot_contour(study)
        fig.write_html(str(results_dir / f"contour_{timestamp}.html"))

        print(f"Visualization plots saved to: {results_dir}")

    except Exception as e:
        print(f"Could not generate visualization plots: {e}")
        print("Make sure plotly is installed.")


def print_study_results(study: optuna.Study, study_name: str) -> None:
    """
    Print summary of Optuna study results.

    Args:
        study: Completed Optuna study
        study_name: Name of the study
    """
    print("\n\n")
    print("=" * 60)
    print(f"Study '{study_name}' completed!")
    print("=" * 60)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best inference error: {study.best_value:.6f}")
    print("\nBest hyperparameters:")

    for param, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {param}: {value:.6f}")
        else:
            print(f"  {param}: {value}")


def main() -> None:
    """
    Main entry point for hyperparameter tuning.

    Parses arguments, creates Optuna study, runs optimization,
    and saves results.
    """
    parser = argparse.ArgumentParser(
        description="Tune hyperparameters for hierarchical network using Optuna"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to base config file'
    )
    parser.add_argument(
        '--n_trials',
        type=int,
        default=50,
        help='Number of Optuna trials'
    )
    parser.add_argument(
        '--study_name',
        type=str,
        default='hierarchical_sensorimotor_tuning',
        help='Name for the Optuna study'
    )
    args = parser.parse_args()

    # Load base config
    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    # Create directory for Optuna results
    results_dir = (
        Path(base_config['saving']['base_dir']) /
        base_config["experiment"]["mode"] /
        "optuna_results"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create Optuna study
    storage_name = f"sqlite:///{results_dir}/{args.study_name}.db"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_name,
        sampler=TPESampler(seed=42),
        direction='minimize',
        load_if_exists=True
    )

    # Run optimization
    print(f"Starting Optuna optimization with {args.n_trials} trials...")
    print(f"Tuning hierarchical network parameters including higher_size, alpha_H, alpha_x")
    study.optimize(
        lambda trial: objective(trial, base_config),
        n_trials=args.n_trials,
        timeout=None,
        n_jobs=1
    )

    # Print results
    print_study_results(study, args.study_name)

    # Save best config
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    best_config_path = save_best_config(study, base_config, results_dir, timestamp)
    print(f"\nBest config saved to: {best_config_path}")

    # Save visualizations
    save_visualizations(study, results_dir, timestamp)


if __name__ == "__main__":
    main()
