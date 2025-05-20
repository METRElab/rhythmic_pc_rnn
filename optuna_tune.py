import argparse
import os
import yaml
import optuna
from optuna.samplers import TPESampler
import copy
from pathlib import Path
import time
import numpy as np

from network import SensorimotorPCRNN
from utils import ExperimentManager, generate_input_sequences
from train import calc_inference_err_sensorimotor


def objective(trial, base_config):
    """Optuna objective function to minimize."""

    # Create a deep copy of the base config to modify
    config = copy.deepcopy(base_config)

    # Define the hyperparameters to tune
    config['experiment']['random_seed'] = trial.suggest_int('random_seed', 1, 1000)
    config['network']['associative_size'] = trial.suggest_int('associative_size', 8, 64)
    config['network']['inference_learning_rate'] = trial.suggest_float('inference_learning_rate', 0.01, 0.5, log=True)
    config['network']['weight_learning_rate'] = trial.suggest_float('weight_learning_rate', 0.01, 0.5, log=True)
    config['network']['n_inference_steps'] = trial.suggest_int('n_inference_steps', 5, 50)

    # Reduce the number of training rounds for faster tuning
    config['experiment']['n_training_rounds'] = 100  # Reduced for hyperparameter search

    # Create a temporary experiment name for this trial
    config['experiment']['name'] = f"optuna_trial_{trial.number}"

    # Initialize experiment manager with the modified config
    temp_config_path = f"temp_config_trial_{trial.number}.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)

    exp_manager = ExperimentManager(temp_config_path)

    # Create network with trial parameters
    network = SensorimotorPCRNN(
        associative_size=config['network']['associative_size'],
        vestibular_size=config['network']['vestibular_size'],
        inference_learning_rate=config['network']['inference_learning_rate'],
        weight_learning_rate=config['network']['weight_learning_rate'],
        n_inference_steps=config['network']['n_inference_steps'],
        random_seed=config["experiment"]["random_seed"]
    )

    # Generate input sequences
    vestibular_seq, beat_seq = generate_input_sequences(
        tempo=config['experiment']['tempo'],
        dt=config['experiment']['dt'],
        duration=config['experiment']['duration'],
        vestibular_size=config['network']['vestibular_size'],
        mode=config['experiment']['mode']
    )

    # Training loop
    n_steps = len(beat_seq)

    # Initialize error accumulators
    accumulated_vest_error = 0
    accumulated_beat_error = 0
    steps_since_last_log = 0
    min_inference_error = np.inf

    for round in range(config['experiment']['n_training_rounds']):
        network.reset_states()

        for step in range(n_steps):
            global_step = round * n_steps + step

            # Get current inputs
            beat = beat_seq[step]
            vestibular = vestibular_seq[step]

            # Training step
            vestibular_pred, beat_pred = network.timestep_train(vestibular, beat)

            # Accumulate errors
            accumulated_vest_error += ((vestibular - vestibular_pred) ** 2).sum().item()
            accumulated_beat_error += (beat - beat_pred).item() ** 2
            steps_since_last_log += 1

            # Report intermediate values for pruning
            if global_step % 200 == 0 and steps_since_last_log > 0:
                current_inference_error = calc_inference_err_sensorimotor(network, config)
                if current_inference_error < min_inference_error:
                    min_inference_error = copy.deepcopy(current_inference_error)

                avg_vest_error = accumulated_vest_error / steps_since_last_log
                trial.report(avg_vest_error, global_step)

                # Enable early stopping if the trial is not promising
                if trial.should_prune():
                    # Clean up temporary config file
                    if os.path.exists(temp_config_path):
                        os.remove(temp_config_path)
                    raise optuna.exceptions.TrialPruned()

    # Compute the final inference error as the objective value
    # inference_error = calc_inference_err_sensorimotor(network, config)

    # Clean up temporary config file
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)

    return min_inference_error
    # return inference_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to base config file')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--study_name', type=str, default='sensorimotor_tuning', help='Name for the Optuna study')
    args = parser.parse_args()

    # Load base config
    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    # Create directory for Optuna results
    results_dir = Path(base_config['saving']['base_dir']) / base_config["experiment"]["mode"] / "optuna_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create Optuna study
    storage_name = f"sqlite:///{results_dir}/{args.study_name}.db"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_name,
        sampler=TPESampler(seed=42),  # Use TPE sampler with seed for reproducibility
        direction='minimize',  # Minimize inference error
        load_if_exists=True    # Continue existing study if it exists
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, base_config),
        n_trials=args.n_trials,
        timeout=None,  # No timeout
        n_jobs=1       # Sequential execution for safety
    )

    # Print study statistics
    print("\n\n")
    print("=" * 50)
    print(f"Study {args.study_name} completed!")
    print("=" * 50)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best inference error: {study.best_value:.6f}")
    print("\nBest hyperparameters:")

    # Extract best parameters
    best_params = study.best_params
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    # Save best hyperparameters to a config file
    best_config = copy.deepcopy(base_config)
    best_config['experiment']['random_seed'] = best_params['random_seed']
    best_config['network']['associative_size'] = best_params['associative_size']
    best_config['network']['inference_learning_rate'] = best_params['inference_learning_rate']
    best_config['network']['weight_learning_rate'] = best_params['weight_learning_rate']
    best_config['network']['n_inference_steps'] = best_params['n_inference_steps']

    # Add timestamp and best trial info to experiment name
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    best_config['experiment']['name'] = f"best_params_{timestamp}_trial_{study.best_trial.number}"

    # Save best config
    best_config_path = results_dir / f"best_config_{timestamp}.yaml"
    with open(best_config_path, 'w') as f:
        yaml.dump(best_config, f)

    print(f"\nBest config saved to: {best_config_path}")

    # Plot optimization history
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(str(results_dir / f"optimization_history_{timestamp}.html"))

        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(str(results_dir / f"param_importances_{timestamp}.html"))

        fig = optuna.visualization.plot_contour(study)
        fig.write_html(str(results_dir / f"contour_{timestamp}.html"))

        print(f"Visualization plots saved to: {results_dir}")
    except:
        print("Could not generate visualization plots. Make sure plotly is installed.")


if __name__ == "__main__":
    main()
