import argparse
import numpy as np

from network import SensorimotorPCRNN, BeatPCRNN
from utils import ExperimentManager, generate_input_sequences


def calc_inference_err_sensorimotor(network, config):
    """Run inference testing and log results."""
    # Generate test sequences
    test_vestibular_seq, test_beat_seq = generate_input_sequences(
        tempo=config['experiment']['tempo'],
        dt=config['experiment']['dt'],
        duration=config['testing']['test_duration'],
        vestibular_size=config['network']['vestibular_size'],
        mode=config['experiment']['mode']
    )

    total_vest_error = 0
    total_steps = 0

    # Run multiple inference rounds
    for _ in range(config['testing']['n_inference_rounds']):
        network.reset_states()
        vestibular_preds = []

        for vestibular, beat in zip(test_vestibular_seq, test_beat_seq):
            # Run inference with only beat input
            vestibular_pred, _, _, _ = network.timestep_inference(vestibular_input=vestibular, beat_input=beat)
            vestibular_preds.append(vestibular_pred)

            # Accumulate error
            total_vest_error += ((vestibular - vestibular_pred) ** 2).sum().item()
            total_steps += 1

    # Log average inference error
    avg_inference_error = total_vest_error / total_steps
    return avg_inference_error


def calc_inference_err_beat(network, config):
    """Run inference testing and log results."""
    # Generate test beat sequence
    test_beat_seq = generate_input_sequences(
        tempo=config['experiment']['tempo'],
        dt=config['experiment']['dt'],
        duration=config['testing']['test_duration'],
        mode=config['experiment']['mode']
    )

    total_beat_error = 0
    total_steps = 0

    # Run multiple inference rounds
    for _ in range(config['testing']['n_inference_rounds']):
        network.reset_states()
        beat_preds = []

        # For each step in the test sequence
        for i in range(len(test_beat_seq) - 1):  # -1 to allow for next beat prediction
            current_beat = test_beat_seq[i]
            # next_beat = test_beat_seq[i + 1]
            next_beat = test_beat_seq[i]

            # Run inference with current beat input
            beat_pred, _ = network.timestep_inference(beat_input=current_beat)
            beat_preds.append(beat_pred)

            # Calculate prediction error (how well does the network predict the next beat)
            total_beat_error += ((next_beat - beat_pred) ** 2).sum().item()
            total_steps += 1

    # Log average inference error
    avg_inference_error = total_beat_error / total_steps if total_steps > 0 else float('inf')
    return avg_inference_error


def train_sensorimotor(exp_manager: ExperimentManager):

    config = exp_manager.config

    # Create network
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
    min_avg_inference_error = np.Inf

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

            # if config['experiment']['mode'] == 'sensorimotor':
            #     accumulated_vest_error += ((vestibular - vestibular_pred) ** 2).sum().item()
            # elif config['experiment']['mode'] == 'doublebeat':
            #     accumulated_vest_error += (vestibular - vestibular_pred).item() ** 2

            accumulated_beat_error += (beat - beat_pred).item() ** 2
            steps_since_last_log += 1

            # Log metrics
            if global_step % config['saving']['log_every'] == 0:
                # Calculate averages
                avg_vest_error = accumulated_vest_error / steps_since_last_log
                avg_beat_error = accumulated_beat_error / steps_since_last_log
                avg_inference_error = calc_inference_err_sensorimotor(network, config)

                # Log training metrics
                metrics = {
                    'vest_error': avg_vest_error,
                    'beat_error': avg_beat_error,
                    'inference_error': avg_inference_error
                }
                exp_manager.log_metrics(metrics, global_step)

                # Run inference testing

                # Reset accumulators
                accumulated_vest_error = 0
                accumulated_beat_error = 0
                steps_since_last_log = 0

                # Save model periodically
                if avg_inference_error < min_avg_inference_error:
                    # min_avg_inference_error = copy.deepcopy(avg_inference_error)
                    min_avg_inference_error = avg_inference_error
                    exp_manager.save_model(network, global_step)


def train_beat(exp_manager: ExperimentManager):

    config = exp_manager.config

    # Create network
    network = BeatPCRNN(
        associative_size=config['network']['associative_size'],
        inference_learning_rate=config['network']['inference_learning_rate'],
        weight_learning_rate=config['network']['weight_learning_rate'],
        n_inference_steps=config['network']['n_inference_steps'],
        random_seed=config["experiment"]["random_seed"]
    )

    # Generate beat sequence
    beat_seq = generate_input_sequences(
        tempo=config['experiment']['tempo'],
        dt=config['experiment']['dt'],
        duration=config['experiment']['duration'],
        mode=config['experiment']['mode']
    )

    # Training loop
    n_steps = len(beat_seq)

    # Initialize error accumulator
    accumulated_beat_error = 0
    steps_since_last_log = 0
    min_avg_inference_error = float('inf')

    for round in range(config['experiment']['n_training_rounds']):
        network.reset_states()

        for step in range(n_steps):
            global_step = round * n_steps + step

            # Get current input
            beat = beat_seq[step]

            # Training step
            beat_pred = network.timestep_train(beat)

            # Accumulate errors
            accumulated_beat_error += (beat - beat_pred).item() ** 2
            steps_since_last_log += 1

            # Log metrics
            if global_step % config['saving']['log_every'] == 0 and steps_since_last_log > 0:
                # Calculate averages
                avg_beat_error = accumulated_beat_error / steps_since_last_log
                avg_inference_error = calc_inference_err_beat(network, config)

                # Log training metrics
                metrics = {
                    'beat_error': avg_beat_error,
                    'inference_error': avg_inference_error
                }
                # print(f"metrics are: {metrics}")
                exp_manager.log_metrics(metrics, global_step)

                # Reset accumulators
                accumulated_beat_error = 0
                steps_since_last_log = 0

                # Save model periodically
                if avg_inference_error < min_avg_inference_error:
                    min_avg_inference_error = avg_inference_error
                    exp_manager.save_model(network, global_step)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    # Initialize experiment
    exp_manager = ExperimentManager(args.config)

    if exp_manager.config['experiment']['mode'] in {"sensorimotor", "doublebeat"}:
        train_sensorimotor(exp_manager)
    elif exp_manager.config['experiment']['mode'] == "beat":
        train_beat(exp_manager)


if __name__ == "__main__":
    train()
