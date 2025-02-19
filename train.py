import argparse
import numpy as np

from network import SensorimotorPredictiveNetworkRNN
from utils import ExperimentManager, generate_input_sequences


def calc_inference_err(network, config):
    """Run inference testing and log results."""
    # Generate test sequences
    test_beat_seq, test_vestibular_seq = generate_input_sequences(
        tempo=config['experiment']['tempo'],
        dt=config['experiment']['dt'],
        duration=config['testing']['test_duration']
    )

    total_vest_error = 0
    total_steps = 0

    # Run multiple inference rounds
    for _ in range(config['testing']['n_inference_rounds']):
        network.reset_states()
        vestibular_preds = []

        for beat, vestibular in zip(test_beat_seq, test_vestibular_seq):
            # Run inference with only beat input
            vestibular_pred = network.timestep_inference(beat)
            vestibular_preds.append(vestibular_pred)

            # Accumulate error
            total_vest_error += ((vestibular - vestibular_pred) ** 2).sum().item()
            total_steps += 1

    # Log average inference error
    avg_inference_error = total_vest_error / total_steps
    return avg_inference_error


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    # Initialize experiment
    exp_manager = ExperimentManager(args.config)
    config = exp_manager.config

    # Create network
    network = SensorimotorPredictiveNetworkRNN(
        associative_size=config['network']['associative_size'],
        vestibular_size=config['network']['vestibular_size'],
        inference_learning_rate=config['network']['inference_learning_rate'],
        weight_learning_rate=config['network']['weight_learning_rate'],
        n_inference_steps=config['network']['n_inference_steps']
    )

    # Generate input sequences
    beat_seq, vestibular_seq = generate_input_sequences(
        tempo=config['experiment']['tempo'],
        dt=config['experiment']['dt'],
        duration=config['experiment']['duration']
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
            accumulated_beat_error += (beat - beat_pred).item() ** 2
            steps_since_last_log += 1

            # Log metrics
            if global_step % config['saving']['log_every'] == 0:
                # Calculate averages
                avg_vest_error = accumulated_vest_error / steps_since_last_log
                avg_beat_error = accumulated_beat_error / steps_since_last_log
                avg_inference_error = calc_inference_err(network, config)

                # Log training metrics
                metrics = {
                    'vest_error': avg_vest_error,
                    'beat_error': avg_beat_error,
                    'inference_error': avg_inference_error
                    # 'training/round': round
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


if __name__ == "__main__":
    train()
