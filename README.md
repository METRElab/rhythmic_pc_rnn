# Sensorimotor Predictive Coding Network

A predictive coding recurrent neural network for learning rhythm through paired auditory-vestibular input. Based on the hypothesis that maternal gait during pregnancy provides scaffolding for rhythm development.

## Requirements

```bash
pip install numpy torch pyyaml tensorboard plotly optuna kaleido
```

## Project Structure

```
├── configs/
│   ├── config_sensorimotor.yaml
│   └── config_sensorimotor_doublebeat.yaml
├── logger.py          # Logging utility
├── network.py         # Predictive coding network
├── utils.py           # Input generation & experiment management
├── train.py           # Training script
├── test_and_plot.py   # Testing & visualization
└── optuna_tune.py     # Hyperparameter tuning
```

## Configuration

Key config options in `config_sensorimotor.yaml`:

```yaml
experiment:
  tempo:
    mode: "range"    # "single" or "range"
    value: 0.5       # used when mode is "single"
    min: 0.4         # used when mode is "range"
    max: 0.8
    step: 0.1        # creates [0.4, 0.5, 0.6, 0.7, 0.8]
```

## Usage

### Training

```bash
python train.py --config configs/config_sensorimotor.yaml
```

Output: `experiments/sensorimotor/{experiment_name}/`
- `config.yaml` - saved configuration
- `training.log` - training logs
- `model_step_{N}.pt` - model checkpoints
- `logs/` - TensorBoard logs

View TensorBoard:
```bash
tensorboard --logdir experiments/sensorimotor/{experiment_name}/logs
```

### Testing

Basic testing (all tempos from config):
```bash
python test_and_plot.py --config experiments/sensorimotor/{exp_name}/config.yaml --model_step 5000
```

With continuation mode (no auditory input after 50% of sequence):
```bash
python test_and_plot.py --config experiments/sensorimotor/{exp_name}/config.yaml --model_step 5000 --continuation
```

Capture predictions before inference optimization:
```bash
python test_and_plot.py --config experiments/sensorimotor/{exp_name}/config.yaml --model_step 5000 --prediction_timing before
```

Output: `experiments/sensorimotor/{exp_name}/inference_plots/`
- HTML interactive plots
- PNG static plots
- `testing.log` - test results

### Hyperparameter Tuning

```bash
python optuna_tune.py --config configs/config_sensorimotor.yaml --n_trials 100 --study_name my_study
```

Output: `experiments/sensorimotor/optuna_results/`
- `best_config_{timestamp}.yaml` - best hyperparameters
- `{study_name}.db` - Optuna study database
- Visualization HTML files

Resume existing study:
```bash
python optuna_tune.py --config configs/config_sensorimotor.yaml --n_trials 50 --study_name my_study
```

## Experiment Modes

- `sensorimotor`: Auditory pulses + vestibular triangular wave (default)
- `doublebeat`: Auditory pulses on both channels (control condition)
