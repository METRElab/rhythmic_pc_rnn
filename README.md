# Sensorimotor Predictive Coding Network

Code for the paper: **"Maternal gait contributes to development of rhythm processing in a predictive processing network model"** (Yousefabadi & Cannon, McMaster University).

## Background

Humans are uniquely musical among primates — we spontaneously perceive and move to rhythmic patterns from early infancy. This project investigates a developmental hypothesis: that maternal gait during pregnancy provides the sensory scaffolding necessary for rhythm learning. A walking mother produces correlated auditory (footsteps) and vestibular (body acceleration) signals that reach the fetus, and this correlated multisensory experience may bootstrap the neural circuits underlying rhythm perception and the urge to move to music.

## How the Model Works

The model is a **predictive coding recurrent neural network** that learns by minimizing prediction error across two sensory channels — auditory and vestibular — using only local Hebbian plasticity (no backpropagation through time).

**Architecture.** An associative layer of recurrent hidden units receives input from two parallel sensory pathways. At each timestep the network generates top-down predictions for (1) its own next hidden state and (2) the current auditory and vestibular inputs. Prediction errors (precision-weighted mismatches between predictions and actual values) drive fast within-timestep state updates via gradient descent on variational free energy, followed by slow Hebbian weight updates.

**Training.** The network is trained on paired stimuli that mimic the sensory consequences of rhythmic locomotion: discrete auditory pulses (footsteps) and a continuous triangular vestibular waveform (trunk acceleration). The continuous vestibular signal is key — it bridges the silent gaps between auditory events, solving the temporal credit assignment problem that makes learning from sparse discrete events alone intractable.

**Key results:**

- **Rhythm learning.** After training on correlated auditory-vestibular input, the network learns to anticipate both modalities, with predictions that lead rather than lag the sensory inputs.
- **Cross-modal prediction (active inference).** When tested with auditory-only input, the trained network spontaneously generates vestibular predictions matching the training waveform — effectively predicting movement from sound alone. Under active inference, these predictions constitute motor commands, providing a computational account for why hearing rhythm evokes the urge to move.
- **Continuation.** The network sustains coherent rhythmic predictions even after all external input ceases, demonstrating autonomous rhythm generation through learned recurrent dynamics.
- **Control conditions.** Training with discrete pulses on both channels (no continuous vestibular signal) fails entirely — the network cannot learn. Training with temporally uncorrelated auditory and vestibular input also fails, confirming that temporal correlation between modalities is essential.

## Requirements

```bash
pip install numpy torch pyyaml tensorboard plotly optuna kaleido scipy matplotlib
```

## Project Structure

```
├── configs/
│   ├── config_sensorimotor.yaml           # Vestibular + correlated auditory
│   ├── config_sensorimotor_doublebeat.yaml # Auditory on both channels (hierarchical)
│   ├── config_beat.yaml                   # Auditory only (no vestibular)
│   ├── config_doublebeat.yaml             # Auditory on both channels
│   ├── config_zero_mean_sensorimotor.yaml # Sensorimotor with zero-mean beat
│   ├── uncorrelated.yaml                  # Vestibular + random auditory (uniform)
│   ├── uncorrelated_poisson.yaml          # Vestibular + random auditory (Poisson)
│   └── uncorrelated_white_noise.yaml      # Vestibular + random auditory (white noise)
├── network.py           # Hierarchical predictive coding RNN (SensorimotorPCRNN)
├── train.py             # Training script
├── test_and_plot.py     # Testing & interactive Plotly visualization
├── utils.py             # Input generation & experiment management
├── logger.py            # Logging utility
├── optuna_tune.py       # Hyperparameter tuning
├── plot_inputs.py       # Standalone input sequence visualization
├── plot_training_log.py # Plot training errors from training.log (Plotly HTML)
├── visualization/       # Publication-ready paper figure pipeline
│   ├── style.py                 # Shared matplotlib config
│   ├── generate_figure_data.py  # Model checkpoints -> .npz data
│   ├── paper_figures.py         # .npz data -> matplotlib figures
│   ├── make_figures.py          # CLI orchestrator
│   └── paper_visualization.py   # Pedagogical/schematic diagrams
└── experiments/         # Output directory (created at runtime)
    ├── sensorimotor/
    ├── doublebeat/
    ├── beat/
    └── uncorrelated/
```

## Configuration

Key config options in `config_sensorimotor.yaml`:

```yaml
experiment:
  tempo:
    mode: "range"         # "single" or "range"
    value: 0.5            # used when mode is "single"
    min: 0.4              # used when mode is "range"
    max: 0.8
    step: 0.1             # creates [0.4, 0.5, 0.6, 0.7, 0.8]
  zero_mean_beat: false   # transform beat pulses to be zero-mean per period
  random_phase: false     # start each sequence at a random phase in the cycle
  random_seed: 111        # seed for reproducibility (used by rng throughout)
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

Limit to a specific number of beats (input zeroed after that beat):
```bash
python test_and_plot.py --config experiments/sensorimotor/{exp_name}/config.yaml --model_step 5000 --n_beats 4
```

Add silent wait time before the input begins:
```bash
python test_and_plot.py --config experiments/sensorimotor/{exp_name}/config.yaml --model_step 5000 --wait_time 0.5
```

Add a free-text note displayed at the bottom of the plot:
```bash
python test_and_plot.py --config experiments/sensorimotor/{exp_name}/config.yaml --model_step 5000 --note "experiment with random phase"
```

Both `auditory_only=True` and `auditory_only=False` conditions are tested and saved automatically as separate plots with compact filenames (e.g. `s5000_t0.5_after.html`, `s5000_t0.5_after_ao.html`).

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

### Paper Figures

Generate all publication-ready figures from trained experiments:

```bash
python -m visualization.make_figures \
    --sensorimotor-config experiments/sensorimotor/{exp_name}/config.yaml \
    --before-step 0 \
    --after-step-sensorimotor 7000 \
    --tempo 0.5 \
    --output-dir visualization/paper_output
```

Optionally include the doublebeat control experiment:

```bash
python -m visualization.make_figures \
    --sensorimotor-config experiments/sensorimotor/{exp_name}/config.yaml \
    --doublebeat-config experiments/doublebeat/{exp_name}/config.yaml \
    --before-step 0 \
    --after-step-sensorimotor 7000 \
    --after-step-doublebeat 1000 \
    --tempo 0.5 \
    --output-dir visualization/paper_output
```

Re-generate figures without re-running inference (for iterating on aesthetics):

```bash
python -m visualization.make_figures \
    --data-dir visualization/paper_output/data \
    --figures-only \
    --output-dir visualization/paper_output
```

Control learning curve smoothing and step range:

```bash
python -m visualization.make_figures \
    --data-dir visualization/paper_output/data \
    --figures-only \
    --output-dir visualization/paper_output \
    --moving-average-window 10 \
    --smoothing-window 50 \
    --lc-start-step 1000 \
    --lc-end-step 50000
```

Overlay another experiment's learning curve for comparison:

```bash
python -m visualization.make_figures \
    --data-dir visualization/paper_output/data \
    --figures-only \
    --output-dir visualization/paper_output \
    --overlay-lc-data other_experiment/data/learning_curve_sensorimotor.npz \
    --lc-label "Sensorimotor" \
    --overlay-lc-label "Uncorrelated control" \
    --overlay-lc-start-step 5000 \
    --overlay-lc-end-step 40000
```

See `visualization/README.md` for full documentation of the figure pipeline.

### Training Log Visualization

Plot all training errors from a `training.log` file as an interactive Plotly HTML:

```bash
python plot_training_log.py experiments/sensorimotor/{exp_name}/training.log
```

With step range and smoothing:

```bash
python plot_training_log.py experiments/sensorimotor/{exp_name}/training.log \
    --start-step 1000 --end-step 50000 \
    --moving-average-window 50 --smoothing-window 30
```

Output: `training_errors.html` next to the log file (or specify `-o path.html`).

## Experiment Modes

- `sensorimotor`: Vestibular triangular wave + correlated auditory pulses (default)
- `doublebeat`: Auditory pulses on both channels (control — no vestibular scaffolding)
- `beat`: Auditory pulses only (no vestibular input)
- `uncorrelated`: Vestibular triangular wave + random uncorrelated auditory pulses
  - `uniform` distribution: inter-pulse intervals drawn from `[ipi_min, ipi_max]`
  - `poisson` distribution: inter-pulse intervals from exponential distribution with rate
  - `white_noise` distribution: each timestep independently has probability `p` of being a pulse
