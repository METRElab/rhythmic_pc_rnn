# Visualization Module

Generates publication-ready paper figures from trained experiment checkpoints.

## Overview

The pipeline has two stages:

1. **Data generation** — loads trained models, runs inference under different conditions, saves raw arrays as `.npz` files.
2. **Figure generation** — loads `.npz` files and produces matplotlib figures (PNG).

These stages are decoupled so you can iterate on figure aesthetics without re-running inference.

## Quick Start

Run from the **project root** (`rhythmic_pc_rnn/`), using the conda environment that has the project dependencies (e.g. `phd_codes_v2`):

```bash
conda activate phd_codes_v2
```

### Full pipeline (data + figures)

```bash
python -m visualization.make_figures \
    --sensorimotor-config experiments/sensorimotor/<exp_name>/config.yaml \
    --before-step 0 \
    --after-step-sensorimotor 7000 \
    --tempo 0.5 \
    --output-dir visualization/paper_output \
    --prediction-timing before
```

python -m visualization.make_figures \
    --sensorimotor-config /Users/matin/mcmaster/cannonlab/phd_codes/predictive_coding/rhythmic_pc_rnn/experiments/paper_revision/good_old_sensorimotor_exp_random_phase_true_20260305_225215/config.yaml \
    --before-step 0 \
    --after-step-sensorimotor 48460 \
    --tempo 0.5 \
    --output-dir visualization/paper_output_good_old_sensorimotor_random_phase \
    --prediction-timing before \
    --moving-average-window 400 \
    --smoothing-window 400 \
    --lc-end-step 60000 \
    --lc-start-step 9000


python -m visualization.make_figures \
    --sensorimotor-config /Users/matin/mcmaster/cannonlab/phd_codes/predictive_coding/rhythmic_pc_rnn/experiments/paper_revision/uncorrelated_control_white_noise_phase_random/config.yaml \
    --before-step 0 \
    --after-step-sensorimotor 6630 \
    --tempo 0.5 \
    --output-dir visualization/paper_output_uncorrelated_white_noise_phase_random \
    --prediction-timing before \
    --moving-average-window 300 \
    --smoothing-window 300 \
    --lc-end-step 61620 \
    --overlay-lc-data /Users/matin/mcmaster/cannonlab/phd_codes/predictive_coding/rhythmic_pc_rnn/visualization/paper_output_good_old_sensorimotor_random_phase/data/learning_curve_sensorimotor.npz \
    --lc-start-step 19000

python -m visualization.make_figures \
    --sensorimotor-config /Users/matin/mcmaster/cannonlab/phd_codes/predictive_coding/rhythmic_pc_rnn/experiments/paper_revision/uncorrelated_control_white_noise_20260212_221431/config.yaml \
    --before-step 0 \
    --after-step-sensorimotor 9240 \
    --tempo 0.5 \
    --output-dir visualization/paper_output_white_noise_new \
    --prediction-timing before \
    --moving-average-window 50 \
    --smoothing-window 100

Optionally include the doublebeat control experiment:

```bash
python -m visualization.make_figures \
    --sensorimotor-config experiments/sensorimotor/<exp_name>/config.yaml \
    --doublebeat-config experiments/doublebeat/<exp_name>/config.yaml \
    --before-step 0 \
    --after-step-sensorimotor 7000 \
    --after-step-doublebeat 1000 \
    --tempo 0.5 \
    --output-dir visualization/paper_output
```

This produces:

```
visualization/paper_output/
    data/
        sensorimotor_before_after.npz
        auditory_only_before_after.npz
        continuation.npz
        continuation_auditory_only.npz
        learning_curve_sensorimotor.npz
        metadata.yaml
        # If --doublebeat-config provided:
        double_auditory_before_after.npz
        learning_curve_doublebeat.npz
    figures/
        figure_1a_learning_curve.png
        figure_1b_before_after.png
        figure_2_auditory_only.png
        figure_3_continuation.png
        figure_3b_continuation_auditory_only.png
        figure_combined.png
        figure_summary.png
        # If --doublebeat-config provided:
        figure_4a_learning_curve_doublebeat.png
        figure_4_double_auditory.png
```

### Re-generate figures only (no inference)

After the data files exist, regenerate figures without re-running the model:

```bash
python -m visualization.make_figures \
    --data-dir visualization/paper_output/data \
    --figures-only \
    --output-dir visualization/paper_output
```

## Using the Python API directly

You can also import the modules in a script or notebook:

```python
from visualization.style import apply_paper_style
from visualization.generate_figure_data import (
    load_model, run_condition_inference, generate_before_after_data,
    extract_tensorboard_learning_curve, save_figure_data, load_figure_data,
)
from visualization.paper_figures import (
    plot_learning_curve, plot_before_after_2x2, plot_continuation,
    plot_double_auditory_1x2, plot_combined_multi_panel,
)

# Apply shared matplotlib style (call once)
apply_paper_style()

# Generate data for one condition
data = generate_before_after_data(
    config_path='experiments/sensorimotor/.../config.yaml',
    before_step=0,
    after_step=7000,
    tempo=0.5,
)

# Plot and save
fig = plot_before_after_2x2(data, max_cycles=6)
fig.savefig('my_figure.png', dpi=300, bbox_inches='tight', facecolor='white')

# Or load previously saved data
data = load_figure_data('visualization/paper_output/data/sensorimotor_before_after.npz')
fig = plot_before_after_2x2(data, max_cycles=4)
```

## Module Reference

### `style.py`

Shared matplotlib configuration. Call `apply_paper_style()` once before generating any figures.

- `PAPER_RCPARAMS` — font sizes, DPI, line widths
- `COLORS` — color constants for all plot elements
- `STYLES` — line style constants (alphas, widths, etc.)
- `apply_paper_style()` — applies rcParams
- `remove_chart_junk(ax)` — removes top/right spines

### `generate_figure_data.py`

Data extraction functions that bridge trained models to visualization data.

| Function | Description |
|---|---|
| `load_model(config_path, model_step)` | Load a trained network from a checkpoint |
| `run_condition_inference(network, config, tempo, ...)` | Run inference and return predictions + input sequences |
| `generate_before_after_data(config_path, before_step, after_step, tempo, ...)` | Generate paired before/after training data |
| `extract_tensorboard_learning_curve(exp_dir, metrics)` | Extract error curves from TensorBoard event files |
| `save_figure_data(data, output_path)` | Save arrays as compressed `.npz` |
| `load_figure_data(path)` | Load `.npz` back to a dict of arrays |
| `generate_all_figure_data(...)` | Master function that produces all `.npz` files for every figure |

### `paper_figures.py`

Plotting functions. Each takes numpy arrays and returns a `matplotlib.Figure`.

| Function | Figures | Layout |
|---|---|---|
| `plot_learning_curve(steps, errors, ...)` | 1a, 4a | Single axes, log-scale y. Supports overlay + legend control |
| `plot_before_after_2x2(data, ...)` | 1b, 2 | 2x2 (vest/aud x before/after) |
| `plot_continuation(data, ...)` | 3 | 3x1 (auditory input + vest pred + aud pred) |
| `plot_double_auditory_1x2(data, ...)` | 4 | 1x2 (before/after, beat only) |
| `plot_combined_multi_panel(data_sm, data_ao, ...)` | Combined | 2x3 (sensorimotor + auditory-only) |
| `plot_summary(lc_steps, lc_errors, data_sm, data_ao, ...)` | Summary | GridSpec: learning curve + 2x2 after-training. Supports overlay + legend control |

`plot_before_after_2x2` handles both Figure 1b and Figure 2 via parameters:

```python
# Figure 1b: standard before/after
fig = plot_before_after_2x2(data, max_cycles=6)

# Figure 2: auditory-only (dotted vestibular ground truth)
fig = plot_before_after_2x2(
    data,
    start_cycle=2, end_cycle=8,
    vest_ground_truth_style=':',
    vest_ground_truth_alpha=0.5,
    vest_label='Original Vestibular Signal (not provided as input)',
)
```

### `make_figures.py`

CLI entry point. See Quick Start above for usage.

## CLI Arguments

```
--sensorimotor-config      Path to sensorimotor experiment config.yaml (required)
--before-step              Model step for "before training" (default: 0)
--after-step-sensorimotor  Final model step for sensorimotor experiment (required)
--tempo                    Tempo in seconds (default: 0.5)
--output-dir               Output directory (default: visualization/paper_output)
--prediction-timing        "before" or "after" inference optimization (default: after)
--figures-only             Skip data generation, only produce figures
--data-dir                 Directory with .npz files (for --figures-only)
--doublebeat-config        Path to doublebeat experiment config.yaml (optional)
--after-step-doublebeat    Final model step for doublebeat (required if --doublebeat-config given)
--moving-average-window    Simple moving average window for learning curves (default: 1, no averaging)
--smoothing-window         uniform_filter1d smoothing window for learning curves (default: 30)
--lc-start-step            First training step to show in learning curves (default: 0).
                           Steps before this are discarded and x-axis is re-zeroed.
--lc-end-step              Last training step to show in learning curves (default: all)
--overlay-lc-data          Path to a second learning_curve_*.npz to overlay with faded colour
--lc-label                 Legend label for the main learning curve (omit for no legend)
--overlay-lc-label         Legend label for the overlay learning curve
--overlay-lc-start-step    First training step to show for the overlay curve (default: 0)
--overlay-lc-end-step      Last training step to show for the overlay curve (default: all)
```

## File Structure

```
visualization/
    __init__.py              # Package init
    style.py                 # Shared matplotlib config
    generate_figure_data.py  # Model -> .npz data pipeline
    paper_figures.py         # Data -> matplotlib figures
    make_figures.py          # CLI orchestrator
    paper_visualization.py   # Pedagogical/schematic diagrams (separate)
    _archive/                # Old notebooks (kept for reference)
```
