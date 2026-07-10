"""
Interactive model step viewer with slider.

Generates two interactive Plotly HTML files (sensorimotor + auditory-only),
each with a slider that lets you scrub through all saved model checkpoints
for an experiment, comparing predictions at different training steps.

Output files are saved to <exp_dir>/inference_plots/:
    step_viewer_sm.html  — sensorimotor condition
    step_viewer_ao.html  — auditory-only condition

Usage:
    python interactive_step_viewer.py \
        --config experiments/sensorimotor/.../config.yaml \
        --tempo 0.5 \
        --prediction-timing after

    # Restrict to a range of steps
    python interactive_step_viewer.py \
        --config experiments/sensorimotor/.../config.yaml \
        --tempo 0.5 \
        --min-step 100 --max-step 5000

    # With continuation mode
    python interactive_step_viewer.py \
        --config experiments/sensorimotor/.../config.yaml \
        --tempo 0.5 \
        --continuation
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import yaml

from test_and_plot import run_inference
from utils import generate_input_sequences, get_tempo_values
from visualization.generate_figure_data import load_model as load_model_from_config


def discover_model_steps(exp_dir: Path) -> List[int]:
    """Find all model_step_*.pt files and return sorted step numbers."""
    pattern = re.compile(r"model_step_(\d+)\.pt$")
    steps = []
    for f in exp_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def build_interactive_figure(
    steps: List[int],
    all_results: Dict[int, Dict[str, np.ndarray]],
    vestibular_seq: torch.Tensor,
    beat_seq: torch.Tensor,
    auditory_only: bool,
    annotation_text: str = "",
) -> go.Figure:
    """
    Build a Plotly figure with a slider over model steps.

    Layout: 4 rows — Beat input, Beat predicted, Vestibular input, Vestibular predicted.
    Each slider position swaps the predicted traces to the corresponding step.
    """
    n_steps_time = len(beat_seq)
    t = np.arange(n_steps_time)

    beat_np = beat_seq.cpu().numpy()
    vest_np = vestibular_seq.cpu().numpy()
    is_2d_vest = vest_np.ndim == 2

    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=[
            "Beat Input",
            "Predicted Beat",
            "Vestibular Input",
            "Predicted Vestibular",
        ],
        shared_xaxes=True,
        vertical_spacing=0.06,
    )

    # Static traces (always visible): input signals
    fig.add_trace(
        go.Scatter(x=t, y=beat_np, mode="lines", name="Beat Input",
                   line=dict(color="red")),
        row=1, col=1,
    )

    if is_2d_vest:
        fig.add_trace(
            go.Scatter(x=t, y=vest_np[:, 0], mode="lines",
                       name="Vestibular ch0", line=dict(color="blue")),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(x=t, y=vest_np[:, 1], mode="lines",
                       name="Vestibular ch1", line=dict(color="cyan")),
            row=3, col=1,
        )
        n_static = 3
    else:
        fig.add_trace(
            go.Scatter(x=t, y=vest_np, mode="lines",
                       name="Vestibular", line=dict(color="blue")),
            row=3, col=1,
        )
        n_static = 2

    # For each model step, add predicted traces (initially invisible except first)
    # We'll track: beat_pred (row2), vest_pred (row4) — possibly 2 channels
    traces_per_step = 1 + (2 if is_2d_vest else 1)  # beat_pred + vest_pred channels

    for i, step in enumerate(steps):
        res = all_results[step]
        visible = i == 0

        # Beat prediction
        fig.add_trace(
            go.Scatter(
                x=t, y=res["beat_pred"],
                mode="lines", name="Predicted Beat",
                line=dict(color="purple"),
                visible=visible,
                showlegend=visible,
            ),
            row=2, col=1,
        )

        # Vestibular prediction
        if is_2d_vest:
            fig.add_trace(
                go.Scatter(
                    x=t, y=res["vest_pred"][:, 0],
                    mode="lines", name="Pred Vest ch0",
                    line=dict(color="green"),
                    visible=visible,
                    showlegend=visible,
                ),
                row=4, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=t, y=res["vest_pred"][:, 1],
                    mode="lines", name="Pred Vest ch1",
                    line=dict(color="magenta"),
                    visible=visible,
                    showlegend=visible,
                ),
                row=4, col=1,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=t, y=res["vest_pred"],
                    mode="lines", name="Predicted Vestibular",
                    line=dict(color="magenta"),
                    visible=visible,
                    showlegend=visible,
                ),
                row=4, col=1,
            )

    # Vertical lines at beat times on all subplots
    beat_times = np.where(beat_np > 0.5)[0]
    for b in beat_times:
        for row in [1, 2, 3, 4]:
            fig.add_vline(
                x=int(b), line_width=1, line_dash="dash",
                line_color="red", row=row, col=1,
            )

    # Build annotation that updates with each slider step
    def _make_annotation(step_val: str) -> dict:
        text = annotation_text.replace("{model_step}", step_val)
        return dict(
            text=text,
            xref="paper", yref="paper",
            x=0.0, y=-0.06,
            showarrow=False,
            font=dict(size=12, color="gray"),
            align="left",
            xanchor="left", yanchor="top",
        )

    # Build slider steps
    slider_steps = []
    for i, step in enumerate(steps):
        # Static traces always visible
        visibility = [True] * n_static
        # For each model step's traces, only the i-th group is visible
        for j in range(len(steps)):
            vis = j == i
            visibility.extend([vis] * traces_per_step)

        ao_label = "auditory-only" if auditory_only else "sensorimotor"
        slider_step = dict(
            method="update",
            args=[
                {"visible": visibility},
                {
                    "title": f"Model Step {step}   ({ao_label})",
                    "annotations": [_make_annotation(str(step))],
                },
            ],
            label=str(step),
        )
        slider_steps.append(slider_step)

    sliders = [dict(
        active=0,
        currentvalue=dict(prefix="Model Step: "),
        pad=dict(t=60),
        steps=slider_steps,
    )]

    ao_label = "auditory-only" if auditory_only else "sensorimotor"
    fig.update_layout(
        sliders=sliders,
        title=f"Model Step {steps[0]}   ({ao_label})",
        height=1100,
        width=3000,
        showlegend=True,
        margin=dict(b=120),
        annotations=[_make_annotation(str(steps[0]))],
    )

    y_labels = ["Beat Input", "Pred Beat", "Vestibular", "Pred Vestibular"]
    for i, label in enumerate(y_labels, start=1):
        fig.update_yaxes(title_text=label, row=i, col=1)

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Interactive viewer: compare model predictions across training steps"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to experiment config.yaml",
    )
    parser.add_argument(
        "--tempo", type=float, default=None,
        help="Tempo in seconds (default: first tempo in config)",
    )
    parser.add_argument(
        "--prediction-timing", type=str, choices=["before", "after"],
        default="after",
        help="Capture predictions before or after inference optimization (default: after)",
    )
    parser.add_argument(
        "--continuation", action="store_true",
        help="Enable continuation mode (no auditory input after 50%% of sequence)",
    )
    parser.add_argument(
        "--min-step", type=int, default=None,
        help="Minimum model step to include (default: all)",
    )
    parser.add_argument(
        "--max-step", type=int, default=None,
        help="Maximum model step to include (default: all)",
    )
    parser.add_argument(
        "--note", type=str, default=None,
        help="Optional note to display at the end of the annotation text",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    exp_dir = config_path.parent

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Discover available steps
    all_steps = discover_model_steps(exp_dir)
    if not all_steps:
        print(f"No model_step_*.pt files found in {exp_dir}")
        return

    # Filter by min/max
    if args.min_step is not None:
        all_steps = [s for s in all_steps if s >= args.min_step]
    if args.max_step is not None:
        all_steps = [s for s in all_steps if s <= args.max_step]

    if not all_steps:
        print("No model steps in the specified range.")
        return

    print(f"Found {len(all_steps)} model steps: {all_steps[0]} .. {all_steps[-1]}")

    # Determine tempo
    if args.tempo is not None:
        tempo = args.tempo
    else:
        tempo = get_tempo_values(config)[0]
    print(f"Tempo: {tempo}")

    # Generate input sequences once
    test_rng = np.random.default_rng(config["experiment"].get("random_seed", 42))
    result = generate_input_sequences(tempo=tempo, config=config, rng=test_rng)

    mode = config["experiment"]["mode"]
    if mode == "beat":
        beat_seq = result
        vestibular_seq = torch.zeros_like(beat_seq)
    else:
        vestibular_seq, beat_seq = result

    dt = config["experiment"]["dt"]
    zero_mean_beat = config["experiment"].get("zero_mean_beat", False)
    if zero_mean_beat:
        steps_per_period = round(tempo / dt)
        beat_baseline = -1.0 / steps_per_period
    else:
        beat_baseline = 0.0

    # Continuation: modify beat_seq for plotting
    if args.continuation:
        continuation_start = int(0.5 * len(beat_seq))
        beat_seq_plot = beat_seq.clone()
        beat_seq_plot[continuation_start + 1 :] = beat_baseline
    else:
        beat_seq_plot = beat_seq

    plots_dir = exp_dir / "inference_plots"
    plots_dir.mkdir(exist_ok=True)

    # Build base annotation parts (model_step is a placeholder updated by the slider)
    mode = config["experiment"]["mode"]
    base_parts = [
        f"config: {config_path.name}",
        f"mode: {mode}",
        "model_step: {model_step}",
        f"tempo: {tempo}",
        f"prediction_timing: {args.prediction_timing}",
        f"continuation: {args.continuation}",
    ]
    if zero_mean_beat:
        base_parts.append(f"zero_mean_beat: True (baseline={beat_baseline:.4f})")
    if args.note:
        base_parts.append(f"note: {args.note}")

    # Run both conditions: sensorimotor (False) and auditory-only (True)
    for auditory_only in [False, True]:
        ao_label = "auditory-only" if auditory_only else "sensorimotor"
        print(f"\n--- {ao_label} condition ---")

        annotation_text = "  |  ".join(base_parts + [f"auditory_only: {auditory_only}"])

        all_results = {}
        for i, step in enumerate(all_steps):
            print(f"  [{i + 1}/{len(all_steps)}] Running inference for step {step}...")
            network, _ = load_model_from_config(config_path, step)
            res = run_inference(
                network=network,
                vestibular_seq=vestibular_seq,
                beat_seq=beat_seq,
                continuation=args.continuation,
                prediction_timing=args.prediction_timing,
                auditory_only=auditory_only,
                beat_baseline=beat_baseline,
            )
            all_results[step] = res

        fig = build_interactive_figure(
            steps=all_steps,
            all_results=all_results,
            vestibular_seq=vestibular_seq,
            beat_seq=beat_seq_plot,
            auditory_only=auditory_only,
            annotation_text=annotation_text,
        )

        ao_tag = "ao" if auditory_only else "sm"
        output_path = plots_dir / f"step_viewer_{ao_tag}.html"
        fig.write_html(str(output_path), auto_open=False)
        print(f"  Saved to {output_path}")


if __name__ == "__main__":
    main()
