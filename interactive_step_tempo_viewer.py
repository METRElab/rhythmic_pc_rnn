"""
Interactive step + tempo viewer.

Like interactive_step_viewer.py, but shows ALL tempos at once: one row per tempo,
two columns (beat channel, vestibular channel). A single slider scrubs through the
saved training checkpoints, swapping every prediction (all tempos, both channels)
at once -- so you can watch the model evolve across training steps AND see how it
does at each tempo simultaneously.

Two HTML files are written to <exp_dir>/inference_plots/:
    step_tempo_viewer_sm.html  -- sensorimotor (both inputs given)
    step_tempo_viewer_ao.html  -- auditory-only (vestibular withheld -> "urge to move")

Usage:
    ~/miniconda3/envs/phd_codes_v2/bin/python interactive_step_tempo_viewer.py \
        --config experiments/revision2/sensorimotor/<exp>/config.yaml
    # optional: --tempos 0.4 0.5 0.6 0.7 0.8  --min-step 0 --max-step 200000
    #           --continuation  --every-nth 2
"""
import argparse
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml

from test_and_plot import run_inference
from utils import generate_input_sequences, get_tempo_values
from visualization.generate_figure_data import load_model as load_model_from_config


def discover_model_steps(exp_dir: Path) -> List[int]:
    pat = re.compile(r"model_step_(\d+)\.pt$")
    return sorted(int(m.group(1)) for f in exp_dir.iterdir()
                  for m in [pat.match(f.name)] if m)


def build_figure(steps, tempos, results, inputs, dt, auditory_only, ann):
    """results[tempo][step] -> dict(beat_pred, vest_pred);  inputs[tempo] -> (vest, beat)."""
    n_t = len(tempos)
    titles = []
    for tempo in tempos:
        titles += [f"tempo {tempo}s — beat", f"tempo {tempo}s — vestibular"]
    fig = make_subplots(rows=n_t, cols=2, subplot_titles=titles,
                        shared_xaxes=True, vertical_spacing=0.04, horizontal_spacing=0.06)

    # Static input traces (always visible)
    for r, tempo in enumerate(tempos, start=1):
        vest_in, beat_in = inputs[tempo]
        t = np.arange(len(beat_in))
        fig.add_trace(go.Scatter(x=t, y=beat_in, mode="lines",
                                 line=dict(color="0.6" if False else "gray", width=1),
                                 name="beat input", showlegend=(r == 1)), row=r, col=1)
        fig.add_trace(go.Scatter(x=t, y=vest_in, mode="lines",
                                 line=dict(color="blue", width=1), name="vestibular input",
                                 showlegend=(r == 1)), row=r, col=2)
    n_static = 2 * n_t

    # Per-step prediction traces (only first step visible initially)
    for i, step in enumerate(steps):
        vis = (i == 0)
        for r, tempo in enumerate(tempos, start=1):
            res = results[tempo][step]
            t = np.arange(len(res["beat_pred"]))
            fig.add_trace(go.Scatter(x=t, y=res["beat_pred"], mode="lines",
                                     line=dict(color="purple", width=1.5), name="beat pred",
                                     visible=vis, showlegend=(vis and r == 1)), row=r, col=1)
            fig.add_trace(go.Scatter(x=t, y=res["vest_pred"], mode="lines",
                                     line=dict(color="red", width=1.5), name="vestibular pred",
                                     visible=vis, showlegend=(vis and r == 1)), row=r, col=2)
    traces_per_step = 2 * n_t

    # Beat-time guide lines
    for r, tempo in enumerate(tempos, start=1):
        _, beat_in = inputs[tempo]
        for b in np.where(np.asarray(beat_in) > 0.5)[0]:
            for c in (1, 2):
                fig.add_vline(x=int(b), line_width=0.5, line_dash="dash",
                              line_color="rgba(255,0,0,0.25)", row=r, col=c)

    # Slider
    label = "auditory-only" if auditory_only else "sensorimotor"
    slider_steps = []
    for i, step in enumerate(steps):
        visibility = [True] * n_static
        for j in range(len(steps)):
            visibility += [j == i] * traces_per_step
        slider_steps.append(dict(method="update", label=str(step),
                                 args=[{"visible": visibility},
                                       {"title": f"{label} — training step {step}   |   {ann}"}]))
    fig.update_layout(
        sliders=[dict(active=0, currentvalue=dict(prefix="training step: "),
                      pad=dict(t=50), steps=slider_steps)],
        title=f"{label} — training step {steps[0]}   |   {ann}",
        height=max(900, 230 * n_t), width=1700, showlegend=True, margin=dict(b=80))
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tempos", type=float, nargs="*", default=None)
    ap.add_argument("--min-step", type=int, default=None)
    ap.add_argument("--max-step", type=int, default=None)
    ap.add_argument("--every-nth", type=int, default=1, help="subsample checkpoints")
    ap.add_argument("--continuation", action="store_true")
    ap.add_argument("--prediction-timing", choices=["before", "after"], default="before",
                    help="'before' = genuine top-down anticipation (correct for plots); "
                         "'after' fits the current input and is misleading for the combined condition")
    args = ap.parse_args()

    config_path = Path(args.config)
    exp_dir = config_path.parent
    config = yaml.safe_load(open(config_path))

    steps = discover_model_steps(exp_dir)
    if args.min_step is not None:
        steps = [s for s in steps if s >= args.min_step]
    if args.max_step is not None:
        steps = [s for s in steps if s <= args.max_step]
    steps = steps[::max(1, args.every_nth)]
    if not steps:
        print(f"No checkpoints found in {exp_dir}")
        return
    tempos = args.tempos if args.tempos else get_tempo_values(config)
    print(f"{len(steps)} checkpoints ({steps[0]}..{steps[-1]}), tempos {tempos}")

    dt = config["experiment"]["dt"]
    zero_mean = config["experiment"].get("zero_mean_beat", False)
    mode = config["experiment"]["mode"]

    # Pre-generate inputs per tempo (fixed phase for a clean, comparable view)
    inputs, baseline = {}, {}
    for tempo in tempos:
        ecfg = dict(config)
        ecfg["experiment"] = dict(config["experiment"])
        ecfg["experiment"]["random_phase"] = False
        ecfg["experiment"]["tempo"] = {"mode": "single", "value": tempo}
        res = generate_input_sequences(tempo=tempo, config=ecfg, rng=np.random.default_rng(0))
        if mode == "beat":
            beat = res; vest = (res * 0.0)
        else:
            vest, beat = res
        inputs[tempo] = (vest.cpu().numpy(), beat.cpu().numpy())
        baseline[tempo] = (-1.0 / round(tempo / dt)) if zero_mean else 0.0

    # Inference for every (checkpoint, tempo, condition)
    res_sm = {tp: {} for tp in tempos}
    res_ao = {tp: {} for tp in tempos}
    for k, step in enumerate(steps):
        print(f"  [{k+1}/{len(steps)}] step {step}")
        network, _ = load_model_from_config(config_path, step)
        for tempo in tempos:
            ecfg = dict(config); ecfg["experiment"] = dict(config["experiment"])
            ecfg["experiment"]["random_phase"] = False
            ecfg["experiment"]["tempo"] = {"mode": "single", "value": tempo}
            r = generate_input_sequences(tempo=tempo, config=ecfg, rng=np.random.default_rng(0))
            vseq, bseq = (r * 0.0, r) if mode == "beat" else r
            for ao, store in ((False, res_sm), (True, res_ao)):
                store[tempo][step] = run_inference(
                    network=network, vestibular_seq=vseq, beat_seq=bseq,
                    continuation=args.continuation, prediction_timing=args.prediction_timing,
                    auditory_only=ao, beat_baseline=baseline[tempo])

    plots_dir = exp_dir / "inference_plots"
    plots_dir.mkdir(exist_ok=True)
    ann = f"{config_path.parent.name}  |  mode={mode}  |  zero_mean={zero_mean}  |  cont={args.continuation}"
    for ao, store, tag in ((False, res_sm, "sm"), (True, res_ao, "ao")):
        fig = build_figure(steps, tempos, store, inputs, dt, ao, ann)
        out = plots_dir / f"step_tempo_viewer_{tag}.html"
        fig.write_html(str(out), auto_open=False)
        print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
