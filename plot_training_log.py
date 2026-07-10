"""
Plot training errors from a training.log file as an interactive Plotly HTML.

Usage:
    python plot_training_log.py <path_to_training.log>

    # Optional: crop to a step range
    python plot_training_log.py <path_to_training.log> --start-step 1000 --end-step 50000
"""

import argparse
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import uniform_filter1d


def parse_training_log(log_path: str):
    """
    Parse a training.log file and extract step-level metrics.

    Returns:
        dict mapping metric names to lists of (step, value) pairs
    """
    # Pattern matches lines like:
    # Step 100 | v_err=0.007069 | b_err=0.026014 | x_err=0.064169 | ...
    step_pattern = re.compile(r'Step\s+(\d+)\s*\|(.+)')
    metric_pattern = re.compile(r'(\w+)=([\d.]+)')

    metrics = {}

    with open(log_path) as f:
        for line in f:
            m = step_pattern.search(line)
            if not m:
                continue

            step = int(m.group(1))
            pairs = metric_pattern.findall(m.group(2))

            for name, value in pairs:
                if name not in metrics:
                    metrics[name] = []
                metrics[name].append((step, float(value)))

    return metrics


def _smooth(values, moving_average_window=1, smoothing_window=1):
    """
    Apply two levels of smoothing (same as make_figures.py learning curves).

    1. moving_average_window: simple moving average (reduces length by window-1)
    2. smoothing_window: uniform_filter1d (preserves length)
    """
    arr = np.array(values)

    if moving_average_window > 1:
        n = len(arr)
        w = moving_average_window
        arr = np.array([arr[i:i + w].mean() for i in range(n - w + 1)])

    if smoothing_window > 1:
        arr = uniform_filter1d(arr, size=smoothing_window, mode='nearest')

    return arr


def plot_metrics(
    metrics, log_path, start_step=0, end_step=None, output_path=None,
    moving_average_window=1, smoothing_window=1,
):
    """
    Create a Plotly figure with all metrics overlaid.
    """
    # Skip non-error metrics (best_vest_inf is a running best, not per-step)
    skip = {'best_vest_inf'}

    fig = go.Figure()

    for name, data in sorted(metrics.items()):
        if name in skip:
            continue
        steps, values = zip(*data)

        # Apply step range filter
        filtered = [
            (s - start_step, v) for s, v in zip(steps, values)
            if s >= start_step and (end_step is None or s <= end_step)
        ]
        if not filtered:
            continue

        f_steps, f_values = zip(*filtered)

        # Apply smoothing
        f_values_smooth = _smooth(f_values, moving_average_window, smoothing_window)
        # Trim steps to match (moving average shortens the array)
        f_steps = list(f_steps)[:len(f_values_smooth)]

        fig.add_trace(go.Scatter(
            x=f_steps,
            y=f_values_smooth.tolist(),
            mode='lines',
            name=name,
        ))

    fig.update_layout(
        title='Training Log Errors',
        xaxis_title='Training Step',
        yaxis_title='Error',
        yaxis_type='linear',
        height=700,
        width=1400,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    )

    if output_path is None:
        output_path = Path(log_path).parent / 'training_errors.html'

    fig.write_html(str(output_path), auto_open=False)
    print(f"Saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Plot training errors from a training.log file'
    )
    parser.add_argument(
        'log_path', type=str,
        help='Path to the training.log file'
    )
    parser.add_argument(
        '--start-step', type=int, default=0,
        help='First step to show (x-axis re-zeroed). Default: 0'
    )
    parser.add_argument(
        '--end-step', type=int, default=None,
        help='Last step to show. Default: all'
    )
    parser.add_argument(
        '-o', '--output', type=str, default=None,
        help='Output HTML path. Default: training_errors.html next to log file'
    )
    parser.add_argument(
        '--moving-average-window', type=int, default=1,
        help='Simple moving average window (default: 1, no averaging)'
    )
    parser.add_argument(
        '--smoothing-window', type=int, default=1,
        help='uniform_filter1d smoothing window (default: 1, no smoothing)'
    )
    args = parser.parse_args()

    metrics = parse_training_log(args.log_path)

    if not metrics:
        print(f"No step data found in {args.log_path}")
        return

    print(f"Found metrics: {', '.join(sorted(metrics.keys()))}")
    print(f"Steps: {metrics[next(iter(metrics))][0][0]} to {metrics[next(iter(metrics))][-1][0]}")

    plot_metrics(
        metrics,
        log_path=args.log_path,
        start_step=args.start_step,
        end_step=args.end_step,
        output_path=args.output,
        moving_average_window=args.moving_average_window,
        smoothing_window=args.smoothing_window,
    )


if __name__ == '__main__':
    main()
