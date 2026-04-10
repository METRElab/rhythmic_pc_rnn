"""
Orchestration script: generate all paper figures from trained experiments.

Usage:
    # Full pipeline: sensorimotor only
    python -m visualization.make_figures \\
        --sensorimotor-config experiments/sensorimotor/.../config.yaml \\
        --before-step 0 \\
        --after-step-sensorimotor 7000 \\
        --tempo 0.5 \\
        --output-dir visualization/paper_output

    # Full pipeline: with doublebeat (optional)
    python -m visualization.make_figures \\
        --sensorimotor-config experiments/sensorimotor/.../config.yaml \\
        --doublebeat-config experiments/doublebeat/.../config.yaml \\
        --before-step 0 \\
        --after-step-sensorimotor 7000 \\
        --after-step-doublebeat 1000 \\
        --tempo 0.5 \\
        --output-dir visualization/paper_output

    # Re-generate figures only (from cached .npz data)
    python -m visualization.make_figures \\
        --data-dir visualization/paper_output/data \\
        --figures-only \\
        --output-dir visualization/paper_output
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from visualization.generate_figure_data import (
    generate_all_figure_data,
    generate_ao_comparison_data,
    load_figure_data,
    save_figure_data,
)
from visualization.paper_figures import (
    plot_ao_comparison,
    plot_before_after_2x2,
    plot_combined_multi_panel,
    plot_continuation,
    plot_double_auditory_1x2,
    plot_learning_curve,
    plot_summary,
)
from visualization.style import apply_paper_style


def _crop_learning_curve(
    steps: np.ndarray,
    errors: np.ndarray,
    start_step: int = 0,
    end_step: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Crop learning curve data to [start_step, end_step] and re-zero the x-axis.

    Args:
        steps: Training step numbers
        errors: Error values
        start_step: First step to include (data before is discarded)
        end_step: Last step to include (None = keep all after start)

    Returns:
        (cropped_steps, cropped_errors) with steps shifted so start_step → 0
    """
    mask = steps >= start_step
    if end_step is not None:
        mask &= steps <= end_step
    steps = steps[mask] - start_step
    errors = errors[mask]
    return steps, errors


def generate_figures(
    data_dir: Path,
    figures_dir: Path,
    moving_average_window: int = 1,
    smoothing_window: int = 30,
    lc_start_step: int = 0,
    lc_end_step: int | None = None,
    overlay_lc_path: Path | None = None,
    lc_label: str | None = None,
    overlay_lc_label: str | None = None,
    overlay_lc_start_step: int = 0,
    overlay_lc_end_step: int | None = None,
    ao_start_cycle: int = 0,
    ao_end_cycle: int = 6,
) -> None:
    """
    Load .npz data files and produce all paper figures.

    Args:
        data_dir: Directory containing the .npz data files
        figures_dir: Directory to save output figures
        moving_average_window: Simple moving average window for learning curves
        smoothing_window: uniform_filter1d window for learning curves
        lc_start_step: First training step to show in learning curves.
            Steps before this are discarded and the x-axis is re-zeroed
            so that this step appears as step 0.
        lc_end_step: Last training step to show (inclusive). None = show all.
        overlay_lc_path: Path to a second .npz learning curve file to overlay
            with faded colour on learning curve plots.
        lc_label: Legend label for the main learning curve. If None, no legend.
        overlay_lc_label: Legend label for the overlay curve. If None, no label.
        overlay_lc_start_step: First training step to show for the overlay curve.
        overlay_lc_end_step: Last training step to show for the overlay curve.
    """
    data_dir = Path(data_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    apply_paper_style()

    # Detect which data files are available
    sm_path = data_dir / 'sensorimotor_before_after.npz'
    ao_path = data_dir / 'auditory_only_before_after.npz'
    cont_path = data_dir / 'continuation.npz'
    cont_ao_path = data_dir / 'continuation_auditory_only.npz'
    lc_sm_path = data_dir / 'learning_curve_sensorimotor.npz'
    lc_db_path = data_dir / 'learning_curve_doublebeat.npz'
    db_path = data_dir / 'double_auditory_before_after.npz'

    ao_cmp_path = data_dir / 'ao_comparison.npz'
    has_doublebeat = lc_db_path.exists() or db_path.exists()
    has_ao_cmp = ao_cmp_path.exists()
    total = 7 + (2 if has_doublebeat else 0) + (1 if has_ao_cmp else 0)
    step = 0

    # Load overlay learning curve if provided, with its own start/end cropping
    overlay_steps = None
    overlay_errors = None
    if overlay_lc_path is not None and Path(overlay_lc_path).exists():
        ov_data = load_figure_data(Path(overlay_lc_path))
        ov_steps = ov_data['beat_error_steps']
        ov_errors = ov_data['beat_error_values'] + ov_data['vest_error_values']
        overlay_steps, overlay_errors = _crop_learning_curve(
            ov_steps, ov_errors, overlay_lc_start_step, overlay_lc_end_step,
        )
        print(f"  Overlay LC loaded from {overlay_lc_path}")

    print("=== Generating paper figures ===")

    # Figure 1a: Learning curve (sensorimotor)
    step += 1
    if lc_sm_path.exists():
        print(f"\n[{step}/{total}] Figure 1a: Learning curve (sensorimotor)")
        lc_data = load_figure_data(lc_sm_path)
        # Sum beat + vest errors for total error
        steps = lc_data['beat_error_steps']
        total_errors = lc_data['beat_error_values'] + lc_data['vest_error_values']
        # total_errors = lc_data['vest_error_values']
        # total_errors = lc_data['beat_error_values']
        steps, total_errors = _crop_learning_curve(
            steps, total_errors, lc_start_step, lc_end_step,
        )
        fig = plot_learning_curve(
            steps, total_errors,
            smoothing_window=smoothing_window,
            moving_average_window=moving_average_window,
            label=lc_label,
            overlay_steps=overlay_steps,
            overlay_errors=overlay_errors,
            overlay_label=overlay_lc_label,
        )
        fig.savefig(
            figures_dir / 'figure_1a_learning_curve.png',
            dpi=300, bbox_inches='tight', facecolor='white',
        )
        print(f"  Saved to {figures_dir / 'figure_1a_learning_curve.png'}")
    else:
        print(f"[{step}/{total}] Skipping Figure 1a (no TensorBoard data)")

    # Figure 1b: Sensorimotor before/after
    step += 1
    if sm_path.exists():
        print(f"[{step}/{total}] Figure 1b: Sensorimotor before/after")
        sm_data = load_figure_data(sm_path)
        fig = plot_before_after_2x2(sm_data, max_cycles=6)
        fig.savefig(
            figures_dir / 'figure_1b_before_after.png',
            dpi=300, bbox_inches='tight', facecolor='white',
        )
        print(f"  Saved to {figures_dir / 'figure_1b_before_after.png'}")

    # Figure 2: Auditory-only
    step += 1
    if ao_path.exists():
        print(f"[{step}/{total}] Figure 2: Auditory-only")
        ao_data = load_figure_data(ao_path)
        fig = plot_before_after_2x2(
            ao_data,
            start_cycle=2,
            end_cycle=8,
            vest_ground_truth_style=':',
            vest_ground_truth_alpha=0.5,
            vest_label='Original Vestibular Signal (not provided as input)',
        )
        fig.savefig(
            figures_dir / 'figure_2_auditory_only.png',
            dpi=300, bbox_inches='tight', facecolor='white',
        )
        print(f"  Saved to {figures_dir / 'figure_2_auditory_only.png'}")

    # Figure 3: Continuation
    step += 1
    if cont_path.exists():
        print(f"[{step}/{total}] Figure 3: Continuation")
        cont_data = load_figure_data(cont_path)
        fig = plot_continuation(cont_data)
        fig.savefig(
            figures_dir / 'figure_3_continuation.png',
            dpi=300, bbox_inches='tight', facecolor='white',
        )
        print(f"  Saved to {figures_dir / 'figure_3_continuation.png'}")

    # Figure 3b: Continuation auditory-only
    step += 1
    if cont_ao_path.exists():
        print(f"[{step}/{total}] Figure 3b: Continuation (auditory-only)")
        cont_ao_data = load_figure_data(cont_ao_path)
        fig = plot_continuation(cont_ao_data)
        fig.savefig(
            figures_dir / 'figure_3b_continuation_auditory_only.png',
            dpi=300, bbox_inches='tight', facecolor='white',
        )
        print(f"  Saved to {figures_dir / 'figure_3b_continuation_auditory_only.png'}")

    # Figure combined: Multi-panel
    step += 1
    if sm_path.exists() and ao_path.exists():
        print(f"[{step}/{total}] Combined multi-panel figure")
        sm_data = load_figure_data(sm_path)
        ao_data = load_figure_data(ao_path)
        fig = plot_combined_multi_panel(
            sm_data, ao_data,
            max_cycles=6, start_cycle_c=2, end_cycle_c=8,
        )
        fig.savefig(
            figures_dir / 'figure_combined.png',
            dpi=300, bbox_inches='tight', facecolor='white',
        )
        print(f"  Saved to {figures_dir / 'figure_combined.png'}")

    # Summary figure: Learning curve + after-training panels
    step += 1
    if lc_sm_path.exists() and sm_path.exists() and ao_path.exists():
        print(f"[{step}/{total}] Summary figure (learning curve + after-training)")
        lc_data = load_figure_data(lc_sm_path)
        sm_data = load_figure_data(sm_path)
        ao_data = load_figure_data(ao_path)
        lc_steps_arr = lc_data['beat_error_steps']
        lc_total_errors = lc_data['beat_error_values'] + lc_data['vest_error_values']
        lc_steps_arr, lc_total_errors = _crop_learning_curve(
            lc_steps_arr, lc_total_errors, lc_start_step, lc_end_step,
        )
        fig = plot_summary(
            lc_steps_arr, lc_total_errors,
            sm_data, ao_data,
            smoothing_window=smoothing_window,
            moving_average_window=moving_average_window,
            lc_label=lc_label,
            overlay_steps=overlay_steps,
            overlay_errors=overlay_errors,
            overlay_label=overlay_lc_label,
        )
        fig.savefig(
            figures_dir / 'figure_summary.png',
            dpi=300, bbox_inches='tight', facecolor='white',
        )
        print(f"  Saved to {figures_dir / 'figure_summary.png'}")

    # AO comparison figure (if data exists)
    if has_ao_cmp:
        step += 1
        print(f"[{step}/{total}] AO comparison figure")
        ao_cmp_data = load_figure_data(ao_cmp_path)
        fig = plot_ao_comparison(
            ao_cmp_data,
            start_cycle=ao_start_cycle,
            end_cycle=ao_end_cycle,
        )
        fig.savefig(
            figures_dir / 'figure_ao_comparison.png',
            dpi=300, bbox_inches='tight', facecolor='white',
        )
        print(f"  Saved to {figures_dir / 'figure_ao_comparison.png'}")

    # Doublebeat figures (only if data exists)
    if has_doublebeat:
        step += 1
        if lc_db_path.exists():
            print(f"[{step}/{total}] Figure 4a: Learning curve (doublebeat)")
            lc_data = load_figure_data(lc_db_path)
            steps = lc_data['beat_error_steps']
            total_errors = lc_data['beat_error_values'] + lc_data['vest_error_values']
            steps, total_errors = _crop_learning_curve(
                steps, total_errors, lc_start_step, lc_end_step,
            )
            fig = plot_learning_curve(
                steps, total_errors,
                smoothing_window=smoothing_window,
                moving_average_window=moving_average_window,
            )
            fig.savefig(
                figures_dir / 'figure_4a_learning_curve_doublebeat.png',
                dpi=300, bbox_inches='tight', facecolor='white',
            )
            print(f"  Saved to {figures_dir / 'figure_4a_learning_curve_doublebeat.png'}")
        else:
            print(f"[{step}/{total}] Skipping Figure 4a (no TensorBoard data)")

        step += 1
        if db_path.exists():
            print(f"[{step}/{total}] Figure 4: Double auditory")
            db_data = load_figure_data(db_path)
            fig = plot_double_auditory_1x2(db_data, start_cycle=0, end_cycle=6)
            fig.savefig(
                figures_dir / 'figure_4_double_auditory.png',
                dpi=300, bbox_inches='tight', facecolor='white',
            )
            print(f"  Saved to {figures_dir / 'figure_4_double_auditory.png'}")

    print(f"\nAll figures saved to {figures_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate all paper figures from trained experiments"
    )

    # Data generation arguments
    parser.add_argument(
        '--sensorimotor-config', type=str,
        help='Path to sensorimotor experiment config.yaml',
    )
    parser.add_argument(
        '--doublebeat-config', type=str,
        help='Path to doublebeat experiment config.yaml',
    )
    parser.add_argument(
        '--before-step', type=int, default=0,
        help='Model step for "before training" (default: 0)',
    )
    parser.add_argument(
        '--after-step-sensorimotor', type=int,
        help='Model step for "after training" (sensorimotor)',
    )
    parser.add_argument(
        '--after-step-doublebeat', type=int,
        help='Model step for "after training" (doublebeat)',
    )
    parser.add_argument(
        '--tempo', type=float, default=0.5,
        help='Tempo in seconds (default: 0.5)',
    )

    # Output arguments
    parser.add_argument(
        '--output-dir', type=str, default='visualization/paper_output',
        help='Output directory (default: visualization/paper_output)',
    )

    # Figures-only mode
    parser.add_argument(
        '--figures-only', action='store_true',
        help='Skip data generation, only produce figures from existing .npz files',
    )
    parser.add_argument(
        '--data-dir', type=str,
        help='Directory with existing .npz files (for --figures-only mode)',
    )
    parser.add_argument(
        '--prediction-timing',
        type=str,
        choices=['before', 'after'],
        default='after',
        help='When to capture predictions: before or after inference optimization'
    )
    parser.add_argument(
        '--moving-average-window', type=int, default=1,
        help='Simple moving average window for learning curves (default: 1, no averaging)',
    )
    parser.add_argument(
        '--smoothing-window', type=int, default=30,
        help='uniform_filter1d smoothing window for learning curves (default: 30)',
    )
    parser.add_argument(
        '--lc-start-step', type=int, default=0,
        help='First training step to show in learning curves (default: 0). '
             'Steps before this are discarded and x-axis is re-zeroed.',
    )
    parser.add_argument(
        '--lc-end-step', type=int, default=None,
        help='Last training step to show in learning curves (default: all)',
    )
    parser.add_argument(
        '--overlay-lc-data', type=str, default=None,
        help='Path to a second learning_curve_*.npz file to overlay '
             'with faded colour on learning curve plots',
    )
    parser.add_argument(
        '--lc-label', type=str, default=None,
        help='Legend label for the main learning curve. '
             'If omitted (along with --overlay-lc-label), no legend is shown.',
    )
    parser.add_argument(
        '--overlay-lc-label', type=str, default=None,
        help='Legend label for the overlay learning curve.',
    )
    parser.add_argument(
        '--overlay-lc-start-step', type=int, default=0,
        help='First training step to show for the overlay learning curve (default: 0). '
             'Steps before this are discarded and x-axis is re-zeroed.',
    )
    parser.add_argument(
        '--overlay-lc-end-step', type=int, default=None,
        help='Last training step to show for the overlay learning curve (default: all)',
    )
    parser.add_argument(
        '--ao-step-1', type=int, default=None,
        help='Model step for middle panel of AO comparison figure',
    )
    parser.add_argument(
        '--ao-step-2', type=int, default=None,
        help='Model step for right panel of AO comparison figure',
    )
    parser.add_argument(
        '--ao-start-cycle', type=int, default=0,
        help='Starting cycle for AO comparison figure (default: 0)',
    )
    parser.add_argument(
        '--ao-end-cycle', type=int, default=6,
        help='Ending cycle for AO comparison figure (default: 6)',
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    # Common kwargs for generate_figures
    fig_kwargs = dict(
        moving_average_window=args.moving_average_window,
        smoothing_window=args.smoothing_window,
        lc_start_step=args.lc_start_step,
        lc_end_step=args.lc_end_step,
        overlay_lc_path=args.overlay_lc_data,
        lc_label=args.lc_label,
        overlay_lc_label=args.overlay_lc_label,
        overlay_lc_start_step=args.overlay_lc_start_step,
        overlay_lc_end_step=args.overlay_lc_end_step,
        ao_start_cycle=args.ao_start_cycle,
        ao_end_cycle=args.ao_end_cycle,
    )

    if args.figures_only:
        # Use provided data-dir or default to output_dir/data
        data_dir = Path(args.data_dir) if args.data_dir else output_dir / 'data'
        if not data_dir.exists():
            print(f"Error: Data directory not found: {data_dir}")
            sys.exit(1)
        generate_figures(data_dir, output_dir / 'figures', **fig_kwargs)
    else:
        # Full pipeline
        if not args.sensorimotor_config:
            print("Error: --sensorimotor-config is required")
            sys.exit(1)
        if args.after_step_sensorimotor is None:
            print("Error: --after-step-sensorimotor is required")
            sys.exit(1)

        # Doublebeat is optional
        doublebeat_config = Path(args.doublebeat_config) if args.doublebeat_config else None
        if doublebeat_config and args.after_step_doublebeat is None:
            print("Error: --after-step-doublebeat is required when --doublebeat-config is provided")
            sys.exit(1)

        generate_all_figure_data(
            sensorimotor_config=Path(args.sensorimotor_config),
            before_step=args.before_step,
            after_step_sensorimotor=args.after_step_sensorimotor,
            tempo=args.tempo,
            output_dir=output_dir,
            doublebeat_config=doublebeat_config,
            after_step_doublebeat=args.after_step_doublebeat,
            prediction_timing=args.prediction_timing,
        )

        # AO comparison: generate data if both steps are specified
        if args.ao_step_1 is not None and args.ao_step_2 is not None:
            print("\n=== Generating AO comparison data ===")
            ao_data = generate_ao_comparison_data(
                config_path=Path(args.sensorimotor_config),
                ao_step_1=args.ao_step_1,
                ao_step_2=args.ao_step_2,
                tempo=args.tempo,
                prediction_timing=args.prediction_timing,
            )
            save_figure_data(ao_data, output_dir / 'data' / 'ao_comparison.npz')
            print(f"  Saved to {output_dir / 'data' / 'ao_comparison.npz'}")

        generate_figures(output_dir / 'data', output_dir / 'figures', **fig_kwargs)


if __name__ == '__main__':
    main()
