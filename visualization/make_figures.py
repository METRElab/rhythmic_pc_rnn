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
    load_figure_data,
)
from visualization.paper_figures import (
    plot_before_after_2x2,
    plot_combined_multi_panel,
    plot_continuation,
    plot_double_auditory_1x2,
    plot_learning_curve,
)
from visualization.style import apply_paper_style


def generate_figures(data_dir: Path, figures_dir: Path) -> None:
    """
    Load .npz data files and produce all paper figures.

    Args:
        data_dir: Directory containing the .npz data files
        figures_dir: Directory to save output figures
    """
    data_dir = Path(data_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    apply_paper_style()

    # Detect which data files are available
    sm_path = data_dir / 'sensorimotor_before_after.npz'
    ao_path = data_dir / 'auditory_only_before_after.npz'
    cont_path = data_dir / 'continuation.npz'
    lc_sm_path = data_dir / 'learning_curve_sensorimotor.npz'
    lc_db_path = data_dir / 'learning_curve_doublebeat.npz'
    db_path = data_dir / 'double_auditory_before_after.npz'

    has_doublebeat = lc_db_path.exists() or db_path.exists()
    total = 7 if has_doublebeat else 5
    step = 0

    print("=== Generating paper figures ===")

    # Figure 1a: Learning curve (sensorimotor)
    step += 1
    if lc_sm_path.exists():
        print(f"\n[{step}/{total}] Figure 1a: Learning curve (sensorimotor)")
        lc_data = load_figure_data(lc_sm_path)
        # Sum beat + vest errors for total error
        steps = lc_data['beat_error_steps']
        total_errors = lc_data['beat_error_values'] + lc_data['vest_error_values']
        fig = plot_learning_curve(steps, total_errors, smoothing_window=30)
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

    # Doublebeat figures (only if data exists)
    if has_doublebeat:
        step += 1
        if lc_db_path.exists():
            print(f"[{step}/{total}] Figure 4a: Learning curve (doublebeat)")
            lc_data = load_figure_data(lc_db_path)
            steps = lc_data['beat_error_steps']
            total_errors = lc_data['beat_error_values'] + lc_data['vest_error_values']
            fig = plot_learning_curve(steps, total_errors, smoothing_window=10)
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

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    if args.figures_only:
        # Use provided data-dir or default to output_dir/data
        data_dir = Path(args.data_dir) if args.data_dir else output_dir / 'data'
        if not data_dir.exists():
            print(f"Error: Data directory not found: {data_dir}")
            sys.exit(1)
        generate_figures(data_dir, output_dir / 'figures')
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

        generate_figures(output_dir / 'data', output_dir / 'figures')


if __name__ == '__main__':
    main()
