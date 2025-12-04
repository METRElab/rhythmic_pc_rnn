"""
Simple logging utility for experiment tracking.

Provides file and console logging with best result tracking.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any


class ExperimentLogger:
    """
    A simple logger for tracking experiment progress.

    Logs to both console and file, tracks best results,
    and provides formatted output for training and testing.

    Attributes:
        exp_dir: Path to experiment directory
        logger: Python logger instance
        best_metrics: Dictionary tracking best values and their steps
    """

    def __init__(
        self, exp_dir: Path, log_filename: str = "training.log", level: str = "INFO"
    ) -> None:
        """
        Initialize the experiment logger.

        Args:
            exp_dir: Path to experiment directory where log file will be saved
            log_filename: Name of the log file
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Track best metrics: {metric_name: {"value": float, "step": int}}
        self.best_metrics: Dict[str, Dict[str, Any]] = {}

        # Create logger
        self.logger = logging.getLogger(f"experiment_{exp_dir.name}")
        self.logger.setLevel(getattr(logging, level.upper()))

        # Clear any existing handlers
        self.logger.handlers = []

        # Create formatters
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        # File handler
        log_path = self.exp_dir / log_filename
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def info(self, message: str) -> None:
        """
        Log an info message.

        Args:
            message: Message to log
        """
        self.logger.info(message)

    def debug(self, message: str) -> None:
        """
        Log a debug message.

        Args:
            message: Message to log
        """
        self.logger.debug(message)

    def warning(self, message: str) -> None:
        """
        Log a warning message.

        Args:
            message: Message to log
        """
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """
        Log an error message.

        Args:
            message: Message to log
        """
        self.logger.error(message)

    def update_best(self, metric_name: str, value: float, step: int) -> bool:
        """
        Update best metric if current value is better (lower).

        Args:
            metric_name: Name of the metric
            value: Current metric value
            step: Current training step

        Returns:
            True if this is a new best value, False otherwise
        """
        if metric_name not in self.best_metrics:
            self.best_metrics[metric_name] = {"value": value, "step": step}
            return True

        if value < self.best_metrics[metric_name]["value"]:
            self.best_metrics[metric_name] = {"value": value, "step": step}
            return True

        return False

    def get_best(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the best value for a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Dictionary with "value" and "step" keys, or None if not tracked
        """
        return self.best_metrics.get(metric_name)

    def log_training_step(
        self,
        step: int,
        vest_error: float,
        beat_error: float,
        vest_inference_error: float,
        beat_inference_error: float,
    ) -> None:
        """
        Log metrics for a training step.

        Args:
            step: Current training step
            vest_error: Vestibular prediction error (training)
            beat_error: Beat prediction error (training)
            vest_inference_error: Vestibular inference error
            beat_inference_error: Beat inference error
        """
        total_error = vest_error + beat_error
        total_inference_error = vest_inference_error + beat_inference_error

        # Update best tracking
        is_new_best = self.update_best(
            "vest_inference_error", vest_inference_error, step
        )
        best_info = self.get_best("vest_inference_error")

        best_str = ""
        if best_info:
            best_str = f" | best_vest_inference={best_info['value']:.6f} (step {best_info['step']})"

        message = (
            f"Step {step} | "
            f"vest_error={vest_error:.6f} | "
            f"beat_error={beat_error:.6f} | "
            f"total_error={total_error:.6f} | "
            f"vest_inference={vest_inference_error:.6f} | "
            f"beat_inference={beat_inference_error:.6f} | "
            f"total_inference={total_inference_error:.6f}"
            f"{best_str}"
        )

        if is_new_best:
            message += " [NEW BEST]"

        self.info(message)

    def log_model_saved(self, step: int, path: Optional[Path] = None) -> None:
        """
        Log that a model was saved.

        Args:
            step: Training step at which model was saved
            path: Optional path where model was saved
        """
        if path:
            self.info(f"Model saved at step {step} to {path}")
        else:
            self.info(f"Model saved at step {step}")

    def log_testing_start(
        self, model_step: int, tempo: float, continuation: bool, prediction_timing: str
    ) -> None:
        """
        Log the start of a testing session.

        Args:
            model_step: Step of the loaded model
            tempo: Tempo being tested
            continuation: Whether continuation mode is enabled
            prediction_timing: "before" or "after" inference
        """
        self.info(
            f"Testing started | model_step={model_step} | tempo={tempo} | "
            f"continuation={continuation} | prediction_timing={prediction_timing}"
        )

    def log_testing_complete(
        self, vest_error: float, beat_error: float, plot_path: Optional[Path] = None
    ) -> None:
        """
        Log the completion of a testing session.

        Args:
            vest_error: Average vestibular error during testing
            beat_error: Average beat error during testing
            plot_path: Optional path where plot was saved
        """
        total_error = vest_error + beat_error
        self.info(
            f"Test complete | vest_error={vest_error:.6f} | "
            f"beat_error={beat_error:.6f} | total_error={total_error:.6f}"
        )
        if plot_path:
            self.info(f"Plot saved: {plot_path}")

    def log_experiment_start(self, config: Dict[str, Any]) -> None:
        """
        Log the start of an experiment with configuration summary.

        Args:
            config: Experiment configuration dictionary
        """
        self.info("=" * 60)
        self.info("EXPERIMENT STARTED")
        self.info("=" * 60)
        self.info(f"Experiment name: {config['experiment']['name']}")
        self.info(f"Mode: {config['experiment']['mode']}")

        tempo_config = config["experiment"]["tempo"]
        if tempo_config["mode"] == "single":
            self.info(f"Tempo: {tempo_config['value']} (single)")
        else:
            self.info(
                f"Tempo: range [{tempo_config['min']}, {tempo_config['max']}] "
                f"step={tempo_config['step']}"
            )

        self.info(f"Training rounds: {config['experiment']['n_training_rounds']}")
        self.info(f"Network size: {config['network']['associative_size']}")
        self.info("=" * 60)

    def log_experiment_end(self) -> None:
        """
        Log the end of an experiment with best results summary.
        """
        self.info("=" * 60)
        self.info("EXPERIMENT COMPLETED")
        self.info("=" * 60)

        if self.best_metrics:
            self.info("Best results:")
            for metric_name, data in self.best_metrics.items():
                self.info(
                    f"  {metric_name}: {data['value']:.6f} at step {data['step']}"
                )

        self.info("=" * 60)


def create_logger(
    exp_dir: Path, log_filename: str = "training.log", level: str = "INFO"
) -> ExperimentLogger:
    """
    Factory function to create an ExperimentLogger.

    Args:
        exp_dir: Path to experiment directory
        log_filename: Name of the log file
        level: Logging level

    Returns:
        Configured ExperimentLogger instance
    """
    return ExperimentLogger(exp_dir, log_filename, level)
