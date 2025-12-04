"""
Predictive coding recurrent neural network for sensorimotor synchronization.

Implements a recurrent predictive coding network with parallel vestibular
and auditory sensory pathways for learning rhythm through prediction error
minimization.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class SensorimotorPCRNN(nn.Module):
    """
    Predictive coding recurrent neural network for sensorimotor learning.

    This network learns to predict both vestibular and auditory (beat) inputs
    through a recurrent associative layer. It implements:
    - Recurrent connections for temporal prediction
    - Parallel sensory pathways for vestibular and beat inputs
    - Hebbian learning for weight updates
    - Fast inference dynamics for state optimization

    Attributes:
        associative_size: Number of units in the associative layer
        vestibular_size: Dimension of vestibular input
        inference_learning_rate: Learning rate for state optimization
        weight_learning_rate: Learning rate for Hebbian weight updates
        n_inference_steps: Number of Euler integration steps per timestep
        Wrec: Recurrent weight matrix
        W_vestibular: Vestibular prediction weights
        W_beat: Beat prediction weights
        x: Current hidden state
        x_prev: Previous hidden state
    """

    def __init__(
        self,
        associative_size: int,
        vestibular_size: int,
        inference_learning_rate: float,
        weight_learning_rate: float,
        n_inference_steps: int,
        random_seed: int = 111,
    ) -> None:
        """
        Initialize the predictive coding network.

        Args:
            associative_size: Number of units in associative layer
            vestibular_size: Dimension of vestibular input (typically 1)
            inference_learning_rate: Learning rate for fast state dynamics
            weight_learning_rate: Learning rate for Hebbian weight updates
            n_inference_steps: Number of Euler steps per timestep
            random_seed: Seed for weight initialization
        """
        super().__init__()

        # Store parameters
        self.associative_size = associative_size
        self.vestibular_size = vestibular_size
        self.inference_learning_rate = inference_learning_rate
        self.weight_learning_rate = weight_learning_rate
        self.n_inference_steps = n_inference_steps
        self.random_seed = random_seed

        # Initialize recurrent weights for associative area
        self.Wrec = nn.Parameter(
            self._init_weights(associative_size, associative_size), requires_grad=False
        )

        # Initialize prediction weights for sensory pathways
        self.W_vestibular = nn.Parameter(
            self._init_weights(vestibular_size, associative_size), requires_grad=False
        )
        self.W_beat = nn.Parameter(
            self._init_weights(1, associative_size), requires_grad=False
        )

        # Initialize states
        self.x = torch.zeros(associative_size, 1)
        self.x_prev = torch.zeros(associative_size, 1)

    def reset_states(self) -> None:
        """Reset network hidden states to zeros."""
        self.x = torch.zeros(self.associative_size, 1)
        self.x_prev = torch.zeros(self.associative_size, 1)

    def _init_weights(self, out_size: int, in_size: int) -> torch.Tensor:
        """
        Initialize weights with small random values.

        Args:
            out_size: Output dimension
            in_size: Input dimension

        Returns:
            Initialized weight tensor of shape [out_size, in_size]
        """
        torch.manual_seed(self.random_seed)
        return torch.randn(out_size, in_size) * 0.05

    @staticmethod
    def _tanh_derivative(x: torch.Tensor) -> torch.Tensor:
        """
        Compute derivative of tanh activation function.

        Args:
            x: Input tensor

        Returns:
            Derivative of tanh at x: 1 - tanh(x)^2
        """
        return 1 - torch.tanh(x) ** 2

    def compute_predictions(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute predictions for all units and sensory inputs.

        Returns:
            Tuple of (mu_rec, mu_v, mu_b):
                - mu_rec: Recurrent prediction from previous state
                - mu_v: Vestibular prediction from current state
                - mu_b: Beat prediction from current state
        """
        # Recurrent predictions use previous state
        mu_rec = self.Wrec @ torch.tanh(self.x_prev)

        # Sensory predictions use current state
        mu_v = self.W_vestibular @ torch.tanh(self.x)
        mu_b = self.W_beat @ torch.tanh(self.x)

        return mu_rec, mu_v, mu_b

    def compute_prediction_errors(
        self,
        vestibular_input: Optional[torch.Tensor],
        beat_input: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute prediction errors for all units and inputs.

        Args:
            vestibular_input: Vestibular sensory input or None
            beat_input: Beat sensory input or None

        Returns:
            Tuple of (e_rec, e_v, e_b):
                - e_rec: Recurrent prediction error
                - e_v: Vestibular prediction error
                - e_b: Beat prediction error
        """
        mu_rec, mu_v, mu_b = self.compute_predictions()

        # Compute errors
        e_rec = self.x - mu_rec
        e_v = (
            vestibular_input - mu_v
            if vestibular_input is not None
            else torch.zeros_like(mu_v)
        )
        e_b = beat_input - mu_b if beat_input is not None else torch.zeros_like(mu_b)

        return e_rec, e_v, e_b

    def optimize_states(
        self,
        vestibular_input: Optional[torch.Tensor],
        beat_input: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Update states to minimize prediction errors via gradient descent.

        Args:
            vestibular_input: Vestibular sensory input or None
            beat_input: Beat sensory input or None

        Returns:
            Recurrent prediction error tensor
        """
        e_rec, e_v, e_b = self.compute_prediction_errors(vestibular_input, beat_input)

        # Start with recurrent errors
        dx = -e_rec

        # Add sensory prediction errors if inputs available
        if vestibular_input is not None:
            dx += e_v * self.W_vestibular.T * self._tanh_derivative(self.x)
        if beat_input is not None:
            dx += e_b * self.W_beat.T * self._tanh_derivative(self.x)

        self.x += self.inference_learning_rate * dx
        return e_rec

    def compute_free_energy(
        self,
        vestibular_input: Optional[torch.Tensor],
        beat_input: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute free energy (sum of squared prediction errors).

        Args:
            vestibular_input: Vestibular sensory input or None
            beat_input: Beat sensory input or None

        Returns:
            Scalar free energy value
        """
        e_rec, e_v, e_b = self.compute_prediction_errors(vestibular_input, beat_input)
        return 0.5 * (torch.sum(e_rec**2) + torch.sum(e_v**2) + e_b**2)

    def update_weights(
        self, e_rec: torch.Tensor, e_v: torch.Tensor, e_b: torch.Tensor
    ) -> None:
        """
        Update weights using Hebbian learning rule.

        Weight updates are proportional to the product of prediction error
        and presynaptic activity.

        Args:
            e_rec: Recurrent prediction error
            e_v: Vestibular prediction error
            e_b: Beat prediction error
        """
        # Compute weight updates
        dWrec = e_rec @ torch.tanh(self.x_prev).T
        dW_vestibular = e_v @ torch.tanh(self.x).T
        dW_beat = e_b @ torch.tanh(self.x).T

        # Apply updates
        self.Wrec += self.weight_learning_rate * dWrec
        self.W_vestibular += self.weight_learning_rate * dW_vestibular
        self.W_beat += self.weight_learning_rate * dW_beat

    def timestep_train(
        self, vestibular_input: torch.Tensor, beat_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process one training timestep with both inputs available.

        Performs state transition, state optimization, and weight learning.

        Args:
            vestibular_input: Vestibular sensory input
            beat_input: Beat sensory input

        Returns:
            Tuple of (vestibular_pred, beat_pred) predictions before optimization
        """
        vestibular_input = vestibular_input.view(self.vestibular_size, 1)
        beat_input = beat_input.view(1, 1)

        # Store current state as previous
        self.x_prev = self.x.clone()
        self.x = self.Wrec @ torch.tanh(self.x_prev)

        # Get predictions before optimizing states
        _, vestibular_pred_before, beat_pred_before = self.compute_predictions()

        # Optimize states
        for _ in range(self.n_inference_steps):
            self.optimize_states(vestibular_input, beat_input)

        # Compute final errors and update weights
        e_rec, e_v, e_b = self.compute_prediction_errors(vestibular_input, beat_input)
        self.update_weights(e_rec, e_v, e_b)

        return vestibular_pred_before, beat_pred_before

    def timestep_inference(
        self,
        vestibular_input: torch.Tensor,
        beat_input: torch.Tensor,
        continuation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process one inference timestep with only beat input driving the network.

        During inference, vestibular predictions are generated from auditory
        input alone, demonstrating cross-modal learning.

        Args:
            vestibular_input: Vestibular input (used for error computation only)
            beat_input: Beat sensory input
            continuation: If True, run without any sensory input (internal continuation)

        Returns:
            Tuple of (vestibular_pred, beat_pred, e_v, e_b):
                - vestibular_pred: Predicted vestibular signal
                - beat_pred: Predicted beat signal
                - e_v: Vestibular prediction error
                - e_b: Beat prediction error
        """
        beat_input = beat_input.view(1, 1)

        # Store current state as previous
        self.x_prev = self.x.clone()
        self.x = self.Wrec @ torch.tanh(self.x_prev)

        # Optimize states using only beat input (or nothing in continuation)
        for _ in range(self.n_inference_steps):
            if continuation:
                self.optimize_states(None, torch.tensor([[0.0]]))
            else:
                self.optimize_states(None, beat_input)

        # Get predictions after inference
        _, vestibular_pred, beat_pred = self.compute_predictions()
        _, e_v, e_b = self.compute_prediction_errors(vestibular_input, beat_input)

        return vestibular_pred, beat_pred, e_v, e_b
