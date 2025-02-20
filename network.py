import torch
import torch.nn as nn
from typing import Tuple, Optional, Any

from torch import Tensor


class SensorimotorPredictiveNetworkRNN(nn.Module):
    def __init__(self, associative_size: int, vestibular_size: int, inference_learning_rate: float,
                 weight_learning_rate: float, n_inference_steps: int):
        """
        Initialize predictive coding network with recurrent associative area for
        sensorimotor synchronization learning.
        """

        super().__init__()  # Call parent class constructor

        # Store parameters
        self.associative_size = associative_size
        self.vestibular_size = vestibular_size
        self.inference_learning_rate = inference_learning_rate
        self.weight_learning_rate = weight_learning_rate
        self.n_inference_steps = n_inference_steps

        # Initialize recurrent weights for associative area
        self.Wrec = nn.Parameter(self._init_weights(associative_size, associative_size), requires_grad=False)  # [associative_size, associative_size]

        # Initialize prediction weights for sensory pathways
        self.W_vestibular = nn.Parameter(self._init_weights(vestibular_size, associative_size), requires_grad=False)
        self.W_beat = nn.Parameter(self._init_weights(1, associative_size), requires_grad=False)  # [1, associative_size]

        # Initialize states
        self.x = torch.zeros(associative_size, 1)  # Current state [associative_size, 1]
        self.x_prev = torch.zeros(associative_size, 1)  # Previous state [associative_size, 1]

    def reset_states(self):
        """Reset network states to initial conditions."""
        self.x = torch.zeros(self.associative_size, 1)
        self.x_prev = torch.zeros(self.associative_size, 1)
        
    def _init_weights(self, out_size: int, in_size: int) -> torch.Tensor:
        """Initialize weights with small random values."""
        torch.manual_seed(111)
        return torch.randn(out_size, in_size) * 0.05

    def _tanh_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Compute derivative of tanh activation function."""
        return 1 - torch.tanh(x) ** 2

    def compute_predictions(self):
        """Compute predictions for all units and sensory inputs."""
        # Recurrent predictions use previous state
        mu_rec = self.Wrec @ torch.tanh(self.x_prev)

        # Sensory predictions use current state
        mu_v = self.W_vestibular @ torch.tanh(self.x)
        mu_b = self.W_beat @ torch.tanh(self.x)

        return mu_rec, mu_v, mu_b

    def compute_prediction_errors(self, vestibular_input, beat_input):
        """Compute prediction errors for all units and inputs."""
        mu_rec, mu_v, mu_b = self.compute_predictions()

        # Compute errors
        e_rec = self.x - mu_rec  # Current state vs prediction from previous state
        e_v = vestibular_input - mu_v if vestibular_input is not None else torch.zeros_like(mu_v)
        e_b = beat_input - mu_b if beat_input is not None else torch.zeros_like(mu_b)

        return e_rec, e_v, e_b

    def optimize_states(self, vestibular_input, beat_input):
        """Compute state updates to minimize prediction errors."""
        # mu_rec, mu_v, mu_b = self.compute_predictions()
        e_rec, e_v, e_b = self.compute_prediction_errors(vestibular_input, beat_input)

        # Start with recurrent errors
        dx = -e_rec

        # Add sensory prediction errors if inputs available
        if vestibular_input is not None:
            # dx += e_v * self.W_vestibular @ self._tanh_derivative(self.x)
            dx += self.W_vestibular.T @ (e_v * self.W_vestibular @ self._tanh_derivative(self.x))
        if beat_input is not None:
            # dx += e_b * self.W_beat @ self._tanh_derivative(self.x)
            dx += self.W_beat.T @ (e_b * self.W_beat @ self._tanh_derivative(self.x))

        self.x -= self.inference_learning_rate * dx

        # return dx

    def compute_free_energy(self, vestibular_input: Optional[torch.Tensor],
                            beat_input: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute free energy (prediction error)."""
        e_rec, e_v, e_b = self.compute_prediction_errors(vestibular_input, beat_input)
        return 0.5 * (torch.sum(e_rec ** 2) + torch.sum(e_v ** 2) + e_b ** 2)

    def update_weights(self, e_rec: torch.Tensor, e_v: torch.Tensor, e_b: torch.Tensor):
        """Update weights based on prediction errors."""
        # Compute weight updates
        dWrec = -e_rec @ torch.tanh(self.x_prev).T  # todo : should it be x_prev ???
        dW_vestibular = -e_v @ torch.tanh(self.x).T
        dW_beat = -e_b @ torch.tanh(self.x).T

        # Apply updates
        self.Wrec -= self.weight_learning_rate * dWrec
        self.W_vestibular -= self.weight_learning_rate * dW_vestibular
        self.W_beat -= self.weight_learning_rate * dW_beat

    def timestep_train(self, vestibular_input: torch.Tensor,
                       beat_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process one training timestep with both inputs available."""
        vestibular_input = vestibular_input.view(self.vestibular_size, 1)
        beat_input = beat_input.view(1, 1)

        # Store current state as previous
        self.x_prev = self.x.clone()
        self.x = self.Wrec @ torch.tanh(self.x_prev)

        # Optimize states
        for _ in range(self.n_inference_steps):
            self.optimize_states(vestibular_input, beat_input)

        # Compute final errors and update weights
        e_rec, e_v, e_b = self.compute_prediction_errors(vestibular_input, beat_input)
        self.update_weights(e_rec, e_v, e_b)

        # Return predictions
        _, vestibular_pred, beat_pred = self.compute_predictions()
        return vestibular_pred, beat_pred

    def timestep_inference(self, beat_input: torch.Tensor) -> tuple[Tensor | Any, Tensor | Any]:
        """Process one inference timestep with only beat input."""
        beat_input = beat_input.view(1, 1)

        # Store current state as previous
        self.x_prev = self.x.clone()
        self.x = self.Wrec @ torch.tanh(self.x_prev)

        # compute predicted beat before optimizing states
        _, vestibular_pred, beat_pred = self.compute_predictions()

        # Optimize states using only beat input
        for _ in range(self.n_inference_steps):
            self.optimize_states(None, beat_input)

        # Return vestibular prediction
        _, _, _ = self.compute_predictions()

        return vestibular_pred, beat_pred
