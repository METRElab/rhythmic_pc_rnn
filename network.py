"""
Hierarchical predictive coding recurrent neural network for sensorimotor synchronization.

Implements a two-layer recurrent predictive coding network with:
- Higher layer H operating on slow timescale
- Associative layer x operating on faster timescale
- Parallel vestibular and auditory sensory pathways
- Leaky integration dynamics for both layers
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any


class SensorimotorPCRNN(nn.Module):
    """
    Hierarchical predictive coding recurrent neural network.

    This network learns to predict both vestibular and auditory (beat) inputs
    through a two-layer hierarchy:
    - Higher layer H: slow timescale, abstract representations
    - Associative layer x: faster timescale, moment-to-moment dynamics

    When higher_size=0, the network reduces to a single-layer model.

    Attributes:
        use_hierarchy: Whether higher layer is enabled
        higher_size: Number of units in higher layer (m)
        associative_size: Number of units in associative layer (n)
        vestibular_size: Dimension of vestibular input
        alpha_H: Timescale for higher layer (slow)
        alpha_x: Timescale for associative layer (faster)
        H: Higher layer state
        H_prev: Previous higher layer state
        x: Associative layer state
        x_prev: Previous associative layer state
    """

    def __init__(
        self,
        higher_size: int,
        associative_size: int,
        vestibular_size: int,
        alpha_H: float,
        alpha_x: float,
        inference_learning_rate_H: float,
        inference_learning_rate_x: float,
        weight_learning_rate_H: float,
        weight_learning_rate_x: float,
        n_inference_steps: int,
        random_seed: int = 111,
    ) -> None:
        """
        Initialize the hierarchical predictive coding network.

        Args:
            higher_size: Number of units in higher layer (0 to disable)
            associative_size: Number of units in associative layer
            vestibular_size: Dimension of vestibular input
            alpha_H: Timescale for higher layer (0 < alpha_H <= 1)
            alpha_x: Timescale for associative layer (0 < alpha_x <= 1)
            inference_learning_rate_H: Learning rate for H state optimization
            inference_learning_rate_x: Learning rate for x state optimization
            weight_learning_rate_H: Learning rate for W_HH and W_Hx
            weight_learning_rate_x: Learning rate for W_rec, W_v, W_b
            n_inference_steps: Number of Euler steps per timestep
            random_seed: Seed for weight initialization
        """
        super().__init__()

        # Determine if hierarchy is enabled
        self.use_hierarchy = higher_size > 0

        # Store dimensions
        self.higher_size = higher_size
        self.associative_size = associative_size
        self.vestibular_size = vestibular_size

        # Store timescales
        self.alpha_H = alpha_H
        self.alpha_x = alpha_x

        # Store learning rates
        self.inference_learning_rate_H = inference_learning_rate_H
        self.inference_learning_rate_x = inference_learning_rate_x
        self.weight_learning_rate_H = weight_learning_rate_H
        self.weight_learning_rate_x = weight_learning_rate_x

        # Store other parameters
        self.n_inference_steps = n_inference_steps
        self.random_seed = random_seed

        # Initialize weights for higher layer (if enabled)
        if self.use_hierarchy:
            self.W_HH = nn.Parameter(
                self._init_weights(higher_size, higher_size),
                requires_grad=False
            )  # [m, m]
            self.W_Hx = nn.Parameter(
                self._init_weights(associative_size, higher_size),
                requires_grad=False
            )  # [n, m]

        # Initialize weights for associative layer
        self.W_rec = nn.Parameter(
            self._init_weights(associative_size, associative_size),
            requires_grad=False
        )  # [n, n]

        # Initialize weights for sensory predictions
        self.W_v = nn.Parameter(
            self._init_weights(vestibular_size, associative_size),
            requires_grad=False
        )  # [v_s, n]
        self.W_b = nn.Parameter(
            self._init_weights(1, associative_size),
            requires_grad=False
        )  # [1, n]

        # Initialize states
        self.reset_states()

    def reset_states(self) -> None:
        """Reset all network states to zeros."""
        if self.use_hierarchy:
            self.H = torch.zeros(self.higher_size, 1)
            self.H_prev = torch.zeros(self.higher_size, 1)

        self.x = torch.zeros(self.associative_size, 1)
        self.x_prev = torch.zeros(self.associative_size, 1)

    def _init_weights(self, out_size: int, in_size: int) -> torch.Tensor:
        """
        Initialize weights with small random values.

        Uses a local torch Generator to avoid polluting the global
        random state (which would affect all subsequent torch operations).

        Args:
            out_size: Output dimension
            in_size: Input dimension

        Returns:
            Initialized weight tensor of shape [out_size, in_size]
        """
        gen = torch.Generator()
        gen.manual_seed(self.random_seed)
        return torch.randn(out_size, in_size, generator=gen) * 0.05

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

    def compute_predictions(self) -> Dict[str, torch.Tensor]:
        """
        Compute predictions for all units and sensory inputs.

        Returns:
            Dictionary containing:
                - mu_H: Higher layer prediction (if hierarchy enabled)
                - mu_x: Associative layer prediction
                - mu_v: Vestibular prediction
                - mu_b: Beat prediction
        """
        predictions = {}

        if self.use_hierarchy:
            # Higher layer prediction (from previous H)
            predictions['mu_H'] = (
                (1 - self.alpha_H) * self.H_prev +
                self.alpha_H * self.W_HH @ torch.tanh(self.H_prev)
            )

            # Associative layer prediction (from previous x + top-down from H)
            predictions['mu_x'] = (
                (1 - self.alpha_x) * self.x_prev +
                self.alpha_x * (
                    self.W_rec @ torch.tanh(self.x_prev) +
                    self.W_Hx @ torch.tanh(self.H)
                )
            )
        else:
            # No hierarchy: simple recurrent prediction with leaky dynamics
            predictions['mu_x'] = (
                (1 - self.alpha_x) * self.x_prev +
                self.alpha_x * self.W_rec @ torch.tanh(self.x_prev)
            )

        # Sensory predictions (from current x)
        predictions['mu_v'] = self.W_v @ torch.tanh(self.x)
        predictions['mu_b'] = self.W_b @ torch.tanh(self.x)

        return predictions

    def compute_prediction_errors(
        self,
        vestibular_input: Optional[torch.Tensor],
        beat_input: Optional[torch.Tensor],
        mu_H: Optional[torch.Tensor] = None,
        mu_x: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute prediction errors for all units and inputs.

        Args:
            vestibular_input: Vestibular sensory input or None
            beat_input: Beat sensory input or None
            mu_H: Stored prediction for H (from state transition)
            mu_x: Stored prediction for x (from state transition)

        Returns:
            Dictionary containing prediction errors:
                - e_H: Higher layer error (if hierarchy enabled)
                - e_x: Associative layer error
                - e_v: Vestibular error
                - e_b: Beat error
        """
        predictions = self.compute_predictions()
        errors = {}

        if self.use_hierarchy:
            # Use stored mu_H if provided, otherwise compute fresh
            stored_mu_H = mu_H if mu_H is not None else predictions['mu_H']
            errors['e_H'] = self.H - stored_mu_H

        # Use stored mu_x if provided, otherwise compute fresh
        stored_mu_x = mu_x if mu_x is not None else predictions['mu_x']
        errors['e_x'] = self.x - stored_mu_x

        # Sensory errors
        if vestibular_input is not None:
            errors['e_v'] = vestibular_input - predictions['mu_v']
        else:
            errors['e_v'] = torch.zeros_like(predictions['mu_v'])

        if beat_input is not None:
            errors['e_b'] = beat_input - predictions['mu_b']
        else:
            errors['e_b'] = torch.zeros_like(predictions['mu_b'])

        return errors

    def optimize_states(
        self,
        vestibular_input: Optional[torch.Tensor],
        beat_input: Optional[torch.Tensor],
        mu_H: Optional[torch.Tensor],
        mu_x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Update states to minimize prediction errors via gradient ascent.

        Args:
            vestibular_input: Vestibular sensory input or None
            beat_input: Beat sensory input or None
            mu_H: Stored prediction for H
            mu_x: Stored prediction for x

        Returns:
            Dictionary of prediction errors
        """
        errors = self.compute_prediction_errors(
            vestibular_input, beat_input, mu_H, mu_x
        )

        if self.use_hierarchy:
            # Gradient for H: -e_H + alpha_x * f'(H) ⊙ (W_Hx.T @ e_x)
            dH = (
                -errors['e_H'] +
                self.alpha_x * self._tanh_derivative(self.H) *
                (self.W_Hx.T @ errors['e_x'])
            )
            self.H = self.H + self.alpha_H * self.inference_learning_rate_H * dH

        # Gradient for x: -e_x + f'(x) ⊙ (W_v.T @ e_v + W_b.T @ e_b)
        dx = (
            -errors['e_x'] +
            self._tanh_derivative(self.x) * (
                self.W_v.T @ errors['e_v'] +
                self.W_b.T @ errors['e_b']
            )
        )
        self.x = self.x + self.alpha_x * self.inference_learning_rate_x * dx

        return errors

    def update_weights(
        self,
        errors: Dict[str, torch.Tensor],
        H_prev: Optional[torch.Tensor],
        x_prev: torch.Tensor
    ) -> None:
        """
        Update weights using Hebbian learning rule.

        Args:
            errors: Dictionary of prediction errors
            H_prev: Previous H state (before transition)
            x_prev: Previous x state (before transition)
        """
        if self.use_hierarchy:
            # W_HH update: alpha_H * e_H @ f(H_prev).T
            dW_HH = self.alpha_H * errors['e_H'] @ torch.tanh(H_prev).T
            self.W_HH.data += self.weight_learning_rate_H * dW_HH

            # W_Hx update: alpha_x * e_x @ f(H).T
            dW_Hx = self.alpha_x * errors['e_x'] @ torch.tanh(self.H).T
            self.W_Hx.data += self.weight_learning_rate_H * dW_Hx

        # W_rec update: alpha_x * e_x @ f(x_prev).T
        dW_rec = self.alpha_x * errors['e_x'] @ torch.tanh(x_prev).T
        self.W_rec.data += self.weight_learning_rate_x * dW_rec

        # W_v update: e_v @ f(x).T
        dW_v = errors['e_v'] @ torch.tanh(self.x).T
        self.W_v.data += self.weight_learning_rate_x * dW_v

        # W_b update: e_b @ f(x).T
        dW_b = errors['e_b'] @ torch.tanh(self.x).T
        self.W_b.data += self.weight_learning_rate_x * dW_b

    def timestep_train(
        self,
        vestibular_input: torch.Tensor,
        beat_input: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Process one training timestep with both inputs available.

        Performs state transition, state optimization, and weight learning.

        Args:
            vestibular_input: Vestibular sensory input
            beat_input: Beat sensory input

        Returns:
            Dictionary containing predictions before optimization:
                - vest_pred: Vestibular prediction
                - beat_pred: Beat prediction
        """
        vestibular_input = vestibular_input.view(self.vestibular_size, 1)
        beat_input = beat_input.view(1, 1)

        # Step 1: Store previous states
        if self.use_hierarchy:
            H_prev_stored = self.H.clone()
        else:
            H_prev_stored = None
        x_prev_stored = self.x.clone()

        self.x_prev = self.x.clone()
        if self.use_hierarchy:
            self.H_prev = self.H.clone()

        # Step 2: State transition with leaky dynamics
        if self.use_hierarchy:
            self.H = (
                (1 - self.alpha_H) * self.H_prev +
                self.alpha_H * self.W_HH @ torch.tanh(self.H_prev)
            )
            self.x = (
                (1 - self.alpha_x) * self.x_prev +
                self.alpha_x * (
                    self.W_rec @ torch.tanh(self.x_prev) +
                    self.W_Hx @ torch.tanh(self.H)
                )
            )
        else:
            self.x = (
                (1 - self.alpha_x) * self.x_prev +
                self.alpha_x * self.W_rec @ torch.tanh(self.x_prev)
            )

        # Step 3: Store predictions (targets for error computation)
        if self.use_hierarchy:
            mu_H = self.H.clone()
        else:
            mu_H = None
        mu_x = self.x.clone()

        # Step 4: Inference loop
        for _ in range(self.n_inference_steps):
            self.optimize_states(vestibular_input, beat_input, mu_H, mu_x)

        # Step 5: Compute final errors
        final_errors = self.compute_prediction_errors(
            vestibular_input, beat_input, mu_H, mu_x
        )

        # Step 6: Update weights
        self.update_weights(final_errors, H_prev_stored, x_prev_stored)

        # Get predictions after optimization
        predictions = self.compute_predictions()
        vest_pred = predictions['mu_v'].clone()
        beat_pred = predictions['mu_b'].clone()

        return {
            'vest_pred': vest_pred,
            'beat_pred': beat_pred
        }

    def timestep_inference(
        self,
        vestibular_input: torch.Tensor,
        beat_input: torch.Tensor,
        continuation: bool = False,
        prediction_timing: str = "after",
        auditory_only: bool = True,
        beat_baseline: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Process one inference timestep (no weight updates).

        Args:
            vestibular_input: Vestibular input (used in inference loop unless
                auditory_only=True, always used for error computation)
            beat_input: Beat sensory input
            continuation: If True, run without sensory input (internal continuation)
            prediction_timing: "before" or "after" inference optimization
            auditory_only: If True, vestibular input is NOT provided to the
                inference loop (only beat). If False (default), both vestibular
                and beat are provided during inference.
            beat_baseline: Baseline value for auditory input during continuation.
                For standard mode this is 0.0; for zero_mean_beat mode this is
                the negative flat value (e.g. -0.1).

        Returns:
            Dictionary containing:
                - vest_pred: Vestibular prediction
                - beat_pred: Beat prediction
                - e_v: Vestibular error
                - e_b: Beat error
                - e_H: Higher layer error (if hierarchy enabled)
                - e_x: Associative layer error
        """
        beat_input = beat_input.view(1, 1)

        # Step 1: Store previous states
        self.x_prev = self.x.clone()
        if self.use_hierarchy:
            self.H_prev = self.H.clone()

        # Step 2: State transition with leaky dynamics
        if self.use_hierarchy:
            self.H = (
                (1 - self.alpha_H) * self.H_prev +
                self.alpha_H * self.W_HH @ torch.tanh(self.H_prev)
            )
            self.x = (
                (1 - self.alpha_x) * self.x_prev +
                self.alpha_x * (
                    self.W_rec @ torch.tanh(self.x_prev) +
                    self.W_Hx @ torch.tanh(self.H)
                )
            )
        else:
            self.x = (
                (1 - self.alpha_x) * self.x_prev +
                self.alpha_x * self.W_rec @ torch.tanh(self.x_prev)
            )

        # Step 3: Store predictions
        if self.use_hierarchy:
            mu_H = self.H.clone()
        else:
            mu_H = None
        mu_x = self.x.clone()

        # Step 3.5: Get before predictions and errors
        predictions_before = self.compute_predictions()
        errors_before = self.compute_prediction_errors(
            vestibular_input, beat_input, mu_H, mu_x
        )

        result_before = {
            'vest_pred': predictions_before['mu_v'],
            'beat_pred': predictions_before['mu_b'],
            'e_v': errors_before['e_v'],
            'e_b': errors_before['e_b'],
            'e_x': errors_before['e_x']
        }

        if self.use_hierarchy:
            result_before['e_H'] = errors_before['e_H']

        # Step 4: Inference loop
        vest_for_inference = None if auditory_only else vestibular_input
        for _ in range(self.n_inference_steps):
            if continuation:
                # No sensory input — use baseline (0 for standard, negative for zero-mean)
                self.optimize_states(None, torch.tensor([[beat_baseline]]), mu_H, mu_x)
            else:
                self.optimize_states(vest_for_inference, beat_input, mu_H, mu_x)

        # Step 5: Get final predictions and errors
        predictions = self.compute_predictions()
        errors = self.compute_prediction_errors(
            vestibular_input, beat_input, mu_H, mu_x
        )

        result_after = {
            'vest_pred': predictions['mu_v'],
            'beat_pred': predictions['mu_b'],
            'e_v': errors['e_v'],
            'e_b': errors['e_b'],
            'e_x': errors['e_x']
        }

        if self.use_hierarchy:
            result_after['e_H'] = errors['e_H']

        if prediction_timing == "before":
            return result_before
        if prediction_timing == "after":
            return result_after
        else:
            raise ValueError("prediction_timing must be 'before' or 'after'")

