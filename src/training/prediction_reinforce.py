"""REINFORCE for Prediction-as-Action (Phase 2b).

Key difference from standard REINFORCE:
- Actions are continuous (predicted next states)
- Uses Gaussian policy instead of Categorical
- No world-model loss (prediction IS the action)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from dataclasses import dataclass

from ..models.baby_model import PredictionModel


@dataclass
class PredictionExperience:
    """Experience tuple for prediction-as-action."""

    state: torch.Tensor  # (state_dim,)
    prediction: torch.Tensor  # (state_dim,) - the "action" = predicted next state
    target: torch.Tensor  # (state_dim,) - actual next state (for metrics)
    reward: float  # Prediction accuracy reward
    log_prob: float  # Log prob of the prediction under Gaussian policy
    mse: float  # MSE for logging


class PredictionREINFORCE:
    """
    REINFORCE for continuous prediction actions.

    This is the core of Phase 2b: prediction IS the action.
    The agent outputs a predicted next state, and gets rewarded for accuracy.
    """

    def __init__(
        self,
        model: PredictionModel,
        lr: float = 1e-3,
        baseline_decay: float = 0.99,
        entropy_coef: float = 0.01,
        device: str = "cpu",
    ):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Running average baseline
        self.baseline = 0.0
        self.baseline_decay = baseline_decay

        # Entropy bonus for exploration
        self.entropy_coef = entropy_coef

    def update(self, experiences: List[PredictionExperience]) -> Dict[str, float]:
        """
        REINFORCE update for prediction-as-action.

        The key insight: policy gradient optimizes prediction accuracy directly
        through the RL reward signal, not through MLE loss.

        Args:
            experiences: List of PredictionExperience tuples

        Returns:
            Dict of metrics for logging
        """
        if not experiences:
            return {}

        # Stack experiences into batches
        states = torch.stack([e.state for e in experiences]).to(self.device)
        predictions = torch.stack([e.prediction for e in experiences]).to(self.device)
        targets = torch.stack([e.target for e in experiences]).to(self.device)
        rewards = torch.tensor(
            [e.reward for e in experiences], dtype=torch.float32, device=self.device
        )
        old_log_probs = torch.tensor(
            [e.log_prob for e in experiences], dtype=torch.float32, device=self.device
        )

        # Forward pass to get current distribution
        mean, std, value = self.model(states)

        # Create Gaussian distribution
        dist = torch.distributions.Normal(mean, std)

        # Compute log probs of the taken predictions
        log_probs = dist.log_prob(predictions).sum(dim=-1)

        # Update baseline (exponential moving average)
        mean_reward = rewards.mean().item()
        self.baseline = (
            self.baseline_decay * self.baseline
            + (1 - self.baseline_decay) * mean_reward
        )

        # Policy gradient with baseline
        advantages = rewards - self.baseline
        policy_loss = -(log_probs * advantages.detach()).mean()

        # Entropy bonus (encourage exploration in prediction space)
        # For Gaussian: entropy = 0.5 * log(2*pi*e*var) per dimension
        entropy = dist.entropy().sum(dim=-1).mean()

        # Total loss (no world-model loss - prediction IS the action!)
        loss = policy_loss - self.entropy_coef * entropy

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Compute metrics
        mse_values = [e.mse for e in experiences]
        mean_mse = sum(mse_values) / len(mse_values)

        # Compute "success" as MSE below threshold (e.g., 0.1)
        successes = [1.0 if e.mse < 0.1 else 0.0 for e in experiences]
        success_rate = sum(successes) / len(successes)

        # Compute prediction accuracy (argmax match)
        pred_argmax = predictions.argmax(dim=-1)
        target_argmax = targets.argmax(dim=-1)
        argmax_accuracy = (pred_argmax == target_argmax).float().mean().item()

        metrics = {
            "loss/total": loss.item(),
            "loss/policy": policy_loss.item(),
            "entropy": entropy.item(),
            "baseline": self.baseline,
            "reward/mean": rewards.mean().item(),
            "reward/std": rewards.std().item() if len(rewards) > 1 else 0.0,
            "prediction/mse": mean_mse,
            "prediction/argmax_accuracy": argmax_accuracy,
            "success_rate": success_rate,
            "std/mean": std.mean().item(),  # Track exploration
        }

        return metrics


class PredictionMLE:
    """
    MLE baseline for comparison with RL.

    This trains the same model architecture but with cross-entropy loss
    instead of RL reward. If RL matches MLE, we've proven that RL
    gradients can replace pretraining gradients.
    """

    def __init__(
        self,
        model: PredictionModel,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def update(self, experiences: List[PredictionExperience]) -> Dict[str, float]:
        """
        MLE update - minimize MSE directly.

        This is the "pretraining" baseline: direct supervision of predictions.
        """
        if not experiences:
            return {}

        # Stack experiences
        states = torch.stack([e.state for e in experiences]).to(self.device)
        targets = torch.stack([e.target for e in experiences]).to(self.device)

        # Forward pass (use mean, ignore std for MLE)
        mean, std, _ = self.model(states)

        # MSE loss (like next-token prediction cross-entropy)
        mse_loss = F.mse_loss(mean, targets)

        # Optimize
        self.optimizer.zero_grad()
        mse_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Compute metrics
        with torch.no_grad():
            pred_argmax = mean.argmax(dim=-1)
            target_argmax = targets.argmax(dim=-1)
            argmax_accuracy = (pred_argmax == target_argmax).float().mean().item()

        metrics = {
            "loss/mse": mse_loss.item(),
            "prediction/argmax_accuracy": argmax_accuracy,
            "std/mean": std.mean().item(),
        }

        return metrics
