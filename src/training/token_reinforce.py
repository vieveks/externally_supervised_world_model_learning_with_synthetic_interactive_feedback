"""REINFORCE for Token Prediction (Phase 3.1).

This extends Phase 2b's prediction REINFORCE to discrete tokens.
Key difference: Categorical distribution instead of Gaussian.

Also includes:
- MLE baseline for comparison
- TD(λ) for credit assignment with delayed rewards
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ..models.token_model import TokenPredictionModel


@dataclass
class TokenExperience:
    """Experience tuple for token prediction."""

    state: torch.Tensor  # One-hot current token (vocab_size,)
    prediction: int  # Predicted next token
    target: int  # Actual next token
    reward: float  # Correctness reward
    log_prob: float  # Log prob of prediction
    correct: bool  # Was prediction correct?


@dataclass
class SequenceExperience:
    """Experience for multi-step sequence prediction."""

    states: List[torch.Tensor]  # Sequence of states
    predictions: List[int]  # Sequence of predictions
    targets: List[int]  # Sequence of actual next tokens
    log_probs: List[float]  # Log probs at each step
    reward: float  # Final reward (delayed)
    correct_count: int  # Number of correct predictions


class TokenREINFORCE:
    """
    REINFORCE for discrete token prediction.

    The core Phase 3.1 algorithm: RL gradients for token prediction.
    """

    def __init__(
        self,
        model: TokenPredictionModel,
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

        # Entropy coefficient for exploration
        self.entropy_coef = entropy_coef

    def update(self, experiences: List[TokenExperience]) -> Dict[str, float]:
        """
        REINFORCE update for token prediction.

        Args:
            experiences: List of TokenExperience tuples

        Returns:
            Dict of metrics
        """
        if not experiences:
            return {}

        # Stack experiences
        states = torch.stack([e.state for e in experiences]).to(self.device)
        predictions = torch.tensor(
            [e.prediction for e in experiences], dtype=torch.long, device=self.device
        )
        targets = torch.tensor(
            [e.target for e in experiences], dtype=torch.long, device=self.device
        )
        rewards = torch.tensor(
            [e.reward for e in experiences], dtype=torch.float32, device=self.device
        )

        # Forward pass
        logits, values = self.model(states)
        dist = Categorical(logits=logits)

        # Compute log probs of taken predictions
        log_probs = dist.log_prob(predictions)

        # Update baseline
        mean_reward = rewards.mean().item()
        self.baseline = (
            self.baseline_decay * self.baseline
            + (1 - self.baseline_decay) * mean_reward
        )

        # Policy gradient with baseline
        advantages = rewards - self.baseline
        policy_loss = -(log_probs * advantages.detach()).mean()

        # Entropy bonus
        entropy = dist.entropy().mean()

        # Total loss
        loss = policy_loss - self.entropy_coef * entropy

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Metrics
        accuracy = (predictions == targets).float().mean().item()
        correct_count = sum(1 for e in experiences if e.correct)

        return {
            "loss/total": loss.item(),
            "loss/policy": policy_loss.item(),
            "entropy": entropy.item(),
            "baseline": self.baseline,
            "reward/mean": mean_reward,
            "reward/std": rewards.std().item() if len(rewards) > 1 else 0.0,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(experiences),
        }


class TokenMLE:
    """
    MLE baseline for comparison with RL.

    Trains with cross-entropy loss (like LLM pretraining).
    If RL matches this, we've proven RL can replace pretraining.
    """

    def __init__(
        self,
        model: TokenPredictionModel,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def update(self, experiences: List[TokenExperience]) -> Dict[str, float]:
        """
        MLE update - minimize cross-entropy directly.

        This is the "pretraining" baseline.
        """
        if not experiences:
            return {}

        # Stack experiences
        states = torch.stack([e.state for e in experiences]).to(self.device)
        targets = torch.tensor(
            [e.target for e in experiences], dtype=torch.long, device=self.device
        )

        # Forward pass
        logits, _ = self.model(states)

        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits, targets)

        # Optimize
        self.optimizer.zero_grad()
        ce_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Metrics
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()

        return {
            "loss/ce": ce_loss.item(),
            "accuracy": accuracy,
        }


class SequenceREINFORCE:
    """
    REINFORCE for sequence prediction with delayed reward.

    For testing credit assignment over multiple steps.
    """

    def __init__(
        self,
        model: TokenPredictionModel,
        lr: float = 1e-3,
        baseline_decay: float = 0.99,
        entropy_coef: float = 0.01,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        self.baseline = 0.0
        self.baseline_decay = baseline_decay
        self.entropy_coef = entropy_coef
        self.gamma = gamma

    def update(self, experiences: List[SequenceExperience]) -> Dict[str, float]:
        """
        REINFORCE update for sequence prediction.

        Reward is given at end of sequence (credit assignment challenge).
        """
        if not experiences:
            return {}

        total_policy_loss = torch.tensor(0.0, device=self.device)
        total_entropy = torch.tensor(0.0, device=self.device)
        total_reward = 0.0
        total_correct = 0
        total_predictions = 0

        for exp in experiences:
            # Recompute log probs with gradient tracking
            seq_log_prob = torch.tensor(0.0, device=self.device)
            for i, state in enumerate(exp.states):
                state_t = state.to(self.device)
                logits, _ = self.model(state_t)
                dist = Categorical(logits=logits)
                seq_log_prob = seq_log_prob + dist.log_prob(
                    torch.tensor(exp.predictions[i], device=self.device)
                )
                total_entropy = total_entropy + dist.entropy()

            # Update baseline
            self.baseline = (
                self.baseline_decay * self.baseline
                + (1 - self.baseline_decay) * exp.reward
            )

            # Advantage
            advantage = exp.reward - self.baseline

            # Policy gradient (all tokens get same signal - credit assignment issue!)
            policy_loss = -seq_log_prob * advantage
            total_policy_loss = total_policy_loss + policy_loss

            total_reward += exp.reward
            total_correct += exp.correct_count
            total_predictions += len(exp.predictions)

        # Average losses
        n = len(experiences)
        policy_loss = total_policy_loss / n
        entropy = total_entropy / (n * len(experiences[0].states))

        # Total loss
        loss = policy_loss - self.entropy_coef * entropy

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            "loss/total": loss.item(),
            "loss/policy": policy_loss.item(),
            "entropy": entropy.item(),
            "baseline": self.baseline,
            "reward/mean": total_reward / n,
            "accuracy": total_correct / total_predictions if total_predictions > 0 else 0,
            "correct_per_seq": total_correct / n,
        }


class TDLambdaREINFORCE:
    """
    TD(λ) enhanced REINFORCE for better credit assignment.

    Uses eligibility traces to propagate reward signal back
    through the sequence more effectively.
    """

    def __init__(
        self,
        model: TokenPredictionModel,
        lr: float = 1e-3,
        gamma: float = 0.99,
        lambda_: float = 0.9,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        device: str = "cpu",
    ):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        self.gamma = gamma
        self.lambda_ = lambda_
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

    def compute_lambda_returns(
        self,
        rewards: List[float],
        values: List[float],
        final_value: float = 0.0,
    ) -> List[float]:
        """
        Compute λ-returns for credit assignment.

        G_t^λ = (1-λ) Σ_{n=1}^∞ λ^{n-1} G_t:t+n

        This smoothly interpolates between:
        - λ=0: TD(0), one-step bootstrap
        - λ=1: Monte Carlo, full return
        """
        T = len(rewards)
        lambda_returns = [0.0] * T

        # Compute from end to start
        G = final_value
        for t in reversed(range(T)):
            # TD target at step t
            td_target = rewards[t] + self.gamma * (
                values[t + 1] if t + 1 < T else final_value
            )

            # λ-return combines TD target with future λ-returns
            G = td_target + self.gamma * self.lambda_ * (G - td_target)
            lambda_returns[t] = G

        return lambda_returns

    def update(self, experiences: List[SequenceExperience]) -> Dict[str, float]:
        """
        TD(λ) update for sequence prediction.

        Uses eligibility traces for better credit assignment.
        """
        if not experiences:
            return {}

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_correct = 0
        total_predictions = 0

        for exp in experiences:
            states = [s.to(self.device) for s in exp.states]
            predictions = exp.predictions
            targets = exp.targets

            # Get values for each state
            values = []
            log_probs = []
            entropies = []

            for i, state in enumerate(states):
                logits, value = self.model(state)
                dist = Categorical(logits=logits)

                values.append(value.item())
                log_probs.append(dist.log_prob(torch.tensor(predictions[i], device=self.device)))
                entropies.append(dist.entropy())

            # Compute per-step rewards (0 except possibly last)
            # For delayed reward, all rewards are 0 except final
            per_step_rewards = [0.0] * len(states)
            per_step_rewards[-1] = exp.reward

            # Compute λ-returns
            lambda_returns = self.compute_lambda_returns(
                per_step_rewards, values, final_value=0.0
            )

            # Policy gradient with λ-returns as targets
            for t in range(len(states)):
                advantage = lambda_returns[t] - values[t]
                total_policy_loss -= log_probs[t] * advantage

                # Value loss
                total_value_loss += (values[t] - lambda_returns[t]) ** 2

                total_entropy += entropies[t]

            total_correct += exp.correct_count
            total_predictions += len(predictions)

        # Average losses
        n = len(experiences)
        T = len(experiences[0].states)
        policy_loss = total_policy_loss / (n * T)
        value_loss = total_value_loss / (n * T)
        entropy = total_entropy / (n * T)

        # Total loss
        loss = (
            policy_loss
            + self.value_coef * value_loss
            - self.entropy_coef * entropy
        )

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            "loss/total": loss.item(),
            "loss/policy": policy_loss.item(),
            "loss/value": value_loss.item(),
            "entropy": entropy.item(),
            "accuracy": total_correct / total_predictions if total_predictions > 0 else 0,
            "lambda": self.lambda_,
        }
