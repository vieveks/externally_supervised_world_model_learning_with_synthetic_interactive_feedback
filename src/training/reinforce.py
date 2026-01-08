"""REINFORCE implementation for WMIL MVP.

FIX #3: Start with REINFORCE, not PPO. Switch to PPO after learning works.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
from dataclasses import dataclass

from ..models.baby_model import BabyModel


@dataclass
class Experience:
    """Single experience tuple."""

    state: torch.Tensor  # (state_dim,)
    action: int
    reward: float  # Scalar reward for policy gradient
    next_state: torch.Tensor  # (state_dim,) - for world model
    log_prob: float
    success: bool  # For logging


class REINFORCE:
    """
    Simple policy gradient for MVP.

    Switch to PPO only after this works.

    Supports:
    - Two-phase training (WM warmup then policy-only)
    - Model-Based Value Expansion (MVE) for querying WM during value estimation
    """

    def __init__(
        self,
        model: BabyModel,
        lr: float = 1e-3,
        baseline_decay: float = 0.99,
        world_model_coef: float = 1.0,
        entropy_coef: float = 0.01,
        mve_horizon: int = 0,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Running average baseline
        self.baseline = 0.0
        self.baseline_decay = baseline_decay

        # Loss coefficients
        self.world_model_coef = world_model_coef
        self.entropy_coef = entropy_coef

        # MVE parameters
        self.mve_horizon = mve_horizon  # 0 = no MVE, 1+ = H-step value expansion
        self.gamma = gamma

        # Training mode: "joint", "wm_only", "policy_only"
        self.training_mode = "joint"

    def set_training_mode(self, mode: str):
        """
        Set training mode.

        Args:
            mode: "joint" (default), "wm_only", or "policy_only"
        """
        assert mode in ["joint", "wm_only", "policy_only"]
        self.training_mode = mode

        if mode == "policy_only":
            # Freeze world-model head
            for param in self.model.world_model.parameters():
                param.requires_grad = False
        else:
            # Unfreeze everything
            for param in self.model.parameters():
                param.requires_grad = True

    def compute_mve_values(
        self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Model-Based Value Expansion targets.

        MVE formula: V_target(s_t) = r_t + γ * V(ŝ_{t+1})

        For H > 1:
            V_target = r_t + γ * r̂_{t+1} + γ² * V(ŝ_{t+2})

        This makes the world model CAUSALLY RELEVANT to value estimation.
        The policy gradient now depends on WM predictions.

        Args:
            states: Current states (batch, state_dim)
            actions: Actions taken (batch,)
            rewards: Actual rewards received (batch,)

        Returns:
            MVE value targets (batch,)
        """
        batch_size = states.size(0)

        with torch.no_grad():
            # Start with current states
            current_states = states
            current_actions = actions
            cumulative_reward = rewards.clone()

            # Roll out H steps using the world model
            for h in range(self.mve_horizon):
                # Predict next state using world model
                predicted_next = self.model.predict_next_state_raw(
                    current_states, current_actions
                )

                # Get value of predicted next state
                _, v_next = self.model(predicted_next)

                if h == self.mve_horizon - 1:
                    # Final step: add discounted value
                    cumulative_reward = cumulative_reward + (self.gamma ** (h + 1)) * v_next
                else:
                    # Intermediate step: would need predicted reward
                    # For now, we only use actual reward at t=0
                    # Future: add reward model for multi-step MVE
                    pass

                # For multi-step MVE, we'd sample next action from policy
                # and continue the rollout. For H=1, this loop runs once.
                if h < self.mve_horizon - 1:
                    action_dist, _ = self.model(predicted_next)
                    current_actions = action_dist.sample()
                    current_states = predicted_next

            return cumulative_reward

    def update(self, experiences: List[Experience]) -> Dict[str, float]:
        """
        REINFORCE update with baseline and world-model loss.

        Respects training_mode:
        - "joint": both policy and WM loss
        - "wm_only": only WM loss (for warmup phase)
        - "policy_only": only policy loss (WM frozen)

        Args:
            experiences: List of Experience tuples from rollout

        Returns:
            Dict of metrics for logging
        """
        if not experiences:
            return {}

        # Stack experiences into batches
        states = torch.stack([e.state for e in experiences]).to(self.device)
        actions = torch.tensor([e.action for e in experiences], device=self.device)
        rewards = torch.tensor(
            [e.reward for e in experiences], dtype=torch.float32, device=self.device
        )
        next_states = torch.stack([e.next_state for e in experiences]).to(self.device)
        old_log_probs = torch.tensor(
            [e.log_prob for e in experiences], dtype=torch.float32, device=self.device
        )

        # Forward pass
        action_dist, _ = self.model(states)
        log_probs = action_dist.log_prob(actions)

        # Compute value targets (MVE or simple rewards)
        if self.mve_horizon > 0:
            # MVE: Use world model to compute value targets
            # This makes WM CAUSALLY RELEVANT to policy gradient
            value_targets = self.compute_mve_values(states, actions, rewards)
        else:
            # Standard: Just use rewards
            value_targets = rewards

        # Update baseline (exponential moving average)
        mean_target = value_targets.mean().item()
        self.baseline = (
            self.baseline_decay * self.baseline
            + (1 - self.baseline_decay) * mean_target
        )

        # Policy gradient with baseline
        advantages = value_targets - self.baseline
        policy_loss = -(log_probs * advantages.detach()).mean()

        # Entropy bonus (exploration)
        entropy = action_dist.entropy().mean()

        # World-model loss (FIX #2: predict raw state, not embedding)
        predicted_next = self.model.predict_next_state_raw(states, actions)
        world_model_loss = F.mse_loss(predicted_next, next_states)

        # Compute loss based on training mode
        if self.training_mode == "wm_only":
            # Phase 1: Only train world-model
            loss = world_model_loss
        elif self.training_mode == "policy_only":
            # Phase 2: Only train policy (WM frozen)
            loss = policy_loss - self.entropy_coef * entropy
        else:
            # Joint training (default)
            loss = (
                policy_loss
                - self.entropy_coef * entropy
                + self.world_model_coef * world_model_loss
            )

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Compute success rate for logging
        successes = [e.success for e in experiences]
        success_rate = sum(successes) / len(successes) if successes else 0.0

        metrics = {
            "loss/total": loss.item(),
            "loss/policy": policy_loss.item(),
            "loss/world_model": world_model_loss.item(),
            "entropy": entropy.item(),
            "baseline": self.baseline,
            "reward/mean": rewards.mean().item(),
            "reward/std": rewards.std().item() if len(rewards) > 1 else 0.0,
            "success_rate": success_rate,
            "training_mode": self.training_mode,
        }

        # Add MVE-specific metrics
        if self.mve_horizon > 0:
            metrics["mve/horizon"] = self.mve_horizon
            metrics["mve/value_target_mean"] = value_targets.mean().item()
            metrics["mve/value_target_std"] = value_targets.std().item() if len(value_targets) > 1 else 0.0

        return metrics
