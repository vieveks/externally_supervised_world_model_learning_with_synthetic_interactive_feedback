"""Baby Model architecture for WMIL.

A minimal transformer with:
- State encoder
- Transformer core
- Action head (policy)
- World-model head (predicts next state)
- Value head (for baseline, optional)
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Tuple


class StateEncoder(nn.Module):
    """Encodes discrete state to continuous embedding."""

    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode state tensor.

        Args:
            state: (batch, state_dim) or (state_dim,)

        Returns:
            Embedding of shape (batch, hidden_dim) or (hidden_dim,)
        """
        return self.encoder(state)


class TransformerCore(nn.Module):
    """Minimal transformer backbone."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process through transformer.

        Args:
            x: (batch, seq_len, hidden_dim)

        Returns:
            (batch, seq_len, hidden_dim)
        """
        return self.transformer(x)


class ActionHead(nn.Module):
    """Policy head: outputs distribution over actions."""

    def __init__(self, hidden_dim: int, num_actions: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, num_actions)

    def forward(self, h: torch.Tensor) -> Categorical:
        """
        Get action distribution.

        Args:
            h: Hidden state (batch, hidden_dim) or (hidden_dim,)

        Returns:
            Categorical distribution over actions
        """
        logits = self.fc(h)
        return Categorical(logits=logits)


class WorldModelHead(nn.Module):
    """
    Predicts next state given current state and action.

    FIX #2: For MVP, predict RAW STATE TENSOR, not embedding.
    This avoids the "drifting encoder" problem early in training.
    """

    def __init__(self, hidden_dim: int, num_actions: int, state_dim: int):
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, hidden_dim)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),  # Output: raw state dim, NOT hidden_dim
        )

    def forward(self, h: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict next state.

        Args:
            h: Current state embedding (batch, hidden_dim)
            action: Action taken (batch,) as long tensor

        Returns:
            Predicted next state as RAW TENSOR (batch, state_dim)
        """
        a_embed = self.action_embed(action)
        combined = torch.cat([h, a_embed], dim=-1)
        return self.predictor(combined)


class ValueHead(nn.Module):
    """Estimates expected return from state (for baseline)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Estimate value.

        Args:
            h: Hidden state (batch, hidden_dim)

        Returns:
            Value estimate (batch,)
        """
        return self.fc(h).squeeze(-1)


class BabyModel(nn.Module):
    """
    Complete baby model with all components.

    Architecture:
        State -> Encoder -> Transformer -> [Action, WorldModel, Value] Heads
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim

        # Components
        self.encoder = StateEncoder(state_dim, hidden_dim)
        self.transformer = TransformerCore(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.action_head = ActionHead(hidden_dim, num_actions)
        self.world_model = WorldModelHead(hidden_dim, num_actions, state_dim)
        self.value_head = ValueHead(hidden_dim)

    def forward(self, state: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        """
        Full forward pass.

        Args:
            state: State tensor (batch, state_dim) or (state_dim,)

        Returns:
            action_dist: Categorical distribution over actions
            value: Value estimate
        """
        # Handle single state (no batch dim)
        squeeze = state.dim() == 1
        if squeeze:
            state = state.unsqueeze(0)

        # Encode and transform
        h = self.encoder(state)
        h = self.transformer(h.unsqueeze(1)).squeeze(1)

        # Get outputs
        action_dist = self.action_head(h)
        value = self.value_head(h)

        if squeeze:
            value = value.squeeze(0)

        return action_dist, value

    def predict_next_state_raw(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """
        World model prediction - predicts RAW next state tensor.

        FIX #2: Use this for MVP to avoid drifting encoder problem.

        Args:
            state: Current state (batch, state_dim)
            action: Action taken (batch,)

        Returns:
            Predicted next state (batch, state_dim)
        """
        # Handle single state
        squeeze = state.dim() == 1
        if squeeze:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)

        h = self.encoder(state)
        h = self.transformer(h.unsqueeze(1)).squeeze(1)
        pred = self.world_model(h, action)

        if squeeze:
            pred = pred.squeeze(0)

        return pred

    def get_embedding(self, state: torch.Tensor) -> torch.Tensor:
        """Get state embedding (for later use)."""
        squeeze = state.dim() == 1
        if squeeze:
            state = state.unsqueeze(0)

        h = self.encoder(state)
        h = self.transformer(h.unsqueeze(1)).squeeze(1)

        if squeeze:
            h = h.squeeze(0)

        return h

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PredictionHead(nn.Module):
    """
    Outputs continuous state prediction (for Prediction-as-Action).

    Instead of discrete action logits, outputs a predicted next state vector.
    Uses Gaussian distribution for RL (mean + learned std).
    """

    def __init__(self, hidden_dim: int, state_dim: int):
        super().__init__()
        self.mean_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        # Learnable log-std (start with reasonable exploration)
        self.log_std = nn.Parameter(torch.zeros(state_dim))

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get prediction distribution parameters.

        Args:
            h: Hidden state (batch, hidden_dim) or (hidden_dim,)

        Returns:
            mean: Predicted state mean (batch, state_dim)
            std: Predicted state std (batch, state_dim)
        """
        mean = self.mean_net(h)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std


class PredictionModel(nn.Module):
    """
    Model for Prediction-as-Action task.

    Key difference from BabyModel:
    - Action = continuous state prediction (not discrete)
    - No separate world model head (prediction IS the action)
    - Uses Gaussian policy for continuous action space

    Architecture:
        State -> Encoder -> Transformer -> PredictionHead -> (mean, std)
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Components
        self.encoder = StateEncoder(state_dim, hidden_dim)
        self.transformer = TransformerCore(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.prediction_head = PredictionHead(hidden_dim, state_dim)
        self.value_head = ValueHead(hidden_dim)

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Args:
            state: State tensor (batch, state_dim) or (state_dim,)

        Returns:
            mean: Predicted next state mean
            std: Predicted next state std
            value: Value estimate
        """
        # Handle single state (no batch dim)
        squeeze = state.dim() == 1
        if squeeze:
            state = state.unsqueeze(0)

        # Encode and transform
        h = self.encoder(state)
        h = self.transformer(h.unsqueeze(1)).squeeze(1)

        # Get outputs
        mean, std = self.prediction_head(h)
        value = self.value_head(h)

        if squeeze:
            mean = mean.squeeze(0)
            std = std.squeeze(0)
            value = value.squeeze(0)

        return mean, std, value

    def sample_prediction(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a prediction from the policy.

        Args:
            state: Current state

        Returns:
            prediction: Sampled predicted next state
            log_prob: Log probability of the sample
        """
        mean, std, _ = self.forward(state)

        # Create Gaussian distribution
        dist = torch.distributions.Normal(mean, std)

        # Sample (reparameterized)
        prediction = dist.rsample()

        # Log prob (sum over state dimensions)
        log_prob = dist.log_prob(prediction).sum(dim=-1)

        return prediction, log_prob

    def get_deterministic_prediction(self, state: torch.Tensor) -> torch.Tensor:
        """Get deterministic prediction (mean) for evaluation."""
        mean, _, _ = self.forward(state)
        return mean

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
