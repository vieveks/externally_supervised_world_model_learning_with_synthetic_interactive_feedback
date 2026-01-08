"""Token Prediction Model for Phase 3.1.

This extends the PredictionModel to discrete tokens.
Key difference: Categorical distribution over vocab instead of Gaussian over state.

Architecture:
    Token → Embedding → Transformer → Prediction Head → Logits over vocab
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Tuple

from .baby_model import TransformerCore, ValueHead


class TokenEmbedding(nn.Module):
    """Embeds discrete tokens into continuous space."""

    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        # Also support one-hot input for compatibility
        self.linear = nn.Linear(vocab_size, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed tokens.

        Args:
            x: Either integer tokens (batch,) or one-hot (batch, vocab_size)

        Returns:
            Embeddings (batch, hidden_dim)
        """
        if x.dtype == torch.long or (x.dtype == torch.int and x.dim() == 1):
            # Integer token input
            return self.embedding(x)
        else:
            # One-hot or soft input
            return self.linear(x)


class TokenPredictionHead(nn.Module):
    """
    Outputs logits over vocabulary for next token prediction.

    This is the discrete equivalent of PredictionHead from Phase 2b.
    """

    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Get logits over vocabulary.

        Args:
            h: Hidden state (batch, hidden_dim)

        Returns:
            Logits (batch, vocab_size)
        """
        return self.net(h)


class TokenPredictionModel(nn.Module):
    """
    Model for Token Prediction Task (Phase 3.1).

    Key difference from PredictionModel:
    - Output: Categorical distribution over vocab (not Gaussian over state)
    - Action: Discrete token selection (not continuous state vector)

    Architecture:
        Token → Embedding → Transformer → Prediction Head → Categorical(vocab)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Components
        self.embedding = TokenEmbedding(vocab_size, hidden_dim)
        self.transformer = TransformerCore(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.prediction_head = TokenPredictionHead(hidden_dim, vocab_size)
        self.value_head = ValueHead(hidden_dim)

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Args:
            state: Token state - either integer (batch,) or one-hot (batch, vocab_size)

        Returns:
            logits: Logits over vocabulary (batch, vocab_size)
            value: Value estimate (batch,)
        """
        # Handle single input (no batch dim)
        squeeze = False
        if state.dim() == 1:
            if state.dtype in [torch.long, torch.int]:
                # Single integer token
                state = state.unsqueeze(0)
            elif state.shape[0] == self.vocab_size:
                # Single one-hot vector
                state = state.unsqueeze(0)
                squeeze = True

        # Embed
        h = self.embedding(state)

        # Transform (add sequence dim, then remove)
        h = self.transformer(h.unsqueeze(1)).squeeze(1)

        # Get outputs
        logits = self.prediction_head(h)
        value = self.value_head(h)

        if squeeze:
            logits = logits.squeeze(0)
            value = value.squeeze(0)

        return logits, value

    def get_distribution(self, state: torch.Tensor) -> Categorical:
        """
        Get categorical distribution over next token.

        Args:
            state: Current token state

        Returns:
            Categorical distribution
        """
        logits, _ = self.forward(state)
        return Categorical(logits=logits)

    def sample_prediction(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a predicted next token from the policy.

        Args:
            state: Current token state

        Returns:
            prediction: Sampled token (batch,) or scalar
            log_prob: Log probability of the sample
        """
        logits, _ = self.forward(state)
        dist = Categorical(logits=logits)
        prediction = dist.sample()
        log_prob = dist.log_prob(prediction)
        return prediction, log_prob

    def get_deterministic_prediction(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get deterministic prediction (argmax) for evaluation.

        Args:
            state: Current token state

        Returns:
            Most likely next token
        """
        logits, _ = self.forward(state)
        return logits.argmax(dim=-1)

    def get_log_prob(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """
        Get log probability of a specific action.

        Args:
            state: Current token state
            action: Token that was selected

        Returns:
            Log probability
        """
        logits, _ = self.forward(state)
        dist = Categorical(logits=logits)
        return dist.log_prob(action)

    def get_entropy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get entropy of the policy distribution.

        Args:
            state: Current token state

        Returns:
            Entropy
        """
        logits, _ = self.forward(state)
        dist = Categorical(logits=logits)
        return dist.entropy()

    def get_hidden(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get hidden representation (for analysis).

        Args:
            state: Token state

        Returns:
            Hidden representation (batch, hidden_dim)
        """
        squeeze = False
        if state.dim() == 1:
            if state.dtype not in [torch.long, torch.int]:
                state = state.unsqueeze(0)
                squeeze = True

        h = self.embedding(state)
        h = self.transformer(h.unsqueeze(1)).squeeze(1)

        if squeeze:
            h = h.squeeze(0)

        return h

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SequenceTokenModel(nn.Module):
    """
    Model for sequence token prediction with memory.

    For multi-step prediction, maintains hidden state across steps.
    Uses LSTM or Transformer with proper masking.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 256,
        max_seq_len: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim) * 0.02
        )

        # Transformer with causal masking
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output heads
        self.prediction_head = TokenPredictionHead(hidden_dim, vocab_size)
        self.value_head = ValueHead(hidden_dim)

    def _get_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(
        self, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for sequence of tokens.

        Args:
            tokens: Token sequence (batch, seq_len) as integers

        Returns:
            logits: Next-token logits (batch, seq_len, vocab_size)
            values: Value estimates (batch, seq_len)
        """
        batch_size, seq_len = tokens.shape

        # Embed tokens
        h = self.embedding(tokens)  # (batch, seq_len, hidden)

        # Add positional encoding
        h = h + self.pos_encoding[:, :seq_len, :]

        # Apply transformer with causal mask
        mask = self._get_causal_mask(seq_len).to(h.device)
        h = self.transformer(h, mask=mask)

        # Get outputs for each position
        logits = self.prediction_head(h)  # (batch, seq_len, vocab)
        values = self.value_head(h)  # (batch, seq_len)

        return logits, values

    def sample_next(
        self, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample next token given sequence so far.

        Args:
            tokens: Token sequence (batch, seq_len)

        Returns:
            next_token: Sampled next token (batch,)
            log_prob: Log probability
        """
        logits, _ = self.forward(tokens)
        # Use logits at last position
        last_logits = logits[:, -1, :]
        dist = Categorical(logits=last_logits)
        next_token = dist.sample()
        log_prob = dist.log_prob(next_token)
        return next_token, log_prob

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
