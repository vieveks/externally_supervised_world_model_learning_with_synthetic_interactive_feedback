"""Token Prediction Task for Phase 3.1.

This extends Phase 2b's prediction-as-action to discrete tokens.
The agent predicts the next token, and gets rewarded for correctness.

Key difference from PredictionTask:
- Actions are discrete tokens (not continuous vectors)
- Uses Categorical distribution (not Gaussian)
- Reward is binary or shaped based on correctness

This is the bridge from state prediction to language prediction.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import random
import torch


@dataclass
class TokenPredictionTask:
    """
    Phase 3.1: Prediction-as-Action for discrete tokens.

    The agent's "action" IS its prediction of the next token.
    Reward = correctness of prediction.

    This mirrors LLM pretraining but with RL gradients:
    - LLM: action = predict next token, loss = cross-entropy
    - Ours: action = predict next token, reward = correctness

    Grammar types:
    1. deterministic_cyclic: 0→1→2→...→(V-1)→0
    2. deterministic_permutation: Fixed random mapping
    3. bigram: P(next|current) learned from fixed distribution
    4. context_free: Simple nested dependencies (e.g., balanced brackets)
    """

    vocab_size: int = 16
    name: str = "token_prediction"
    grammar_type: str = "deterministic_cyclic"  # or "deterministic_permutation", "bigram"
    reward_type: str = "binary"  # or "shaped"

    # For sequence prediction with delayed reward
    sequence_length: int = 1  # Number of predictions before reward
    reward_delay: int = 0  # Additional delay after sequence

    # Internal state
    _permutation: List[int] = field(default_factory=list, repr=False)
    _bigram_probs: Dict[int, List[float]] = field(default_factory=dict, repr=False)
    _current_token: int = field(default=0, repr=False)
    _step_count: int = field(default=0, repr=False)
    _pending_rewards: List[float] = field(default_factory=list, repr=False)

    def __post_init__(self):
        """Initialize grammar-specific structures."""
        if self.grammar_type == "deterministic_permutation" and not self._permutation:
            # Create a random but fixed permutation
            self._permutation = list(range(self.vocab_size))
            random.shuffle(self._permutation)

        if self.grammar_type == "bigram" and not self._bigram_probs:
            # Create random but fixed bigram probabilities
            for token in range(self.vocab_size):
                probs = [random.random() for _ in range(self.vocab_size)]
                total = sum(probs)
                self._bigram_probs[token] = [p / total for p in probs]

    @property
    def state_dim(self) -> int:
        """State is one-hot of current token."""
        return self.vocab_size

    @property
    def num_actions(self) -> int:
        """Action is predicting next token."""
        return self.vocab_size

    def get_next_token(self, current: int) -> int:
        """
        Get the correct next token given current token.

        This is the ground truth the agent must learn to predict.
        """
        if self.grammar_type == "deterministic_cyclic":
            return (current + 1) % self.vocab_size

        elif self.grammar_type == "deterministic_permutation":
            return self._permutation[current]

        elif self.grammar_type == "bigram":
            # Sample from bigram distribution
            probs = self._bigram_probs[current]
            return random.choices(range(self.vocab_size), weights=probs)[0]

        else:
            raise ValueError(f"Unknown grammar type: {self.grammar_type}")

    def get_deterministic_next(self, current: int) -> int:
        """
        Get deterministic next token (for evaluation).

        For stochastic grammars, returns most likely next token.
        """
        if self.grammar_type == "deterministic_cyclic":
            return (current + 1) % self.vocab_size

        elif self.grammar_type == "deterministic_permutation":
            return self._permutation[current]

        elif self.grammar_type == "bigram":
            # Return argmax of bigram distribution
            probs = self._bigram_probs[current]
            return probs.index(max(probs))

        else:
            raise ValueError(f"Unknown grammar type: {self.grammar_type}")

    def encode_token(self, token: int) -> List[float]:
        """Encode token as one-hot vector."""
        state = [0.0] * self.vocab_size
        state[token] = 1.0
        return state

    def reset(self) -> Tuple[int, List[float]]:
        """
        Reset to a new episode.

        Returns:
            current_token: The starting token
            state: One-hot encoding of current token
        """
        self._current_token = random.randint(0, self.vocab_size - 1)
        self._step_count = 0
        self._pending_rewards = []
        return self._current_token, self.encode_token(self._current_token)

    def step(self, predicted_token: int) -> Tuple[List[float], float, bool, Dict]:
        """
        Take a step: agent predicts next token, environment judges.

        Args:
            predicted_token: Agent's prediction of next token

        Returns:
            next_state: One-hot encoding of actual next token
            reward: Correctness reward (possibly delayed)
            done: Episode finished?
            info: Additional information
        """
        # Get actual next token
        actual_next = self.get_next_token(self._current_token)

        # Compute immediate correctness
        correct = (predicted_token == actual_next)
        immediate_reward = 1.0 if correct else 0.0

        # Handle reward delay
        if self.reward_delay > 0:
            self._pending_rewards.append(immediate_reward)
            if len(self._pending_rewards) > self.reward_delay:
                reward = self._pending_rewards.pop(0)
            else:
                reward = 0.0  # No reward yet (delayed)
        else:
            reward = immediate_reward

        # Apply reward shaping if configured
        if self.reward_type == "shaped" and not correct:
            # Partial credit based on "distance" in embedding space
            # (For now, just binary - can add shaping later)
            reward = reward

        # Transition to next state
        self._current_token = actual_next
        self._step_count += 1

        # Episode termination
        done = self._step_count >= self.sequence_length

        # Flush pending rewards at end of episode
        if done and self._pending_rewards:
            reward = sum(self._pending_rewards) / len(self._pending_rewards)
            self._pending_rewards = []

        info = {
            "correct": correct,
            "predicted": predicted_token,
            "actual": actual_next,
            "step": self._step_count,
        }

        return self.encode_token(self._current_token), reward, done, info

    def get_transition_matrix(self) -> torch.Tensor:
        """
        Get the full transition matrix for analysis.

        Returns:
            T[i, j] = P(next=j | current=i)
        """
        T = torch.zeros(self.vocab_size, self.vocab_size)

        if self.grammar_type == "deterministic_cyclic":
            for i in range(self.vocab_size):
                T[i, (i + 1) % self.vocab_size] = 1.0

        elif self.grammar_type == "deterministic_permutation":
            for i in range(self.vocab_size):
                T[i, self._permutation[i]] = 1.0

        elif self.grammar_type == "bigram":
            for i in range(self.vocab_size):
                for j in range(self.vocab_size):
                    T[i, j] = self._bigram_probs[i][j]

        return T


@dataclass
class SequenceTokenTask:
    """
    Extended token prediction with multi-step sequences.

    For testing credit assignment over longer horizons.
    Agent must predict a sequence of tokens, reward given at end.
    """

    vocab_size: int = 16
    sequence_length: int = 5  # Number of predictions per episode
    grammar_type: str = "deterministic_cyclic"
    reward_aggregation: str = "mean"  # "mean", "sum", "final"

    _base_task: TokenPredictionTask = field(default=None, repr=False)
    _predictions: List[int] = field(default_factory=list, repr=False)
    _actuals: List[int] = field(default_factory=list, repr=False)
    _step: int = field(default=0, repr=False)

    def __post_init__(self):
        self._base_task = TokenPredictionTask(
            vocab_size=self.vocab_size,
            grammar_type=self.grammar_type,
            sequence_length=1,  # Base task is single-step
            reward_delay=0,
        )

    @property
    def state_dim(self) -> int:
        return self.vocab_size

    @property
    def num_actions(self) -> int:
        return self.vocab_size

    def reset(self) -> Tuple[int, List[float]]:
        """Reset to new episode."""
        self._predictions = []
        self._actuals = []
        self._step = 0
        return self._base_task.reset()

    def step(self, predicted_token: int) -> Tuple[List[float], float, bool, Dict]:
        """
        Take a step in the sequence.

        Reward is delayed until end of sequence.
        """
        # Get actual next token
        actual_next = self._base_task.get_next_token(self._base_task._current_token)

        # Record prediction
        self._predictions.append(predicted_token)
        self._actuals.append(actual_next)

        # Transition
        self._base_task._current_token = actual_next
        self._step += 1

        # Check if episode done
        done = self._step >= self.sequence_length

        # Compute reward only at end
        if done:
            correct_count = sum(
                1 for p, a in zip(self._predictions, self._actuals) if p == a
            )
            if self.reward_aggregation == "mean":
                reward = correct_count / self.sequence_length
            elif self.reward_aggregation == "sum":
                reward = float(correct_count)
            elif self.reward_aggregation == "final":
                # Only reward if ALL predictions correct
                reward = 1.0 if correct_count == self.sequence_length else 0.0
            else:
                reward = correct_count / self.sequence_length
        else:
            reward = 0.0  # Delayed until end

        info = {
            "step": self._step,
            "predicted": predicted_token,
            "actual": actual_next,
            "correct": predicted_token == actual_next,
            "sequence_predictions": self._predictions.copy(),
            "sequence_actuals": self._actuals.copy(),
        }

        return self._base_task.encode_token(actual_next), reward, done, info

    def get_transition_matrix(self) -> torch.Tensor:
        """Get transition matrix from base task."""
        return self._base_task.get_transition_matrix()
