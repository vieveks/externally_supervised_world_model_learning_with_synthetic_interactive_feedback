"""
Ambiguous Grammar Task - Week 1 of Language Upgrade

This implements grammars with TRUE ambiguity - multiple valid continuations
with equal or near-equal probability.

Key difference from stochastic bigrams:
- Bigrams had dominant token (50%) + secondary (20%)
- Ambiguous grammars have multiple tokens with EQUAL probability
- This forces policies to maintain distributions, not just argmax

Example:
    x → y (50%) | z (50%)

This is the first genuine "language phenomenon" - ambiguity.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import random
import torch
import numpy as np


@dataclass
class AmbiguousGrammar:
    """
    Grammar with true ambiguity - multiple equally valid continuations.

    This tests whether RL can maintain a distribution over tokens,
    not just collapse to a single argmax.

    Key properties:
    1. Some prefixes have multiple valid continuations with EQUAL probability
    2. Oracle accuracy < 100% (due to ambiguity)
    3. Requires maintaining policy entropy
    """

    vocab_size: int = 16
    ambiguity_level: str = "high"  # "low" (30/70), "medium" (40/60), "high" (50/50)
    num_ambiguous_tokens: int = 8  # How many tokens have ambiguous continuations
    branching_factor: int = 2  # Number of valid continuations per ambiguous token

    # Internal state
    _transitions: Dict[int, List[Tuple[int, float]]] = field(default_factory=dict, repr=False)
    _current_token: int = field(default=0, repr=False)
    _step_count: int = field(default=0, repr=False)

    def __post_init__(self):
        """Generate ambiguous grammar structure."""
        self._generate_grammar()

    def _generate_grammar(self):
        """
        Generate grammar with controlled ambiguity.

        Structure:
        - First half of vocab: ambiguous (multiple valid next tokens)
        - Second half of vocab: deterministic or low-entropy
        """
        # Set ambiguity probabilities based on level
        if self.ambiguity_level == "low":
            probs = [0.7, 0.3]  # 70/30 split
        elif self.ambiguity_level == "medium":
            probs = [0.6, 0.4]  # 60/40 split
        else:  # high
            probs = [0.5, 0.5]  # 50/50 split (true ambiguity)

        # Normalize if branching > 2
        if self.branching_factor > 2:
            probs = [1.0 / self.branching_factor] * self.branching_factor

        for token in range(self.vocab_size):
            if token < self.num_ambiguous_tokens:
                # Ambiguous: multiple valid continuations
                # Pick random next tokens
                next_tokens = random.sample(
                    range(self.vocab_size),
                    min(self.branching_factor, self.vocab_size)
                )
                self._transitions[token] = [(t, p) for t, p in zip(next_tokens, probs)]
            else:
                # Deterministic or low-entropy: single clear continuation
                next_token = (token + 1) % self.vocab_size
                self._transitions[token] = [(next_token, 1.0)]

    @property
    def state_dim(self) -> int:
        """State is one-hot of current token."""
        return self.vocab_size

    @property
    def num_actions(self) -> int:
        """Action is predicting next token."""
        return self.vocab_size

    def get_valid_continuations(self, token: int) -> List[Tuple[int, float]]:
        """
        Get all valid next tokens with their probabilities.

        Returns:
            List of (next_token, probability) tuples
        """
        return self._transitions[token]

    def is_ambiguous(self, token: int) -> bool:
        """Check if this token has ambiguous continuations."""
        continuations = self._transitions[token]
        return len(continuations) > 1 and all(p > 0.25 for _, p in continuations)

    def sample_next_token(self, current: int) -> int:
        """
        Sample next token according to grammar distribution.

        For ambiguous tokens, this samples from the distribution.
        For deterministic tokens, returns the single valid next token.
        """
        continuations = self._transitions[current]
        next_tokens = [t for t, _ in continuations]
        probs = [p for _, p in continuations]
        return random.choices(next_tokens, weights=probs)[0]

    def get_oracle_distribution(self, token: int) -> torch.Tensor:
        """
        Get the oracle probability distribution for next token.

        This is what an ideal learner should output.
        """
        dist = torch.zeros(self.vocab_size)
        for next_token, prob in self._transitions[token]:
            dist[next_token] = prob
        return dist

    def compute_oracle_accuracy(self) -> float:
        """
        Compute theoretical maximum accuracy (always predict argmax).

        For ambiguous tokens, accuracy = max_prob
        For deterministic tokens, accuracy = 1.0
        """
        total = 0.0
        for token in range(self.vocab_size):
            continuations = self._transitions[token]
            max_prob = max(p for _, p in continuations)
            total += max_prob
        return total / self.vocab_size

    def compute_entropy(self, token: int) -> float:
        """
        Compute entropy of next-token distribution.

        H = -sum(p * log(p))

        High entropy = high ambiguity
        Low entropy = deterministic or near-deterministic
        """
        continuations = self._transitions[token]
        probs = [p for _, p in continuations]
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        return entropy

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
        return self._current_token, self.encode_token(self._current_token)

    def step(self, predicted_token: int) -> Tuple[List[float], float, bool, Dict]:
        """
        Take a step: agent predicts next token, environment judges.

        Args:
            predicted_token: Agent's prediction of next token

        Returns:
            next_state: One-hot encoding of actual next token
            reward: Correctness reward (1.0 if correct, 0.0 otherwise)
            done: Episode finished
            info: Additional information including ambiguity
        """
        # Sample actual next token from grammar distribution
        actual_next = self.sample_next_token(self._current_token)

        # Compute reward (binary correctness)
        correct = (predicted_token == actual_next)
        reward = 1.0 if correct else 0.0

        # Check if this was an ambiguous decision point
        was_ambiguous = self.is_ambiguous(self._current_token)
        oracle_dist = self.get_oracle_distribution(self._current_token)
        entropy = self.compute_entropy(self._current_token)

        # Transition to next state
        self._current_token = actual_next
        self._step_count += 1

        # Episode termination (single-step for now)
        done = True

        info = {
            "correct": correct,
            "predicted": predicted_token,
            "actual": actual_next,
            "was_ambiguous": was_ambiguous,
            "oracle_distribution": oracle_dist,
            "entropy": entropy,
            "step": self._step_count,
        }

        return self.encode_token(self._current_token), reward, done, info

    def get_transition_matrix(self) -> torch.Tensor:
        """
        Get full transition matrix for analysis.

        Returns:
            T[i, j] = P(next=j | current=i)
        """
        T = torch.zeros(self.vocab_size, self.vocab_size)
        for token in range(self.vocab_size):
            for next_token, prob in self._transitions[token]:
                T[token, next_token] = prob
        return T

    def analyze_grammar(self) -> Dict:
        """
        Analyze grammar properties for validation.

        Returns statistics about ambiguity, entropy, etc.
        """
        num_ambiguous = sum(1 for t in range(self.vocab_size) if self.is_ambiguous(t))
        avg_entropy = np.mean([self.compute_entropy(t) for t in range(self.vocab_size)])
        oracle_acc = self.compute_oracle_accuracy()

        return {
            "vocab_size": self.vocab_size,
            "num_ambiguous_tokens": num_ambiguous,
            "ambiguity_percentage": num_ambiguous / self.vocab_size,
            "average_entropy": avg_entropy,
            "max_entropy": np.log(self.vocab_size),
            "normalized_entropy": avg_entropy / np.log(self.vocab_size),
            "oracle_accuracy": oracle_acc,
            "branching_factor": self.branching_factor,
            "ambiguity_level": self.ambiguity_level,
        }


@dataclass
class SequenceAmbiguousTask:
    """
    Multi-step ambiguous grammar with delayed reward.

    Tests credit assignment when:
    1. Multiple decisions are ambiguous
    2. Reward is delayed to end of sequence
    """

    vocab_size: int = 16
    sequence_length: int = 5
    ambiguity_level: str = "high"

    _base_task: AmbiguousGrammar = field(default=None, repr=False)
    _predictions: List[int] = field(default_factory=list, repr=False)
    _actuals: List[int] = field(default_factory=list, repr=False)
    _ambiguous_steps: List[bool] = field(default_factory=list, repr=False)
    _step: int = field(default=0, repr=False)

    def __post_init__(self):
        self._base_task = AmbiguousGrammar(
            vocab_size=self.vocab_size,
            ambiguity_level=self.ambiguity_level,
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
        self._ambiguous_steps = []
        self._step = 0
        return self._base_task.reset()

    def step(self, predicted_token: int) -> Tuple[List[float], float, bool, Dict]:
        """
        Take a step in the sequence.

        Reward is delayed until end of sequence.
        """
        # Get actual next token
        actual_next = self._base_task.sample_next_token(self._base_task._current_token)
        was_ambiguous = self._base_task.is_ambiguous(self._base_task._current_token)

        # Record prediction
        self._predictions.append(predicted_token)
        self._actuals.append(actual_next)
        self._ambiguous_steps.append(was_ambiguous)

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
            reward = correct_count / self.sequence_length
        else:
            reward = 0.0  # Delayed until end

        info = {
            "step": self._step,
            "predicted": predicted_token,
            "actual": actual_next,
            "correct": predicted_token == actual_next,
            "was_ambiguous": was_ambiguous,
            "sequence_predictions": self._predictions.copy(),
            "sequence_actuals": self._actuals.copy(),
            "ambiguous_steps": self._ambiguous_steps.copy(),
        }

        return self._base_task.encode_token(actual_next), reward, done, info

    def get_transition_matrix(self) -> torch.Tensor:
        """Get transition matrix from base task."""
        return self._base_task.get_transition_matrix()
