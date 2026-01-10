"""
Compositional Grammar Task - Week 2 of Language Upgrade

This implements a simple compositional structure to test whether RL
learns generalizable compositional rules or just memorizes training examples.

Key Design:
- Simple 2-token compositions: PREFIX + SUFFIX
- Train on most combinations
- Test on held-out compositions
- If RL generalizes → structure learning works despite distributional collapse

Example:
    Training: A→X, A→Y, B→X, B→Y, C→X
    Test: C→Y (never seen)
    Question: Does model predict Y after C, despite never seeing C→Y?

This is the critical Week 2 experiment that tests if structure learned
in Week 1 actually composes.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set
import random
import torch
import numpy as np


@dataclass
class CompositionalGrammar:
    """
    Compositional grammar with systematic train/test splits.

    Tests whether RL learns compositional structure that generalizes
    to novel combinations of seen primitives.

    Design:
    - vocab_size tokens total
    - Split into two classes: PREFIXES and SUFFIXES
    - Each valid sequence is PREFIX → SUFFIX
    - Hold out specific compositions for testing
    """

    vocab_size: int = 16
    num_prefixes: int = 4  # How many prefix tokens (e.g., A, B, C, D)
    num_suffixes: int = 4  # How many suffix tokens (e.g., X, Y, Z, W)
    held_out_fraction: float = 0.25  # Fraction of compositions to hold out

    # Internal state
    _prefix_tokens: List[int] = field(default_factory=list, repr=False)
    _suffix_tokens: List[int] = field(default_factory=list, repr=False)
    _train_compositions: Set[Tuple[int, int]] = field(default_factory=set, repr=False)
    _test_compositions: Set[Tuple[int, int]] = field(default_factory=set, repr=False)
    _current_token: int = field(default=0, repr=False)
    _target_suffix: int = field(default=0, repr=False)  # The correct suffix for this episode
    _step_count: int = field(default=0, repr=False)

    def __post_init__(self):
        """Generate compositional structure and splits."""
        self._generate_grammar()

    def _generate_grammar(self):
        """
        Generate compositional grammar with train/test splits.

        Strategy:
        1. Create DETERMINISTIC mapping: each prefix -> ONE specific suffix
        2. This makes the task learnable (not ambiguous)
        3. Hold out some (prefix, suffix) pairs for generalization testing
        """
        # Assign token IDs
        self._prefix_tokens = list(range(self.num_prefixes))
        self._suffix_tokens = list(range(self.num_prefixes, self.num_prefixes + self.num_suffixes))

        # Create deterministic 1-to-1 mapping
        # Prefix i maps to suffix i (wrapping if needed)
        all_compositions = [
            (p, self._suffix_tokens[p % len(self._suffix_tokens)])
            for p in self._prefix_tokens
        ]

        # Randomly split into train/test
        num_test = max(1, int(len(all_compositions) * self.held_out_fraction))
        random.shuffle(all_compositions)

        self._test_compositions = set(all_compositions[:num_test])
        self._train_compositions = set(all_compositions[num_test:])

    @property
    def state_dim(self) -> int:
        """State is one-hot of current token."""
        return self.vocab_size

    @property
    def num_actions(self) -> int:
        """Action is predicting next token."""
        return self.vocab_size

    def is_valid_composition(self, prefix: int, suffix: int) -> bool:
        """Check if this composition is valid in the grammar."""
        return (prefix, suffix) in self._train_compositions or (prefix, suffix) in self._test_compositions

    def is_train_composition(self, prefix: int, suffix: int) -> bool:
        """Check if this composition was seen during training."""
        return (prefix, suffix) in self._train_compositions

    def is_test_composition(self, prefix: int, suffix: int) -> bool:
        """Check if this composition is held-out for testing."""
        return (prefix, suffix) in self._test_compositions

    def get_correct_suffix(self, prefix: int) -> List[int]:
        """
        Get all valid suffixes for a given prefix.

        In training mode: only return train suffixes
        In test mode: return all suffixes (including held-out)
        """
        # Get all valid suffixes for this prefix
        valid_suffixes = []
        for suffix in self._suffix_tokens:
            if self.is_valid_composition(prefix, suffix):
                valid_suffixes.append(suffix)
        return valid_suffixes

    def encode_token(self, token: int) -> List[float]:
        """Encode token as one-hot vector."""
        state = [0.0] * self.vocab_size
        if 0 <= token < self.vocab_size:
            state[token] = 1.0
        return state

    def reset(self, test_mode: bool = False) -> Tuple[int, List[float]]:
        """
        Reset to a new episode.

        Args:
            test_mode: If True, sample from test compositions only

        Returns:
            current_token: The starting prefix token
            state: One-hot encoding of current token
        """
        # Sample a specific composition (prefix, suffix pair)
        if test_mode and self._test_compositions:
            # Sample from held-out compositions only
            composition = random.choice(list(self._test_compositions))
        else:
            # Sample from training compositions
            if self._train_compositions:
                composition = random.choice(list(self._train_compositions))
            else:
                # Fallback: random composition
                composition = (
                    random.choice(self._prefix_tokens),
                    random.choice(self._suffix_tokens)
                )

        self._current_token = composition[0]  # prefix
        self._target_suffix = composition[1]  # suffix (the correct answer)
        self._step_count = 0
        return self._current_token, self.encode_token(self._current_token)

    def step(self, predicted_token: int, test_mode: bool = False) -> Tuple[List[float], float, bool, Dict]:
        """
        Take a step: agent predicts suffix given prefix.

        Args:
            predicted_token: Agent's prediction of suffix
            test_mode: If True, use held-out compositions (ignored - composition set at reset)

        Returns:
            next_state: One-hot encoding of actual suffix
            reward: Correctness reward (1.0 if correct, 0.0 otherwise)
            done: Episode finished
            info: Additional information
        """
        # Current token is the prefix
        prefix = self._current_token

        # The correct answer was determined at reset
        actual_next = self._target_suffix

        # Compute reward (binary correctness)
        correct = (predicted_token == actual_next)
        reward = 1.0 if correct else 0.0

        # Check if this is a held-out composition
        is_held_out = self.is_test_composition(prefix, actual_next)

        # Transition to next state
        self._current_token = actual_next
        self._step_count += 1

        # Episode termination (single-step for now)
        done = True

        info = {
            "correct": correct,
            "predicted": predicted_token,
            "actual": actual_next,
            "prefix": prefix,
            "suffix": actual_next,
            "is_held_out": is_held_out,
            "step": self._step_count,
        }

        return self.encode_token(self._current_token), reward, done, info

    def analyze_splits(self) -> Dict:
        """
        Analyze train/test split properties.

        Returns statistics about compositional structure and splits.
        """
        total_compositions = len(self._train_compositions) + len(self._test_compositions)

        # Analyze coverage
        prefixes_in_train = set(p for p, _ in self._train_compositions)
        suffixes_in_train = set(s for _, s in self._train_compositions)
        prefixes_in_test = set(p for p, _ in self._test_compositions)
        suffixes_in_test = set(s for _, s in self._test_compositions)

        return {
            "total_compositions": total_compositions,
            "train_compositions": len(self._train_compositions),
            "test_compositions": len(self._test_compositions),
            "held_out_fraction": len(self._test_compositions) / total_compositions if total_compositions > 0 else 0,
            "num_prefixes": self.num_prefixes,
            "num_suffixes": self.num_suffixes,
            "prefixes_in_train": len(prefixes_in_train),
            "suffixes_in_train": len(suffixes_in_train),
            "prefixes_in_test": len(prefixes_in_test),
            "suffixes_in_test": len(suffixes_in_test),
            "all_primitives_in_train": (len(prefixes_in_train) == self.num_prefixes and
                                       len(suffixes_in_train) == self.num_suffixes),
        }

    def get_train_compositions(self) -> List[Tuple[int, int]]:
        """Get list of training compositions."""
        return list(self._train_compositions)

    def get_test_compositions(self) -> List[Tuple[int, int]]:
        """Get list of held-out test compositions."""
        return list(self._test_compositions)
