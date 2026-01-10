"""Factorized Compositional Grammar for Week 2 (v2).

True compositional structure with independent slots.

Grammar:
    A → {x, y}  (slot A has 2 choices)
    B → {a, b}  (slot B has 2 choices)
    Sequences: A B

Train/Test Split:
    Train: x-a, x-b, y-a (3 of 4 combinations)
    Test: y-b (held-out combination)

Key Property:
    Each symbol appears in multiple contexts.
    Generalization requires recombining known parts.
    Tests if model learns factorized structure: P(A, B) = P(A) × P(B)
"""

import random
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Dict


@dataclass
class FactorizedGrammar:
    """
    Factorized compositional grammar with 2 slots.

    Structure:
        Slot A: num_a_values choices (e.g., x, y)
        Slot B: num_b_values choices (e.g., a, b)
        Sequences: A B (2 tokens)

    Train/Test Split:
        Hold out specific (A, B) combinations
        Each symbol appears in training (no novel primitives)
    """

    vocab_size: int = 16
    num_a_values: int = 2  # How many values for slot A (e.g., x, y)
    num_b_values: int = 2  # How many values for slot B (e.g., a, b)
    held_out_fraction: float = 0.25  # Fraction of combinations to hold out

    # Internal state
    _a_tokens: List[int] = field(default_factory=list, repr=False)
    _b_tokens: List[int] = field(default_factory=list, repr=False)
    _train_sequences: Set[Tuple[int, int]] = field(default_factory=set, repr=False)
    _test_sequences: Set[Tuple[int, int]] = field(default_factory=set, repr=False)
    _current_sequence: List[int] = field(default_factory=list, repr=False)
    _step_count: int = field(default=0, repr=False)

    def __post_init__(self):
        """Generate factorized grammar and splits."""
        self._generate_grammar()

    def _generate_grammar(self):
        """
        Generate factorized compositional structure.

        Creates all A × B combinations, then holds out some for testing.
        """
        # Assign token IDs
        # A tokens: 0, 1, 2, ... (first num_a_values)
        # B tokens: num_a_values, num_a_values+1, ... (next num_b_values)
        self._a_tokens = list(range(self.num_a_values))
        self._b_tokens = list(range(self.num_a_values, self.num_a_values + self.num_b_values))

        # Generate all possible sequences: A × B
        all_sequences = [
            (a, b) for a in self._a_tokens for b in self._b_tokens
        ]

        # Randomly split into train/test
        num_test = max(1, int(len(all_sequences) * self.held_out_fraction))
        random.shuffle(all_sequences)

        self._test_sequences = set(all_sequences[:num_test])
        self._train_sequences = set(all_sequences[num_test:])

        # Verify all primitives appear in training
        train_a = set(a for a, b in self._train_sequences)
        train_b = set(b for a, b in self._train_sequences)

        assert len(train_a) == self.num_a_values, \
            f"Not all A values in training! Got {len(train_a)}, need {self.num_a_values}"
        assert len(train_b) == self.num_b_values, \
            f"Not all B values in training! Got {len(train_b)}, need {self.num_b_values}"

    @property
    def state_dim(self) -> int:
        """State is one-hot of current token sequence."""
        return self.vocab_size

    @property
    def num_actions(self) -> int:
        """Action is predicting next token."""
        return self.vocab_size

    def get_train_sequences(self) -> Set[Tuple[int, int]]:
        """Get training sequences."""
        return self._train_sequences.copy()

    def get_test_sequences(self) -> Set[Tuple[int, int]]:
        """Get test sequences."""
        return self._test_sequences.copy()

    def is_train_sequence(self, a: int, b: int) -> bool:
        """Check if (a, b) sequence is in training set."""
        return (a, b) in self._train_sequences

    def is_test_sequence(self, a: int, b: int) -> bool:
        """Check if (a, b) sequence is in test set."""
        return (a, b) in self._test_sequences

    def encode_token(self, token: int) -> List[float]:
        """One-hot encode a token."""
        state = [0.0] * self.vocab_size
        if 0 <= token < self.vocab_size:
            state[token] = 1.0
        return state

    def reset(self, test_mode: bool = False) -> Tuple[int, List[float]]:
        """
        Reset to a new episode.

        Args:
            test_mode: If True, sample from test sequences only

        Returns:
            first_token: The A token (first in sequence)
            state: One-hot encoding of first token
        """
        # Sample a sequence
        if test_mode and self._test_sequences:
            sequence = random.choice(list(self._test_sequences))
        else:
            if self._train_sequences:
                sequence = random.choice(list(self._train_sequences))
            else:
                # Fallback
                sequence = (
                    random.choice(self._a_tokens),
                    random.choice(self._b_tokens)
                )

        self._current_sequence = list(sequence)  # [a_token, b_token]
        self._step_count = 0

        # Return first token (A)
        first_token = self._current_sequence[0]
        return first_token, self.encode_token(first_token)

    def step(self, predicted_token: int) -> Tuple[List[float], float, bool, Dict]:
        """
        Take a step: predict next token in sequence.

        Step 0: Given A, predict B

        Args:
            predicted_token: Agent's prediction

        Returns:
            next_state: One-hot of next token (or terminal state)
            reward: Correctness reward (1.0 if correct, 0.0 otherwise)
            done: Episode finished
            info: Additional information
        """
        if self._step_count == 0:
            # Step 0: Predict B given A
            correct_token = self._current_sequence[1]  # B token
            correct = (predicted_token == correct_token)
            reward = 1.0 if correct else 0.0

            # Check if this sequence is held-out
            a_token = self._current_sequence[0]
            is_held_out = self.is_test_sequence(a_token, correct_token)

            self._step_count += 1
            done = True  # Single-step prediction for now

            info = {
                "correct": correct,
                "predicted": predicted_token,
                "actual": correct_token,
                "sequence": tuple(self._current_sequence),
                "is_held_out": is_held_out,
                "step": self._step_count,
            }

            return self.encode_token(correct_token), reward, done, info

        else:
            # Should not reach here in single-step mode
            raise ValueError("Episode already finished!")

    def analyze_splits(self) -> Dict:
        """Analyze train/test split properties."""
        train_a = set(a for a, b in self._train_sequences)
        train_b = set(b for a, b in self._train_sequences)
        test_a = set(a for a, b in self._test_sequences)
        test_b = set(b for a, b in self._test_sequences)

        return {
            "total_sequences": len(self._train_sequences) + len(self._test_sequences),
            "train_sequences": len(self._train_sequences),
            "test_sequences": len(self._test_sequences),
            "num_a_values": self.num_a_values,
            "num_b_values": self.num_b_values,
            "a_values_in_train": len(train_a),
            "b_values_in_train": len(train_b),
            "a_values_in_test": len(test_a),
            "b_values_in_test": len(test_b),
            "all_primitives_in_train": (
                len(train_a) == self.num_a_values and
                len(train_b) == self.num_b_values
            ),
        }
