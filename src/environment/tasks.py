"""Task definitions for WMIL.

Tasks:
- PatternEcho: Single-step copy task (MVP)
- SequenceNext: Multi-step sequence prediction task (Phase 2)
- DelayedMatch: Multi-step navigation with delayed reward (Phase 3)
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Protocol
import random


class Task(Protocol):
    """Protocol for all tasks."""
    num_actions: int
    name: str

    def generate_instance(self, difficulty: int = 0) -> Tuple[int, dict]: ...
    def compute_reward(self, action: int, target: int) -> float: ...
    def get_next_state_target(self, current_target: int, action: int) -> int: ...


@dataclass
class PatternEchoTask:
    """
    Simplest possible task: output the target number shown in state.

    State: One-hot encoding of target (e.g., target=3 -> [0,0,0,1,0,0,0,0])
    Goal: Output action == target
    Reward: 1.0 if correct, 0.0 otherwise

    This tests: Can the baby learn to "echo" what it sees?
    """

    num_actions: int = 8
    name: str = "pattern_echo"

    def generate_instance(self, difficulty: int = 0) -> Tuple[int, dict]:
        """
        Generate a task instance.

        Args:
            difficulty: Not used for MVP (always full range)

        Returns:
            target: The correct action (0 to num_actions-1)
            metadata: Additional info for debugging
        """
        # For MVP, difficulty is ignored - use full range
        target = random.randint(0, self.num_actions - 1)
        return target, {"difficulty": difficulty}

    def compute_reward(self, action: int, target: int) -> float:
        """
        Compute reward for action given target.

        Args:
            action: The action taken by the baby
            target: The correct target action

        Returns:
            1.0 if correct, 0.0 otherwise
        """
        return 1.0 if action == target else 0.0

    def get_next_state_target(self, current_target: int, action: int) -> int:
        """
        Get next state's target after action.

        For PatternEcho with T=1, this doesn't matter much,
        but we generate a new random target for consistency.
        """
        return random.randint(0, self.num_actions - 1)


@dataclass
class SequenceNextTask:
    """
    Multi-step sequence prediction task.

    The baby must learn a deterministic sequence and predict the next element.

    Example sequence (length=4): [2, 5, 1, 7, 2, 5, 1, 7, ...]
    - State shows current position in sequence
    - Baby must output the NEXT element in the sequence
    - This requires learning the transition dynamics

    Why this matters:
    - Single-step reactive policies will fail
    - The world-model must learn the sequence structure
    - Prediction accuracy directly affects task performance

    State encoding: One-hot of current sequence element
    Action: Predict next element (0 to num_actions-1)
    Reward: 1.0 if prediction matches next element, 0.0 otherwise
    """

    num_actions: int = 8
    sequence_length: int = 4  # Length of repeating pattern
    name: str = "sequence_next"

    # The sequence is generated once per task instance
    _sequence: List[int] = field(default_factory=list, repr=False)
    _position: int = field(default=0, repr=False)

    def __post_init__(self):
        """Generate a random sequence on initialization."""
        if not self._sequence:
            self._sequence = [random.randint(0, self.num_actions - 1)
                             for _ in range(self.sequence_length)]

    def generate_instance(self, difficulty: int = 0) -> Tuple[int, dict]:
        """
        Generate a task instance.

        For SequenceNext, we optionally regenerate the sequence based on difficulty.

        Args:
            difficulty: 0 = keep same sequence, >0 = regenerate with longer length

        Returns:
            target: Current element (what the state shows)
            metadata: Sequence info for debugging
        """
        # Difficulty can increase sequence length
        if difficulty > 0:
            self.sequence_length = min(4 + difficulty, self.num_actions)
            self._sequence = [random.randint(0, self.num_actions - 1)
                             for _ in range(self.sequence_length)]

        # Start at random position in sequence
        self._position = random.randint(0, self.sequence_length - 1)
        current = self._sequence[self._position]

        return current, {
            "difficulty": difficulty,
            "sequence": self._sequence.copy(),
            "position": self._position,
            "next_correct": self._sequence[(self._position + 1) % self.sequence_length]
        }

    def compute_reward(self, action: int, target: int) -> float:
        """
        Compute reward for action.

        For SequenceNext, reward is based on whether the action
        correctly predicts the NEXT element in the sequence.

        Note: 'target' here is the CURRENT element (what state shows).
        The correct action is the NEXT element.

        Args:
            action: The action taken (prediction of next element)
            target: The current element (what state shows)

        Returns:
            1.0 if action == next element, 0.0 otherwise
        """
        # Find position of current element in sequence
        # (In case of duplicates, use stored position)
        next_idx = (self._position + 1) % self.sequence_length
        correct_next = self._sequence[next_idx]
        return 1.0 if action == correct_next else 0.0

    def get_next_state_target(self, current_target: int, action: int) -> int:
        """
        Get next state's target after action.

        The sequence advances deterministically regardless of action.

        Args:
            current_target: Current element
            action: Action taken (doesn't affect transition)

        Returns:
            Next element in sequence
        """
        # Advance position in sequence
        self._position = (self._position + 1) % self.sequence_length
        return self._sequence[self._position]

    def reset_sequence(self, new_sequence: List[int] = None):
        """
        Reset or set a new sequence.

        Args:
            new_sequence: Optional specific sequence to use
        """
        if new_sequence:
            self._sequence = new_sequence
            self.sequence_length = len(new_sequence)
        else:
            self._sequence = [random.randint(0, self.num_actions - 1)
                             for _ in range(self.sequence_length)]
        self._position = 0


@dataclass
class DelayedMatchTask:
    """
    Multi-step navigation task with delayed reward.

    This task REQUIRES a world model because:
    1. Reward is delayed (only at end of episode)
    2. Actions affect state transitions
    3. Must plan: target → action → position → match?

    Episode structure (episode_length=2):
        Step 0: State = [target_one_hot, phase=0]
                Action taken, position updated
                Reward = 0

        Step 1: State = [position_one_hot, phase=1]
                Reward = 1.0 if position == target, else 0.0

    State encoding: [position_one_hot (N), phase (1)] = N+1 dims
        - phase=0: showing target
        - phase=1: showing current position

    Transition dynamics (1D circular grid):
        - Actions 0 to N-1: move to that position directly
        - This is simplified: action = desired position

    Why world-model helps:
        - Without WM: Only 1 sparse reward signal per episode
        - With WM: Can learn action→position mapping, enabling planning
    """

    num_actions: int = 8  # Also number of positions
    name: str = "delayed_match"

    # Episode state
    _target: int = field(default=0, repr=False)
    _position: int = field(default=0, repr=False)
    _phase: int = field(default=0, repr=False)  # 0=target shown, 1=position shown

    def generate_instance(self, difficulty: int = 0) -> Tuple[int, dict]:
        """
        Generate a new episode.

        Returns the target (shown in phase 0).
        """
        self._target = random.randint(0, self.num_actions - 1)
        self._position = random.randint(0, self.num_actions - 1)  # Random start
        self._phase = 0  # Start by showing target

        return self._target, {
            "difficulty": difficulty,
            "target": self._target,
            "start_position": self._position,
            "phase": self._phase,
        }

    def compute_reward(self, action: int, target: int) -> float:
        """
        Compute reward.

        Reward is ONLY given at phase 1 (end of episode).
        At phase 0, always returns 0.

        Args:
            action: Action taken
            target: Current state value (target in phase 0, position in phase 1)

        Returns:
            1.0 if phase=1 and position==target, else 0.0
        """
        if self._phase == 0:
            # Phase 0: No reward yet, just update position based on action
            return 0.0
        else:
            # Phase 1: Check if we reached target
            return 1.0 if self._position == self._target else 0.0

    def get_next_state_target(self, current_target: int, action: int) -> int:
        """
        Transition to next state.

        Phase 0 → Phase 1: Action determines new position
        Phase 1 → Done: Episode ends

        Returns the value to encode in next state's one-hot.
        """
        if self._phase == 0:
            # Action directly sets position (simplified dynamics)
            self._position = action % self.num_actions
            self._phase = 1
            return self._position  # Next state shows position
        else:
            # Episode should end after phase 1
            # Return current position (doesn't matter, episode ends)
            return self._position

    def get_state_dim(self) -> int:
        """State dimension: position one-hot + phase indicator."""
        return self.num_actions + 1

    def encode_state(self, value: int, phase: int) -> List[float]:
        """
        Encode state as vector.

        Args:
            value: Position or target to encode
            phase: Current phase (0 or 1)

        Returns:
            List of floats: [one_hot..., phase_indicator]
        """
        state = [0.0] * (self.num_actions + 1)
        state[value] = 1.0
        state[self.num_actions] = float(phase)
        return state

    @property
    def current_phase(self) -> int:
        """Get current phase."""
        return self._phase

    @property
    def target(self) -> int:
        """Get current target."""
        return self._target


@dataclass
class PredictionTask:
    """
    Phase 2b: Prediction-as-Action task.

    THE KEY INSIGHT: The agent's "action" IS its prediction of the next state.
    Reward = accuracy of prediction.

    This unifies prediction and control:
    - No separate policy head (outputs state prediction)
    - No separate WM head (prediction IS the action)
    - Agent learns to predict by being rewarded for accuracy
    - RL gradients directly optimize prediction quality

    Why this matters:
    - In LLM pretraining: action = predict next token, reward = log P(correct)
    - Here: action = predict next state, reward = -MSE or threshold
    - If RL reward shapes representations as well as MLE, we've proven the thesis

    Episode structure (T=1 for simplicity):
        State: One-hot encoding of current position
        Action: Predicted next state (continuous vector, state_dim dimensions)
        Dynamics: Deterministic transition (e.g., circular shift)
        Reward: -MSE(prediction, actual) or 1.0 if MSE < threshold

    State encoding: One-hot of position (N dimensions)
    Action space: Continuous (N dimensions) - the predicted next state

    Ablation: reward_delay parameter for delayed reward experiments.
        reward_delay=0: Immediate reward (default)
        reward_delay=k: Reward given after k steps (credit assignment challenge)
    """

    num_positions: int = 8
    name: str = "prediction"
    dynamics_type: str = "circular_shift"  # or "random_fixed", "identity"
    reward_type: str = "negative_mse"  # or "threshold_binary"
    threshold: float = 0.1  # For threshold_binary reward
    reward_delay: int = 0  # Ablation: delay reward by k steps

    # For random_fixed dynamics: a fixed permutation
    _permutation: List[int] = field(default_factory=list, repr=False)
    _position: int = field(default=0, repr=False)

    def __post_init__(self):
        """Initialize dynamics if needed."""
        if self.dynamics_type == "random_fixed" and not self._permutation:
            # Create a random but fixed permutation
            self._permutation = list(range(self.num_positions))
            random.shuffle(self._permutation)

    @property
    def num_actions(self) -> int:
        """Action dimension = state dimension (predicting next state)."""
        return self.num_positions

    @property
    def state_dim(self) -> int:
        """State is one-hot of position."""
        return self.num_positions

    @property
    def action_continuous(self) -> bool:
        """This task has continuous actions (state predictions)."""
        return True

    def generate_instance(self, difficulty: int = 0) -> Tuple[int, dict]:
        """
        Generate a new episode.

        Returns the starting position.
        """
        self._position = random.randint(0, self.num_positions - 1)
        return self._position, {
            "difficulty": difficulty,
            "position": self._position,
            "dynamics": self.dynamics_type,
        }

    def get_dynamics_next(self, position: int) -> int:
        """
        Apply deterministic dynamics to get next position.

        This is the ground truth the agent must learn to predict.
        """
        if self.dynamics_type == "circular_shift":
            # Simple: position advances by 1
            return (position + 1) % self.num_positions
        elif self.dynamics_type == "random_fixed":
            # Fixed permutation (learnable but not trivial)
            return self._permutation[position]
        elif self.dynamics_type == "identity":
            # Trivial: state doesn't change (for debugging)
            return position
        else:
            raise ValueError(f"Unknown dynamics: {self.dynamics_type}")

    def encode_state(self, position: int) -> List[float]:
        """Encode position as one-hot vector."""
        state = [0.0] * self.num_positions
        state[position] = 1.0
        return state

    def compute_reward(self, predicted_state: List[float], actual_position: int) -> float:
        """
        Compute reward based on prediction accuracy.

        Args:
            predicted_state: Agent's predicted next state (continuous vector)
            actual_position: The actual next position (used to create target)

        Returns:
            Reward value (higher = better prediction)
        """
        # Create target one-hot
        target = [0.0] * self.num_positions
        target[actual_position] = 1.0

        # Compute MSE
        mse = sum((p - t) ** 2 for p, t in zip(predicted_state, target)) / len(target)

        if self.reward_type == "negative_mse":
            # Reward = -MSE (higher is better, 0 is perfect)
            return -mse
        elif self.reward_type == "threshold_binary":
            # Reward = 1 if MSE < threshold, else 0
            return 1.0 if mse < self.threshold else 0.0
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")

    def step(self, predicted_state: List[float]) -> Tuple[List[float], float, bool]:
        """
        Take a step: agent predicts next state, environment judges accuracy.

        Args:
            predicted_state: Agent's prediction of next state

        Returns:
            next_state: The actual next state (one-hot)
            reward: Prediction accuracy reward
            done: Always True (T=1 episodes)
        """
        # Get actual next position from dynamics
        actual_next_position = self.get_dynamics_next(self._position)

        # Compute reward based on prediction accuracy
        reward = self.compute_reward(predicted_state, actual_next_position)

        # Transition to next state
        self._position = actual_next_position
        next_state = self.encode_state(self._position)

        # T=1: episode always ends after one step
        done = True

        return next_state, reward, done

    def get_target_state(self) -> List[float]:
        """Get the correct next state (for computing metrics)."""
        next_pos = self.get_dynamics_next(self._position)
        return self.encode_state(next_pos)


@dataclass
class NavigationTask:
    """
    T=3 Navigation task with RELATIVE movement dynamics.

    This task STRUCTURALLY requires a world model because:
    1. Actions cause RELATIVE position changes (not absolute)
    2. Must plan 2 moves to reach target from random start
    3. Credit assignment across multiple steps
    4. No simple mapping from observation to action

    Episode structure (T=3):
        Step 0: State = [start_pos, target_pos, phase=0]
                Agent sees where it is AND where to go
                Takes action → moves relatively
                Reward = 0

        Step 1: State = [current_pos, target_pos, phase=1]
                Agent sees updated position, same target
                Takes action → moves relatively
                Reward = 0

        Step 2: State = [final_pos, target_pos, phase=2]
                Reward = 1.0 if final_pos == target, else 0.0

    State encoding: [position_one_hot (N), target_one_hot (N), phase_one_hot (3)]
        Total: 2*N + 3 dimensions

    Action semantics (8 actions for 8 positions):
        Actions 0-7: Move to make position = (current + action - 3) mod 8
        So: action=0 → move -3, action=3 → stay, action=7 → move +4

    Why world-model is ESSENTIAL:
        - Policy sees (pos, target, phase) but correct action depends on
          PLANNING: what sequence of relative moves reaches target?
        - Without WM: Only sparse reward at end, no signal for intermediate actions
        - With WM: Can simulate trajectories, learn action→position dynamics
    """

    num_positions: int = 8
    num_actions: int = 8  # Relative moves
    name: str = "navigation"
    episode_length: int = 3

    # Episode state
    _target: int = field(default=0, repr=False)
    _position: int = field(default=0, repr=False)
    _phase: int = field(default=0, repr=False)

    def generate_instance(self, difficulty: int = 0) -> Tuple[int, dict]:
        """
        Generate a new episode.

        Returns initial state info.
        """
        self._target = random.randint(0, self.num_positions - 1)
        self._position = random.randint(0, self.num_positions - 1)
        self._phase = 0

        return self._position, {
            "difficulty": difficulty,
            "target": self._target,
            "start_position": self._position,
            "phase": self._phase,
        }

    def compute_reward(self, action: int, target: int) -> float:
        """
        Compute reward. Only given at final phase.
        """
        if self._phase < self.episode_length - 1:
            return 0.0
        else:
            return 1.0 if self._position == self._target else 0.0

    def get_next_state_target(self, current_target: int, action: int) -> int:
        """
        Apply relative movement and advance phase.

        Action semantics: action causes position change of (action - 3)
        - action=0: move -3
        - action=3: stay (move 0)
        - action=7: move +4
        """
        if self._phase < self.episode_length - 1:
            # Apply relative movement
            delta = action - 3  # Convert action to relative move
            self._position = (self._position + delta) % self.num_positions
            self._phase += 1

        return self._position

    def get_state_dim(self) -> int:
        """State dimension: pos_one_hot + target_one_hot + phase_one_hot."""
        return self.num_positions + self.num_positions + self.episode_length

    def encode_state(self) -> List[float]:
        """
        Encode current state as vector.

        Returns:
            [position_one_hot, target_one_hot, phase_one_hot]
        """
        state = [0.0] * self.get_state_dim()

        # Position one-hot (first N dims)
        state[self._position] = 1.0

        # Target one-hot (next N dims)
        state[self.num_positions + self._target] = 1.0

        # Phase one-hot (last 3 dims)
        state[2 * self.num_positions + self._phase] = 1.0

        return state

    @property
    def current_phase(self) -> int:
        return self._phase

    @property
    def target(self) -> int:
        return self._target

    @property
    def position(self) -> int:
        return self._position

    def is_done(self) -> bool:
        """Check if episode is complete."""
        return self._phase >= self.episode_length - 1
