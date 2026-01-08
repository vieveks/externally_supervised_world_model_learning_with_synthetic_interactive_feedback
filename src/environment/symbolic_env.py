"""Symbolic environment for WMIL."""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, List, Optional
import torch

from .tasks import PatternEchoTask, DelayedMatchTask, NavigationTask
from .parent import DeterministicParent


@dataclass
class RewardVector:
    """Vector-valued reward to preserve signal clarity."""

    goal: float = 0.0  # From environment (task success)
    pred: float = 0.0  # Intrinsic: world-model prediction accuracy

    def scalar(self, goal_weight: float = 1.0, pred_weight: float = 0.1) -> float:
        """Weighted sum for policy gradient."""
        return goal_weight * self.goal + pred_weight * self.pred

    def to_dict(self) -> Dict[str, float]:
        """Convert to dict for logging."""
        return {"goal": self.goal, "pred": self.pred}


@dataclass
class State:
    """
    State representation for symbolic environment.

    Supports:
    - Simple one-hot encoding (PatternEcho, SequenceNext)
    - Extended encoding with phase indicator (DelayedMatch)
    """

    target: int
    num_actions: int
    phase: int = 0  # For multi-phase tasks like DelayedMatch
    include_phase: bool = False  # Whether to include phase in encoding

    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        """
        Convert to tensor.

        Returns:
            - If include_phase=False: (num_actions,) one-hot
            - If include_phase=True: (num_actions+1,) one-hot + phase indicator
        """
        if self.include_phase:
            tensor = torch.zeros(self.num_actions + 1, device=device)
            tensor[self.target] = 1.0
            tensor[self.num_actions] = float(self.phase)
        else:
            tensor = torch.zeros(self.num_actions, device=device)
            tensor[self.target] = 1.0
        return tensor

    @property
    def dim(self) -> int:
        """Get state dimension."""
        return self.num_actions + 1 if self.include_phase else self.num_actions

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, num_actions: int, include_phase: bool = False) -> "State":
        """Create State from tensor."""
        if include_phase:
            target = tensor[:num_actions].argmax().item()
            phase = int(tensor[num_actions].item())
            return cls(target=target, num_actions=num_actions, phase=phase, include_phase=True)
        else:
            target = tensor.argmax().item()
            return cls(target=target, num_actions=num_actions)


@dataclass
class NavigationState:
    """
    State for NavigationTask with richer encoding.

    Encodes: [position_one_hot, target_one_hot, phase_one_hot]
    """

    position: int
    target: int
    phase: int
    num_positions: int
    episode_length: int = 3

    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        """
        Convert to tensor: [pos_one_hot, target_one_hot, phase_one_hot]
        """
        dim = 2 * self.num_positions + self.episode_length
        tensor = torch.zeros(dim, device=device)

        # Position one-hot
        tensor[self.position] = 1.0

        # Target one-hot
        tensor[self.num_positions + self.target] = 1.0

        # Phase one-hot
        tensor[2 * self.num_positions + self.phase] = 1.0

        return tensor

    @property
    def dim(self) -> int:
        return 2 * self.num_positions + self.episode_length


class SymbolicEnv:
    """
    Gym-like environment wrapper for WMIL.

    FIX #1: Episodes end on time limit ONLY, not on success.

    Supports:
    - Single-step tasks (PatternEcho, SequenceNext)
    - Multi-step tasks with phases (DelayedMatch)
    - Navigation tasks with richer state encoding (NavigationTask)
    """

    def __init__(
        self,
        task,  # Can be PatternEchoTask, SequenceNextTask, DelayedMatchTask, or NavigationTask
        parent: DeterministicParent,
        max_steps: int = 1,
        device: str = "cpu"
    ):
        """
        Initialize environment.

        Args:
            task: Task definition
            parent: Parent providing dynamics/rewards
            max_steps: Episode length (1 for MVP single-step)
            device: Torch device
        """
        self.task = task
        self.parent = parent
        self.max_steps = max_steps
        self.device = device

        # Check task type for appropriate handling
        self.is_delayed_match = hasattr(task, 'current_phase') and not isinstance(task, NavigationTask)
        self.is_navigation = isinstance(task, NavigationTask)

        # State tracking
        self.current_target: int = 0
        self.step_count: int = 0
        self.episode_success: bool = False

    def reset(self, difficulty: int = 0):
        """
        Start a new episode.

        Args:
            difficulty: Difficulty level (ignored for MVP)

        Returns:
            Initial state (State or NavigationState depending on task)
        """
        self.current_target, metadata = self.parent.generate_task_instance(difficulty)
        self.step_count = 0
        self.episode_success = False

        if self.is_navigation:
            return NavigationState(
                position=self.task.position,
                target=self.task.target,
                phase=self.task.current_phase,
                num_positions=self.task.num_positions,
                episode_length=self.task.episode_length
            )
        elif self.is_delayed_match:
            return State(
                target=self.current_target,
                num_actions=self.task.num_actions,
                phase=self.task.current_phase,
                include_phase=True
            )
        else:
            return State(target=self.current_target, num_actions=self.task.num_actions)

    def step(self, action: int):
        """
        Take action in environment.

        FIX #1: Done ONLY on time limit, NOT on success.

        Args:
            action: Action to take (0 to num_actions-1)

        Returns:
            next_state: New state (State or NavigationState)
            reward: RewardVector with goal reward
            done: True if episode ended (time limit only)
            info: Debug info including success flag
        """
        # Compute reward
        r_goal = self.parent.get_reward(self.current_target, action)

        # Get next target (for world-model prediction target)
        next_target = self.parent.get_transition(self.current_target, action)

        # Update state
        self.current_target = next_target
        self.step_count += 1

        # FIX #1: Done ONLY on time limit
        done = self.step_count >= self.max_steps

        # Track success separately for logging
        success = r_goal > 0.9
        self.episode_success = self.episode_success or success

        # Build info dict
        info = {
            "success": success,
            "episode_success": self.episode_success,
            "step": self.step_count,
            "target": self.current_target,
        }

        # Create next state with appropriate encoding
        if self.is_navigation:
            next_state = NavigationState(
                position=self.task.position,
                target=self.task.target,
                phase=self.task.current_phase,
                num_positions=self.task.num_positions,
                episode_length=self.task.episode_length
            )
            info["phase"] = self.task.current_phase
            info["position"] = self.task.position
            info["target_goal"] = self.task.target
        elif self.is_delayed_match:
            next_state = State(
                target=self.current_target,
                num_actions=self.task.num_actions,
                phase=self.task.current_phase,
                include_phase=True
            )
            info["phase"] = self.task.current_phase
            info["target_goal"] = self.task.target
        else:
            next_state = State(target=self.current_target, num_actions=self.task.num_actions)

        reward = RewardVector(goal=r_goal)

        return next_state, reward, done, info

    @property
    def state_dim(self) -> int:
        """Dimension of state tensor."""
        if self.is_navigation:
            return self.task.get_state_dim()  # 2*N + episode_length
        elif self.is_delayed_match:
            return self.task.num_actions + 1  # +1 for phase indicator
        return self.task.num_actions

    @property
    def action_dim(self) -> int:
        """Number of possible actions."""
        return self.task.num_actions
