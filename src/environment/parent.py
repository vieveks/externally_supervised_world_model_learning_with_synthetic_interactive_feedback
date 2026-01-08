"""Parent environment implementations.

MVP: DeterministicParent only - no LLM until this works.
"""

from typing import Tuple, Union
import torch

from .tasks import PatternEchoTask, SequenceNextTask, DelayedMatchTask, NavigationTask

# Type alias for supported tasks
TaskType = Union[PatternEchoTask, SequenceNextTask, DelayedMatchTask, NavigationTask]


class DeterministicParent:
    """
    Rule-based environment for MVP.

    No LLM calls - pure Python logic for fast iteration and debugging.
    Supports multiple task types.
    """

    def __init__(self, task: TaskType):
        """
        Initialize with a task.

        Args:
            task: The task defining rules and rewards
        """
        self.task = task

    def generate_task_instance(self, difficulty: int = 0) -> Tuple[int, dict]:
        """
        Generate a new task instance.

        Returns:
            target: The target action for this instance
            metadata: Debug info
        """
        return self.task.generate_instance(difficulty)

    def get_reward(self, target: int, action: int) -> float:
        """
        Compute reward for action given current target.

        Args:
            target: The correct target action
            action: The action taken

        Returns:
            Reward value (0.0 or 1.0 for PatternEcho)
        """
        return self.task.compute_reward(action, target)

    def get_transition(self, current_target: int, action: int) -> int:
        """
        Get next state's target after action.

        Args:
            current_target: Current target
            action: Action taken

        Returns:
            New target for next state
        """
        return self.task.get_next_state_target(current_target, action)
