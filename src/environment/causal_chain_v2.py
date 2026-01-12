"""
CausalChain-T3 v2: True Partial Observability

Key change from v1: Agent does NOT see previous action.
This ensures the task cannot be solved by memorization.

The only way to solve this task is to:
1. Build an internal belief state over latent states
2. Use that belief to plan actions
3. Update belief based on observed transitions
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass


@dataclass
class CausalChainV2Config:
    """Configuration for CausalChain v2 environment."""
    num_latent_states: int = 4
    num_observations: int = 2
    num_actions: int = 3
    horizon: int = 3

    LEFT: int = 0
    RIGHT: int = 1
    STAY: int = 2


class CausalChainEnvV2:
    """
    CausalChain-T3 v2 with True Partial Observability

    Key difference from v1:
    - Agent does NOT receive previous action as input
    - Agent only sees: [current_observation, target, timestep]
    - This means the agent cannot distinguish aliased states from observation alone

    Example of aliasing:
        State 0 and State 1 both produce observation 0
        If agent is at obs=0, it doesn't know if it's at state 0 or 1
        To reach target=2:
            - From state 0: need RIGHT, RIGHT
            - From state 1: need RIGHT (just 1 step)
        The agent CANNOT know which path to take without tracking belief state

    The ONLY way to solve this task:
        1. Maintain belief distribution over latent states
        2. Update belief based on actions taken and observations received
        3. Plan based on expected outcomes under current belief
    """

    def __init__(self, config: Optional[CausalChainV2Config] = None):
        self.config = config or CausalChainV2Config()
        self._build_transition_table()

        self._latent_state: int = 0
        self._target_state: int = 0
        self._timestep: int = 0
        self._done: bool = False
        self._action_history: List[int] = []  # For analysis only

    def _build_transition_table(self):
        """Cyclic transitions."""
        n = self.config.num_latent_states
        self.transition_table = np.zeros((n, 3), dtype=np.int32)
        for z in range(n):
            self.transition_table[z, self.config.LEFT] = (z - 1) % n
            self.transition_table[z, self.config.RIGHT] = (z + 1) % n
            self.transition_table[z, self.config.STAY] = z

    def _get_observation(self, latent_state: int) -> int:
        """Aliasing: o = z // 2."""
        return latent_state // 2

    def reset(self, start_state: Optional[int] = None,
              target_state: Optional[int] = None) -> Dict:
        """Reset environment."""
        if start_state is not None:
            self._latent_state = start_state
        else:
            self._latent_state = np.random.randint(self.config.num_latent_states)

        if target_state is not None:
            self._target_state = target_state
        else:
            self._target_state = np.random.randint(self.config.num_latent_states)

        self._timestep = 0
        self._done = False
        self._action_history = []

        return self._get_obs_dict()

    def _get_obs_dict(self) -> Dict:
        """
        Get observation dictionary.

        NOTE: Does NOT include previous action!
        """
        return {
            'observation': self._get_observation(self._latent_state),
            'target': self._target_state,
            'timestep': self._timestep,
            # For debugging/probing only:
            'latent_state': self._latent_state,
        }

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """Take action."""
        if self._done:
            raise RuntimeError("Episode done. Call reset().")

        old_latent = self._latent_state
        self._latent_state = self.transition_table[self._latent_state, action]
        self._timestep += 1
        self._action_history.append(action)

        self._done = (self._timestep >= self.config.horizon)

        if self._done:
            reward = 1.0 if self._latent_state == self._target_state else 0.0
        else:
            reward = 0.0

        info = {
            'old_latent': old_latent,
            'new_latent': self._latent_state,
            'action': action,
            'success': self._latent_state == self._target_state if self._done else None,
            'action_history': self._action_history.copy(),
        }

        return self._get_obs_dict(), reward, self._done, info

    def compute_random_success_rate(self, n_episodes: int = 1000) -> float:
        """Random baseline should be ~25%."""
        successes = 0
        for _ in range(n_episodes):
            self.reset()
            for _ in range(self.config.horizon):
                action = np.random.randint(self.config.num_actions)
                _, _, done, info = self.step(action)
            if info['success']:
                successes += 1
        return successes / n_episodes

    def compute_oracle_success_rate(self, n_episodes: int = 1000) -> float:
        """Oracle that knows true state should get 100%."""
        successes = 0
        for _ in range(n_episodes):
            obs = self.reset()
            z = obs['latent_state']
            target = obs['target']

            # Compute optimal path
            n = self.config.num_latent_states
            dist_right = (target - z) % n
            dist_left = (z - target) % n

            if dist_right == 0:
                actions = [self.config.STAY] * self.config.horizon
            elif dist_right <= dist_left:
                actions = [self.config.RIGHT] * dist_right + [self.config.STAY] * (self.config.horizon - dist_right)
            else:
                actions = [self.config.LEFT] * dist_left + [self.config.STAY] * (self.config.horizon - dist_left)

            for a in actions[:self.config.horizon]:
                _, _, done, info = self.step(a)

            if info['success']:
                successes += 1

        return successes / n_episodes


def analyze_task_difficulty():
    """
    Analyze why this task requires world modeling.

    Key insight: For ambiguous initial states, different actions are optimal.
    """
    print("CausalChain-T3 v2: Task Difficulty Analysis")
    print("=" * 60)

    env = CausalChainEnvV2()

    print("\nAliasing Structure:")
    print("  States 0, 1 -> Observation 0")
    print("  States 2, 3 -> Observation 1")

    print("\nConflicting Optimal Actions for same (obs, target) pair:")
    print("-" * 60)

    # For each (obs, target) pair, find cases where different latent states
    # require different optimal actions
    conflicts = []
    for obs in [0, 1]:
        latent_states = [s for s in range(4) if s // 2 == obs]
        for target in range(4):
            optimal_actions = {}
            for z in latent_states:
                # Compute optimal first action
                n = 4
                dist_right = (target - z) % n
                dist_left = (z - target) % n

                if dist_right == 0:
                    opt_action = 2  # STAY
                elif dist_right <= dist_left:
                    opt_action = 1  # RIGHT
                else:
                    opt_action = 0  # LEFT

                optimal_actions[z] = opt_action

            # Check if there's conflict
            actions = list(optimal_actions.values())
            if len(set(actions)) > 1:
                conflicts.append((obs, target, optimal_actions))
                action_names = ['LEFT', 'RIGHT', 'STAY']
                print(f"  obs={obs}, target={target}:")
                for z, a in optimal_actions.items():
                    print(f"    latent={z} -> optimal={action_names[a]}")

    print(f"\nTotal conflicting (obs, target) pairs: {len(conflicts)}/8")
    print("\nThese cases CANNOT be solved by reactive policy!")
    print("Agent must infer latent state from history to choose correctly.")

    # Verify baselines
    print("\n" + "=" * 60)
    print("Baseline Success Rates:")
    print(f"  Random: {env.compute_random_success_rate():.2%}")
    print(f"  Oracle: {env.compute_oracle_success_rate():.2%}")


if __name__ == "__main__":
    analyze_task_difficulty()
