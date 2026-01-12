"""
CausalChain-T3 Environment for Phase 4

A minimal environment that requires causal, multi-step reasoning:
- Partial observability (aliased states)
- Non-commutative transitions
- Delayed reward (only at T=3)

This task CANNOT be solved by:
1. Reactive policies (same obs → different correct actions)
2. Memorization (too many combinations)
3. One-step prediction alone (need multi-step planning)
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass


@dataclass
class CausalChainConfig:
    """Configuration for CausalChain environment."""
    num_latent_states: int = 4      # |Z| = 4
    num_observations: int = 2        # |O| = 2 (aliased)
    num_actions: int = 3             # LEFT, RIGHT, STAY
    horizon: int = 3                 # T = 3 steps

    # Action indices
    LEFT: int = 0
    RIGHT: int = 1
    STAY: int = 2


class CausalChainEnv:
    """
    CausalChain-T3 Environment

    Latent state space: z ∈ {0, 1, 2, 3}
    Observation space: o ∈ {0, 1} where o = z // 2
    Action space: a ∈ {LEFT=0, RIGHT=1, STAY=2}

    Transitions are cyclic:
        LEFT:  z → (z - 1) mod 4
        RIGHT: z → (z + 1) mod 4
        STAY:  z → z

    Key property: Non-commutative (LEFT then RIGHT ≠ RIGHT then LEFT for some paths)
    Actually for this simple cyclic case they commute, but aliasing still breaks reactive policies.

    Episode:
        - z_0 ~ Uniform({0,1,2,3})
        - z_target ~ Uniform({0,1,2,3})
        - Run T=3 steps
        - Reward at final step: r_T = 1 if z_T == z_target else 0
    """

    def __init__(self, config: Optional[CausalChainConfig] = None):
        self.config = config or CausalChainConfig()

        # Build transition table
        self._build_transition_table()

        # Episode state
        self._latent_state: int = 0
        self._target_state: int = 0
        self._timestep: int = 0
        self._done: bool = False

    def _build_transition_table(self):
        """
        Build the transition table: T[z, a] = z'

        Default: Cyclic transitions
            LEFT:  z → (z - 1) mod 4
            RIGHT: z → (z + 1) mod 4
            STAY:  z → z
        """
        n_states = self.config.num_latent_states
        n_actions = self.config.num_actions

        self.transition_table = np.zeros((n_states, n_actions), dtype=np.int32)

        for z in range(n_states):
            self.transition_table[z, self.config.LEFT] = (z - 1) % n_states
            self.transition_table[z, self.config.RIGHT] = (z + 1) % n_states
            self.transition_table[z, self.config.STAY] = z

    def set_transition_table(self, table: np.ndarray):
        """
        Set a custom transition table for intervention tests.

        Args:
            table: Shape (num_latent_states, num_actions), table[z, a] = z'
        """
        assert table.shape == (self.config.num_latent_states, self.config.num_actions)
        self.transition_table = table.astype(np.int32)

    def _get_observation(self, latent_state: int) -> int:
        """
        Map latent state to observation (aliasing function).

        Default: o = z // 2
            States {0, 1} → Observation 0
            States {2, 3} → Observation 1
        """
        return latent_state // 2

    def reset(self,
              start_state: Optional[int] = None,
              target_state: Optional[int] = None) -> Dict:
        """
        Reset the environment.

        Args:
            start_state: Optional fixed start state (for testing)
            target_state: Optional fixed target state (for testing)

        Returns:
            obs_dict: {
                'observation': int,      # Aliased observation
                'target': int,           # Goal state (always visible)
                'timestep': int,         # Current timestep
            }
        """
        # Sample or set initial state
        if start_state is not None:
            self._latent_state = start_state
        else:
            self._latent_state = np.random.randint(self.config.num_latent_states)

        # Sample or set target state
        if target_state is not None:
            self._target_state = target_state
        else:
            self._target_state = np.random.randint(self.config.num_latent_states)

        self._timestep = 0
        self._done = False

        return self._get_obs_dict()

    def _get_obs_dict(self) -> Dict:
        """Get current observation dictionary."""
        return {
            'observation': self._get_observation(self._latent_state),
            'target': self._target_state,
            'timestep': self._timestep,
            'latent_state': self._latent_state,  # For debugging/probing only
        }

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Take a step in the environment.

        Args:
            action: Action index (0=LEFT, 1=RIGHT, 2=STAY)

        Returns:
            obs_dict: Next observation
            reward: 0 for intermediate steps, 1 or 0 at final step
            done: True if episode finished
            info: Additional info (latent state for probing)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset().")

        # Transition
        old_latent = self._latent_state
        self._latent_state = self.transition_table[self._latent_state, action]
        self._timestep += 1

        # Check if done
        self._done = (self._timestep >= self.config.horizon)

        # Reward only at final step
        if self._done:
            reward = 1.0 if self._latent_state == self._target_state else 0.0
        else:
            reward = 0.0

        info = {
            'old_latent': old_latent,
            'new_latent': self._latent_state,
            'action': action,
            'success': self._latent_state == self._target_state if self._done else None,
        }

        return self._get_obs_dict(), reward, self._done, info

    def get_optimal_action_sequence(self, start: int, target: int) -> List[int]:
        """
        Compute optimal action sequence from start to target.

        For cyclic transitions, the optimal path is the shorter direction.

        Args:
            start: Starting latent state
            target: Target latent state

        Returns:
            List of actions to reach target (may be shorter than horizon)
        """
        n = self.config.num_latent_states

        # Distance going right (positive direction)
        dist_right = (target - start) % n
        # Distance going left (negative direction)
        dist_left = (start - target) % n

        if dist_right == 0:
            # Already at target
            return [self.config.STAY] * self.config.horizon
        elif dist_right <= dist_left:
            # Go right
            actions = [self.config.RIGHT] * dist_right
        else:
            # Go left
            actions = [self.config.LEFT] * dist_left

        # Pad with STAY if needed
        while len(actions) < self.config.horizon:
            actions.append(self.config.STAY)

        return actions[:self.config.horizon]

    def compute_optimal_success_rate(self, n_episodes: int = 1000) -> float:
        """Verify optimal policy achieves 100% success."""
        successes = 0
        for _ in range(n_episodes):
            obs = self.reset()
            start = obs['latent_state']
            target = obs['target']

            actions = self.get_optimal_action_sequence(start, target)

            for a in actions:
                obs, reward, done, info = self.step(a)

            if info['success']:
                successes += 1

        return successes / n_episodes

    def compute_random_success_rate(self, n_episodes: int = 1000) -> float:
        """Compute random policy success rate (should be ~25%)."""
        successes = 0
        for _ in range(n_episodes):
            obs = self.reset()

            for _ in range(self.config.horizon):
                action = np.random.randint(self.config.num_actions)
                obs, reward, done, info = self.step(action)

            if info['success']:
                successes += 1

        return successes / n_episodes


class CausalChainEnvNonCommutative(CausalChainEnv):
    """
    Variant with truly non-commutative transitions.

    This version has transitions where the order of actions matters
    even for the same start/end pair.

    Transition table (designed to be non-commutative):
        z=0: LEFT→3, RIGHT→1, STAY→0
        z=1: LEFT→0, RIGHT→3, STAY→1   # Note: RIGHT goes to 3, not 2
        z=2: LEFT→1, RIGHT→3, STAY→2
        z=3: LEFT→0, RIGHT→2, STAY→3   # Note: LEFT goes to 0, not 2

    This breaks commutativity:
        From z=0: RIGHT then LEFT = 1 → 0
        From z=0: LEFT then RIGHT = 3 → 2
    """

    def _build_transition_table(self):
        """Build non-commutative transition table."""
        self.transition_table = np.array([
            [3, 1, 0],  # z=0: LEFT→3, RIGHT→1, STAY→0
            [0, 3, 1],  # z=1: LEFT→0, RIGHT→3, STAY→1
            [1, 3, 2],  # z=2: LEFT→1, RIGHT→3, STAY→2
            [0, 2, 3],  # z=3: LEFT→0, RIGHT→2, STAY→3
        ], dtype=np.int32)


def verify_non_commutativity(env: CausalChainEnv) -> bool:
    """
    Check if the environment has non-commutative transitions.

    Returns True if there exists some state z where:
        T[T[z, a1], a2] != T[T[z, a2], a1]
    """
    T = env.transition_table
    n_states = env.config.num_latent_states
    n_actions = env.config.num_actions

    for z in range(n_states):
        for a1 in range(n_actions):
            for a2 in range(n_actions):
                if a1 != a2:
                    # z → a1 → a2
                    path1 = T[T[z, a1], a2]
                    # z → a2 → a1
                    path2 = T[T[z, a2], a1]

                    if path1 != path2:
                        return True

    return False


if __name__ == "__main__":
    # Quick test
    print("Testing CausalChain-T3 Environment")
    print("=" * 50)

    # Standard environment
    env = CausalChainEnv()
    print(f"\nStandard (cyclic) environment:")
    print(f"  Transition table:\n{env.transition_table}")
    print(f"  Non-commutative: {verify_non_commutativity(env)}")
    print(f"  Random success rate: {env.compute_random_success_rate():.2%}")
    print(f"  Optimal success rate: {env.compute_optimal_success_rate():.2%}")

    # Non-commutative environment
    env_nc = CausalChainEnvNonCommutative()
    print(f"\nNon-commutative environment:")
    print(f"  Transition table:\n{env_nc.transition_table}")
    print(f"  Non-commutative: {verify_non_commutativity(env_nc)}")
    print(f"  Random success rate: {env_nc.compute_random_success_rate():.2%}")

    # Example episode
    print(f"\nExample episode (standard env):")
    obs = env.reset()
    print(f"  Start: z={obs['latent_state']}, o={obs['observation']}, target={obs['target']}")

    optimal_actions = env.get_optimal_action_sequence(obs['latent_state'], obs['target'])
    action_names = ['LEFT', 'RIGHT', 'STAY']
    print(f"  Optimal actions: {[action_names[a] for a in optimal_actions]}")

    for i, a in enumerate(optimal_actions):
        obs, reward, done, info = env.step(a)
        print(f"  Step {i+1}: action={action_names[a]}, z={obs['latent_state']}, o={obs['observation']}, r={reward}")

    print(f"  Success: {info['success']}")
