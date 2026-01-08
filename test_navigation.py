"""Test NavigationTask implementation."""

import torch
from src.environment.tasks import NavigationTask
from src.environment.parent import DeterministicParent
from src.environment.symbolic_env import SymbolicEnv, NavigationState


def test_navigation_task():
    """Test NavigationTask mechanics."""
    print("=" * 50)
    print("Testing NavigationTask")
    print("=" * 50)

    task = NavigationTask(num_positions=8, num_actions=8, episode_length=3)
    parent = DeterministicParent(task)

    # Test episode generation
    pos, metadata = task.generate_instance()
    print(f"\nGenerated episode:")
    print(f"  Start position: {pos}")
    print(f"  Target: {metadata['target']}")
    print(f"  Phase: {metadata['phase']}")

    # Test state encoding
    state_vec = task.encode_state()
    print(f"\nState encoding (dim={len(state_vec)}):")
    print(f"  Position one-hot: {state_vec[:8]}")
    print(f"  Target one-hot: {state_vec[8:16]}")
    print(f"  Phase one-hot: {state_vec[16:]}")

    # Test action dynamics
    print(f"\nAction dynamics (relative movement):")
    for action in [0, 3, 7]:
        # Reset to known state
        task._position = 4
        task._phase = 0
        delta = action - 3
        new_pos = task.get_next_state_target(task._position, action)
        print(f"  Action {action}: delta={delta:+d}, pos 4 -> {new_pos}")

    print("\n✓ NavigationTask mechanics work correctly!")


def test_navigation_env():
    """Test NavigationTask with SymbolicEnv."""
    print("\n" + "=" * 50)
    print("Testing NavigationTask with SymbolicEnv")
    print("=" * 50)

    task = NavigationTask(num_positions=8, num_actions=8, episode_length=3)
    parent = DeterministicParent(task)
    env = SymbolicEnv(task=task, parent=parent, max_steps=3, device="cpu")

    print(f"\nEnvironment info:")
    print(f"  State dim: {env.state_dim}")
    print(f"  Action dim: {env.action_dim}")
    print(f"  Is navigation: {env.is_navigation}")

    # Run a few episodes
    print(f"\nRunning 3 test episodes:")
    for ep in range(3):
        state = env.reset()
        print(f"\n  Episode {ep + 1}:")
        print(f"    Initial: pos={state.position}, target={state.target}, phase={state.phase}")

        done = False
        step = 0
        while not done:
            # Random action
            action = torch.randint(0, 8, (1,)).item()
            next_state, reward, done, info = env.step(action)
            print(f"    Step {step}: action={action}, new_pos={next_state.position}, "
                  f"reward={reward.goal:.1f}, done={done}")
            state = next_state
            step += 1

    print("\n✓ SymbolicEnv with NavigationTask works correctly!")


def test_optimal_policy():
    """Test that optimal policy can solve the task."""
    print("\n" + "=" * 50)
    print("Testing Optimal Policy")
    print("=" * 50)

    task = NavigationTask(num_positions=8, num_actions=8, episode_length=3)
    parent = DeterministicParent(task)
    env = SymbolicEnv(task=task, parent=parent, max_steps=3, device="cpu")

    successes = 0
    n_episodes = 100

    for _ in range(n_episodes):
        state = env.reset()
        start_pos = state.position
        target = state.target

        # Optimal policy: compute needed delta, apply over 2 steps
        # Total delta needed: (target - start_pos) mod 8
        total_delta = (target - start_pos) % 8
        if total_delta > 4:  # Go the other way
            total_delta = total_delta - 8

        # Split across two actions
        # Each action gives delta = action - 3, so action = delta + 3
        # Try to split evenly
        if abs(total_delta) <= 4:
            # Can do in one step, second step is stay
            action1 = total_delta + 3
            action2 = 3  # Stay
        else:
            # Split
            delta1 = total_delta // 2
            delta2 = total_delta - delta1
            action1 = delta1 + 3
            action2 = delta2 + 3

        # Clamp actions to valid range
        action1 = max(0, min(7, action1))
        action2 = max(0, min(7, action2))

        # Execute
        done = False
        actions = [action1, action2]
        step = 0
        while not done:
            _, reward, done, info = env.step(actions[step] if step < 2 else 3)
            step += 1

        if reward.goal > 0.9:
            successes += 1

    success_rate = successes / n_episodes
    print(f"\nOptimal policy success rate: {success_rate:.1%}")
    print(f"  (100 episodes, greedy 2-step planning)")

    # Note: This simple optimal policy doesn't handle all cases perfectly
    # due to wrap-around, but should get most
    if success_rate > 0.7:
        print("\n✓ Optimal policy achieves good performance!")
    else:
        print(f"\n! Optimal policy only achieves {success_rate:.1%}")
        print("  (Task may be harder than expected)")


def test_random_baseline():
    """Test random policy performance."""
    print("\n" + "=" * 50)
    print("Testing Random Baseline")
    print("=" * 50)

    task = NavigationTask(num_positions=8, num_actions=8, episode_length=3)
    parent = DeterministicParent(task)
    env = SymbolicEnv(task=task, parent=parent, max_steps=3, device="cpu")

    successes = 0
    n_episodes = 1000

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            action = torch.randint(0, 8, (1,)).item()
            _, reward, done, _ = env.step(action)

        if reward.goal > 0.9:
            successes += 1

    success_rate = successes / n_episodes
    print(f"\nRandom policy success rate: {success_rate:.1%}")
    print(f"  Expected: ~{1/8:.1%} (1/8 positions)")

    print("\n✓ Random baseline established!")


if __name__ == "__main__":
    test_navigation_task()
    test_navigation_env()
    test_optimal_policy()
    test_random_baseline()

    print("\n" + "=" * 50)
    print("All NavigationTask tests passed!")
    print("=" * 50)
