"""
Test suite for CausalChain-T3 Environment

Validates:
1. Transition dynamics work correctly
2. Observation aliasing is correct
3. Random baseline is ~25%
4. Optimal policy achieves 100%
5. Non-commutative variant breaks action order
"""

import numpy as np
from src.environment.causal_chain import (
    CausalChainEnv,
    CausalChainEnvNonCommutative,
    CausalChainConfig,
    verify_non_commutativity,
)


def test_transition_dynamics():
    """Test that transitions follow the cyclic pattern."""
    env = CausalChainEnv()
    config = env.config

    print("Test 1: Transition Dynamics")
    print("-" * 40)

    # Test each transition
    expected = {
        (0, config.LEFT): 3,
        (0, config.RIGHT): 1,
        (0, config.STAY): 0,
        (1, config.LEFT): 0,
        (1, config.RIGHT): 2,
        (1, config.STAY): 1,
        (2, config.LEFT): 1,
        (2, config.RIGHT): 3,
        (2, config.STAY): 2,
        (3, config.LEFT): 2,
        (3, config.RIGHT): 0,
        (3, config.STAY): 3,
    }

    all_passed = True
    for (z, a), expected_z_next in expected.items():
        actual = env.transition_table[z, a]
        passed = actual == expected_z_next
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"  T[{z}, {['L','R','S'][a]}] = {actual}, expected {expected_z_next}: {status}")

    print(f"  Overall: {'PASS' if all_passed else 'FAIL'}\n")
    return all_passed


def test_observation_aliasing():
    """Test that observation aliasing works correctly."""
    env = CausalChainEnv()

    print("Test 2: Observation Aliasing")
    print("-" * 40)

    # States 0,1 should map to observation 0
    # States 2,3 should map to observation 1
    expected = {0: 0, 1: 0, 2: 1, 3: 1}

    all_passed = True
    for z, expected_o in expected.items():
        obs = env.reset(start_state=z, target_state=0)
        actual_o = obs['observation']
        passed = actual_o == expected_o
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"  z={z} -> o={actual_o}, expected {expected_o}: {status}")

    print(f"  Overall: {'PASS' if all_passed else 'FAIL'}\n")
    return all_passed


def test_random_baseline():
    """Test that random policy achieves ~25% success."""
    env = CausalChainEnv()

    print("Test 3: Random Baseline (~25%)")
    print("-" * 40)

    rate = env.compute_random_success_rate(n_episodes=5000)
    expected = 0.25
    tolerance = 0.03  # Allow 3% deviation

    passed = abs(rate - expected) < tolerance
    status = "PASS" if passed else "FAIL"
    print(f"  Random success rate: {rate:.2%}")
    print(f"  Expected: {expected:.2%} +/- {tolerance:.2%}")
    print(f"  Overall: {status}\n")
    return passed


def test_optimal_policy():
    """Test that optimal policy achieves 100% success."""
    env = CausalChainEnv()

    print("Test 4: Optimal Policy (100%)")
    print("-" * 40)

    rate = env.compute_optimal_success_rate(n_episodes=1000)
    passed = rate == 1.0
    status = "PASS" if passed else "FAIL"
    print(f"  Optimal success rate: {rate:.2%}")
    print(f"  Overall: {status}\n")
    return passed


def test_optimal_action_sequences():
    """Test that optimal action sequences are correct."""
    env = CausalChainEnv()
    config = env.config

    print("Test 5: Optimal Action Sequences")
    print("-" * 40)

    # Test specific cases
    test_cases = [
        (0, 0, [config.STAY, config.STAY, config.STAY]),  # Already there
        (0, 1, [config.RIGHT, config.STAY, config.STAY]),  # 1 step right
        (0, 2, [config.RIGHT, config.RIGHT, config.STAY]),  # 2 steps right
        (0, 3, [config.LEFT, config.STAY, config.STAY]),  # 1 step left (shorter than 3 right)
        (1, 3, [config.RIGHT, config.RIGHT, config.STAY]),  # 2 steps right
        (3, 1, [config.RIGHT, config.RIGHT, config.STAY]),  # 2 steps (either direction works, algorithm picks right)
    ]

    all_passed = True
    for start, target, expected_actions in test_cases:
        actual = env.get_optimal_action_sequence(start, target)
        passed = actual == expected_actions
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"

        action_names = ['L', 'R', 'S']
        actual_str = [action_names[a] for a in actual]
        expected_str = [action_names[a] for a in expected_actions]
        print(f"  ({start}->{target}): {actual_str}, expected {expected_str}: {status}")

    print(f"  Overall: {'PASS' if all_passed else 'FAIL'}\n")
    return all_passed


def test_non_commutativity():
    """Test that non-commutative variant has non-commutative transitions."""
    env_standard = CausalChainEnv()
    env_nc = CausalChainEnvNonCommutative()

    print("Test 6: Non-Commutativity")
    print("-" * 40)

    standard_nc = verify_non_commutativity(env_standard)
    nc_nc = verify_non_commutativity(env_nc)

    print(f"  Standard env is non-commutative: {standard_nc} (expected: False)")
    print(f"  NC env is non-commutative: {nc_nc} (expected: True)")

    passed = (not standard_nc) and nc_nc
    print(f"  Overall: {'PASS' if passed else 'FAIL'}\n")
    return passed


def test_episode_structure():
    """Test that episodes have correct structure."""
    env = CausalChainEnv()
    config = env.config

    print("Test 7: Episode Structure")
    print("-" * 40)

    obs = env.reset()

    # Check initial observation dict
    checks = [
        ('observation' in obs, "has 'observation' key"),
        ('target' in obs, "has 'target' key"),
        ('timestep' in obs, "has 'timestep' key"),
        (obs['timestep'] == 0, "timestep starts at 0"),
        (0 <= obs['observation'] < config.num_observations, "observation in range"),
        (0 <= obs['target'] < config.num_latent_states, "target in range"),
    ]

    all_passed = True
    for passed, desc in checks:
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"  {desc}: {status}")

    # Test episode length
    for t in range(config.horizon):
        obs, reward, done, info = env.step(config.STAY)

        expected_done = (t == config.horizon - 1)
        passed = done == expected_done
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"  Step {t+1} done={done}, expected={expected_done}: {status}")

    print(f"  Overall: {'PASS' if all_passed else 'FAIL'}\n")
    return all_passed


def test_delayed_reward():
    """Test that reward is only given at final step."""
    env = CausalChainEnv()
    config = env.config

    print("Test 8: Delayed Reward")
    print("-" * 40)

    # Run multiple episodes and check reward timing
    all_passed = True
    for _ in range(10):
        obs = env.reset()
        rewards = []

        for t in range(config.horizon):
            action = np.random.randint(config.num_actions)
            obs, reward, done, info = env.step(action)
            rewards.append(reward)

        # First T-1 rewards should be 0
        intermediate_zero = all(r == 0 for r in rewards[:-1])
        # Final reward should be 0 or 1
        final_valid = rewards[-1] in [0.0, 1.0]

        passed = intermediate_zero and final_valid
        all_passed = all_passed and passed

    status = "PASS" if all_passed else "FAIL"
    print(f"  Intermediate rewards are 0: {intermediate_zero}")
    print(f"  Final reward is 0 or 1: {final_valid}")
    print(f"  Overall: {status}\n")
    return all_passed


def test_intervention():
    """Test that we can swap transition tables for intervention tests."""
    env = CausalChainEnv()

    print("Test 9: Transition Table Intervention")
    print("-" * 40)

    # Original table
    original_table = env.transition_table.copy()

    # New table (reverse directions)
    new_table = np.array([
        [1, 3, 0],  # Swapped LEFT and RIGHT
        [2, 0, 1],
        [3, 1, 2],
        [0, 2, 3],
    ], dtype=np.int32)

    env.set_transition_table(new_table)

    # Verify change
    passed = np.array_equal(env.transition_table, new_table)
    status = "PASS" if passed else "FAIL"
    print(f"  Table swap successful: {status}")

    # Verify we can reset back
    env.set_transition_table(original_table)
    passed2 = np.array_equal(env.transition_table, original_table)
    status2 = "PASS" if passed2 else "FAIL"
    print(f"  Table restore successful: {status2}")

    overall = passed and passed2
    print(f"  Overall: {'PASS' if overall else 'FAIL'}\n")
    return overall


def run_all_tests():
    """Run all tests and report summary."""
    print("=" * 60)
    print("CausalChain-T3 Environment Test Suite")
    print("=" * 60)
    print()

    tests = [
        ("Transition Dynamics", test_transition_dynamics),
        ("Observation Aliasing", test_observation_aliasing),
        ("Random Baseline", test_random_baseline),
        ("Optimal Policy", test_optimal_policy),
        ("Optimal Action Sequences", test_optimal_action_sequences),
        ("Non-Commutativity", test_non_commutativity),
        ("Episode Structure", test_episode_structure),
        ("Delayed Reward", test_delayed_reward),
        ("Transition Intervention", test_intervention),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"  ERROR: {e}\n")
            results.append((name, False))

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed_count = sum(1 for _, p in results if p)
    total = len(results)

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    print()
    print(f"Total: {passed_count}/{total} tests passed")

    if passed_count == total:
        print("\nAll tests passed! Environment is ready for Phase 4 experiments.")
    else:
        print("\nSome tests failed. Please fix before proceeding.")

    return passed_count == total


if __name__ == "__main__":
    run_all_tests()
