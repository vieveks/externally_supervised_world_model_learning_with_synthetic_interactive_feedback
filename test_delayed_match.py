#!/usr/bin/env python3
"""Test script for DelayedMatch task."""

from src.environment.tasks import DelayedMatchTask
from src.environment.parent import DeterministicParent
from src.environment.symbolic_env import SymbolicEnv, State

# Test the task directly
print("=== Testing DelayedMatchTask ===")
task = DelayedMatchTask(num_actions=8)
print(f"Task: {task.name}")

# Generate instance
target, meta = task.generate_instance()
print(f"Target: {target}")
print(f"Metadata: {meta}")
print(f"Current phase: {task.current_phase}")

# Test reward at phase 0 (should be 0)
r0 = task.compute_reward(target, target)
print(f"Reward at phase 0 (correct action): {r0} (should be 0)")

# Test transition
next_val = task.get_next_state_target(target, target)  # Action = target
print(f"Next state value after action={target}: {next_val}")
print(f"Phase after transition: {task.current_phase}")

# Test reward at phase 1
r1_correct = task.compute_reward(0, next_val)  # Action doesn't matter at phase 1
print(f"Reward at phase 1 (position={next_val}, target={target}): {r1_correct}")

# Test full environment
print("\n=== Testing SymbolicEnv with DelayedMatch ===")
task2 = DelayedMatchTask(num_actions=8)
parent = DeterministicParent(task2)
env = SymbolicEnv(task=task2, parent=parent, max_steps=2)

# Reset
state = env.reset()
print(f"Initial state - target: {state.target}, phase: {state.phase}")
print(f"State tensor shape: {state.to_tensor().shape}")
print(f"State tensor: {state.to_tensor()}")

# Step 0: Take action matching target
action = state.target  # Optimal action
next_state, reward, done, info = env.step(action)
print(f"\nAfter step 0 (action={action}):")
print(f"  Next state - target: {next_state.target}, phase: {next_state.phase}")
print(f"  Reward: {reward.goal} (should be 0)")
print(f"  Done: {done}")
print(f"  Info: {info}")

# Step 1: Episode ends
next_state2, reward2, done2, info2 = env.step(0)  # Action doesn't matter
print(f"\nAfter step 1:")
print(f"  Reward: {reward2.goal} (should be 1.0 if we matched)")
print(f"  Done: {done2}")
print(f"  Success: {info2.get('success')}")

# Test wrong action
print("\n=== Testing wrong action ===")
task3 = DelayedMatchTask(num_actions=8)
parent3 = DeterministicParent(task3)
env3 = SymbolicEnv(task=task3, parent=parent3, max_steps=2)

state = env3.reset()
wrong_action = (state.target + 1) % 8  # Wrong action
print(f"Target: {state.target}, Wrong action: {wrong_action}")

next_state, reward, done, info = env3.step(wrong_action)
print(f"After wrong action - position: {next_state.target}")

next_state2, reward2, done2, info2 = env3.step(0)
print(f"Final reward: {reward2.goal} (should be 0.0)")

print("\n=== All tests passed! ===")
