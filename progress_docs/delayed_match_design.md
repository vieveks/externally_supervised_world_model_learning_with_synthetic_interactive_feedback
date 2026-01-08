# DelayedMatch Task Design

## Why Current Tasks Don't Require World-Model

| Task | Problem |
|------|---------|
| PatternEcho | State directly encodes answer |
| SequenceNext | Can memorize as lookup table |

Both are **Markov lookup problems**: `π(a|s) = δ(a = f(s))`

## Requirements for World-Model to Matter

A task must have **at least two** of:
1. Multi-step episodes (T ≥ 3)
2. Delayed reward (reward only at end)
3. Partial observability (current state insufficient)
4. Action-dependent transitions (wrong early action ruins future)

## DelayedMatch Task Design

### Core Idea
The baby must **remember a target** shown at the start, then **navigate** through intermediate states to reach it.

### Episode Structure (T=3 steps)

```
Step 0: State shows TARGET (e.g., "reach position 5")
        Baby outputs action → transitions to intermediate state
        Reward: 0 (always)

Step 1: State shows CURRENT POSITION (e.g., "you are at 2")
        Baby outputs action → transitions toward/away from target
        Reward: 0 (always)

Step 2: State shows FINAL POSITION
        Reward: 1.0 if final_position == target, else 0.0
```

### Why This Requires World-Model

1. **Delayed reward**: No signal until step 2
2. **Memory required**: Target shown only at step 0, must remember
3. **Planning required**: Must predict how actions affect position
4. **Credit assignment**: Must figure out which action mattered

### State Encoding

Two types of states:
- **Target state**: One-hot encoding of target position + flag
- **Position state**: One-hot encoding of current position + flag

State vector: `[position_one_hot (8), is_target_phase (1)]` = 9 dims

### Transition Dynamics

Simple grid world in 1D:
- Action 0-3: Move left (amount = action)
- Action 4-7: Move right (amount = action - 4)
- Position wraps around: `new_pos = (pos + delta) % num_positions`

### Why World-Model Should Help

1. **Prediction enables credit assignment**
   - WM learns: "action X at position Y leads to position Z"
   - Policy can use this to plan

2. **Sparse reward problem**
   - Without WM: Only 1 reward signal per 3-step episode
   - With WM: Dense prediction signal at every step

3. **Multi-step reasoning**
   - Must chain: target → action1 → pos1 → action2 → final_pos
   - Reactive policy cannot do this

### Expected Results

| Condition | Expected Performance |
|-----------|---------------------|
| With WM | Should learn (~70%+) |
| Without WM | Should struggle (~random or slightly above) |

### Simplified Version First

Start with T=2 (even simpler):
```
Step 0: See target, output action
Step 1: See result, get reward
```

This tests if delayed reward alone creates WM advantage.

### Configuration

```yaml
task: delayed_match
num_actions: 8
num_positions: 8
episode_length: 3
state_dim: 9  # 8 position + 1 phase flag
```

---

## Ablation Results (T=2 simplified version)

### Experimental Setup
- 10,000 training steps
- 64 episodes per batch
- Episode length: 2 steps (simplified)

### Results

| Condition | Final Success | Best Success | Final Entropy | WM Loss |
|-----------|---------------|--------------|---------------|---------|
| With WM (coef=1.0) | 98.44% | 100% | 0.37 | 0.0055 |
| Without WM (coef=0.0) | 96.88% | 100% | 0.30 | 0.26 |

### Analysis

**Unexpected Finding**: Both conditions achieved ~97-98% success!

This suggests the T=2 DelayedMatch task is **still too simple**:
- The policy learns `action = target` directly (a lookup)
- Delayed reward by 1 step isn't enough to require planning
- The task reduces to: "see target → output target"

**Key Differences**:
1. **WM Loss**: With WM dropped to 0.0055, without stayed ~0.26
   - World-model learns transitions perfectly when trained
   - Without WM signal, predictions remain random
2. **Entropy**: Similar convergence (~0.3-0.4)
   - Both found deterministic policies

### Why T=2 Doesn't Require World-Model

```
T=2 Episode:
Step 0: See target=5, output action=5 → position=5
Step 1: Get reward=1.0 (position matches target)
```

The policy just needs to learn: `action = target`
- No intermediate state to navigate through
- No planning required
- Direct stimulus-response suffices

### Next Steps: T=3 or More Complex Task

To truly require world-model, need:
1. **T=3+ steps**: Multiple actions before reward
2. **State aliasing**: Same state, different correct actions
3. **Non-trivial dynamics**: Navigation through intermediate states

Options:
- Extend DelayedMatch to T=3 with actual navigation
- Create a task with state aliasing (requires memory)
- Add stochasticity to transitions

---

## NavigationTask Results (T=3 with relative movement)

### Task Design

NavigationTask introduces **relative movement dynamics**:
- State: [position_one_hot(8), target_one_hot(8), phase_one_hot(3)] = 19 dims
- Actions: Relative moves (action 0 = -3, action 3 = stay, action 7 = +4)
- Episode: 2 actions before reward at step 2
- Challenge: Must plan 2-step path from random start to target

**Why this should require world-model:**
1. Cannot solve with simple lookup (action ≠ target)
2. Must predict: action → new_position
3. Must plan: start → action1 → intermediate → action2 → target

### Baselines
- Random: ~12.5% (1/8 positions)
- Optimal: 100% (greedy 2-step planning)

### Ablation Results (20,000 steps)

| Condition | Final Success | Best Success | Final Entropy | WM Loss |
|-----------|---------------|--------------|---------------|---------|
| **Without WM** (coef=0.0) | **50.00%** | **59.38%** | 0.81 | 0.24 |
| With WM (coef=1.0) | 23.44% | 25.00% | 1.58 | 0.05 |

### Analysis

**UNEXPECTED: No-WM outperforms With-WM!**

This is the opposite of what theory predicted. Why?

1. **World-model loss dominates gradient**
   - WM coefficient = 1.0 may be too high
   - Policy gradient gets drowned out by WM loss
   - Model optimizes for prediction rather than task success

2. **WM loss decreases but doesn't help**
   - With WM: loss 0.26 → 0.05 (good learning!)
   - But success only 23% (barely above random)
   - WM is learning dynamics but not using them for planning

3. **No-WM learns faster**
   - Without WM distraction, policy can focus on reward
   - Entropy decreases faster (1.9 → 0.8)
   - Policy finds partial solution through trial-and-error

4. **Entropy tells the story**
   - With WM: entropy stays high (1.58) - policy uncertain
   - Without WM: entropy drops (0.81) - policy more decisive

### Hypothesis: WM Coefficient Too High

The world-model loss may be interfering with policy learning rather than helping it. The WM learns accurate dynamics but the model doesn't use this knowledge for planning.

### Next Steps

1. **Lower WM coefficient**: Try 0.1 or 0.01 instead of 1.0
2. **Separate WM and policy updates**: Different learning rates
3. **Curriculum**: Start without WM, add later
4. **Architecture**: Separate WM and policy heads more

### Key Insight

Learning accurate world models is not sufficient for planning.
The model must also learn to USE the world model for decision-making.
This is a fundamental challenge in model-based RL.

---

## Frozen WM Experiment (Two-Phase Training)

### Motivation

The guide pointed out a critical distinction:
- **Auxiliary WM**: WM trained as loss but never queried → creates gradient competition
- **Planning WM**: WM used for imagination/rollouts → enables model-based reasoning

To isolate gradient interference, we test:
1. Phase 1: Train WM only (5000 steps)
2. Phase 2: Freeze WM, train policy only (20000 steps)

### Three-Way Comparison (20,000 policy steps)

| Condition | Final Success | Best Success | WM Loss | Entropy |
|-----------|---------------|--------------|---------|---------|
| **No WM** (coef=0.0) | **50.00%** | **59.38%** | 0.24 | 0.81 |
| Frozen WM (2-phase) | 43.75% | 43.75% | 0.15 | 0.92 |
| Joint WM (coef=1.0) | 23.44% | 25.00% | 0.05 | 1.58 |

### Analysis

1. **No WM still wins**: Removing WM loss entirely gives best performance
2. **Frozen WM helps vs Joint**: 43% vs 23% - removing gradient competition helps
3. **But doesn't beat No-WM**: The WM learned during warmup doesn't help policy

### What This Tells Us

The frozen WM experiment shows:

1. **Gradient interference is real**: Frozen WM (43%) >> Joint WM (23%)
2. **But WM alone isn't enough**: Frozen WM (43%) < No WM (50%)

The WM learns dynamics but the representations it creates **don't help** the policy
because the policy never **uses** the WM for planning.

### The Fundamental Lesson

> A world model trained as an auxiliary loss is invisible to the policy
> unless explicitly queried.

Current architecture:
```
shared transformer
├── policy head (trained by reward)
├── value head
└── world-model head (trained by prediction loss)
```

This creates **representation multitasking**, not **planning**.

### What Would Actually Help

To make WM useful for planning:
1. **Rollout-based planning**: Use WM to simulate future states
2. **Value expansion**: Use WM to compute multi-step value targets
3. **Latent imagination**: Train policy inside WM-generated trajectories (Dreamer-style)

### Paper-Worthy Statement

> We find that accurate world models learned from interaction can degrade
> performance when used purely as auxiliary losses, highlighting the necessity
> of explicit planning mechanisms. Removing gradient competition via two-phase
> training partially recovers performance, but still underperforms pure policy
> learning—suggesting that world models must be actively queried during decision-
> making to provide benefit.
