# Phase 4: World Learning - Implementation Updates

## Overview

This document tracks the implementation progress for Phase 4 experiments.

**Goal**: Test whether prediction-driven RL can acquire causal, multi-step world knowledge under delayed reward.

---

## Update Log

### 2026-01-12: Environment Implementation Started

**What we're building**: The CausalChain-T3 environment

**Why this task is decisive**:
1. **Partial observability**: Agent sees aliased observations, not true latent state
2. **Non-commutative transitions**: Order of actions matters (LEFT→RIGHT ≠ RIGHT→LEFT)
3. **Delayed reward**: Only at T=3, no intermediate feedback
4. **Cannot be solved reactively**: Same observation requires different actions depending on hidden state

**Environment Design**:
```
Latent states: z ∈ {0, 1, 2, 3}
Observations:  o ∈ {0, 1} where o = z // 2 (states 0,1 → obs 0; states 2,3 → obs 1)
Actions:       a ∈ {0=LEFT, 1=RIGHT, 2=STAY}

Transition table (cyclic with direction):
  z=0: LEFT→3, RIGHT→1, STAY→0
  z=1: LEFT→0, RIGHT→2, STAY→1
  z=2: LEFT→1, RIGHT→3, STAY→2
  z=3: LEFT→2, RIGHT→0, STAY→3

Episode:
  - Sample z_0 uniformly from {0,1,2,3}
  - Sample z_target uniformly from {0,1,2,3}
  - Run for T=3 steps
  - Reward = 1 if z_T == z_target, else 0
```

**Random baseline**: 25% (1/4 chance of landing on target)
**Optimal policy**: 100% (any state reachable in ≤2 steps)

---

## Files Created

| File | Purpose | Status |
|------|---------|--------|
| `src/environment/causal_chain.py` | CausalChain-T3 environment | Complete |
| `test_causal_chain.py` | Environment validation tests | Complete (9/9 pass) |
| `src/agents/causal_agent.py` | Agent with prediction head | Complete |
| `run_phase4_experiments.py` | Main experiment runner | Complete |

---

### 2026-01-12: Agent Implementation Complete

**CausalAgent Architecture**:
```
Input: [obs_onehot, prev_action_onehot, target_onehot, timestep_onehot]
       Total input dim: 2 + 4 + 4 + 3 = 13

Encoder: 2-layer MLP (13 → 64 → 64)

Heads:
  - Policy: Linear(64 → 3) → Categorical over actions
  - Value: Linear(64 → 1) → State value estimate
  - Prediction: MLP(64 + 3 → 64 → 2) → Next observation prediction

Total parameters: ~9,800
```

**Training Loop Implemented for Three Conditions**:

| Condition | Architecture | Reward | Expected Result |
|-----------|--------------|--------|-----------------|
| A | Policy + Value | Sparse task | FAILS (no state disambiguation) |
| B | Full + Pred as Action | Prediction accuracy | Learns dynamics, fails task |
| C | Full | Sparse task + aux pred | THE QUESTION |

---

### 2026-01-12: First Experiment Results - SURPRISING FINDING

**Condition A Results (Reactive RL - 10,000 episodes)**:

```
Episode   500 | Success: 24.80%   (random baseline)
Episode  5000 | Success: 44.60%   (learning starts)
Episode  7500 | Success: 84.20%   (rapid improvement)
Episode 10000 | Success: 94.80%

Final Success Rate: 100.00%
Random Baseline: 25.00%
Relative Improvement: 300.0%
```

**CRITICAL FINDING**: The "Reactive RL" baseline (Condition A) achieved **100% success rate**!

This means the CausalChain-T3 task, as designed, CAN be solved without a world model.

**Why This Happened**:

The agent receives: `[observation, prev_action, target, timestep]`

With all this information, the task becomes a supervised mapping problem:
- The agent learns: (obs, prev_action, target, timestep) → optimal_action
- With 2 × 4 × 4 × 3 = 96 possible input states
- Each maps to a deterministic correct action
- The neural network can memorize this mapping!

**Problem Identified**:

The task does NOT actually require multi-step planning or world modeling because:
1. Previous action + current observation provides enough info to infer latent state
2. Given inferred latent state + target + timestep, optimal action is deterministic
3. Network can learn this mapping without understanding WHY it works

**Implication**:

We need to redesign the task to:
1. Remove `prev_action` from the observation (true partial observability)
2. OR use an environment where history doesn't disambiguate state
3. OR require actual rollout/planning to solve

---

### 2026-01-12: Phase 4 v2 Results - ANOTHER SURPRISING FINDING!

**Redesigned Task (v2)**:
- Agent does NOT see `prev_action`
- Input: `[observation, target, timestep]` only
- Theoretical reactive ceiling: **50%** (computed analytically)

**Results (15,000 episodes)**:

| Condition | Architecture | Final Success | Expected |
|-----------|-------------|---------------|----------|
| A | Feedforward | **100%** | ~50% |
| B | LSTM | 78.5% | >90% |
| C | FF + Prediction | **100%** | ~50% |

**CRITICAL PARADOX**:
1. The feedforward (reactive) agent achieved **100%** despite theoretical ceiling of 50%
2. The LSTM (with memory) achieved only **78.5%** - WORSE than feedforward!
3. Prediction head made no difference (A = C)

**Why Feedforward Beat the Ceiling**:

The theoretical ceiling assumes:
- Agent commits to ONE fixed action sequence for each (obs, target)
- With 50/50 probability over aliased states, best fixed sequence succeeds 50%

But the feedforward agent can learn:
- Different actions for different **timesteps** given same (obs, target)
- This creates an implicit strategy that adapts based on observed transitions

**Example**: For obs=0, target=2 at t=0:
- If z=0: optimal = RIGHT, RIGHT, STAY
- If z=1: optimal = RIGHT, STAY, STAY

Agent might learn: "Always go RIGHT at t=0, then see what obs I get at t=1"
- If obs at t=1 is 1 (states 2,3): already at 2 or 3, can adjust
- If obs at t=1 is 0 (states 0,1): something else, adjust

**The agent implicitly learns a belief-update strategy by exploiting the timestep dimension!**

**Why LSTM Failed**:
- Credit assignment through time is hard
- LSTM has many more parameters (23,942 vs 9,542)
- May need different hyperparameters or architecture

**New Interpretation**:

The task CAN be solved without explicit world modeling because:
1. The environment is highly structured (only 4 states, 2 observations)
2. The horizon is short (T=3)
3. The agent can learn implicit state inference from observation changes
4. The timestep input allows different policies at each step

**This is NOT what we wanted to test!**

The task is still too simple - the feedforward network can effectively memorize
a decision tree that handles all cases.

**Next Steps**:

Need to redesign the task further:
1. **Remove timestep from input** - force pure reactive behavior
2. **OR increase state space** - make memorization harder
3. **OR use longer horizons** - make planning more necessary
4. **OR use partial observability with no information gain** - make belief tracking essential

---
