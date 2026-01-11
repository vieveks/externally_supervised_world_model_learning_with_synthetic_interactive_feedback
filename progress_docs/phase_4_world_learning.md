# Phase 4: World Learning Through Interaction

## Official Definition

**One-Sentence Goal:**
> Determine whether prediction-driven RL can acquire causal, multi-step world knowledge under delayed reward, or whether prediction alone is insufficient without explicit planning mechanisms.

**Start Date:** January 2026
**Status:** NOT STARTED

---

## What Phase 4 Is NOT

The coding agent should **NOT**:

- ❌ Add likelihood / cross-entropy losses
- ❌ Add teacher forcing
- ❌ Add demonstrations
- ❌ Add KL to oracle
- ❌ Add entropy bonuses as a "fix"
- ❌ Scale model size
- ❌ Switch to natural language

If any of these appear, Phase 4 is violated.

---

## Core Hypothesis (Testable, Falsifiable)

### H₄ (Primary Hypothesis)

> **Prediction-as-Action RL can learn accurate one-step world models, but fails to use them for multi-step decision-making under delayed reward unless the architecture explicitly enables planning or imagination.**

This is deliberately strong. It allows both positive and negative results.

### What This Tests

| Phase | Question | Answer |
|-------|----------|--------|
| Phase 3 | Can RL learn predictive representations? | ✅ Yes |
| Phase 4 | Can RL acquire and USE world knowledge? | ❓ Unknown |

The key distinction:
- **Phase 3**: Prediction = the task (one-step accuracy)
- **Phase 4**: Prediction = tool for solving a different task (multi-step planning)

---

## The Single Decisive Task: CausalChain-T3

### Why This Task

The task must:
1. ❌ Cannot be solved reactively (no state → action mapping)
2. ❌ Cannot be solved by memorization (combinatorial)
3. ✅ Forces causal, multi-step reasoning
4. ✅ Requires understanding action consequences

### Environment Specification

```
CausalChain-T3 Environment
==========================

Latent State Space:
  z_t ∈ {0, 1, 2, 3}  (4 hidden states)

Observable State:
  o_t = g(z_t)  where g is many-to-one (aliased)
  Example: o_t ∈ {A, B} where states {0,1} → A, {2,3} → B

Action Space:
  a_t ∈ {LEFT, RIGHT, STAY}  (3 actions)

Transition Dynamics (unknown to agent):
  z_{t+1} = f(z_t, a_t)

  Example transition table:
  | z_t | LEFT | RIGHT | STAY |
  |-----|------|-------|------|
  |  0  |  3   |   1   |  0   |
  |  1  |  0   |   2   |  1   |
  |  2  |  1   |   3   |  2   |
  |  3  |  2   |   0   |  3   |

  Key: Non-commutative (LEFT then RIGHT ≠ RIGHT then LEFT)

Episode Structure:
  - Horizon: T = 3 steps
  - Initial state: z_0 sampled uniformly
  - Target state: z_target sampled at episode start
  - Reward: ONLY at final step
    r_T = 1 if z_T == z_target else 0

Observation at each step:
  - o_t (aliased observation)
  - z_target (goal, always visible)
  - t (timestep)
```

### Why This Task Is Decisive

1. **No reactive solution**: Same observation can require different actions depending on hidden state
2. **No memorization**: 4 × 4 × 3³ = 432 possible (start, goal, action sequence) combinations
3. **Forces world model**: Must predict z_{t+1} = f(z_t, a_t) to plan
4. **Delayed reward**: No intermediate feedback on whether actions are correct

### Random Baseline

- Random policy: 1/4 = 25% success rate (4 possible final states)
- Optimal policy: 100% (always reachable in ≤3 steps)

---

## Agent Architecture

### Minimal Architecture (Justified)

```
Input: [o_t, a_{t-1}, z_target, t]
         ↓
    ┌─────────────┐
    │   Encoder   │  (shared, 2-layer MLP, hidden=64)
    │   h_t       │
    └─────────────┘
         ↓
    ┌────┴────┬────────┐
    ↓         ↓        ↓
┌───────┐ ┌───────┐ ┌───────────┐
│Policy │ │Value  │ │Prediction │
│ π(a|h)│ │ V(h)  │ │ ô_{t+1}   │
└───────┘ └───────┘ └───────────┘
```

### Head Specifications

| Head | Output | Training Signal |
|------|--------|-----------------|
| Policy | π(a\|h) ∈ Δ³ | Policy gradient from task reward |
| Value | V(h) ∈ ℝ | TD/MC from task reward |
| Prediction | ô_{t+1} ∈ Δ² | **Condition-dependent** (see below) |

### Key Constraint

> The prediction head does NOT receive direct supervision in Condition C.
> It must learn to predict through the task reward signal alone.

---

## Training Regimes (The Ablation)

### Condition A: Reactive RL (Baseline)

```
Architecture: Policy + Value only (no prediction head)
Reward: Task reward only (sparse, at T=3)
Training: Standard REINFORCE/PPO

Expected Result: ❌ FAILS
Reason: Cannot distinguish aliased states
```

### Condition B: Prediction-as-Action (Phase 3 Style)

```
Architecture: Full (with prediction head)
Action: a_t = ô_{t+1} (prediction IS the action)
Reward: r_t = -||ô_{t+1} - o_{t+1}||² (prediction accuracy)

Expected Result:
  ✅ Learns one-step dynamics
  ❌ Fails navigation task (wrong objective)

Purpose: Verify world model CAN be learned
```

### Condition C: Prediction + Delayed Reward (PHASE 4 CORE)

```
Architecture: Full (with prediction head)
Action: a_t ∈ {LEFT, RIGHT, STAY} (environment actions)
Reward: r_T = 1 if z_T == z_target else 0 (sparse)

Prediction head training options:
  C1: Auxiliary loss (prediction + task reward)
  C2: No auxiliary loss (prediction head learns from task gradient only)

Expected Result: THIS IS THE QUESTION
  - If C succeeds: RL can acquire AND use world knowledge
  - If C fails: Prediction alone insufficient for planning
```

### Training Hyperparameters

```python
config = {
    "horizon": 3,
    "num_latent_states": 4,
    "num_observations": 2,  # aliased
    "num_actions": 3,

    "hidden_dim": 64,
    "learning_rate": 3e-4,
    "gamma": 0.99,

    "total_episodes": 50000,
    "eval_interval": 1000,
    "eval_episodes": 500,

    "entropy_coef": 0.01,  # standard, not a "fix"
    "value_coef": 0.5,
    "prediction_coef": 0.1,  # only for C1
}
```

---

## Metrics (What Actually Matters)

### Primary Metrics

| Metric | What It Tests | Target |
|--------|---------------|--------|
| **Task Success Rate** | Can agent solve navigation? | > 50% (above random 25%) |
| **Optimal Action Rate** | Does agent take shortest path? | > 70% |

### Diagnostic Metrics

| Metric | What It Tests | Method |
|--------|---------------|--------|
| **World Model Accuracy** | Does prediction head learn dynamics? | Eval ô vs o on held-out transitions |
| **Latent Probe Accuracy** | Does h_t encode z_t? | Linear probe z_t from h_t |
| **Counterfactual Probe** | Does h_t encode f(z_t, a)? | Probe z_{t+1} from (h_t, a) |

### Intervention Tests

```
Test 1: Transition Table Swap
- Train on transition table T1
- Eval on modified table T2
- If performance degrades predictably → agent uses world model
- If performance unchanged → agent memorized

Test 2: Goal Generalization
- Train with goals {0, 1, 2}
- Eval with held-out goal {3}
- Tests compositional use of world model
```

### Metrics NOT to Use

- ❌ Accuracy (ambiguous for multi-step)
- ❌ Entropy (not the question)
- ❌ KL divergence (no oracle)
- ❌ Loss curves alone (not informative)

---

## Expected Outcomes and Interpretation

### Outcome 1: Full Failure

```
Condition A: Fails (expected)
Condition B: Learns dynamics, fails task (expected)
Condition C: Learns dynamics, STILL fails task

Interpretation:
  World models are learned but NOT USED for decision-making.
  Planning mechanism is missing.

Conclusion:
  Prediction alone is INSUFFICIENT for world learning.
  Explicit planning/imagination required.

Publishability: HIGH (clean negative result)
```

### Outcome 2: Partial Success

```
Condition C: Task success > 50% but < 90%
            Unstable across seeds
            Some goals harder than others

Interpretation:
  Implicit planning is weak but present.
  Architecture supports some multi-step reasoning.

Conclusion:
  Prediction provides WEAK planning signal.
  May need architectural support for robust planning.

Publishability: MEDIUM-HIGH (nuanced finding)
```

### Outcome 3: Strong Success

```
Condition C: Task success > 90%
            Consistent across seeds
            Probes show causal structure
            Intervention tests pass

Interpretation:
  RL CAN acquire and use world knowledge through interaction.
  No pretraining required.

Conclusion:
  Prediction-as-Action enables baby-like world learning.
  This is the strongest possible result.

Publishability: VERY HIGH (major positive result)
```

---

## Implementation Checklist

### Week 1: Environment

- [ ] Implement `CausalChainEnv` class
  - [ ] Latent state dynamics
  - [ ] Observation aliasing function
  - [ ] Configurable transition tables
  - [ ] Episode reset with random start/goal
  - [ ] Sparse reward at T=3

- [ ] Implement test suite
  - [ ] Verify transitions are non-commutative
  - [ ] Verify aliasing works correctly
  - [ ] Verify random baseline ~25%
  - [ ] Verify optimal policy achieves 100%

### Week 2: Agent Architecture

- [ ] Implement `CausalAgent` class
  - [ ] Shared encoder
  - [ ] Policy head
  - [ ] Value head
  - [ ] Prediction head (optional)

- [ ] Implement training loops
  - [ ] Condition A: Reactive RL
  - [ ] Condition B: Prediction-as-Action
  - [ ] Condition C: Prediction + Delayed Reward

### Week 3: Experiments

- [ ] Run Condition A (baseline)
- [ ] Run Condition B (verify world model learning)
- [ ] Run Condition C1 (auxiliary prediction loss)
- [ ] Run Condition C2 (no auxiliary loss)
- [ ] 5 seeds per condition

### Week 4: Analysis

- [ ] Compute all primary metrics
- [ ] Run linear probes for latent state
- [ ] Run counterfactual probes
- [ ] Run intervention tests
- [ ] Generate plots and tables

### Week 5: Documentation

- [ ] Write results summary
- [ ] Interpret findings
- [ ] Draft paper outline (if results are clear)

---

## File Structure

```
src/
  environment/
    causal_chain.py          # CausalChain-T3 environment
    test_causal_chain.py     # Environment tests

  agents/
    causal_agent.py          # Agent with prediction head

  training/
    train_causal.py          # Training loop for all conditions

  analysis/
    causal_probes.py         # Linear probes for latent state
    intervention_tests.py    # Transition table swap tests

experiments/
  run_phase4_experiments.py  # Main experiment runner

results/
  phase4_causal/
    condition_a/
    condition_b/
    condition_c1/
    condition_c2/

progress_docs/
  phase_4_world_learning.md  # This file
```

---

## Literature Support

### Papers Supporting This Design

1. **Curiosity-Driven Exploration** (Pathak et al., 2017)
   - Shows prediction error as intrinsic reward
   - Supports Condition B design

2. **World Models** (Ha & Schmidhuber, 2018)
   - Latent dynamics learning for RL
   - Supports prediction head architecture

3. **Dreamer** (Hafner et al., 2020)
   - End-to-end world model + policy learning
   - Supports joint training approach

4. **MuZero** (Schrittwieser et al., 2020)
   - Implicit world models for planning
   - Supports latent state representation

5. **Predictive State Representations** (Littman et al.)
   - State as future prediction
   - Theoretical grounding for our approach

### What This Work Adds

Unlike prior work:
- We test whether prediction ALONE enables planning (not explicit MCTS/imagination)
- We use a minimal task that isolates the world-learning question
- We diagnose success/failure with causal probes

---

## Strategic Context

### How Phase 4 Connects to Prior Work

| Phase | Question | Status |
|-------|----------|--------|
| 1-2 | Can RL learn predictions? | ✅ Yes |
| 3 | Does RL learn distributions? | ✅ No (collapse) |
| 4 | Can RL learn and USE world knowledge? | ❓ This phase |
| 5 | Can explicit planning help? | Future work |

### What Phase 4 Unlocks

Regardless of outcome, Phase 4 provides:

1. **Clean answer** to: "Is prediction alone enough for baby-like world learning?"

2. **Principled justification** for (if needed):
   - Planning modules
   - Imagination/rollouts
   - Tree search
   - Hierarchical RL

3. **Natural transition** to Phase 5:
   - If Phase 4 fails: Add explicit planning
   - If Phase 4 succeeds: Scale to harder tasks

---

## Final Directive

> **Do not optimize Phase 4 for success. Optimize it for clarity.**

A clean failure is more valuable than a noisy success.

The goal is scientific understanding, not benchmark performance.

---

*Document created: January 2026*
*Last updated: January 2026*
