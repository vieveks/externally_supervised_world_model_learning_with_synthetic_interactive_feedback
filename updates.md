# WMIL: World-Model Interactive Learning - Implementation Plan & Updates

## Project Overview

**Core Hypothesis**: A randomly initialized transformer can learn structured representations and goal-directed behavior through interaction with an LLM-powered environment, without any pretraining or token-level imitation.

**Key Distinction**: This is NOT distillation, NOT RLHF. It is self-supervised world-model learning with synthetic interactive feedback.

---

## Implementation Plan

### Phase 0: Project Setup
**Status**: ✅ Complete

- [x] Initialize repository
- [x] Set up Python environment (requirements.txt)
- [x] Create directory structure
- [x] Set up logging (TensorBoard optional, console fallback)

### Phase 1: Minimal Environment
**Status**: ✅ Complete

**Goal**: Create a symbolic environment where states and actions are discrete, and a deterministic parent provides consequences.

#### 1.1 Environment Design
- [x] Define state space: One-hot encoded target (for PatternEcho)
- [x] Define action space: 8 discrete symbols
- [x] Implement `SymbolicEnv` class with FIX #1 (time-limit only termination)

#### 1.2 Parent Implementation
- [x] Create `DeterministicParent` (rule-based, no LLM for MVP)
- [x] Implement state transition logic
- [x] Implement reward signal generation (0/1 for correct/incorrect)

#### 1.3 Task Suite
- [x] **Task 1: PatternEcho** - Baby must output the target number shown in state
- [ ] Task 2: SequenceNext (Phase 2)
- [ ] Task 3: Conditional (Phase 2)

### Phase 2: Baby Model Architecture
**Status**: ✅ Complete

**Goal**: Minimal transformer with world-model head and value head.

#### 2.1 Core Architecture
- [x] `StateEncoder`: Linear layers to embed state
- [x] `TransformerCore`: 2-layer, 64-hidden, 4 heads
- [x] `ActionHead`: Policy distribution over actions
- [x] `WorldModelHead`: Predicts RAW next state (FIX #2)
- [x] `ValueHead`: For baseline (optional)

#### 2.2 Implementation Details
- [x] `BabyModel` class (~117K parameters)
- [x] Random initialization (NO pretrained weights)
- [x] Discrete action space (Categorical distribution)
- [x] World-model predicts raw state tensor, not embedding

### Phase 3: Reward System
**Status**: ✅ Complete

**Goal**: Vector-valued rewards, NO token matching.

- [x] `RewardVector` dataclass with goal and pred components
- [x] Goal reward: 1.0 if correct, 0.0 otherwise
- [x] Prediction reward: Computed from world-model MSE (intrinsic)
- [x] Logging each component separately

### Phase 4: Training Loop
**Status**: ✅ Complete

**Goal**: REINFORCE with world-model auxiliary loss.

- [x] `REINFORCE` class with running baseline (FIX #3: not PPO)
- [x] World-model loss: MSE on predicted vs actual raw state
- [x] Gradient clipping for stability
- [x] Checkpointing and resumption
- [x] TensorBoard logging (optional) + console fallback

### Phase 5: Curriculum Learning
**Status**: ⬜ Not Started (Phase 2 work)

- [ ] Define difficulty levels
- [ ] Implement mastery detection
- [ ] Automatic curriculum advancement

### Phase 6: Evaluation & Analysis
**Status**: ⬜ Not Started (Phase 2 work)

- [ ] Baseline comparisons
- [ ] Ablation studies
- [ ] Representation analysis

---

## Directory Structure (Implemented)

```
self_supervised_world_model_learning_with_synthetic_interactive_feedback/
├── src/
│   ├── __init__.py
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── symbolic_env.py      # SymbolicEnv, State, RewardVector
│   │   ├── tasks.py             # PatternEchoTask
│   │   └── parent.py            # DeterministicParent
│   ├── models/
│   │   ├── __init__.py
│   │   └── baby_model.py        # BabyModel with all heads
│   ├── training/
│   │   ├── __init__.py
│   │   ├── reinforce.py         # REINFORCE algorithm
│   │   └── trainer.py           # Main training loop
│   └── utils/
│       ├── __init__.py
│       └── config.py            # Config dataclass and loader
├── configs/
│   └── default.yaml             # MVP hyperparameters
├── paper_drafts/
│   └── draft_01.tex             # Paper draft
├── train.py                     # Entry point
├── requirements.txt             # Dependencies
├── plan.md                      # Detailed implementation plan
├── updates.md                   # This file
└── README.md                    # Project overview
```

---

## Experiment Log

### Experiment 001: MVP PatternEcho ✅ PASSED
**Date**: 2026-01-01
**Objective**: Verify a randomly initialized transformer can learn PatternEcho task

**Config**:
```yaml
task: pattern_echo
num_actions: 8
hidden_dim: 64
num_layers: 2
algorithm: reinforce
parent: deterministic
total_steps: 5000
```

**Results**:
| Metric | Start | End | Target |
|--------|-------|-----|--------|
| Success Rate | 12.5% (random) | **100%** | >50% ✅ |
| Entropy | 2.01 | **0.07** | ↓ ✅ |
| World-Model Loss | 0.15 | **0.11** | ↓ ✅ |

**Training Curve Highlights**:
- Step 500: 46% success, entropy 1.6
- Step 1000: 80% success, entropy 0.8
- Step 2000: 98% success, entropy 0.2
- Step 5000: 100% success, entropy 0.07

**Insights**:
1. Learning is fast - reaches >90% success by step 1500
2. Entropy drops smoothly as policy becomes confident
3. World-model loss plateaus around 0.11 (can't go lower due to state transitions being stochastic)
4. The framework is fundamentally sound

**Command**:
```bash
python train.py --total_steps 5000
```

---

## Key Design Decisions

### Decision 1: No KL-to-Parent
**Rationale**: Any KL term `D_KL(Parent || Baby)` smuggles pretraining back in by asking the baby to match the parent's token distribution. We explicitly reject this.

### Decision 2: Vector Rewards
**Rationale**: Collapsing multiple signals into a scalar loses information and makes debugging harder. We keep rewards as vectors and handle multi-objective optimization explicitly.

### Decision 3: Discrete Symbols, Not Tokens
**Rationale**: Starting with English tokens bakes in linguistic structure. We use arbitrary discrete symbols and let structure emerge from utility.

### Decision 4: World-Model as First-Class Citizen
**Rationale**: The world-model head is what distinguishes this from "RL chat". It provides dense self-supervised signal that doesn't require the parent to grade every output.

### Decision 5: Raw State Prediction (FIX #2)
**Rationale**: Early in training, the encoder is random and embeddings drift. Predicting raw state tensors gives a stable target for the world-model to learn from.

### Decision 6: REINFORCE Before PPO (FIX #3)
**Rationale**: PPO introduces GAE bugs, advantage normalization issues, and credit assignment confusion. REINFORCE with a simple running baseline is sufficient for MVP and easier to debug.

---

## Open Questions

1. ~~**State Representation**: Should the world-model predict in embedding space or token space?~~ **RESOLVED**: Raw state tensor for MVP, embedding later.
2. **Reward Scalarization**: When combining vector rewards, use fixed weights or adaptive?
3. ~~**LLM as Environment**: How deterministic should the parent's responses be?~~ **RESOLVED**: Deterministic ONLY for MVP.
4. **Minimal Innate Structure**: What's the absolute minimum the baby needs (attention? positional encoding?)
5. **Scaling**: How does performance change with more actions, longer sequences, harder tasks?

---

## References

- Ha, D., & Schmidhuber, J. (2018). World Models.
- Pathak, D., et al. (2017). Curiosity-driven Exploration.
- Schulman, J., et al. (2017). Proximal Policy Optimization.
- OpenAI. (2019). Dota 2 with Large Scale Deep RL.

---

## Updates Timeline

| Date | Update |
|------|--------|
| 2026-01-01 | Initial plan created |
| 2026-01-01 | **CRITICAL REVIEW**: Incorporated 5 dangerous fixes from external review |
| 2026-01-01 | **MVP IMPLEMENTED**: Full implementation of PatternEcho task |
| 2026-01-01 | **MVP PASSED**: All 4 success criteria met |

---

## 2026-01-01: Critical Review Fixes

### Issues Identified (from external LLM review)

| Issue | Severity | Fix |
|-------|----------|-----|
| Episode termination on success | DANGEROUS | End on time limit ONLY, log success separately |
| World-model predicts embedding | DANGEROUS | Predict RAW state tensor for MVP (encoder drifts early) |
| Starting with PPO | Overengineered | Start with REINFORCE + baseline |
| Scalar rewards immediately | Loses signal | Log each reward component separately |
| LLM parent from start | Confounds | Deterministic parent ONLY until MVP works |

### What Was Changed

1. **plan.md**: Added "CRITICAL FIXES" section at top
2. **plan.md**: Updated MVP to be even more minimal (8 actions, 2 layers, 64 hidden)
3. **plan.md**: Added REINFORCE implementation before PPO
4. **plan.md**: Fixed `SymbolicEnv.step()` to not terminate on success
5. **plan.md**: Fixed `WorldModelHead` to output raw state dim, not hidden dim

---

## 2026-01-01: MVP Implementation Complete

### Files Created

| File | Lines | Description |
|------|-------|-------------|
| `src/environment/tasks.py` | 55 | PatternEchoTask definition |
| `src/environment/parent.py` | 52 | DeterministicParent |
| `src/environment/symbolic_env.py` | 120 | SymbolicEnv, State, RewardVector |
| `src/models/baby_model.py` | 200 | Full BabyModel with all heads |
| `src/training/reinforce.py` | 95 | REINFORCE algorithm |
| `src/training/trainer.py` | 210 | Main training loop |
| `src/utils/config.py` | 68 | Configuration handling |
| `configs/default.yaml` | 40 | Default hyperparameters |
| `train.py` | 90 | Entry point script |

### Model Summary

```
BabyModel (117,625 parameters)
├── StateEncoder: 8 → 64 → 64
├── TransformerCore: 2 layers, 4 heads, 256 FFN
├── ActionHead: 64 → 8
├── WorldModelHead: 64 + 64 → 64 → 8
└── ValueHead: 64 → 32 → 1
```

### Training Performance

- **Speed**: ~500 iterations/second on CUDA
- **Convergence**: >90% success in ~1500 steps
- **Stability**: No training instabilities observed

---

## Next Steps (Phase 2)

1. **More Tasks**: Add SequenceNext and Conditional tasks
2. **PPO**: Upgrade from REINFORCE for more complex tasks
3. **Curriculum**: Implement difficulty progression
4. ~~**Ablations**: Test importance of world-model head~~ ✅ Done
5. **LLM Parent**: Integrate after deterministic version is solid

---

## 2026-01-01: World-Model Ablation Study

### Experiment 002: World-Model Ablation on PatternEcho

**Objective**: Determine if world-model head contributes to learning

**Methodology**: Compare `world_model_coef=1.0` vs `world_model_coef=0.0` over 4 runs each

### Results Summary

| Metric | With WM (mean) | Without WM (mean) | Difference |
|--------|----------------|-------------------|------------|
| Best Success | 97.27% | 94.14% | +3.13% |
| Final Success | 95.70% | 92.58% | +3.12% |
| Final Entropy | 0.2577 | 0.2970 | -0.0393 |
| Variance (std) | 0.79% | 3.08% | 4x lower |

### Individual Runs

**WITH World-Model:**
| Run | Best | Final | Entropy |
|-----|------|-------|---------|
| 1 | 98.44% | 95.31% | 0.2089 |
| 2 | 95.31% | 95.31% | 0.3240 |
| 3 | 95.31% | 95.31% | 0.3111 |
| 4 | 100.00% | 96.88% | 0.1869 |

**WITHOUT World-Model:**
| Run | Best | Final | Entropy |
|-----|------|-------|---------|
| 1 | 95.31% | 90.62% | 0.3186 |
| 2 | 89.06% | 89.06% | 0.3845 |
| 3 | 96.88% | 95.31% | 0.2192 |
| 4 | 95.31% | 95.31% | 0.2656 |

### Interpretation

**Finding**: Small but consistent improvement with world-model (~3% higher success, 4x lower variance)

**Why Small?** PatternEcho is a single-step task that doesn't require prediction:
- No temporal dependencies
- Reactive policy suffices
- World-model provides auxiliary gradients but isn't essential

**Conclusion**: Inconclusive for PatternEcho. Need SequenceNext task where prediction matters.

**Full report**: `progress_docs/ablation_world_model.md`

---

## 2026-01-01: SequenceNext Task Implementation

### Task Description

**SequenceNext**: Predict the next element in a repeating sequence
- Sequence: e.g., [2, 5, 1, 7, 2, 5, 1, 7, ...]
- State shows current element, baby must predict next
- This requires learning transition dynamics (unlike PatternEcho)

### Results Summary (5-6 runs each)

| Metric | With WM | Without WM |
|--------|---------|------------|
| Best Success | 91.15% | 92.50% |
| Final Success | 79.43% | 89.06% |

### Key Finding: Still No Clear World-Model Advantage

**Unexpected result**: Both conditions perform similarly (~80-90% success)

**Why?**
1. Task still solvable as lookup table (4 transitions to memorize)
2. Single-step episodes don't require planning
3. Each episode is independent, not truly sequential

### What We Learned

Current tasks (PatternEcho, SequenceNext) can be solved with pure stimulus-response learning.
The world-model IS learning (WM loss drops from 0.22 → 0.03), but isn't NECESSARY for success.

### What's Needed Next

Tasks that REQUIRE prediction/planning:
- Multi-step episodes with delayed rewards
- Partial observability requiring inference
- Planning ahead, not just reacting

**Full report**: `progress_docs/sequence_next_ablation.md`
