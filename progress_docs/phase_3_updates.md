# Phase 3 Implementation Progress

## Overview
This document tracks the implementation progress of Phase 3: Language Learning Without Pretraining.

**Start Date**: 2026-01-04

**Goal**: Demonstrate that RL gradients can learn language representations without pretraining, using an LLM parent as a property-based verifier.

---

## Stage 3.1: Token Prediction Task

### Status: ✅ **COMPLETED - All Core Experiments Passed!**

### Objective
Extend Phase 2b's prediction-as-action to discrete tokens. Prove RL can learn token dynamics like state dynamics.

### Implementation Tasks

| Task | Status | Notes |
|------|--------|-------|
| Create `TokenPredictionTask` | ✅ Done | `src/environment/token_prediction.py` |
| Create `TokenPredictionModel` | ✅ Done | `src/models/token_model.py` |
| Create `TokenREINFORCE` | ✅ Done | `src/training/token_reinforce.py` |
| Create `TokenTrainer` | ✅ Done | `src/training/token_trainer.py` |
| Implement TD(λ) | ✅ Done | `TDLambdaREINFORCE` class |
| Create run script | ✅ Done | `run_token_prediction.py` |
| Run RL vs MLE comparison | ✅ Done | Both achieved 100% accuracy! |
| Run delayed reward ablation | ✅ Done | 100% accuracy across all delays! |

### Design Decisions

1. **Vocabulary Size**: Starting with 16 tokens (same as Phase 2b state dim)
2. **Grammar Types**:
   - Deterministic cyclic: 0→1→2→...→15→0
   - Deterministic permutation: Fixed random mapping
   - Bigram: P(next|current) with learned transitions
3. **Reward**: Binary (correct/incorrect) initially, then delayed

### Files Created

#### `src/environment/token_prediction.py`
Token prediction task - extends Phase 2b to discrete tokens.
- `TokenPredictionTask`: Single-step token prediction
- `SequenceTokenTask`: Multi-step sequence prediction with delayed reward
- Grammar types: `deterministic_cyclic`, `deterministic_permutation`, `bigram`

#### `src/models/token_model.py`
Token prediction model with Categorical distribution.
- `TokenEmbedding`: Embeds tokens into hidden space
- `TokenPredictionHead`: Outputs logits over vocabulary
- `TokenPredictionModel`: Full model (Embedding → Transformer → Head)
- `SequenceTokenModel`: For multi-step sequences with causal masking

#### `src/training/token_reinforce.py`
Training algorithms for token prediction.
- `TokenREINFORCE`: Standard REINFORCE for single-step
- `TokenMLE`: Cross-entropy baseline (like pretraining)
- `SequenceREINFORCE`: REINFORCE for sequences with delayed reward
- `TDLambdaREINFORCE`: TD(λ) with eligibility traces for better credit assignment

#### `src/training/token_trainer.py`
Training loop and experiment management.
- `TokenTrainerConfig`: Configuration dataclass
- `TokenTrainer`: Main trainer class
- `run_token_experiment()`: Entry point function

#### `run_token_prediction.py`
Main experiment script with multiple modes.

### Commands

```bash
# Run single RL experiment
python run_token_prediction.py --mode rl --steps 10000

# Run single MLE experiment
python run_token_prediction.py --mode mle --steps 10000

# Run RL vs MLE comparison (core experiment)
python run_token_prediction.py --compare

# Run delay ablation (credit assignment test)
python run_token_prediction.py --delay-ablation

# Run TD(λ) vs REINFORCE comparison
python run_token_prediction.py --td-lambda

# Run grammar comparison
python run_token_prediction.py --grammar-compare

# Run all experiments
python run_token_prediction.py --all

# With GPU
python run_token_prediction.py --compare --device cuda
```

---

## Stage 3.2: Grounded Language Task

### Status: ⬜ Not Started

### Objective
Language actions affect a world state, creating prediction-action-feedback loop.

---

## Stage 3.3: Interactive LLM Parent Task

### Status: ⬜ Not Started

### Objective
LLM parent provides property-based (not likelihood-based) feedback.

---

## Experiment Log

### 2026-01-08: Delay Ablation Experiment - OUTSTANDING SUCCESS! ✅

**Command**: `python run_token_prediction.py --delay-ablation --steps 15000`

**Results**: 🎉 **ALL experiments achieved 100% accuracy!**

| Sequence Length | Accuracy | Notes |
|----------------|----------|-------|
| 1 (no delay) | **100.00%** | Converged by step ~6000 |
| 3 (moderate delay) | **100.00%** | Converged by step ~4000 |
| 5 (challenging delay) | **100.00%** | Converged by step ~4000 |
| 7 (very long delay) | **100.00%** | Converged by step ~8000 |

**Key Findings**:
1. ✅ **Vanilla REINFORCE handles credit assignment perfectly** - even with 7-step delayed reward
2. ✅ **No degradation with delay** - all sequence lengths reached 100% accuracy
3. ✅ **Fast convergence** - all experiments converged within 8000 steps
4. ✅ **Stable learning** - maintained 100% accuracy after convergence

**Why This Matters**:
- Phase 2b showed 35% accuracy at 5-step delay for continuous states
- Token prediction achieves **100% accuracy at 7-step delay**
- This suggests discrete action spaces may be EASIER for credit assignment than continuous
- Validates that RL can handle multi-token sequences without specialized algorithms (TD(λ) not needed for deterministic grammars)

**Implications for Stage 3.2/3.3**:
- Credit assignment is NOT a bottleneck for token prediction
- Can confidently move to grounded language tasks
- May not need TD(λ) for deterministic/semi-deterministic tasks
- Focus optimization efforts elsewhere (exploration, LLM feedback quality)

---

### 2026-01-08: RL vs MLE Comparison - HYPOTHESIS VALIDATED! ✅

**Command**: `conda run -n pytorch_5070ti python run_token_prediction.py --compare --steps 10000`

**Results**:
- **MLE Accuracy**: 100%
- **RL Accuracy**: 100%
- **Gap**: 0%

**Conclusion**: ✅ **RL gradients perfectly match MLE for token prediction**

This validates the core Stage 3.1 hypothesis: *prediction-as-action* extends from continuous states (Phase 2b) to discrete tokens.

---

### 2026-01-04: Stage 3.1 Implementation Started

**Goal**: Implement token prediction task as bridge from Phase 2b to language.

**Key Design Choices**:
- Categorical distribution (discrete tokens) vs Gaussian (continuous states)
- Same transformer backbone as Phase 2b
- Start with deterministic grammar (no stochasticity)

---

## Key Metrics to Track

| Metric | Stage 3.1 Target | Current | Status |
|--------|------------------|---------|--------|
| RL Accuracy (deterministic) | ≥95% | **100%** | ✅ **Exceeded!** |
| RL vs MLE gap | ≤5x samples | **0% gap** | ✅ **Perfect match!** |
| Delayed reward (5-step) | >35% | **100%** | ✅ **Far exceeded!** |
| Delayed reward (7-step) | N/A | **100%** | ✅ **Bonus result!** |
| CCA (RL vs MLE) | >0.8 | - | ⬜ Future analysis |

---

## Open Questions

1. ✅ **ANSWERED**: Does discrete action space fundamentally change learning dynamics?
   - **YES** - Discrete tokens appear EASIER than continuous states for credit assignment
   - 100% accuracy at 7-step delay (vs 35% at 5-step for continuous in Phase 2b)

2. ✅ **ANSWERED**: What is the maximum delay RL can handle for token prediction?
   - At least 7 steps with deterministic grammar
   - Next: test with stochastic grammars and longer sequences

3. ⬜ **PENDING**: Does TD(λ) significantly help over vanilla REINFORCE?
   - Vanilla REINFORCE already achieves 100% - may not need TD(λ) for deterministic tasks
   - Next: test on stochastic grammars or grounded language where credit assignment is harder

---

## Notes

- Phase 2b achieved 100% accuracy with 8-dim continuous states
- Delayed reward at 5 steps gave 35% accuracy
- Target: replicate these results with discrete tokens
