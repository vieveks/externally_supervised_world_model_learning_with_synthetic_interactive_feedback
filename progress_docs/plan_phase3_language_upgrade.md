# Phase 3: Language Learning Upgrade Plan

## CRITICAL REALITY CHECK

**Current Status**: Phase 3.1 (deterministic + stochastic) is NOT yet a language paper.

**Why**: We have discrete token prediction, but we have NOT demonstrated:
- Ambiguity (multiple valid continuations)
- Compositional generalization (seen parts in new combinations)
- Long-range dependencies
- Partial observability
- Semantic constraints

**What We Actually Have**: Strong foundations for prediction-as-action on discrete tokens

**What We Need**: Three mandatory upgrades to make this a genuine language learning paper

---

## The Three Mandatory Upgrades (NON-NEGOTIABLE)

### 🔴 Upgrade 1: Ambiguous Grammar

**Problem**: Current grammars are deterministic or have unique argmax. No true ambiguity.

**Solution**: Implement grammar with genuine ambiguity

**Example Grammar**:
```
S → A B | A C
A → x
B → y
C → z

Prefix: x
Valid continuations: y (50%) or z (50%)
```

**Why This Matters**:
- First appearance of "language-like uncertainty"
- Deterministic policies fail (must maintain distribution)
- MLE learns conditional distribution easily
- RL must handle ambiguity + credit assignment simultaneously

**Implementation**:
```python
@dataclass
class AmbiguousGrammar:
    """
    Grammar with genuine ambiguity.

    Key property: Same prefix → multiple valid continuations
    Requires: Maintaining distribution, not just argmax
    """

    rules: Dict[str, List[Tuple[List[str], float]]]
    # Example: {"x": [("y", 0.5), ("z", 0.5)]}

    def get_valid_continuations(self, prefix: List[str]) -> List[Tuple[str, float]]:
        """Return all valid next tokens with probabilities."""

    def sample_sequence(self) -> List[str]:
        """Generate sequence following grammar rules."""
```

**Success Criteria**:
- RL maintains entropy in policy (doesn't collapse to argmax)
- RL accuracy ≥ 80% (matching MLE)
- Policy entropy > 0.5 at ambiguous decision points

**Experiment Design**:
1. Compare RL vs MLE on ambiguous grammar
2. Measure policy entropy at ambiguous points
3. Test: Does RL maintain distribution or collapse to single mode?

---

### 🔴 Upgrade 2: Compositional Generalization

**Problem**: Current tasks can be solved by memorization. No test of composition.

**Solution**: Train/test split that breaks memorization

**Example (SCAN-style)**:
```
Train:
  x y a
  x y b
  x z a

Test:
  x z b  (never seen in training)
```

**Why This Matters**:
- Distinguishes memorization from true compositional learning
- Standard test in compositional generalization literature
- Failure here is also publishable (important negative result)

**Implementation**:
```python
class CompositionalSplit:
    """
    Train/test split for compositional generalization.

    Inspired by SCAN, but much simpler.
    """

    def generate_splits(
        self,
        vocab_size: int,
        composition_depth: int = 2
    ) -> Tuple[List[Sequence], List[Sequence]]:
        """
        Generate train and test splits.

        Train: All combinations except held-out compositions
        Test: Novel compositions of seen elements
        """

    def evaluate_compositionality(
        self,
        model,
        test_sequences: List[Sequence]
    ) -> float:
        """
        Measure generalization to novel compositions.

        Returns accuracy on held-out compositions.
        """
```

**Success Criteria**:
- RL achieves >50% on held-out compositions (above chance)
- If RL ≈ MLE on composition → strong claim
- If RL < MLE → still publishable (shows limitation)

**Experiment Design**:
1. Train on all-but-one composition
2. Test on held-out composition
3. Compare RL vs MLE generalization
4. Analyze which compositions transfer better

---

### 🔴 Upgrade 3: Representation Analysis

**Problem**: Have CCA for states, need it for ambiguous + compositional cases

**Solution**: Analyze representations on linguistically meaningful dimensions

**What to Analyze**:
```python
class RepresentationAnalysis:
    """
    Analyze learned representations on language-relevant dimensions.
    """

    def cca_on_ambiguous_prefixes(
        self,
        rl_model,
        mle_model,
        ambiguous_contexts: List[str]
    ) -> float:
        """
        Do RL and MLE learn similar representations
        for ambiguous contexts?
        """

    def cca_on_compositions(
        self,
        rl_model,
        mle_model,
        held_out_compositions: List[str]
    ) -> float:
        """
        Do representations generalize to novel compositions?
        """

    def linear_probe_compositionality(
        self,
        model,
        probe_task: str = "constituent_identity"
    ) -> float:
        """
        Can we linearly decode compositional structure?

        Example probes:
        - Which constituent is this? (A vs B vs C)
        - What position in composition? (first vs second)
        """
```

**Success Criteria**:
- CCA similarity (RL vs MLE) > 0.7 on ambiguous cases
- CCA similarity > 0.6 on compositional cases
- Linear probes show structured representations

**Key Claim This Enables**:
> "RL recovers the same internal abstractions as supervised language modeling."

This is the high-impact sentence.

---

## What We Must NOT Do (Will Kill Paper)

❌ **Do NOT introduce**:
- English text
- Wikipedia/web corpora
- LLM teachers that provide likelihood
- GPT reward models
- "Parent LLM supervision" with probability distributions

**Why**: The moment you do, reviewers say "you reintroduced pretraining through the back door"

**Stay synthetic until the claim is airtight**

---

## The Correct High-Impact Paper Structure

### Title (Example)
"Prediction as Action for Synthetic Language Learning: Reinforcement Learning Without Likelihood Supervision"

**Notice**:
- "Synthetic Language" → honest about scope
- "Without Likelihood Supervision" → sharp contrast with MLE
- No GPT claims

### Core Claims (What We ARE Allowed to Say)

1. ✅ RL can learn ambiguous token distributions
2. ✅ RL can learn compositional structure
3. ✅ RL learns representations aligned with MLE
4. ✅ Credit assignment depends on action geometry and entropy

**This is already a strong language paper**

### What Makes This "Language" Not "Tokens"

Must hit ≥3 of these criteria:

| Criterion | Our Approach |
|-----------|--------------|
| **Ambiguity** | ✅ Ambiguous grammar (multiple valid continuations) |
| **Compositionality** | ✅ Train/test split with held-out compositions |
| **Long-range dependency** | ⚠️ Can add if needed (nested structures) |
| **Partial observability** | ⚠️ Can add if needed (hidden state) |
| **Semantic constraint** | ⚠️ Implicit in grounded tasks (Stage 3.2) |

**We need ≥3. We have 2 mandatory, can add 1-2 more if needed.**

---

## Detailed 4-6 Week Execution Plan

### Week 1: Ambiguous Grammar

**Tasks**:
- [ ] Implement AmbiguousGrammar class
- [ ] Verify oracle ceiling (should be <100% due to ambiguity)
- [ ] Ensure policy maintains entropy (doesn't collapse)
- [ ] Test edge case: extreme ambiguity (uniform distribution)

**Deliverable**: Working ambiguous grammar with verified properties

**Success Check**: Policy entropy > 0.5 at ambiguous decision points

---

### Week 2: Compositional Generalization

**Tasks**:
- [ ] Implement CompositionalSplit class
- [ ] Generate train/test splits systematically
- [ ] Run RL vs MLE on compositional splits
- [ ] Measure generalization to held-out compositions

**Deliverable**: Compositional generalization results

**Success Check**: Accuracy on held-out > chance baseline

**Key Question**: Does RL ≈ MLE on composition?
- If yes → strong claim (RL generalizes like MLE)
- If no → still publishable (shows limitation clearly)

---

### Week 3: Representation Analysis

**Tasks**:
- [ ] CCA on ambiguous prefixes (RL vs MLE)
- [ ] CCA on compositional cases
- [ ] Linear probes for compositional structure
- [ ] Visualization of learned representations

**Deliverable**: Comprehensive representation analysis

**Success Check**: CCA > 0.7 on ambiguous, >0.6 on compositional

**Key Claim**:
> "RL and MLE learn similar internal abstractions despite different optimization objectives"

---

### Week 4: Paper Writing

**Tasks**:
- [ ] Write Introduction (framing as language learning)
- [ ] Write Methods (ambiguity + composition + representations)
- [ ] Write Results (all three upgrades)
- [ ] Write Discussion (scope, limitations, future work)

**Deliverable**: Complete draft

**Key Framing**:
- Emphasize why this matters for LLMs
- NOT that we solve LLMs
- Position as conceptual/foundational work

---

### Optional Week 5-6: Scaling

**Only if Weeks 1-4 succeed**:
- [ ] Increase vocab to 64 tokens
- [ ] Increase sequence length to 20-30
- [ ] Test if results hold at larger scale

**Why Optional**: Core claims don't depend on scale if foundations are solid

---

## Revised Success Criteria

### Stage 3.1b: Ambiguous Token Prediction ✅

**From Previous**:
- ✅ RL handles stochastic (60% accuracy, matches MLE)
- ✅ CCA = 1.000 (perfect alignment)

**New Requirements**:
- [ ] RL handles true ambiguity (maintains distribution)
- [ ] Policy entropy > 0.5 at ambiguous points
- [ ] RL accuracy ≥ 80% on ambiguous grammar

### Stage 3.1c: Compositional Generalization (NEW)

**Requirements**:
- [ ] Implement train/test split with held-out compositions
- [ ] RL achieves >50% on held-out (above chance)
- [ ] Compare RL vs MLE generalization gap
- [ ] Publishable either way (success OR interesting failure)

### Stage 3.1d: Representation Analysis (NEW)

**Requirements**:
- [ ] CCA on ambiguous contexts > 0.7
- [ ] CCA on compositions > 0.6
- [ ] Linear probes show compositional structure
- [ ] Visualizations show structured representations

---

## What Changes in Paper Strategy

### Paper 2 (Phase 3 - Updated Scope)

**Old Title**: "From States to Tokens"

**New Title**: "Prediction as Action for Synthetic Language Learning: Reinforcement Learning Without Likelihood Supervision"

**Old Scope**: Deterministic + stochastic token prediction

**New Scope**:
1. Deterministic grammars (Stage 3.1a - DONE)
2. Stochastic grammars (Stage 3.1b - DONE)
3. **Ambiguous grammars (Stage 3.1b upgrade)**
4. **Compositional generalization (Stage 3.1c - NEW)**
5. **Representation analysis (Stage 3.1d - NEW)**

**Key Claims**:
- RL can learn language structure without likelihood supervision
- Ambiguity is handled (maintains distributions)
- Composition emerges (generalizes to novel combinations)
- Representations align with MLE (despite different optimization)

**This is now a genuine language paper**

---

## Risk Analysis & Mitigation

### Risk 1: Ambiguous Grammar Too Hard

**Symptom**: RL accuracy << MLE, policy collapses to argmax

**Mitigation**:
- Start with mild ambiguity (70/30 split, not 50/50)
- Gradually increase ambiguity
- Report results across ambiguity levels

**Publishable Either Way**:
- Success → RL handles ambiguity
- Failure → RL has fundamental limitation with ambiguity (important negative result)

### Risk 2: Compositional Generalization Fails

**Symptom**: Both RL and MLE fail on held-out compositions

**Mitigation**:
- Simplify composition structure
- Test on easier splits first
- Report partial success (some compositions work)

**Publishable Either Way**:
- Success → Approach enables composition
- Failure → Clear limitation for future work

### Risk 3: Representation Misalignment

**Symptom**: CCA < 0.5 on ambiguous or compositional cases

**Mitigation**:
- Analyze where divergence occurs
- Report layer-wise CCA (might align at some layers)
- Use other similarity metrics (RSA, CKA)

**Publishable Either Way**:
- High CCA → Same representations
- Low CCA → Different but equally effective representations

---

## Decision Points

### After Week 1 (Ambiguous Grammar)

**Check**: Does RL handle ambiguity?
- ✅ If yes → Continue to Week 2
- ❌ If no → Write paper on limitation, report negative result

### After Week 2 (Compositional)

**Check**: Does RL generalize compositionally?
- ✅ If yes → Continue to Week 3
- ❌ If no → Still publishable (2/3 upgrades complete)

### After Week 3 (Representations)

**Check**: Are all three upgrades successful?
- ✅ All 3 → High-impact language paper
- ✅ 2 of 3 → Still strong conceptual paper
- ⚠️ 1 of 3 → Reconsider scope

### After Week 4 (Paper Draft)

**Check**: Is story coherent and compelling?
- ✅ If yes → Submit to top venue (ICLR/NeurIPS)
- ⚠️ If uncertain → Get feedback, iterate

---

## Updated Timeline

### Completed

- ✅ Stage 3.1a: Deterministic grammars (100% accuracy)
- ✅ Stage 3.1b: Stochastic grammars (60% accuracy, CCA=1.000)

### Immediate Next Steps (This Month)

**Week 1 (Starting Now)**:
- [ ] Implement ambiguous grammar
- [ ] Test RL vs MLE on ambiguity
- [ ] Verify policy maintains distribution

**Week 2**:
- [ ] Implement compositional splits
- [ ] Test generalization to held-out compositions
- [ ] RL vs MLE comparison

**Week 3**:
- [ ] CCA on ambiguous contexts
- [ ] CCA on compositional cases
- [ ] Linear probes for structure

**Week 4**:
- [ ] Write paper draft
- [ ] Create figures/tables
- [ ] Get feedback

### Optional (If Time Allows)

**Week 5-6**:
- [ ] Vocabulary scaling (64, 128 tokens)
- [ ] Sequence length scaling (20-30 steps)
- [ ] Additional ablations

---

## What We're NOT Doing (For Now)

### Postponed to Future Work

❌ Stage 3.2 (Grounded Language)
- Requires full language capabilities first
- Too ambitious before proving core claims

❌ Stage 3.3 (LLM Parent)
- Requires grounding first
- Risk of "reward hacking" criticism

❌ English Text
- Stay synthetic until foundations solid
- Avoid "you just added pretraining" critique

❌ Large Vocabularies (>128)
- Not needed for conceptual claims
- Can scale later if needed

### Why This is the Right Scope

**Minimal viable language paper**:
1. Ambiguity ✓
2. Composition ✓
3. Representations ✓

**This is sufficient for high-impact publication**

Grounding and LLM parents are future work (separate paper)

---

## Files to Create/Modify

### New Files

```
src/environment/
├── ambiguous_grammar.py         # Ambiguous grammar task
├── compositional_grammar.py     # Compositional splits

src/training/
├── ambiguous_trainer.py         # Training on ambiguous grammars

src/evaluation/
├── compositionality_eval.py     # Compositional generalization metrics
├── representation_analysis.py   # CCA + linear probes + viz

experiments/
├── run_ambiguous.py            # Ambiguous grammar experiments
├── run_compositional.py        # Compositional generalization
├── run_representation.py       # Representation analysis
```

### Modified Files

```
progress_docs/
├── plan_phase3.md              # Update with new structure
├── phase_3_updates.md          # Document new experiments

paper_drafts/
├── new_paper_phase3.tex        # Rewrite with language framing
```

---

## The Key Insight

**Old Framing** (insufficient):
> "We extend prediction-as-action to discrete tokens"

**New Framing** (correct):
> "We show that RL can learn language structure—ambiguity, composition, and aligned representations—without likelihood supervision"

**Why This Matters**:
- Reviewers care about language phenomena, not just tokens
- Three upgrades provide language phenomena
- Each upgrade is well-motivated in linguistics/NLP
- Together they make a compelling language learning story

---

## Honest Assessment

### What We Have Now

✅ Solid foundations (deterministic + stochastic)
✅ Perfect technical execution (100% accuracy, CCA=1.000)
✅ Clear experimental methodology

### What We're Missing

❌ Language phenomena (ambiguity, composition)
❌ Language framing and claims
❌ Comparison to language learning benchmarks

### The Gap

**Current**: Strong discrete prediction paper
**Target**: Strong language learning paper

**Bridge**: Three mandatory upgrades (4-6 weeks work)

### Realistic Outcome

**Best case**: All three upgrades succeed → high-impact language paper at top venue

**Expected case**: 2/3 upgrades succeed → strong conceptual language paper

**Worst case**: 1/3 succeeds → narrow scope, but still publishable

**All outcomes are acceptable and publishable**

---

## Next Immediate Actions

1. **Read this plan carefully**
2. **Decide**: Do we commit to the language upgrade? (vs publishing what we have as discrete prediction paper)
3. **If yes to upgrade**: Start Week 1 (ambiguous grammar) immediately
4. **If no to upgrade**: Write current results as discrete prediction paper

**Recommendation**: Do the upgrade. It's 4-6 weeks for a much stronger story.

---

*This plan transforms Phase 3 from "discrete token prediction" to "synthetic language learning without likelihood supervision"—a genuinely high-impact claim.*
