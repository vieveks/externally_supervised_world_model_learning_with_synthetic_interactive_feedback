# Phase 3: Language Learning Without Pretraining

## Executive Summary

**Goal**: Identify the minimal interactive conditions under which language and world models can emerge without offline pretraining.

> **Critical Framing**: The goal of Phase 3 is NOT to compete with pretrained LLMs in scale or performance. It is to demonstrate that the *architectural principles* validated in Phase 2b extend to language—proving that pretraining is not fundamentally necessary, even if it remains practically efficient.

Phase 2b established the core insight: **RL gradients can replace MLE when prediction is the action**. Phase 3 extends this to language—the ultimate prediction task.

**The Central Hypothesis**:
> If prediction-as-action works for state dynamics, it should work for language dynamics. The baby predicts the next token(s), the LLM parent judges properties of the output (not likelihood), and RL gradients shape language representations—no pretraining required.

---

## What Phase 2b Proved (Foundation)

| Finding | Implication for Phase 3 |
|---------|------------------------|
| RL achieves 100% prediction accuracy | RL can optimize prediction objectives |
| RL matches MLE on representations (CCA=0.878) | Representations emerge from RL, not just MLE |
| Delayed reward degrades gracefully (100%→35% at 5 steps) | Credit assignment is the key challenge |
| Prediction-as-action unifies learning and using | Language modeling IS prediction-as-action |

---

## The Four Missing Components

Based on your original goal and the guide's analysis, Phase 3 must address four gaps:

### 1. Interactive Text Environment (Not a Dataset)

**Current State**: PredictionTask uses deterministic state transitions.

**Required**: An environment where:
- Agent outputs tokens → environment responds
- Data distribution depends on agent's behavior (policy-dependent)
- No teacher forcing (agent never sees the "correct" answer directly)

**Key Insight**: Pretraining uses static corpora. Babies learn from interactive feedback.

### 2. Multi-Step Linguistic Credit Assignment

**Current State**: Phase 2b showed 35% accuracy at 5-step delay.

**Challenge**: Language requires credit assignment across 10-100+ tokens.

**Solution Approaches**:
- TD(λ) with eligibility traces
- Dense intermediate rewards from LLM critic
- Hierarchical credit assignment (word → phrase → sentence)

### 3. Language-World Grounding Loop

**Current State**: Phase 2b predicts abstract state vectors.

**Required**: Text actions that affect a world state, creating a closed loop:
```
Text prediction → World change → Sensory feedback → New text
```

### 4. Dense Parent Guidance (Not Teacher Forcing)

**Current State**: DeterministicParent provides binary rewards.

**Required**: LLM parent that provides:
- Rich feedback (grammaticality, semantic correctness, pragmatic appropriateness)
- Per-token or per-phrase guidance (dense, not sparse)
- Interactive responses that depend on agent's outputs

> **⚠️ CRITICAL CONSTRAINT**: The LLM parent must act as a **verifier**, not a **teacher**.
>
> **Allowed signals** (properties of child's output):
> - Grammaticality: "Is this well-formed?"
> - Semantic consistency: "Does this make sense given the world state?"
> - Task success: "Did this achieve the communicative goal?"
> - Fluency: "Does this sound natural?"
>
> **Disallowed signals** (implicit teacher forcing):
> - Next-token likelihood: "How likely is this token?"
> - Hidden reference comparison: "How close to what I would say?"
> - Target distribution: "What should come next?"
>
> If we allow likelihood-based feedback, reviewers will correctly argue we've replaced cross-entropy with a noisy proxy for cross-entropy. The parent must judge **output properties**, not **output probability**.

---

## Why This Is NOT "Noisy MLE"

A skeptical reviewer might argue: "If the LLM parent provides feedback correlated with prediction quality, isn't this just MLE with a noisy proxy?"

**The key differences**:

| Property | MLE Pretraining | Our Approach |
|----------|-----------------|--------------|
| **Target signal** | P(correct_token \| context) | Properties(output) given context |
| **Gradient source** | Per-token cross-entropy | Scalar/sparse reward |
| **Teacher forcing** | Yes (sees correct answer) | No (never sees answer) |
| **Data distribution** | Static corpus | Policy-dependent (interactive) |
| **Feedback type** | Full distribution | Binary/scalar properties |

**The crucial distinction**: MLE optimizes *likelihood*—"what is the probability of this specific token?" Our approach optimizes *properties*—"is this output grammatical, consistent, fluent?"

Multiple outputs can satisfy the same properties. "The cat sat on the mat" and "The cat sat on the floor" might both score 1.0 on grammaticality and consistency, even though they have very different likelihoods under MLE.

**Why this matters**: If the baby can learn language by optimizing properties (not likelihood), it proves that the statistical regularity of language can emerge from functional constraints—not just from maximizing probability.

---

## Literature Foundation

### Emergent Language Research
- [Emergent Language Survey (2025)](https://link.springer.com/article/10.1007/s10458-025-09691-y): Comprehensive taxonomy of language emergence in multi-agent RL
- [Language Grounded Multi-Agent RL (NeurIPS 2024)](https://neurips.cc/virtual/2024/poster/96086): Shows language grounding stabilizes emergent communication

### Credit Assignment for Language
- [GRPO-λ: Credit Assignment for LLM Reasoning](https://arxiv.org/html/2510.00194): Eligibility traces for token-level credit assignment
- [Sequence Compression for Credit Assignment](https://arxiv.org/html/2405.03878v2): Chunked-TD for faster credit assignment
- [Sutton & Barto Ch.12: Eligibility Traces](http://www.incompleteideas.net/book/ebook/node72.html): Foundational TD(λ) theory

### LLM as Dense Reward Signal
- [DRLC: Dense Rewards from LLM Critic (EMNLP 2024)](https://arxiv.org/html/2401.07382v1): LLM provides per-token rewards
- [Dense Reward for Free in RLHF](https://arxiv.org/html/2402.00782v1): Using attention weights to densify rewards
- [Text2Reward](https://openreview.net/forum?id=tUM39YTRxH): LLM generates dense reward functions

### Language Grounding Environments
- [SILG Benchmark](https://proceedings.neurips.cc/paper/2021/file/b3e3e393c77e35a4a3f3cbd1e429b5dc-Paper.pdf): Symbolic Interactive Language Grounding
- [BabyAI-Text](https://arxiv.org/html/2302.02662v4): Text-based navigation with procedural tasks
- [Grounded Language in 3D Environments](https://arxiv.org/abs/1706.06551): DeepMind's language grounding work

### Developmental Language Acquisition
- [Communicative Success as Learning Signal (2025)](https://arxiv.org/abs/2505.05970): Developmentally plausible rewards for interactive LMs
- [Trial-and-Demonstration Framework](https://arxiv.org/html/2405.13828): Interactive language learning with corrective feedback
- [Predictive Coding for Speech](https://www.sciencedirect.com/science/article/abs/pii/S016763931630139X): Brain-inspired language acquisition

### Active Inference Connection
- [Free Energy Principle](https://en.wikipedia.org/wiki/Free_energy_principle): Unifies perception, action, and learning
- [Active Inference Process Theory](https://activeinference.github.io/papers/process_theory.pdf): Mathematical framework

---

## Phase 3 Experimental Design

### Overview: Three-Stage Progression

```
Stage 3.1: Token Prediction Task (minimal)
    └─ Prove: RL can learn token dynamics like state dynamics

Stage 3.2: Grounded Language Task (grounded)
    └─ Prove: Language can emerge from world interaction

Stage 3.3: Interactive Parent Task (full)
    └─ Prove: LLM parent enables language acquisition without pretraining
```

---

## Stage 3.1: Token Prediction Task

### Goal
Replicate Phase 2b success on discrete tokens instead of continuous states.

### Why This Stage?
- Direct bridge from Phase 2b (state prediction → token prediction)
- No LLM required yet (deterministic grammar)
- Validates the architecture before adding complexity

### Task Specification

```python
@dataclass
class TokenPredictionTask:
    """
    Prediction-as-Action for discrete tokens.

    Key difference from Phase 2b:
    - Actions are discrete tokens (not continuous vectors)
    - Uses softmax policy (Categorical) instead of Gaussian
    - Reward = correctness of next-token prediction

    This is EXACTLY what LLM pretraining does, but with RL gradients.
    """
    vocab_size: int = 16  # Small vocabulary to start
    sequence_type: str = "deterministic_grammar"  # or "markov", "ngram"

    # Grammar rules (deterministic transitions)
    # Example: A→B→C→A (cyclic), D→E→D (loop), etc.
    grammar: Dict[int, int] = field(default_factory=dict)

    def step(self, predicted_token: int) -> Tuple[int, float, bool]:
        """
        Agent predicts next token, environment judges.

        Returns:
            actual_next: The true next token (revealed after prediction)
            reward: 1.0 if correct, 0.0 otherwise (or shaped)
            done: Episode termination
        """
```

### Architecture Changes

```python
class TokenPredictionModel(nn.Module):
    """
    Like PredictionModel but for discrete tokens.

    Changes from Phase 2b:
    - Output: logits over vocab_size tokens (not continuous state)
    - Distribution: Categorical (not Gaussian)
    - Embedding: Token embedding layer (not state encoder)
    """

    def __init__(self, vocab_size: int, hidden_dim: int = 64, ...):
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.transformer = TransformerCore(...)
        self.prediction_head = nn.Linear(hidden_dim, vocab_size)  # Logits
        self.value_head = ValueHead(hidden_dim)
```

### Experimental Conditions

| Condition | Description | Expected Result |
|-----------|-------------|-----------------|
| RL (REINFORCE) | Policy gradient with prediction reward | Should reach 100% |
| MLE (CE Loss) | Cross-entropy on next token | Baseline (fast convergence) |
| RL + Delay | Reward given after k tokens | Accuracy drops with delay |

### Grammar Types to Test

1. **Deterministic Cyclic** (easiest):
   - Tokens: 0→1→2→3→0→...
   - Equivalent to Phase 2b circular shift

2. **Deterministic Grammar** (medium):
   - Rules: S→AB, A→aA|a, B→bB|b
   - Requires learning structure

3. **Bigram/Trigram** (harder):
   - P(next|current) or P(next|prev2)
   - Stochastic but learnable

4. **Context-Free** (hardest for this stage):
   - Nested dependencies: anbn
   - Requires memory

### Success Criteria for Stage 3.1

- [ ] RL achieves ≥95% accuracy on deterministic grammar
- [ ] RL matches MLE within 5x sample efficiency
- [ ] Representations are similar (CCA > 0.8)
- [ ] Delayed reward shows graceful degradation

### Key Questions This Answers

1. Does discrete action space change anything fundamentally?
2. Can RL learn symbolic transitions as well as continuous ones?
3. What is the credit assignment horizon for token prediction?

---

## Stage 3.2: Grounded Language Task

### Goal
Language actions affect a world state, creating a prediction-action-feedback loop.

### Why This Stage?
- Introduces grounding (language ↔ world)
- Still deterministic (no LLM yet)
- Tests whether language emerges from world interaction

### Task Design: TextWorld Navigation

```python
@dataclass
class GroundedLanguageTask:
    """
    Text commands control a grid world agent.

    State: Grid world (agent position, objects, goal)
    Action: Text command (e.g., "go north", "pick apple")
    Dynamics: Command → world state change
    Observation: Text description of new state

    The twist: Agent must PREDICT the text description.
    Reward = accuracy of prediction.

    This creates a language-world grounding loop:
        Text action → World change → Text observation → Predict next observation
    """

    grid_size: int = 4
    vocab_size: int = 32  # Small controlled vocabulary
    max_command_length: int = 3  # "go north now"

    # World state
    agent_pos: Tuple[int, int]
    objects: Dict[Tuple[int, int], str]

    def execute_command(self, command_tokens: List[int]) -> List[int]:
        """
        Execute text command, return observation tokens.

        This is the grounding function:
            f: (world_state, command) → new_world_state → observation
        """

    def step(self, predicted_observation: List[int]) -> Tuple[List[int], float, bool]:
        """
        Agent predicts what it will observe after command.
        Reward = prediction accuracy.
        """
```

### The Grounding Loop

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Agent sees: "You are at (1,2). Apple is north."           │
│       ↓                                                     │
│  Agent outputs: "go north"                                  │
│       ↓                                                     │
│  World executes: agent moves to (1,3)                       │
│       ↓                                                     │
│  Agent PREDICTS observation: "You are at (1,3). Got apple." │
│       ↓                                                     │
│  Actual observation: "You are at (1,3). Got apple."         │
│       ↓                                                     │
│  Reward: high (prediction matched)                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Architecture: Sequence-to-Sequence Prediction

```python
class GroundedLanguageModel(nn.Module):
    """
    Encoder-decoder for grounded language prediction.

    Input: current observation tokens + command tokens
    Output: predicted next observation tokens

    This is a mini language model with world grounding.
    """

    def __init__(self, vocab_size: int, hidden_dim: int = 128):
        self.encoder = TransformerEncoder(...)
        self.decoder = TransformerDecoder(...)

    def forward(self, obs_tokens, cmd_tokens) -> Distribution:
        """
        Returns distribution over next observation tokens.
        """
```

### Training: Autoregressive Prediction with RL

```python
def generate_prediction(self, obs, cmd, max_len=10):
    """
    Autoregressively generate predicted observation.
    Each token sampled from policy, accumulate log_probs.

    This is RL-based language generation:
    - No teacher forcing (unlike MLE)
    - Reward at end of sequence (credit assignment challenge!)
    """
    predicted_tokens = []
    log_probs = []

    for _ in range(max_len):
        logits = self.model(obs, cmd, predicted_tokens)
        dist = Categorical(logits=logits)
        token = dist.sample()
        predicted_tokens.append(token)
        log_probs.append(dist.log_prob(token))

    return predicted_tokens, sum(log_probs)
```

### Credit Assignment Challenge

**Problem**: Reward is given after entire prediction sequence (5-10 tokens).

**Solution Options**:

1. **Dense per-token reward** (if we had oracle):
   - r_t = 1 if token_t matches target_t
   - Easy but unrealistic

2. **TD(λ) with eligibility traces**:
   - Learned value function V(partial_sequence)
   - Traces propagate reward back

3. **Sequence-level baseline**:
   - REINFORCE with reward - baseline(obs, cmd)
   - High variance but simple

4. **Token-level shaping** (Phase 3.3 adds this):
   - LLM critic provides per-token feedback

### Experimental Conditions

| Condition | Reward Signal | Expected Result |
|-----------|---------------|-----------------|
| Sequence-level | Final match score | Baseline (high variance) |
| Per-token dense | Per-position match | Upper bound (oracle) |
| TD(λ) learned | Value + traces | Should approach dense |
| MLE baseline | Cross-entropy loss | Reference (fast) |

### Success Criteria for Stage 3.2

- [ ] Agent learns correct command→observation mappings
- [ ] Prediction accuracy > 70% on held-out commands
- [ ] TD(λ) improves over sequence-level reward
- [ ] Some language structure emerges (basic compositionality)

### ⚠️ Expected Difficulty

Stage 3.2 combines several hard problems simultaneously:
- Sequence generation (autoregressive)
- Delayed reward (end of sequence)
- Grounding (language ↔ world)
- No teacher forcing

**Realistic expectations**:
- Learning will be slow and high-variance
- Clean convergence is unlikely
- Multiple runs with different seeds will be needed
- Some configurations may fail entirely

This is acceptable for a research paper—we are probing the limits of the approach, not claiming production-ready results.

---

## Stage 3.3: Interactive LLM Parent Task

### Goal
LLM parent provides dense, interactive feedback—enabling language acquisition without pretraining.

### Why This Stage?
- The full hypothesis test
- LLM replaces oracle with realistic feedback
- Interactive (policy-dependent) data generation

### The LLM Parent Architecture

```python
class LLMParent:
    """
    An LLM that acts as a VERIFIER (not teacher) for the baby model.

    ⚠️ CRITICAL DESIGN PRINCIPLE:
    The parent judges PROPERTIES of the child's output, NOT likelihood.
    The parent NEVER has access to a "correct" reference answer.

    Key properties:
    1. Interactive: Responds to baby's outputs
    2. Dense: Provides per-token or per-phrase feedback
    3. Non-teacher-forcing: Never gives the "correct" answer
    4. Property-based: Judges grammaticality, consistency, success—NOT probability

    ALLOWED feedback signals:
    - Grammaticality: "Is this well-formed English?"
    - Semantic consistency: "Does this make sense given the world?"
    - Task success: "Did this achieve the goal?"
    - Fluency: "Does this sound natural?"

    DISALLOWED feedback signals:
    - Likelihood: "How probable is this continuation?"
    - Reference comparison: "How close to the 'right' answer?"
    - Target distribution: "What should come next?"

    This is fundamentally different from pretraining:
    - Pretraining: optimizes P(target | context)
    - LLM Parent: optimizes properties(output) given context
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()

    def score_prediction(
        self,
        context: str,
        prediction: str,
        world_state: Dict = None  # For grounded tasks
    ) -> Dict[str, float]:
        """
        Score the baby's prediction on PROPERTY dimensions.

        Returns:
            grammaticality: Is it grammatical? (0-1)
            semantic: Does it make sense given world state? (0-1)
            task_success: Does it achieve the communicative goal? (0-1)
            fluency: Does it sound natural? (0-1)

        NOTE: We explicitly DO NOT score:
            - likelihood: How probable is this?
            - correctness: Does it match a hidden reference?

        The parent judges OUTPUT PROPERTIES, not OUTPUT PROBABILITY.
        """

    def provide_feedback(
        self,
        context: str,
        prediction: str
    ) -> str:
        """
        Provide corrective feedback (not the answer).

        Example (ALLOWED):
            Context: "The cat sat on the ___"
            Prediction: "floor"
            Feedback: "That's grammatical and makes sense!"
            (Even if "mat" was the 'expected' answer)

        Example (DISALLOWED):
            Feedback: "The answer should be 'mat'"
            Feedback: "That's unlikely given the context"
        """
```

### Three Feedback Modes

#### Mode 1: Score-Only (Minimal)
```python
def score_only(self, context: str, prediction: str, world_state: Dict = None) -> float:
    """
    Just a scalar score. Most like Phase 2b.
    Tests: Can RL work with LLM-generated rewards?

    ⚠️ NOTE: We score PROPERTIES, not LIKELIHOOD.
    """
    prompt = f"""
    Context: {context}
    Generated text: {prediction}
    {"World state: " + str(world_state) if world_state else ""}

    Score this text from 0-1 based on these PROPERTIES:
    - Is it grammatically correct?
    - Does it make sense given the context/world?
    - Does it sound natural and fluent?

    DO NOT score based on:
    - How likely or probable this specific text is
    - Whether it matches some expected answer

    Return only a number between 0 and 1.
    """
    return float(self.query(prompt))
```

#### Mode 2: Dense Token Scores (DRLC-style)
```python
def dense_scores(self, context: str, prediction: str, world_state: Dict = None) -> List[float]:
    """
    Per-token PROPERTY scores from LLM critic.
    Based on DRLC (Dense Rewards from LLM Critic).

    ⚠️ NOTE: Scores reflect grammaticality/consistency at each position,
    NOT token probability or likelihood.
    """
    prompt = f"""
    Context: {context}
    Generated text: {prediction}
    {"World state: " + str(world_state) if world_state else ""}

    For each token, rate how well it maintains these properties (0-1):
    - Grammatical correctness up to this point
    - Semantic consistency with context/world
    - Natural flow and fluency

    DO NOT rate based on probability or how "expected" each token is.
    Rate based on whether the text WORKS, not whether it's LIKELY.

    Format: token1:score1, token2:score2, ...
    """
    # Parse response into per-token scores
```

#### Mode 3: Interactive Dialogue (Full)
```python
def interactive_dialogue(
    self,
    context: str,
    baby_utterance: str
) -> Tuple[str, float]:
    """
    Full interactive mode: Parent responds AND scores.

    This creates a dialogue loop:
        Baby: "I want go store"
        Parent: "You want to go TO the store? Okay, let's go!"
        (Implicit correction, positive reinforcement)
    """
    prompt = f"""
    You are a patient parent teaching a child to speak.
    The child said: "{baby_utterance}"
    Context: {context}

    Respond naturally, gently correcting any errors by modeling
    correct speech. Do NOT explicitly point out mistakes.
    Also rate the child's utterance quality (0-1).

    Format:
    Response: [your response]
    Score: [0-1]
    """
```

### Task Design: Language Game

```python
@dataclass
class LanguageGameTask:
    """
    The baby plays language games with the LLM parent.

    Games:
    1. Sentence completion: Parent gives context, baby completes
    2. Description: Parent shows scene, baby describes
    3. Question answering: Parent asks, baby answers
    4. Dialogue: Open-ended conversation

    All games reward PREDICTION accuracy:
    - Baby predicts what a competent speaker would say
    - Parent judges how close it is (without giving answer)
    """

    game_type: str = "completion"
    vocab_size: int = 1000  # Or use subword tokenizer
    max_length: int = 20

    def generate_episode(self) -> Dict:
        """Generate a game instance."""
        if self.game_type == "completion":
            return self._generate_completion()
        elif self.game_type == "description":
            return self._generate_description()
        # ...

    def _generate_completion(self) -> Dict:
        """
        Use LLM to generate a completion task.

        The parent creates a prompt, and the baby must predict
        a reasonable completion. Parent scores (doesn't solve).
        """
        context = self.parent.generate_context()
        return {"context": context, "game": "completion"}
```

### Reward Shaping with LLM

```python
class LLMRewardShaper:
    """
    Shapes sparse completion rewards into dense token rewards.

    Based on:
    - DRLC (Dense Rewards from LLM Critic)
    - Text2Reward (LLM-generated reward functions)
    """

    def shape_rewards(
        self,
        tokens: List[int],
        final_score: float
    ) -> List[float]:
        """
        Distribute final score across tokens.

        Options:
        1. Uniform: r_t = final_score / len(tokens)
        2. LLM-weighted: Ask LLM which tokens contributed most
        3. Attention-based: Use parent's attention as weights
        """
```

### The Full Training Loop

```python
def train_step_interactive(self):
    """
    One step of interactive language learning.

    Key differences from pretraining:
    1. No teacher forcing
    2. Reward from LLM parent (not cross-entropy)
    3. Data is policy-dependent
    4. Dense feedback from LLM critic
    """

    # 1. Parent generates context
    context = self.parent.generate_context()

    # 2. Baby generates prediction (no teacher forcing!)
    predicted_tokens, log_probs = self.baby.generate(context)

    # 3. Parent provides feedback
    if self.feedback_mode == "score_only":
        reward = self.parent.score_only(context, predicted_tokens)
        rewards = [reward]  # Sparse
    elif self.feedback_mode == "dense":
        rewards = self.parent.dense_scores(context, predicted_tokens)
    elif self.feedback_mode == "interactive":
        response, reward = self.parent.interactive_dialogue(
            context, predicted_tokens
        )
        rewards = [reward]  # Could be densified
        # Response becomes part of next context!

    # 4. RL update
    loss = self.compute_policy_gradient(log_probs, rewards)
    loss.backward()
    self.optimizer.step()
```

### Curriculum Design

```python
class LanguageCurriculum:
    """
    Progressive difficulty for language learning.

    Inspired by:
    - BabyLM challenge (limited data)
    - Developmental linguistics
    - Curriculum learning research
    """

    levels = [
        # Level 0: Single word prediction
        {"task": "completion", "context_len": 1, "pred_len": 1},

        # Level 1: Short phrase completion
        {"task": "completion", "context_len": 3, "pred_len": 2},

        # Level 2: Sentence completion
        {"task": "completion", "context_len": 5, "pred_len": 5},

        # Level 3: Description (grounded)
        {"task": "description", "scene_complexity": "simple"},

        # Level 4: Question answering
        {"task": "qa", "question_type": "factual"},

        # Level 5: Dialogue
        {"task": "dialogue", "turns": 2},
    ]

    def advance(self, success_rate: float, threshold: float = 0.8):
        """Advance to next level if mastery achieved."""
```

### Experimental Conditions

| Condition | Feedback Type | Credit Assignment | Expected |
|-----------|---------------|-------------------|----------|
| Sparse LLM | Score at end | Sequence REINFORCE | Baseline |
| Dense LLM | Per-token scores | Token REINFORCE | Better |
| TD(λ) + LLM | Score + traces | Eligibility traces | Best? |
| Interactive | Dialogue + score | Response as context | Novel |
| MLE Oracle | Cross-entropy | Direct | Upper bound |

### Success Criteria for Stage 3.3

- [ ] Baby learns to generate grammatical sentences (perplexity improves)
- [ ] Performance scales with curriculum level
- [ ] Dense feedback outperforms sparse
- [ ] Qualitative: Baby's outputs become more human-like over training
- [ ] Representations learned are useful for downstream tasks

---

## Credit Assignment: Deep Dive

### The Core Challenge

Phase 2b showed 35% accuracy at 5-step delay. Language requires credit assignment over 10-100+ tokens. This section details solutions.

### Solution 1: TD(λ) with Eligibility Traces

```python
class TDLambdaLanguage:
    """
    TD(λ) for language prediction.

    Key insight from GRPO-λ paper:
    - Eligibility traces can be computed from log-probs
    - No explicit critic needed (critic-free TD)
    """

    def __init__(self, lambda_: float = 0.9, gamma: float = 0.99):
        self.lambda_ = lambda_
        self.gamma = gamma

    def compute_lambda_returns(
        self,
        rewards: List[float],
        values: List[float]
    ) -> List[float]:
        """
        Compute λ-returns for credit assignment.

        G_t^λ = (1-λ) Σ_{n=1}^∞ λ^{n-1} G_t:t+n

        where G_t:t+n is n-step return.
        """
        T = len(rewards)
        lambda_returns = [0.0] * T

        # Backward pass
        G = 0
        for t in reversed(range(T)):
            G = rewards[t] + self.gamma * (
                self.lambda_ * G + (1 - self.lambda_) * values[t+1]
                if t+1 < T else 0
            )
            lambda_returns[t] = G

        return lambda_returns
```

### Solution 2: Dense Rewards from LLM Critic (DRLC)

```python
class DRLCCritic:
    """
    Dense Rewards from LLM Critic.

    Based on EMNLP 2024 paper.
    LLM provides per-token feedback.
    """

    def __init__(self, critic_model: str = "gpt-4o-mini"):
        self.critic = OpenAI(model=critic_model)

    def get_dense_rewards(
        self,
        context: str,
        generated: List[str]
    ) -> List[float]:
        """
        Ask LLM to rate each token's contribution.

        Prompt engineering is crucial here.
        """
        prompt = f"""
        Context: {context}
        Generated text: {' '.join(generated)}

        Rate each generated token from 0-1 based on how appropriate
        it is given the context and preceding tokens.
        Consider: grammar, meaning, coherence, relevance.

        Format: token1:0.8, token2:0.6, token3:0.9, ...
        """
        response = self.critic.complete(prompt)
        return self._parse_scores(response, len(generated))
```

### Solution 3: Attention-Based Reward Redistribution

```python
class AttentionRewardRedistribution:
    """
    Use parent model's attention to redistribute rewards.

    Based on "Dense Reward for Free in RLHF" paper.

    Intuition: Tokens the parent "attended to" when scoring
    are likely the important ones.
    """

    def redistribute(
        self,
        final_reward: float,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Redistribute final reward using attention.

        attention_weights: (num_tokens,) from parent's last layer
        """
        # Normalize attention to sum to 1
        weights = attention_weights / attention_weights.sum()

        # Redistribute reward
        token_rewards = final_reward * weights

        return token_rewards
```

### Solution 4: Hierarchical Credit Assignment

```python
class HierarchicalCredit:
    """
    Credit assignment at multiple levels:
    - Token level
    - Word level
    - Phrase level
    - Sentence level

    Intuition: Some rewards apply to words, others to phrases.
    """

    def assign_hierarchical(
        self,
        tokens: List[int],
        word_boundaries: List[int],
        phrase_boundaries: List[int],
        rewards: Dict[str, float]  # {"word": w, "phrase": p, "sent": s}
    ) -> List[float]:
        """
        Combine multi-level rewards into per-token rewards.
        """
        token_rewards = [0.0] * len(tokens)

        # Sentence-level reward distributed uniformly
        sent_r = rewards.get("sent", 0) / len(tokens)

        # Word-level reward distributed within words
        for start, end in zip(word_boundaries[:-1], word_boundaries[1:]):
            word_r = rewards.get("word", 0) / (end - start)
            for i in range(start, end):
                token_rewards[i] += sent_r + word_r

        return token_rewards
```

---

## Implementation Plan

### Directory Structure

```
src/
├── environment/
│   ├── token_prediction.py      # Stage 3.1
│   ├── grounded_language.py     # Stage 3.2
│   └── language_game.py         # Stage 3.3
├── models/
│   ├── token_model.py           # Discrete token prediction
│   ├── seq2seq_model.py         # Grounded language
│   └── language_model.py        # Full language model
├── training/
│   ├── token_reinforce.py       # Stage 3.1 training
│   ├── grounded_reinforce.py    # Stage 3.2 training
│   ├── language_reinforce.py    # Stage 3.3 training
│   └── td_lambda.py             # Credit assignment
├── parent/
│   ├── llm_parent.py            # LLM interaction
│   ├── reward_shaper.py         # Dense reward from LLM
│   └── curriculum.py            # Curriculum manager
└── evaluation/
    ├── language_metrics.py      # Perplexity, BLEU, etc.
    └── representation_probe.py  # Linear probes
```

### Phase 3 Timeline

| Stage | Description | Key Deliverable |
|-------|-------------|-----------------|
| 3.1a | Token prediction model | TokenPredictionTask working |
| 3.1b | RL vs MLE comparison | Accuracy parity demonstrated |
| 3.1c | Credit assignment ablation | TD(λ) improvement quantified |
| 3.2a | Grounded environment | TextWorld navigation task |
| 3.2b | Seq2seq prediction | Language grounding demonstrated |
| 3.2c | Compositional generalization | Novel command handling |
| 3.3a | LLM parent integration | Score-only feedback working |
| 3.3b | Dense feedback | DRLC-style rewards working |
| 3.3c | Interactive training | Full dialogue loop |
| 3.3d | Curriculum learning | Progressive difficulty |
| Paper | Write results | Phase 3 paper draft |

---

## Evaluation Metrics

### Prediction Accuracy
- **Token accuracy**: % of correctly predicted tokens
- **Sequence accuracy**: % of fully correct sequences
- **Perplexity**: Cross-entropy on held-out data

### Language Quality
- **Grammaticality**: LLM-judged or parser-based
- **Semantic coherence**: Embedding similarity to references
- **Fluency**: LLM rating or human evaluation

### Representation Quality
- **Linear probes**: Can we decode linguistic features?
- **CCA similarity**: RL vs MLE representations
- **Transfer**: Performance on downstream tasks

### Credit Assignment Effectiveness
- **Delay tolerance**: Accuracy vs reward delay curve
- **Variance reduction**: TD(λ) vs Monte Carlo

---

## Risk Analysis

| Risk | Mitigation |
|------|------------|
| LLM API costs | Start with small vocab, cache responses |
| Credit assignment fails at scale | Implement TD(λ), dense rewards early |
| LLM feedback too noisy | Calibrate prompts, ensemble scores |
| Curriculum too aggressive | Conservative advancement thresholds |
| Baby model too small | Scale up if needed (but start small) |
| Emergent language is "code" not language | Use LLM to enforce naturalness |

---

## Questions to Resolve

### Before Starting

1. **Tokenization**: Character-level, subword (BPE), or word-level?
   - Recommendation: Start with small word vocab, move to subword

2. **LLM choice**: GPT-4o-mini, Claude Haiku, or local model?
   - Recommendation: GPT-4o-mini for cost/quality balance

3. **Grounding environment**: Custom TextWorld or existing (BabyAI)?
   - Recommendation: Custom for control, inspired by BabyAI

4. **Evaluation**: How to measure "language acquisition"?
   - Recommendation: Multiple metrics (accuracy, perplexity, human eval)

### Design Decisions

5. **Dense vs sparse feedback**: Which to prioritize?
   - Recommendation: Start sparse, add dense as needed

6. **Autoregressive vs parallel**: Token generation style?
   - Recommendation: Autoregressive (more realistic)

7. **Curriculum**: Predefined levels or adaptive?
   - Recommendation: Predefined with mastery gates

---

## Connection to Original Goal

From your initial conversation:

> "What would be required to show that an LLM as a dense guide for a baby-like architecture can learn language and world model like LLMs without pretraining?"

**Phase 3 answers this directly**:

1. **LLM as dense guide**: Stages 3.3a-c implement LLM parent providing feedback
2. **Baby-like architecture**: Our PredictionModel extended to language
3. **Learn language**: Token prediction → grounded language → dialogue
4. **World model**: Stage 3.2 grounds language in world dynamics
5. **Without pretraining**: RL gradients only, no MLE pretraining phase

**The key insight remains**:
> Prediction-as-action unifies learning and using. When the baby's "action" is predicting language, and the "reward" is prediction accuracy judged by a parent, language acquisition emerges from interaction—not from pretraining on static corpora.

---

## Summary: Phase 3 Success Criteria (Updated 2026-01-08)

### ✅ Stage 3.1a: COMPLETED - Deterministic Token Prediction
**Status**: All experiments passed with outstanding results

**Achievements**:
- ✅ RL matches MLE: Both 100% accuracy on deterministic grammars
- ✅ Credit assignment: 100% at 7-step delay (far exceeding 35% at 5-step for continuous)
- ✅ Vanilla REINFORCE suffices for deterministic grammars
- ✅ Discrete action spaces show superior credit assignment vs continuous

**Key Insight**: The challenge is NOT credit assignment for deterministic sequences—it's stochasticity and partial observability.

### ⬜ Stage 3.1b: NEXT - Stochastic Token Prediction
**Goal**: Test whether RL handles probabilistic token transitions

**Tasks**:
- [ ] Implement bigram grammar: $P(x_{t+1}|x_t)$ with learned transition matrix
- [ ] Test RL vs MLE on stochastic transitions
- [ ] Measure credit assignment degradation with stochasticity
- [ ] CCA analysis: Do RL and MLE learn similar representations?

**Success Criteria**:
- RL achieves ≥80% accuracy on stochastic grammar (MLE baseline for comparison)
- Credit assignment maintains >70% at 5-step delay
- CCA similarity (RL vs MLE) > 0.7

**Why This Matters**: Stochasticity is the ACTUAL test—determinism was too easy. This determines if the approach extends beyond toy problems.

### ⬜ Stage 3.1c: Vocabulary Scaling
**Goal**: Show results aren't artifact of small vocab

**Tasks**:
- [ ] Scale vocabulary to 64 tokens
- [ ] Scale vocabulary to 128 tokens
- [ ] Test if convergence degrades with vocab size

**Success Criteria**:
- RL maintains ≥90% accuracy at 64-token vocab
- RL maintains ≥80% accuracy at 128-token vocab
- Sample efficiency gap remains <10× vs MLE

### ⬜ Stage 3.2: Grounded Language Task
**Goal**: Tokens cause world-state changes (language grounding)

**Critical Dependency**: Must complete 3.1b (stochastic) first

**Tasks**:
- [ ] Design command→observation task (e.g., "move left" changes grid position)
- [ ] Implement sequence generation with world grounding
- [ ] Test RL on grounded prediction task

**Success Criteria**:
- Agent learns correct command→observation mappings (>70% accuracy)
- Some compositional structure emerges (test on novel combinations)
- TD(λ) helps with multi-token commands (vs vanilla REINFORCE)

**Expected Difficulty**: High—this combines sequence generation + delayed reward + grounding + no teacher forcing

### ⬜ Stage 3.3: Interactive LLM Parent Task
**Goal**: Property-based feedback from LLM (not likelihood-based)

**Critical Dependency**: Must complete 3.2 first

**Tasks**:
- [ ] Implement LLM verifier (judges grammaticality, consistency, fluency)
- [ ] Test score-only feedback mode
- [ ] Test dense per-token feedback mode
- [ ] Compare LLM feedback vs oracle feedback

**Success Criteria**:
- Baby generates grammatical sentences (>80% LLM-judged)
- Property-based feedback enables learning (without likelihood signals)
- Performance approaches oracle-feedback baseline

### What Phase 3.1 Results Changed

**Original Concern**: Credit assignment would be a major bottleneck
**Reality**: Credit assignment is EASY for deterministic sequences, even with 7-step delay

**Original Plan**: TD(λ) would be necessary
**Reality**: Vanilla REINFORCE suffices for deterministic grammars

**Original Expectation**: Discrete tokens would be similar difficulty to continuous
**Reality**: Discrete tokens are MUCH EASIER (100% vs 35% at comparable delays)

**Updated Focus**: The real challenges are:
1. **Stochasticity** (not determinism)
2. **Partial observability** (not fully observable)
3. **Compositionality** (not memorization)
4. **Scale** (not toy problems)

### Revised Timeline

**Immediate (Stage 3.1b - Stochastic)**:
- [ ] Week 1: Implement bigram grammar
- [ ] Week 2: Run stochastic experiments
- [ ] Week 3: CCA analysis
- [ ] **Decision point**: If stochastic fails badly, stop here and write paper on deterministic results only

**Short-term (Stage 3.1c - Scale)**:
- [ ] Week 4-5: Vocabulary scaling experiments
- [ ] Write Phase 3.1 paper (deterministic + stochastic + scale)

**Medium-term (Stage 3.2 - Grounding)**:
- Only if 3.1b succeeds
- 2-3 weeks implementation + experiments

**Long-term (Stage 3.3 - LLM Parent)**:
- Only if 3.2 succeeds
- 3-4 weeks implementation + experiments

### What Would Falsify the Hypothesis?

**Already falsified**:
- ❌ "RL fundamentally can't do credit assignment for sequences" - FALSIFIED (100% at 7-step)
- ❌ "Discrete tokens are as hard as continuous" - FALSIFIED (discrete is much easier)

**Still testable**:
- ⚠️ "RL can handle stochastic token transitions" - PENDING (Stage 3.1b)
- ⚠️ "Property-based feedback suffices (no likelihood needed)" - PENDING (Stage 3.3)
- ⚠️ "Approach scales beyond toy vocabularies" - PENDING (Stage 3.1c)

### Paper Strategy

**Paper 1 (Already Submitted)**: Prediction-as-Action for continuous states
- Status: Under review at TMLR
- Do NOT touch until reviews come back

**Paper 2 (Phase 3.1 - This Work)**: Extension to discrete tokens
- Title: "From States to Tokens: Prediction-as-Action Scales to Discrete Sequential Prediction"
- Scope: Deterministic grammars ONLY (+ stochastic if 3.1b succeeds)
- Target: TMLR or ICLR
- Timeline: Write after 3.1b/3.1c complete

**Paper 3 (Phase 3.2/3.3 - Future)**: Grounded language + LLM parent
- Only if 3.1b succeeds AND we proceed to 3.2/3.3
- Much more ambitious scope
- Target: NeurIPS or ICLR

---

## Next Immediate Steps

1. **Implement stochastic bigram grammar** (Stage 3.1b)
2. **Run stochastic experiments** (RL vs MLE, delay ablation)
3. **CCA analysis** (RL vs MLE representations)
4. **Decision point**: Continue to grounding OR write paper on what we have
5. **Integrate LLM parent**: Stage 3.3 infrastructure
6. **Run full experiments**: Collect results for paper

---

*Phase 3 represents the culmination of the original vision: proving that language can emerge from interaction, not pretraining. The foundation from Phase 2b is solid. Now we scale to language.*
