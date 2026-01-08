# WMIL Implementation Plan

> **Goal**: Prove that a randomly initialized transformer can learn non-random, goal-directed behavior through interaction with an LLM-powered environment.

---

## ⚠️ CRITICAL FIXES (from review)

Before implementing, these issues MUST be addressed:

### Fix 1: Episode Termination (DANGEROUS)
**Problem**: `done = r_goal > 0.9` creates shortcut where agent optimizes for early termination, not understanding.

**Solution**: Episodes end ONLY on time limit. Log success separately.
```python
# WRONG
done = self.step_count >= self.max_steps or r_goal > 0.9

# RIGHT
done = self.step_count >= self.max_steps
info['success'] = r_goal > 0.9  # Track separately
```

### Fix 2: World-Model Prediction Target (DANGEROUS)
**Problem**: Early in training, encoder is random → embeddings drift → world-model learns noise.

**Solution**: For MVP, predict RAW STATE TENSOR, not embedding.
```python
# WRONG (initially)
actual_next = self.model.get_state_embedding(next_state)
loss = MSE(predicted, actual_next)

# RIGHT (for MVP)
actual_next = next_state.to_tensor()  # Raw state
loss = MSE(predicted, actual_next)

# LATER: Switch to embedding-space once behavior stabilizes
```

### Fix 3: Start with REINFORCE, Not PPO
**Problem**: PPO introduces GAE bugs, advantage normalization, credit assignment confusion.

**Solution**: REINFORCE + baseline first. Switch to PPO after learning is confirmed.
```python
# Phase 1: REINFORCE
loss = -log_prob * (reward - baseline)

# Phase 2: PPO (after learning works)
# ... full clipped surrogate
```

### Fix 4: Log Reward Components Separately
**Problem**: Collapsing to scalar immediately loses debugging signal.

**Solution**: Log each component, verify goal improves without sacrificing prediction.
```python
logger.log({
    'reward/goal': reward.goal,
    'reward/pred': reward.pred,
    'reward/total': reward.scalar(),
})
```

### Fix 5: NO LLM Until Deterministic Works
**Problem**: LLM adds latency, nondeterminism, confounds.

**Solution**: LLM integration is Phase 3+. MVP uses ONLY deterministic parent.

---

## Phase 0: Project Setup
**Status**: 🔄 In Progress
**Priority**: HIGH
**Estimated Files**: 3-4

### Objectives
- Set up Python environment with all dependencies
- Create directory structure
- Configure experiment tracking

### Tasks

#### 0.1 Create Directory Structure
```
src/
├── __init__.py
├── environment/
│   ├── __init__.py
│   ├── symbolic_env.py
│   ├── tasks.py
│   └── parent_llm.py
├── models/
│   ├── __init__.py
│   ├── baby_model.py
│   ├── heads.py
│   └── encoder.py
├── training/
│   ├── __init__.py
│   ├── buffer.py
│   ├── ppo.py
│   ├── rewards.py
│   └── trainer.py
├── curriculum/
│   ├── __init__.py
│   └── curriculum.py
└── utils/
    ├── __init__.py
    ├── logging.py
    └── config.py
configs/
└── default.yaml
experiments/
└── .gitkeep
tests/
└── .gitkeep
```

#### 0.2 Dependencies (requirements.txt)
```
torch>=2.0.0
numpy
pyyaml
wandb  # or tensorboard
tqdm
pytest
# For LLM parent:
openai  # or anthropic, or transformers for local
```

#### 0.3 Config System
- Create `configs/default.yaml` with all hyperparameters
- Create `src/utils/config.py` to load/validate configs

### Deliverables
- [ ] All directories created with `__init__.py` files
- [ ] `requirements.txt` with pinned versions
- [ ] `configs/default.yaml` with sensible defaults
- [ ] Config loader utility

### Exit Criteria
- `pip install -r requirements.txt` works
- `python -c "from src import *"` works

---

## Phase 1: Symbolic Environment
**Status**: ⬜ Not Started
**Priority**: CRITICAL
**Estimated Files**: 4

### Objectives
- Create a minimal environment where learning is possible but non-trivial
- Integrate LLM as environment dynamics/reward oracle
- Define 3 simple tasks with clear success criteria

### Tasks

#### 1.1 State Representation (`src/environment/symbolic_env.py`)

**State Structure**:
```python
@dataclass
class State:
    # Example: {"shape": 2, "color": 1, "position": [0, 1], "sequence": [3, 1, 4]}
    attributes: Dict[str, Any]

    def to_tensor(self) -> torch.Tensor:
        """Convert to fixed-size tensor for model input"""
        pass

    def to_prompt(self) -> str:
        """Convert to text for LLM parent"""
        pass
```

**Design Decisions**:
- Use integer indices for categorical attributes (shape: 0-4, color: 0-7)
- Fixed-size tensor representation (pad if needed)
- States must be comparable (for world-model prediction loss)

#### 1.2 Action Space

**Action Design**:
```python
NUM_ACTIONS = 16  # Start small

class ActionSpace:
    def __init__(self, size: int = 16):
        self.size = size
        # Actions are just integers 0..size-1
        # NO semantic meaning attached initially

    def sample(self) -> int:
        return random.randint(0, self.size - 1)
```

**Critical**: Actions are arbitrary symbols. The baby must LEARN that action 7 means "select red" in one context.

#### 1.3 Task Definitions (`src/environment/tasks.py`)

**Task 1: Pattern Echo**
```python
class PatternEchoTask:
    """
    State: {"target": 5}
    Goal: Output action == target
    Reward: 1.0 if correct, 0.0 otherwise

    This is the simplest possible task - can the baby learn
    to output the number it sees?
    """
```

**Task 2: Sequence Next**
```python
class SequenceNextTask:
    """
    State: {"sequence": [2, 4, 6, ?]}
    Goal: Output the next number in pattern
    Reward: 1.0 if correct, 0.0 otherwise

    Requires: Learning arithmetic patterns
    """
```

**Task 3: Conditional Response**
```python
class ConditionalTask:
    """
    State: {"shape": "circle", "color": "red"}
    Rule: If shape==circle -> action 0-3
          If shape==square -> action 4-7
          If color==red -> add 8 to action

    Goal: Output correct action for state
    Reward: 1.0 if correct, 0.0 otherwise

    Requires: Learning conditional logic
    """
```

#### 1.4 LLM Parent Integration (`src/environment/parent_llm.py`)

**Parent Responsibilities**:
1. **State Transitions**: Given (state, action), produce next_state
2. **Reward Signals**: Judge if action was appropriate
3. **Task Generation**: Create new task instances at given difficulty

```python
class ParentLLM:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()  # or Anthropic, or local

    def get_transition(self, state: State, action: int, task: Task) -> State:
        """
        Ask LLM: "Given this state and action, what's the next state?"
        For simple tasks, this might be deterministic.
        """
        pass

    def get_reward(self, state: State, action: int, task: Task) -> float:
        """
        Ask LLM: "Was this action correct for this state and task?"
        Returns: 0.0 to 1.0

        CRITICAL: Do NOT return token probabilities or explanations
        that could be used for gradient computation.
        """
        pass

    def generate_task_instance(self, task_type: str, difficulty: int) -> Tuple[State, Task]:
        """Generate a new problem at given difficulty level"""
        pass
```

**LLM-Free Fallback**:
For faster iteration, implement deterministic versions of each task that don't require LLM calls:
```python
class DeterministicParent:
    """Rule-based environment for debugging/fast iteration"""
    pass
```

#### 1.5 Environment Wrapper

```python
class SymbolicEnv:
    def __init__(self, task: Task, parent: DeterministicParent, max_steps: int = 1):
        self.task = task
        self.parent = parent
        self.max_steps = max_steps  # For MVP: T=1 (single-step)

    def reset(self, difficulty: int = 0) -> State:
        """Start new episode"""
        self.state, self.task_instance = self.parent.generate_task_instance(
            self.task, difficulty
        )
        self.step_count = 0
        self.episode_success = False
        return self.state

    def step(self, action: int) -> Tuple[State, RewardVector, bool, dict]:
        """
        Returns:
            next_state: New state after action
            reward: Vector [r_goal]
            done: Episode finished? (TIME LIMIT ONLY - Fix #1)
            info: Debug info including success flag
        """
        next_state = self.parent.get_transition(self.state, action, self.task_instance)
        r_goal = self.parent.get_reward(self.state, action, self.task_instance)

        self.state = next_state
        self.step_count += 1

        # FIX #1: Done ONLY on time limit, NOT on success
        done = self.step_count >= self.max_steps

        # Track success separately for logging
        success = r_goal > 0.9
        self.episode_success = self.episode_success or success

        info = {
            'success': success,
            'episode_success': self.episode_success,
            'step': self.step_count
        }

        return next_state, RewardVector(goal=r_goal), done, info
```

### Deliverables
- [ ] `State` class with tensor/prompt conversion
- [ ] `ActionSpace` class (16 discrete actions)
- [ ] 3 task implementations (PatternEcho, SequenceNext, Conditional)
- [ ] `ParentLLM` wrapper (with LLM-free fallback)
- [ ] `SymbolicEnv` gym-like wrapper
- [ ] Unit tests for each task

### Exit Criteria
- Can run episodes with random policy
- Reward signals make sense (random policy gets ~0, oracle gets 1.0)
- Environment runs fast enough (<100ms per step with deterministic parent)

---

## Phase 2: Baby Model Architecture
**Status**: ⬜ Not Started
**Priority**: CRITICAL
**Estimated Files**: 4

### Objectives
- Build minimal transformer from scratch (no pretrained anything)
- Implement all heads (action, world-model, value)
- Ensure architecture is small enough to train quickly

### Tasks

#### 2.1 State Encoder (`src/models/encoder.py`)

```python
class StateEncoder(nn.Module):
    """
    Converts discrete state attributes to continuous embeddings.

    Input: State tensor (batch, state_dim)
    Output: Embedded state (batch, hidden_dim)
    """
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        # Option 1: Simple MLP
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Option 2: Separate embeddings per attribute
        # self.shape_embed = nn.Embedding(num_shapes, embed_dim)
        # self.color_embed = nn.Embedding(num_colors, embed_dim)
        # ...

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state)
```

#### 2.2 Transformer Core (`src/models/baby_model.py`)

```python
class TransformerCore(nn.Module):
    """
    Minimal transformer backbone.

    - No pretrained weights
    - Small: 2-4 layers, 128-256 hidden
    - Standard architecture (attention + FFN)
    """
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_dim)
        return self.transformer(x)
```

#### 2.3 Heads (`src/models/heads.py`)

```python
class ActionHead(nn.Module):
    """Policy head: outputs distribution over actions"""
    def __init__(self, hidden_dim: int, num_actions: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, num_actions)

    def forward(self, h: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.fc(h)
        return Categorical(logits=logits)


class WorldModelHead(nn.Module):
    """
    Predicts next state given current state and action.

    FIX #2: For MVP, predict RAW STATE TENSOR, not embedding.
    This avoids the "drifting encoder" problem early in training.
    """
    def __init__(self, hidden_dim: int, action_dim: int, state_dim: int):
        super().__init__()
        self.action_embed = nn.Embedding(action_dim, hidden_dim)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)  # Output: raw state dim, NOT hidden_dim
        )

    def forward(self, h: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        h: current state embedding (batch, hidden_dim)
        action: action taken (batch,)
        returns: predicted next state as RAW TENSOR (batch, state_dim)
        """
        a_embed = self.action_embed(action)
        combined = torch.cat([h, a_embed], dim=-1)
        return self.predictor(combined)


class ValueHead(nn.Module):
    """Estimates expected return from state"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h).squeeze(-1)
```

#### 2.4 Full Baby Model

```python
class BabyModel(nn.Module):
    """
    Complete baby model with all components.

    Architecture:
        State -> Encoder -> Transformer -> [Action, WorldModel, Value] Heads
    """
    def __init__(self, config: Config):
        super().__init__()
        self.encoder = StateEncoder(config.state_dim, config.hidden_dim)
        self.transformer = TransformerCore(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads
        )
        self.action_head = ActionHead(config.hidden_dim, config.num_actions)
        self.world_model = WorldModelHead(config.hidden_dim, config.num_actions)
        self.value_head = ValueHead(config.hidden_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Full forward pass.

        Returns: (action_dist, value)
        """
        h = self.encoder(state)
        h = self.transformer(h.unsqueeze(1)).squeeze(1)  # Add/remove seq dim

        action_dist = self.action_head(h)
        value = self.value_head(h)

        return action_dist, value

    def predict_next_state_raw(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        World model prediction - predicts RAW next state tensor.
        FIX #2: Use this for MVP to avoid drifting encoder problem.
        """
        h = self.encoder(state)
        h = self.transformer(h.unsqueeze(1)).squeeze(1)
        return self.world_model(h, action)  # Returns raw state dim

    def get_state_embedding(self, state: torch.Tensor) -> torch.Tensor:
        """Get embedding (for later use when switching to embedding-space prediction)"""
        h = self.encoder(state)
        return self.transformer(h.unsqueeze(1)).squeeze(1)
```

### Deliverables
- [ ] `StateEncoder` module
- [ ] `TransformerCore` module
- [ ] All three heads (Action, WorldModel, Value)
- [ ] `BabyModel` combining everything
- [ ] Unit tests (forward pass shapes, gradient flow)

### Exit Criteria
- Model forward pass works with dummy input
- Gradients flow to all parameters
- Parameter count < 1M (for fast iteration)

---

## Phase 3: Reward System
**Status**: ⬜ Not Started
**Priority**: HIGH
**Estimated Files**: 1

### Objectives
- Implement vector-valued reward computation
- Keep reward components separate for analysis
- Add predictability reward (intrinsic)

### Tasks

#### 3.1 Reward Vector Definition

```python
@dataclass
class RewardVector:
    goal: float = 0.0       # From environment (LLM judge)
    pred: float = 0.0       # Intrinsic: world-model prediction accuracy
    cons: float = 0.0       # Optional: consistency bonus

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([self.goal, self.pred, self.cons])

    def scalar(self, weights: Tuple[float, float, float] = (1.0, 0.1, 0.1)) -> float:
        """Weighted sum for policy gradient"""
        return weights[0] * self.goal + weights[1] * self.pred + weights[2] * self.cons
```

#### 3.2 Reward Computation

```python
class RewardComputer:
    def __init__(self, pred_scale: float = 0.1):
        self.pred_scale = pred_scale

    def compute_prediction_reward(
        self,
        predicted_embedding: torch.Tensor,
        actual_embedding: torch.Tensor
    ) -> float:
        """
        Intrinsic reward for accurate world-model prediction.

        Higher reward = better prediction = baby understands dynamics
        """
        mse = F.mse_loss(predicted_embedding, actual_embedding)
        # Convert loss to reward (lower loss = higher reward)
        # Clip to prevent extreme values
        reward = torch.exp(-mse).item()  # Range: (0, 1]
        return reward * self.pred_scale

    def combine_rewards(
        self,
        env_reward: RewardVector,
        pred_reward: float
    ) -> RewardVector:
        """Add intrinsic prediction reward to environment reward"""
        return RewardVector(
            goal=env_reward.goal,
            pred=pred_reward,
            cons=env_reward.cons
        )
```

### Deliverables
- [ ] `RewardVector` dataclass
- [ ] `RewardComputer` with prediction reward
- [ ] Logging for each reward component

### Exit Criteria
- Reward values are in reasonable range
- Prediction reward decreases as world model improves

---

## Phase 4: Training Loop
**Status**: ⬜ Not Started
**Priority**: CRITICAL
**Estimated Files**: 3

### Objectives
- Implement PPO (or simpler REINFORCE first)
- Add world-model auxiliary loss
- Create full training loop with logging

### Tasks

#### 4.1 Experience Buffer (`src/training/buffer.py`)

```python
@dataclass
class Experience:
    state: torch.Tensor
    action: int
    reward: RewardVector
    next_state: torch.Tensor
    done: bool
    log_prob: float
    value: float
    predicted_next_embedding: torch.Tensor
    actual_next_embedding: torch.Tensor


class RolloutBuffer:
    def __init__(self, capacity: int = 2048):
        self.capacity = capacity
        self.buffer: List[Experience] = []

    def add(self, exp: Experience):
        self.buffer.append(exp)

    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Convert to tensors for training"""
        pass

    def clear(self):
        self.buffer = []

    def compute_returns(self, gamma: float = 0.99, lam: float = 0.95):
        """Compute GAE returns"""
        pass
```

#### 4.2 REINFORCE Implementation (`src/training/reinforce.py`) — START HERE

```python
class REINFORCE:
    """
    Simple policy gradient for MVP. Switch to PPO only after this works.
    """
    def __init__(
        self,
        model: BabyModel,
        lr: float = 1e-3,
        baseline_decay: float = 0.99,
        world_model_coef: float = 1.0,
        entropy_coef: float = 0.01
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.baseline = 0.0  # Running average baseline
        self.baseline_decay = baseline_decay
        self.world_model_coef = world_model_coef
        self.entropy_coef = entropy_coef

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Simple REINFORCE update with baseline and world-model loss.
        """
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']  # Scalar rewards
        next_states_raw = batch['next_states_raw']  # FIX #2: Raw tensors

        # Forward pass
        action_dist, _ = self.model(states)
        log_probs = action_dist.log_prob(actions)

        # Update baseline
        mean_reward = rewards.mean().item()
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * mean_reward

        # Policy gradient with baseline
        advantages = rewards - self.baseline
        policy_loss = -(log_probs * advantages).mean()

        # Entropy bonus (exploration)
        entropy = action_dist.entropy().mean()

        # World-model loss (FIX #2: predict raw state, not embedding)
        predicted_next = self.model.predict_next_state_raw(states, actions)
        world_model_loss = F.mse_loss(predicted_next, next_states_raw)

        # Total loss
        loss = policy_loss - self.entropy_coef * entropy + self.world_model_coef * world_model_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'entropy': entropy.item(),
            'world_model_loss': world_model_loss.item(),
            'baseline': self.baseline,
            'mean_reward': mean_reward
        }
```

#### 4.3 PPO Implementation (`src/training/ppo.py`) — PHASE 2 ONLY

```python
class PPO:
    def __init__(
        self,
        model: BabyModel,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        world_model_coef: float = 1.0,
        max_grad_norm: float = 0.5
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.world_model_coef = world_model_coef
        self.max_grad_norm = max_grad_norm

    def update(self, buffer: RolloutBuffer, epochs: int = 4) -> Dict[str, float]:
        """
        PPO update with world-model auxiliary loss.

        Returns: Dict of loss components for logging
        """
        batch = buffer.get_batch()

        for _ in range(epochs):
            # Policy loss (clipped surrogate)
            action_dist, values = self.model(batch['states'])
            log_probs = action_dist.log_prob(batch['actions'])

            ratio = torch.exp(log_probs - batch['old_log_probs'])
            surr1 = ratio * batch['advantages']
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch['advantages']
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values, batch['returns'])

            # Entropy bonus
            entropy = action_dist.entropy().mean()

            # World-model loss
            predicted_next = self.model.predict_next_state(batch['states'], batch['actions'])
            world_model_loss = F.mse_loss(predicted_next, batch['actual_next_embeddings'])

            # Total loss
            loss = (
                policy_loss
                + self.value_coef * value_loss
                - self.entropy_coef * entropy
                + self.world_model_coef * world_model_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'world_model_loss': world_model_loss.item()
        }
```

#### 4.3 Main Trainer (`src/training/trainer.py`)

```python
class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.env = SymbolicEnv(...)
        self.model = BabyModel(config)
        self.ppo = PPO(self.model, ...)
        self.buffer = RolloutBuffer()
        self.reward_computer = RewardComputer()

        # Logging
        self.logger = WandbLogger(config)  # or TensorBoard

    def collect_rollouts(self, num_steps: int) -> Dict[str, float]:
        """Collect experience from environment"""
        state = self.env.reset()
        episode_rewards = []

        for _ in range(num_steps):
            state_tensor = state.to_tensor()

            with torch.no_grad():
                action_dist, value = self.model(state_tensor)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)

                # World model prediction
                predicted_next = self.model.predict_next_state(state_tensor, action)

            next_state, env_reward, done, info = self.env.step(action.item())

            # Get actual next state embedding
            with torch.no_grad():
                actual_next = self.model.get_state_embedding(next_state.to_tensor())

            # Compute prediction reward
            pred_reward = self.reward_computer.compute_prediction_reward(
                predicted_next, actual_next
            )
            reward = self.reward_computer.combine_rewards(env_reward, pred_reward)

            self.buffer.add(Experience(
                state=state_tensor,
                action=action.item(),
                reward=reward,
                next_state=next_state.to_tensor(),
                done=done,
                log_prob=log_prob.item(),
                value=value.item(),
                predicted_next_embedding=predicted_next,
                actual_next_embedding=actual_next
            ))

            episode_rewards.append(reward.scalar())

            if done:
                state = self.env.reset()
            else:
                state = next_state

        return {'mean_reward': np.mean(episode_rewards)}

    def train(self, total_steps: int):
        """Main training loop"""
        steps = 0

        while steps < total_steps:
            # Collect rollouts
            rollout_stats = self.collect_rollouts(self.config.rollout_steps)
            steps += self.config.rollout_steps

            # Compute returns
            self.buffer.compute_returns()

            # PPO update
            update_stats = self.ppo.update(self.buffer)

            # Log
            self.logger.log({
                'step': steps,
                **rollout_stats,
                **update_stats
            })

            # Clear buffer
            self.buffer.clear()

            # Checkpointing
            if steps % self.config.save_every == 0:
                self.save_checkpoint(steps)

        self.logger.finish()
```

### Deliverables
- [ ] `RolloutBuffer` with GAE computation
- [ ] `PPO` class with all loss components
- [ ] `Trainer` with full training loop
- [ ] Checkpointing and resumption
- [ ] Logging integration (wandb/tensorboard)

### Exit Criteria
- Training runs without errors
- Losses decrease over time
- Can resume from checkpoint

---

## Phase 5: Curriculum Learning
**Status**: ⬜ Not Started
**Priority**: MEDIUM
**Estimated Files**: 1

### Objectives
- Implement progressive difficulty increase
- Define mastery criteria
- (Optional) Action space expansion

### Tasks

#### 5.1 Curriculum Manager

```python
class CurriculumManager:
    def __init__(
        self,
        initial_level: int = 0,
        max_level: int = 5,
        mastery_threshold: float = 0.8,
        window_size: int = 100
    ):
        self.level = initial_level
        self.max_level = max_level
        self.mastery_threshold = mastery_threshold
        self.recent_rewards = deque(maxlen=window_size)

    def update(self, reward: float) -> bool:
        """
        Update with latest reward, return True if level advanced.
        """
        self.recent_rewards.append(reward)

        if len(self.recent_rewards) == self.recent_rewards.maxlen:
            success_rate = np.mean([r > 0.5 for r in self.recent_rewards])

            if success_rate >= self.mastery_threshold and self.level < self.max_level:
                self.level += 1
                self.recent_rewards.clear()
                return True

        return False

    def get_difficulty(self) -> int:
        return self.level
```

#### 5.2 Difficulty Levels per Task

```python
DIFFICULTY_CONFIGS = {
    'pattern_echo': {
        0: {'target_range': (0, 4)},    # Only 5 possible targets
        1: {'target_range': (0, 8)},    # 9 possible targets
        2: {'target_range': (0, 15)},   # All 16 actions
    },
    'sequence_next': {
        0: {'sequence_length': 2, 'pattern': 'increment'},
        1: {'sequence_length': 3, 'pattern': 'increment'},
        2: {'sequence_length': 3, 'pattern': 'arithmetic'},
    },
    'conditional': {
        0: {'num_conditions': 1},
        1: {'num_conditions': 2},
        2: {'num_conditions': 3, 'nested': True},
    }
}
```

### Deliverables
- [ ] `CurriculumManager` class
- [ ] Difficulty configs for each task
- [ ] Integration with trainer

### Exit Criteria
- Curriculum advances when mastery achieved
- Difficulty correlates with task complexity

---

## Phase 6: Evaluation & Analysis
**Status**: ⬜ Not Started
**Priority**: HIGH
**Estimated Files**: 2

### Objectives
- Quantify learning progress
- Analyze learned representations
- Compare against baselines

### Tasks

#### 6.1 Metrics

```python
class Evaluator:
    def evaluate(self, model: BabyModel, env: SymbolicEnv, num_episodes: int = 100):
        """
        Comprehensive evaluation.

        Returns:
            success_rate: Fraction of episodes with correct final action
            world_model_mse: Average prediction error
            policy_entropy: How deterministic is the policy?
            per_difficulty_success: Success rate at each difficulty
        """
        pass
```

#### 6.2 Representation Analysis

```python
def analyze_representations(model: BabyModel, states: List[State]):
    """
    Analyze structure in learned embeddings.

    - t-SNE visualization
    - Clustering by state attributes
    - Linear probing (can you predict attributes from embeddings?)
    """
    pass
```

#### 6.3 Baselines

1. **Random Policy**: Lower bound
2. **RL-Only**: Remove world-model head, train with goal reward only
3. **KL-Distill**: Add KL term to parent (to show this is different)

### Deliverables
- [ ] `Evaluator` class with all metrics
- [ ] Representation analysis scripts
- [ ] Baseline implementations
- [ ] Plotting utilities

### Exit Criteria
- Can generate all figures for paper
- Baselines implemented and comparable

---

## Implementation Order

```
Phase 0 (Setup)
     ↓
Phase 1 (Environment) ← START HERE
     ↓
Phase 2 (Model)
     ↓
Phase 3 (Rewards)
     ↓
Phase 4 (Training) ← FIRST RESULTS
     ↓
Phase 5 (Curriculum)
     ↓
Phase 6 (Evaluation) ← PAPER-READY
```

## 🎯 LOCKED MVP (Non-Negotiable First Experiment)

**This is the ONLY thing that matters until it works.**

### Configuration
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Task | PatternEcho ONLY | Simplest possible |
| Actions | 8 | Smaller = faster learning |
| Model | 2-layer, hidden=64 | ~50K params, trains in minutes |
| Algorithm | REINFORCE + baseline | No PPO complexity |
| Parent | Deterministic ONLY | No LLM confounds |
| Episodes | Fixed-length (T=1) | Single-step task |
| Steps | 10,000 | Quick iteration |
| World-model target | Raw state tensor | Not embedding (Fix #2) |

### Success Criteria (ALL must be true)
1. ✅ Success rate > 12.5% (random = 1/8 = 12.5%)
2. ✅ Success rate > 50% by step 5,000
3. ✅ Policy entropy ↓ over training
4. ✅ World-model MSE ↓ over training

### If MVP Fails
**STOP.** Do not add complexity. Debug:
1. Is the gradient flowing? (`torch.autograd.grad`)
2. Is the reward signal correct? (oracle policy should get 1.0)
3. Is the state encoding correct? (can a linear probe recover target?)

### Code to Run MVP
```bash
python -m src.training.trainer \
    --task pattern_echo \
    --num_actions 8 \
    --hidden_dim 64 \
    --num_layers 2 \
    --algorithm reinforce \
    --parent deterministic \
    --max_steps 10000
```

### What MVP Proves
If this works, you have demonstrated:
- A randomly initialized model can learn goal-directed behavior
- World-model learning provides useful signal
- The framework is fundamentally sound

Only THEN proceed to Phase 2 (more tasks, PPO, curriculum).

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| LLM calls too slow | Implement deterministic fallback first |
| Reward too sparse | Add prediction reward as dense signal |
| Model doesn't learn | Start smaller (2 layers, 64 hidden) |
| World-model loss dominates | Tune λ weights carefully |
| Curriculum too aggressive | Increase mastery threshold |

---

## Questions to Resolve Before Coding

1. **LLM Choice**: OpenAI API? Local model? Anthropic?
2. **Hardware**: GPU available? If not, keep model tiny.
3. **Logging**: Wandb or TensorBoard?
4. **Priority**: Start with Phase 1 (environment) or Phase 2 (model)?
