"""
CausalAgent for Phase 4 Experiments

Architecture:
    Input: [observation, previous_action, target, timestep]
           ↓
    ┌─────────────┐
    │   Encoder   │  (shared, 2-layer MLP)
    │   h_t       │
    └─────────────┘
           ↓
    ┌────┴────┬────────┐
    ↓         ↓        ↓
┌───────┐ ┌───────┐ ┌───────────┐
│Policy │ │Value  │ │Prediction │
│ π(a|h)│ │ V(h)  │ │ ô_{t+1}   │
└───────┘ └───────┘ └───────────┘

Three configurations:
- Condition A: Policy + Value only (no prediction head)
- Condition B: Prediction-as-Action (prediction IS the action)
- Condition C: Full architecture with prediction head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np


@dataclass
class CausalAgentConfig:
    """Configuration for CausalAgent."""
    # Environment dimensions
    num_observations: int = 2       # Aliased observation space
    num_actions: int = 3            # LEFT, RIGHT, STAY
    num_latent_states: int = 4      # For target encoding
    horizon: int = 3                # Episode length

    # Architecture
    hidden_dim: int = 64
    num_hidden_layers: int = 2

    # Which heads to include
    use_prediction_head: bool = True
    use_value_head: bool = True

    # Training
    learning_rate: float = 3e-4
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    prediction_coef: float = 0.1    # Weight for auxiliary prediction loss


class CausalAgent(nn.Module):
    """
    Agent for CausalChain-T3 environment.

    Input features:
        - observation: one-hot (num_observations)
        - previous_action: one-hot (num_actions) + 1 for "no previous action"
        - target: one-hot (num_latent_states)
        - timestep: one-hot (horizon)

    Total input dim: num_observations + (num_actions + 1) + num_latent_states + horizon
    """

    def __init__(self, config: Optional[CausalAgentConfig] = None):
        super().__init__()
        self.config = config or CausalAgentConfig()

        # Calculate input dimension
        self.input_dim = (
            self.config.num_observations +      # observation
            self.config.num_actions + 1 +       # prev_action (including "none")
            self.config.num_latent_states +     # target
            self.config.horizon                 # timestep
        )

        # Shared encoder
        encoder_layers = []
        in_dim = self.input_dim
        for _ in range(self.config.num_hidden_layers):
            encoder_layers.append(nn.Linear(in_dim, self.config.hidden_dim))
            encoder_layers.append(nn.ReLU())
            in_dim = self.config.hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Policy head
        self.policy_head = nn.Linear(self.config.hidden_dim, self.config.num_actions)

        # Value head (optional)
        if self.config.use_value_head:
            self.value_head = nn.Linear(self.config.hidden_dim, 1)
        else:
            self.value_head = None

        # Prediction head (optional)
        if self.config.use_prediction_head:
            # Predicts next observation given current state and action
            # Input: hidden + action one-hot
            self.prediction_head = nn.Sequential(
                nn.Linear(self.config.hidden_dim + self.config.num_actions, self.config.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.config.hidden_dim, self.config.num_observations)
            )
        else:
            self.prediction_head = None

    def _encode_input(self,
                      observation: torch.Tensor,
                      prev_action: torch.Tensor,
                      target: torch.Tensor,
                      timestep: torch.Tensor) -> torch.Tensor:
        """
        Encode input features into one-hot vectors and concatenate.

        Args:
            observation: (batch,) int tensor
            prev_action: (batch,) int tensor (-1 for no previous action)
            target: (batch,) int tensor
            timestep: (batch,) int tensor

        Returns:
            (batch, input_dim) float tensor
        """
        batch_size = observation.shape[0]
        device = observation.device

        # One-hot encode observation
        obs_onehot = F.one_hot(observation.long(), self.config.num_observations).float()

        # One-hot encode previous action (with extra dim for "none")
        # prev_action = -1 means no previous action, encode as index num_actions
        prev_action_shifted = prev_action.long() + 1  # -1 -> 0, 0 -> 1, etc.
        prev_action_onehot = F.one_hot(prev_action_shifted, self.config.num_actions + 1).float()

        # One-hot encode target
        target_onehot = F.one_hot(target.long(), self.config.num_latent_states).float()

        # One-hot encode timestep
        timestep_onehot = F.one_hot(timestep.long(), self.config.horizon).float()

        # Concatenate all features
        features = torch.cat([obs_onehot, prev_action_onehot, target_onehot, timestep_onehot], dim=-1)

        return features

    def get_hidden(self,
                   observation: torch.Tensor,
                   prev_action: torch.Tensor,
                   target: torch.Tensor,
                   timestep: torch.Tensor) -> torch.Tensor:
        """Get hidden representation from encoder."""
        features = self._encode_input(observation, prev_action, target, timestep)
        hidden = self.encoder(features)
        return hidden

    def forward(self,
                observation: torch.Tensor,
                prev_action: torch.Tensor,
                target: torch.Tensor,
                timestep: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all heads.

        Returns:
            Dict with keys:
                - 'hidden': (batch, hidden_dim) encoder output
                - 'policy_logits': (batch, num_actions)
                - 'value': (batch, 1) if value head exists
        """
        hidden = self.get_hidden(observation, prev_action, target, timestep)

        outputs = {
            'hidden': hidden,
            'policy_logits': self.policy_head(hidden),
        }

        if self.value_head is not None:
            outputs['value'] = self.value_head(hidden)

        return outputs

    def predict_next_obs(self,
                         hidden: torch.Tensor,
                         action: torch.Tensor) -> torch.Tensor:
        """
        Predict next observation given hidden state and action.

        Args:
            hidden: (batch, hidden_dim) from encoder
            action: (batch,) int tensor

        Returns:
            (batch, num_observations) logits for next observation
        """
        if self.prediction_head is None:
            raise RuntimeError("Prediction head not enabled")

        action_onehot = F.one_hot(action, self.config.num_actions).float()
        pred_input = torch.cat([hidden, action_onehot], dim=-1)
        pred_logits = self.prediction_head(pred_input)

        return pred_logits

    def get_action(self,
                   observation: torch.Tensor,
                   prev_action: torch.Tensor,
                   target: torch.Tensor,
                   timestep: torch.Tensor,
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            observation, prev_action, target, timestep: Input tensors
            deterministic: If True, return argmax action

        Returns:
            action: (batch,) sampled action
            log_prob: (batch,) log probability of action
            entropy: (batch,) entropy of policy
        """
        outputs = self.forward(observation, prev_action, target, timestep)
        logits = outputs['policy_logits']

        dist = Categorical(logits=logits)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy

    def evaluate_action(self,
                        observation: torch.Tensor,
                        prev_action: torch.Tensor,
                        target: torch.Tensor,
                        timestep: torch.Tensor,
                        action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy for given actions.

        Returns:
            log_prob: (batch,) log probability of action
            entropy: (batch,) entropy of policy
            value: (batch,) value estimate (or zeros if no value head)
        """
        outputs = self.forward(observation, prev_action, target, timestep)
        logits = outputs['policy_logits']

        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        if 'value' in outputs:
            value = outputs['value'].squeeze(-1)
        else:
            value = torch.zeros_like(log_prob)

        return log_prob, entropy, value


class ReplayBuffer:
    """Simple replay buffer for storing episode transitions."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, transition: Dict):
        """Add transition to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Dict:
        """Sample batch of transitions."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        # Stack into tensors
        return {
            key: torch.stack([t[key] for t in batch])
            for key in batch[0].keys()
        }

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer = []
        self.position = 0


def collect_episode(env, agent: CausalAgent, device: torch.device) -> Dict:
    """
    Collect one episode of experience.

    Returns:
        Dict with episode data:
            - observations: (T,) int
            - prev_actions: (T,) int
            - targets: (T,) int
            - timesteps: (T,) int
            - actions: (T,) int
            - rewards: (T,) float
            - log_probs: (T,) float
            - values: (T,) float
            - latent_states: (T+1,) int (for probing)
            - success: bool
    """
    obs_dict = env.reset()

    observations = []
    prev_actions = []
    targets = []
    timesteps = []
    actions = []
    rewards = []
    log_probs = []
    values = []
    entropies = []
    latent_states = [obs_dict['latent_state']]
    hiddens = []

    prev_action = -1  # No previous action at start

    done = False
    while not done:
        # Convert to tensors
        obs_t = torch.tensor([obs_dict['observation']], device=device)
        prev_a_t = torch.tensor([prev_action], device=device)
        target_t = torch.tensor([obs_dict['target']], device=device)
        timestep_t = torch.tensor([obs_dict['timestep']], device=device)

        # Get action
        with torch.no_grad():
            action, log_prob, entropy = agent.get_action(obs_t, prev_a_t, target_t, timestep_t)
            outputs = agent.forward(obs_t, prev_a_t, target_t, timestep_t)

            if 'value' in outputs:
                value = outputs['value'].squeeze()
            else:
                value = torch.tensor(0.0, device=device)

            hidden = outputs['hidden'].squeeze(0)

        # Store
        observations.append(obs_dict['observation'])
        prev_actions.append(prev_action)
        targets.append(obs_dict['target'])
        timesteps.append(obs_dict['timestep'])
        actions.append(action.item())
        log_probs.append(log_prob.item())
        values.append(value.item())
        entropies.append(entropy.item())
        hiddens.append(hidden.cpu())

        # Step environment
        action_int = action.item()
        obs_dict, reward, done, info = env.step(action_int)

        rewards.append(reward)
        latent_states.append(obs_dict['latent_state'] if not done else info['new_latent'])
        prev_action = action_int

    return {
        'observations': torch.tensor(observations),
        'prev_actions': torch.tensor(prev_actions),
        'targets': torch.tensor(targets),
        'timesteps': torch.tensor(timesteps),
        'actions': torch.tensor(actions),
        'rewards': torch.tensor(rewards),
        'log_probs': torch.tensor(log_probs),
        'values': torch.tensor(values),
        'entropies': torch.tensor(entropies),
        'latent_states': torch.tensor(latent_states),
        'hiddens': torch.stack(hiddens),
        'success': info['success'],
    }


if __name__ == "__main__":
    # Quick test
    print("Testing CausalAgent")
    print("=" * 50)

    from src.environment.causal_chain import CausalChainEnv

    device = torch.device("cpu")

    # Create environment and agent
    env = CausalChainEnv()
    config = CausalAgentConfig()
    agent = CausalAgent(config).to(device)

    print(f"Agent architecture:")
    print(f"  Input dim: {agent.input_dim}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Num parameters: {sum(p.numel() for p in agent.parameters())}")

    # Collect an episode
    episode = collect_episode(env, agent, device)

    print(f"\nEpisode collected:")
    print(f"  Length: {len(episode['actions'])}")
    print(f"  Observations: {episode['observations'].tolist()}")
    print(f"  Actions: {episode['actions'].tolist()}")
    print(f"  Rewards: {episode['rewards'].tolist()}")
    print(f"  Latent states: {episode['latent_states'].tolist()}")
    print(f"  Success: {episode['success']}")

    # Test forward pass shapes
    print(f"\nOutput shapes:")
    obs = torch.tensor([0])
    prev_a = torch.tensor([-1])
    target = torch.tensor([2])
    timestep = torch.tensor([0])

    outputs = agent.forward(obs, prev_a, target, timestep)
    print(f"  Hidden: {outputs['hidden'].shape}")
    print(f"  Policy logits: {outputs['policy_logits'].shape}")
    print(f"  Value: {outputs['value'].shape}")

    # Test prediction head
    if agent.prediction_head is not None:
        pred = agent.predict_next_obs(outputs['hidden'], torch.tensor([1]))
        print(f"  Prediction logits: {pred.shape}")

    print("\nAll tests passed!")
