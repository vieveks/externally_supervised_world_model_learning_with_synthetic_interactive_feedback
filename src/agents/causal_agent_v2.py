"""
CausalAgent v2 for True Partial Observability

Key difference from v1: Does NOT receive previous action as input.
Must maintain internal state/memory to solve the task.

Architecture options:
- A: Feedforward (reactive) - expected to fail
- B: RNN/LSTM - can maintain belief state
- C: Feedforward with prediction head - can prediction help?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import numpy as np


@dataclass
class CausalAgentV2Config:
    """Configuration for CausalAgent v2."""
    num_observations: int = 2
    num_actions: int = 3
    num_latent_states: int = 4
    horizon: int = 3

    hidden_dim: int = 64
    use_rnn: bool = False  # If True, use LSTM for memory
    use_prediction_head: bool = True
    use_value_head: bool = True

    learning_rate: float = 3e-4
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    prediction_coef: float = 0.1


class CausalAgentV2(nn.Module):
    """
    Agent for CausalChain v2 (no prev_action input).

    Input features:
        - observation: one-hot (num_observations)
        - target: one-hot (num_latent_states)
        - timestep: one-hot (horizon)

    Total input dim: num_observations + num_latent_states + horizon
    """

    def __init__(self, config: Optional[CausalAgentV2Config] = None):
        super().__init__()
        self.config = config or CausalAgentV2Config()

        # Input dimension (NO prev_action!)
        self.input_dim = (
            self.config.num_observations +
            self.config.num_latent_states +
            self.config.horizon
        )

        # Encoder
        if self.config.use_rnn:
            self.encoder = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=self.config.hidden_dim,
                batch_first=True
            )
            self.hidden_state = None
        else:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.config.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                nn.ReLU(),
            )

        # Policy head
        self.policy_head = nn.Linear(self.config.hidden_dim, self.config.num_actions)

        # Value head
        if self.config.use_value_head:
            self.value_head = nn.Linear(self.config.hidden_dim, 1)
        else:
            self.value_head = None

        # Prediction head
        if self.config.use_prediction_head:
            self.prediction_head = nn.Sequential(
                nn.Linear(self.config.hidden_dim + self.config.num_actions, self.config.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.config.hidden_dim, self.config.num_observations)
            )
        else:
            self.prediction_head = None

    def _encode_input(self, observation: torch.Tensor,
                      target: torch.Tensor,
                      timestep: torch.Tensor) -> torch.Tensor:
        """Encode inputs to feature vector."""
        obs_onehot = F.one_hot(observation.long(), self.config.num_observations).float()
        target_onehot = F.one_hot(target.long(), self.config.num_latent_states).float()
        timestep_onehot = F.one_hot(timestep.long(), self.config.horizon).float()

        features = torch.cat([obs_onehot, target_onehot, timestep_onehot], dim=-1)
        return features

    def reset_hidden(self, batch_size: int = 1, device: torch.device = None):
        """Reset RNN hidden state for new episode."""
        if self.config.use_rnn:
            device = device or next(self.parameters()).device
            self.hidden_state = (
                torch.zeros(1, batch_size, self.config.hidden_dim, device=device),
                torch.zeros(1, batch_size, self.config.hidden_dim, device=device)
            )

    def get_hidden(self, observation: torch.Tensor,
                   target: torch.Tensor,
                   timestep: torch.Tensor) -> torch.Tensor:
        """Get hidden representation."""
        features = self._encode_input(observation, target, timestep)

        if self.config.use_rnn:
            # features: (batch, input_dim) -> (batch, 1, input_dim)
            features = features.unsqueeze(1)
            output, self.hidden_state = self.encoder(features, self.hidden_state)
            hidden = output.squeeze(1)  # (batch, hidden_dim)
        else:
            hidden = self.encoder(features)

        return hidden

    def forward(self, observation: torch.Tensor,
                target: torch.Tensor,
                timestep: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        hidden = self.get_hidden(observation, target, timestep)

        outputs = {
            'hidden': hidden,
            'policy_logits': self.policy_head(hidden),
        }

        if self.value_head is not None:
            outputs['value'] = self.value_head(hidden)

        return outputs

    def predict_next_obs(self, hidden: torch.Tensor,
                         action: torch.Tensor) -> torch.Tensor:
        """Predict next observation."""
        if self.prediction_head is None:
            raise RuntimeError("Prediction head not enabled")

        action_onehot = F.one_hot(action.long(), self.config.num_actions).float()
        pred_input = torch.cat([hidden, action_onehot], dim=-1)
        return self.prediction_head(pred_input)

    def get_action(self, observation: torch.Tensor,
                   target: torch.Tensor,
                   timestep: torch.Tensor,
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        outputs = self.forward(observation, target, timestep)
        logits = outputs['policy_logits']

        dist = Categorical(logits=logits)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy

    def evaluate_action(self, observation: torch.Tensor,
                        target: torch.Tensor,
                        timestep: torch.Tensor,
                        action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for batch."""
        outputs = self.forward(observation, target, timestep)
        logits = outputs['policy_logits']

        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        if 'value' in outputs:
            value = outputs['value'].squeeze(-1)
        else:
            value = torch.zeros_like(log_prob)

        return log_prob, entropy, value


def collect_episode_v2(env, agent: CausalAgentV2, device: torch.device) -> Dict:
    """
    Collect episode with v2 environment (no prev_action).
    """
    obs_dict = env.reset()

    # Reset RNN hidden state if using RNN
    if agent.config.use_rnn:
        agent.reset_hidden(batch_size=1, device=device)

    observations = []
    targets = []
    timesteps = []
    actions = []
    rewards = []
    log_probs = []
    values = []
    entropies = []
    hiddens = []
    latent_states = [obs_dict['latent_state']]

    done = False
    while not done:
        obs_t = torch.tensor([obs_dict['observation']], device=device)
        target_t = torch.tensor([obs_dict['target']], device=device)
        timestep_t = torch.tensor([obs_dict['timestep']], device=device)

        with torch.no_grad():
            action, log_prob, entropy = agent.get_action(obs_t, target_t, timestep_t)
            outputs = agent.forward(obs_t, target_t, timestep_t)

            if 'value' in outputs:
                value = outputs['value'].squeeze()
            else:
                value = torch.tensor(0.0, device=device)

            hidden = outputs['hidden'].squeeze(0)

        observations.append(obs_dict['observation'])
        targets.append(obs_dict['target'])
        timesteps.append(obs_dict['timestep'])
        actions.append(action.item())
        log_probs.append(log_prob.item())
        values.append(value.item())
        entropies.append(entropy.item())
        hiddens.append(hidden.cpu())

        obs_dict, reward, done, info = env.step(action.item())

        rewards.append(reward)
        latent_states.append(info['new_latent'])

    return {
        'observations': torch.tensor(observations),
        'targets': torch.tensor(targets),
        'timesteps': torch.tensor(timesteps),
        'actions': torch.tensor(actions),
        'rewards': torch.tensor(rewards),
        'log_probs': torch.tensor(log_probs),
        'values': torch.tensor(values),
        'entropies': torch.tensor(entropies),
        'hiddens': torch.stack(hiddens),
        'latent_states': torch.tensor(latent_states),
        'success': info['success'],
    }


if __name__ == "__main__":
    from src.environment.causal_chain_v2 import CausalChainEnvV2

    print("Testing CausalAgent v2")
    print("=" * 50)

    device = torch.device("cpu")

    env = CausalChainEnvV2()
    config = CausalAgentV2Config(use_rnn=False)
    agent = CausalAgentV2(config).to(device)

    print(f"Agent architecture (Feedforward):")
    print(f"  Input dim: {agent.input_dim}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Params: {sum(p.numel() for p in agent.parameters())}")

    episode = collect_episode_v2(env, agent, device)
    print(f"\nEpisode:")
    print(f"  Observations: {episode['observations'].tolist()}")
    print(f"  Actions: {episode['actions'].tolist()}")
    print(f"  Latent states: {episode['latent_states'].tolist()}")
    print(f"  Success: {episode['success']}")

    # Test RNN version
    print("\n" + "=" * 50)
    config_rnn = CausalAgentV2Config(use_rnn=True)
    agent_rnn = CausalAgentV2(config_rnn).to(device)
    print(f"Agent architecture (LSTM):")
    print(f"  Params: {sum(p.numel() for p in agent_rnn.parameters())}")

    episode_rnn = collect_episode_v2(env, agent_rnn, device)
    print(f"\nEpisode (LSTM):")
    print(f"  Success: {episode_rnn['success']}")
