"""
Phase 4 v2: True Partial Observability Experiments

Key change from v1: Agent does NOT see previous action.
This ensures the task CANNOT be solved by memorization.

Three conditions:
    A) Feedforward (reactive) - expected to FAIL (ceiling ~50%)
    B) LSTM (with memory) - can maintain belief state
    C) Feedforward + Prediction head - can implicit world model help?

The critical question: Can any architecture solve this without explicit planning?
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass, asdict

from src.environment.causal_chain_v2 import CausalChainEnvV2
from src.agents.causal_agent_v2 import CausalAgentV2, CausalAgentV2Config, collect_episode_v2


@dataclass
class ExperimentConfigV2:
    """Configuration for Phase 4 v2 experiments."""
    num_episodes: int = 20000
    eval_interval: int = 500
    eval_episodes: int = 200

    hidden_dim: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    prediction_coef: float = 0.1

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "results/phase4_v2"


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def compute_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    returns = torch.zeros_like(rewards)
    running_return = 0.0
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
    return returns


def train_feedforward(config: ExperimentConfigV2, use_prediction: bool = False) -> Dict:
    """
    Train feedforward agent (Condition A or C).

    Args:
        use_prediction: If True, add auxiliary prediction loss (Condition C)
    """
    condition = "C" if use_prediction else "A"
    print("\n" + "=" * 60)
    print(f"CONDITION {condition}: Feedforward {'+ Prediction' if use_prediction else '(Reactive)'}")
    print("=" * 60)

    set_seed(config.seed)
    device = torch.device(config.device)

    env = CausalChainEnvV2()

    agent_config = CausalAgentV2Config(
        hidden_dim=config.hidden_dim,
        use_rnn=False,
        use_prediction_head=use_prediction,
        use_value_head=True,
    )
    agent = CausalAgentV2(agent_config).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate)

    metrics = {
        'episodes': [],
        'success_rates': [],
        'policy_losses': [],
        'value_losses': [],
        'prediction_losses': [],
        'entropies': [],
    }

    episode_successes = []
    pred_losses = []

    for episode_idx in range(config.num_episodes):
        episode = collect_episode_v2(env, agent, device)
        episode_successes.append(float(episode['success']))

        returns = compute_returns(episode['rewards'], config.gamma)

        observations = episode['observations'].to(device)
        targets = episode['targets'].to(device)
        timesteps = episode['timesteps'].to(device)
        actions = episode['actions'].to(device)
        returns = returns.to(device)
        hiddens = episode['hiddens'].to(device)

        # Re-forward for gradients (feedforward, so no hidden state issue)
        outputs = agent.forward(observations, targets, timesteps)
        logits = outputs['policy_logits']
        values = outputs['value'].squeeze(-1) if 'value' in outputs else torch.zeros_like(returns)

        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropies = dist.entropy()

        advantages = returns - values.detach()
        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, returns)

        # Prediction loss (if enabled)
        if use_prediction and len(observations) > 1:
            pred_logits = agent.predict_next_obs(hiddens[:-1], actions[:-1])
            next_obs = observations[1:]
            pred_loss = F.cross_entropy(pred_logits, next_obs.long())
            pred_losses.append(pred_loss.item())
        else:
            pred_loss = torch.tensor(0.0, device=device)

        entropy_loss = -entropies.mean()

        loss = (policy_loss +
                config.value_coef * value_loss +
                config.prediction_coef * pred_loss +
                config.entropy_coef * entropy_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode_idx + 1) % config.eval_interval == 0:
            success_rate = np.mean(episode_successes[-config.eval_interval:])
            pred_loss_avg = np.mean(pred_losses[-config.eval_interval:]) if pred_losses else 0

            metrics['episodes'].append(episode_idx + 1)
            metrics['success_rates'].append(success_rate)
            metrics['policy_losses'].append(policy_loss.item())
            metrics['value_losses'].append(value_loss.item())
            metrics['prediction_losses'].append(pred_loss_avg)
            metrics['entropies'].append(entropies.mean().item())

            print(f"Episode {episode_idx + 1:5d} | "
                  f"Success: {success_rate:.2%} | "
                  f"Entropy: {entropies.mean().item():.4f}" +
                  (f" | PredLoss: {pred_loss_avg:.4f}" if use_prediction else ""))

    final_rate = evaluate_agent_v2(env, agent, device, config.eval_episodes)
    metrics['final_success_rate'] = final_rate

    print(f"\nFinal Success Rate: {final_rate:.2%}")
    print(f"Random Baseline: 25.00%")

    return metrics, agent


def train_lstm(config: ExperimentConfigV2) -> Dict:
    """
    Condition B: LSTM agent with memory.

    The LSTM can maintain belief state across timesteps.
    """
    print("\n" + "=" * 60)
    print("CONDITION B: LSTM (Memory)")
    print("=" * 60)

    set_seed(config.seed)
    device = torch.device(config.device)

    env = CausalChainEnvV2()

    agent_config = CausalAgentV2Config(
        hidden_dim=config.hidden_dim,
        use_rnn=True,  # KEY: Use LSTM
        use_prediction_head=True,
        use_value_head=True,
    )
    agent = CausalAgentV2(agent_config).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate)

    metrics = {
        'episodes': [],
        'success_rates': [],
        'policy_losses': [],
        'entropies': [],
    }

    episode_successes = []

    for episode_idx in range(config.num_episodes):
        episode = collect_episode_v2(env, agent, device)
        episode_successes.append(float(episode['success']))

        returns = compute_returns(episode['rewards'], config.gamma)

        # For LSTM, we need to re-forward the whole sequence
        observations = episode['observations'].to(device)
        targets = episode['targets'].to(device)
        timesteps = episode['timesteps'].to(device)
        actions = episode['actions'].to(device)
        returns = returns.to(device)

        # Reset hidden state and forward through sequence
        agent.reset_hidden(batch_size=1, device=device)

        all_log_probs = []
        all_entropies = []
        all_values = []

        for t in range(len(observations)):
            outputs = agent.forward(
                observations[t:t+1],
                targets[t:t+1],
                timesteps[t:t+1]
            )

            logits = outputs['policy_logits']
            dist = torch.distributions.Categorical(logits=logits)
            log_prob = dist.log_prob(actions[t:t+1])
            entropy = dist.entropy()

            all_log_probs.append(log_prob)
            all_entropies.append(entropy)
            if 'value' in outputs:
                all_values.append(outputs['value'].squeeze())

        log_probs = torch.cat(all_log_probs)
        entropies = torch.cat(all_entropies)
        values = torch.stack(all_values) if all_values else torch.zeros_like(returns)

        advantages = returns - values.detach()
        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_loss = -entropies.mean()

        loss = (policy_loss +
                config.value_coef * value_loss +
                config.entropy_coef * entropy_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode_idx + 1) % config.eval_interval == 0:
            success_rate = np.mean(episode_successes[-config.eval_interval:])

            metrics['episodes'].append(episode_idx + 1)
            metrics['success_rates'].append(success_rate)
            metrics['policy_losses'].append(policy_loss.item())
            metrics['entropies'].append(entropies.mean().item())

            print(f"Episode {episode_idx + 1:5d} | "
                  f"Success: {success_rate:.2%} | "
                  f"Entropy: {entropies.mean().item():.4f}")

    final_rate = evaluate_agent_v2(env, agent, device, config.eval_episodes, use_rnn=True)
    metrics['final_success_rate'] = final_rate

    print(f"\nFinal Success Rate: {final_rate:.2%}")
    print(f"Random Baseline: 25.00%")

    return metrics, agent


def evaluate_agent_v2(env, agent, device, num_episodes: int,
                      use_rnn: bool = False) -> float:
    """Evaluate agent success rate."""
    successes = 0

    for _ in range(num_episodes):
        obs_dict = env.reset()

        if use_rnn:
            agent.reset_hidden(batch_size=1, device=device)

        done = False
        while not done:
            obs_t = torch.tensor([obs_dict['observation']], device=device)
            target_t = torch.tensor([obs_dict['target']], device=device)
            timestep_t = torch.tensor([obs_dict['timestep']], device=device)

            with torch.no_grad():
                action, _, _ = agent.get_action(
                    obs_t, target_t, timestep_t,
                    deterministic=True
                )

            obs_dict, _, done, info = env.step(action.item())

        if info['success']:
            successes += 1

    return successes / num_episodes


def compute_theoretical_ceiling():
    """
    Compute theoretical ceiling for reactive (feedforward) policy.

    For each (obs, target, timestep), find the action that maximizes
    success probability across the aliased latent states.
    """
    from src.environment.causal_chain_v2 import CausalChainEnvV2

    env = CausalChainEnvV2()

    # For each possible input state, compute best action
    # Input: (obs, target, timestep)
    # For obs in {0, 1}, this corresponds to latent states {0,1} or {2,3}

    total_success = 0
    total_cases = 0

    for initial_latent in range(4):
        for target in range(4):
            # What observation does this latent give?
            obs = initial_latent // 2

            # A reactive policy sees (obs, target, 0) at timestep 0
            # It must commit to an action without knowing if it's latent 0 or 1 (if obs=0)

            # Try each possible action sequence
            best_success_prob = 0

            for a0 in range(3):
                for a1 in range(3):
                    for a2 in range(3):
                        # Simulate this action sequence
                        z = initial_latent
                        z = env.transition_table[z, a0]
                        z = env.transition_table[z, a1]
                        z = env.transition_table[z, a2]

                        if z == target:
                            # This works for this specific initial state
                            pass

            # For a reactive policy, count success rate assuming uniform
            # over aliased states with same observation
            total_cases += 1

    # Actually let's just simulate
    print("\nTheoretical Analysis: Reactive Policy Ceiling")
    print("-" * 50)

    # Group initial states by observation
    obs_to_states = {0: [0, 1], 1: [2, 3]}

    total_optimal = 0
    total_reactive_best = 0
    n_configs = 0

    for target in range(4):
        for obs in [0, 1]:
            latent_states = obs_to_states[obs]

            # For each action sequence, compute success rate
            best_seq_success = 0
            best_seq = None

            for a0 in range(3):
                for a1 in range(3):
                    for a2 in range(3):
                        successes = 0
                        for z0 in latent_states:
                            z = z0
                            z = env.transition_table[z, a0]
                            z = env.transition_table[z, a1]
                            z = env.transition_table[z, a2]
                            if z == target:
                                successes += 1

                        success_rate = successes / len(latent_states)
                        if success_rate > best_seq_success:
                            best_seq_success = success_rate
                            best_seq = (a0, a1, a2)

            total_reactive_best += best_seq_success
            total_optimal += 1  # Oracle always succeeds
            n_configs += 1

    reactive_ceiling = total_reactive_best / n_configs
    print(f"Reactive policy ceiling: {reactive_ceiling:.2%}")
    print(f"(This assumes uniform distribution over aliased states)")

    return reactive_ceiling


def run_all_conditions_v2(config: ExperimentConfigV2) -> Dict:
    """Run all three conditions."""
    results = {}

    # Compute theoretical ceiling first
    ceiling = compute_theoretical_ceiling()

    # Condition A: Feedforward (reactive)
    metrics_a, _ = train_feedforward(config, use_prediction=False)
    results['A'] = metrics_a

    # Condition B: LSTM (memory)
    metrics_b, _ = train_lstm(config)
    results['B'] = metrics_b

    # Condition C: Feedforward + Prediction
    metrics_c, _ = train_feedforward(config, use_prediction=True)
    results['C'] = metrics_c

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 4 v2 EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"\nBaselines:")
    print(f"  Random: 25.00%")
    print(f"  Reactive Ceiling: {ceiling:.2%}")
    print(f"  Oracle: 100.00%")

    print(f"\nResults:")
    print(f"  A) Feedforward (reactive):  {results['A']['final_success_rate']:.2%}")
    print(f"  B) LSTM (memory):           {results['B']['final_success_rate']:.2%}")
    print(f"  C) Feedforward + Pred:      {results['C']['final_success_rate']:.2%}")

    print("\n" + "-" * 60)
    print("INTERPRETATION:")

    if results['B']['final_success_rate'] > 0.9:
        print("LSTM achieves near-optimal! Memory enables belief tracking.")
    elif results['B']['final_success_rate'] > ceiling + 0.1:
        print("LSTM exceeds reactive ceiling. Memory helps but not optimal.")
    else:
        print("LSTM does not exceed reactive ceiling. Task harder than expected.")

    if results['C']['final_success_rate'] > results['A']['final_success_rate'] + 0.05:
        print("Prediction head helps feedforward agent (implicit world model).")
    else:
        print("Prediction head does NOT help feedforward agent.")

    results['ceiling'] = ceiling
    return results


def save_results(results: Dict, config: ExperimentConfigV2):
    """Save results to JSON."""
    os.makedirs(config.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{config.save_dir}/phase4_v2_results_{timestamp}.json"

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    results_json = {}
    for k, v in results.items():
        if isinstance(v, dict):
            results_json[k] = {k2: convert(v2) for k2, v2 in v.items()}
        else:
            results_json[k] = convert(v)

    with open(filename, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, default="all",
                        choices=["A", "B", "C", "all"])
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = ExperimentConfigV2(
        num_episodes=args.episodes,
        seed=args.seed,
    )

    print("Phase 4 v2: True Partial Observability")
    print("=" * 60)
    print(f"Episodes: {config.num_episodes}")
    print(f"Device: {config.device}")

    if args.condition == "all":
        results = run_all_conditions_v2(config)
        save_results(results, config)
    elif args.condition == "A":
        train_feedforward(config, use_prediction=False)
    elif args.condition == "B":
        train_lstm(config)
    elif args.condition == "C":
        train_feedforward(config, use_prediction=True)
