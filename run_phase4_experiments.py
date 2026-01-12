"""
Phase 4: World Learning Through Interaction - Experiment Runner

Tests whether prediction-driven RL can acquire causal, multi-step world knowledge.

Three conditions:
    A) Reactive RL: Policy + Value only (no prediction head)
    B) Prediction-as-Action: Prediction IS the action, reward = prediction accuracy
    C) Prediction + Delayed Reward: Full architecture, sparse task reward

Expected outcomes:
    A) Fails (cannot distinguish aliased states)
    B) Learns one-step dynamics, but fails navigation task
    C) THIS IS THE QUESTION - can RL learn AND use world knowledge?
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt

from src.environment.causal_chain import CausalChainEnv, CausalChainConfig
from src.agents.causal_agent import CausalAgent, CausalAgentConfig, collect_episode


@dataclass
class ExperimentConfig:
    """Configuration for Phase 4 experiments."""
    # Training
    num_episodes: int = 20000
    eval_interval: int = 500
    eval_episodes: int = 200

    # Agent
    hidden_dim: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    prediction_coef: float = 0.1

    # Experiment
    condition: str = "C"  # A, B, or C
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Output
    save_dir: str = "results/phase4_causal"


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def compute_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute discounted returns."""
    returns = torch.zeros_like(rewards)
    running_return = 0.0
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
    return returns


def train_condition_a(config: ExperimentConfig) -> Dict:
    """
    Condition A: Reactive RL (Baseline)

    - Policy + Value only
    - No prediction head
    - Sparse task reward at T=3

    Expected: FAILS (cannot distinguish aliased states)
    """
    print("\n" + "=" * 60)
    print("CONDITION A: Reactive RL (Baseline)")
    print("=" * 60)

    set_seed(config.seed)
    device = torch.device(config.device)

    # Create environment
    env = CausalChainEnv()

    # Create agent WITHOUT prediction head
    agent_config = CausalAgentConfig(
        hidden_dim=config.hidden_dim,
        use_prediction_head=False,  # KEY: No prediction head
        use_value_head=True,
    )
    agent = CausalAgent(agent_config).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate)

    # Training metrics
    metrics = {
        'episodes': [],
        'success_rates': [],
        'policy_losses': [],
        'value_losses': [],
        'entropies': [],
    }

    episode_successes = []

    for episode_idx in range(config.num_episodes):
        # Collect episode
        episode = collect_episode(env, agent, device)
        episode_successes.append(float(episode['success']))

        # Compute returns
        returns = compute_returns(episode['rewards'], config.gamma)

        # Convert to device
        observations = episode['observations'].to(device)
        prev_actions = episode['prev_actions'].to(device)
        targets = episode['targets'].to(device)
        timesteps = episode['timesteps'].to(device)
        actions = episode['actions'].to(device)
        returns = returns.to(device)

        # Get log probs and values
        log_probs, entropies, values = agent.evaluate_action(
            observations, prev_actions, targets, timesteps, actions
        )

        # Compute advantages
        advantages = returns - values.detach()

        # Policy loss (REINFORCE with baseline)
        policy_loss = -(log_probs * advantages).mean()

        # Value loss
        value_loss = F.mse_loss(values, returns)

        # Entropy bonus
        entropy_loss = -entropies.mean()

        # Total loss
        loss = (policy_loss +
                config.value_coef * value_loss +
                config.entropy_coef * entropy_loss)

        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluation
        if (episode_idx + 1) % config.eval_interval == 0:
            success_rate = np.mean(episode_successes[-config.eval_interval:])
            metrics['episodes'].append(episode_idx + 1)
            metrics['success_rates'].append(success_rate)
            metrics['policy_losses'].append(policy_loss.item())
            metrics['value_losses'].append(value_loss.item())
            metrics['entropies'].append(entropies.mean().item())

            print(f"Episode {episode_idx + 1:5d} | "
                  f"Success: {success_rate:.2%} | "
                  f"Policy Loss: {policy_loss.item():.4f} | "
                  f"Entropy: {entropies.mean().item():.4f}")

    # Final evaluation
    final_success_rate = evaluate_agent(env, agent, device, config.eval_episodes)
    metrics['final_success_rate'] = final_success_rate

    print(f"\nFinal Success Rate: {final_success_rate:.2%}")
    print(f"Random Baseline: 25.00%")
    print(f"Relative Improvement: {(final_success_rate - 0.25) / 0.25 * 100:.1f}%")

    return metrics, agent


def train_condition_b(config: ExperimentConfig) -> Dict:
    """
    Condition B: Prediction-as-Action

    - Action IS the prediction of next observation
    - Reward = prediction accuracy
    - Tests if world model CAN be learned

    Expected: Learns one-step dynamics, but doesn't solve navigation task
    """
    print("\n" + "=" * 60)
    print("CONDITION B: Prediction-as-Action")
    print("=" * 60)

    set_seed(config.seed)
    device = torch.device(config.device)

    # Create environment
    env = CausalChainEnv()

    # Create agent with prediction head
    agent_config = CausalAgentConfig(
        hidden_dim=config.hidden_dim,
        use_prediction_head=True,
        use_value_head=True,
        num_actions=2,  # Action = predict next observation (0 or 1)
    )
    agent = CausalAgent(agent_config).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate)

    # Training metrics
    metrics = {
        'episodes': [],
        'prediction_accuracies': [],
        'task_success_rates': [],
        'policy_losses': [],
    }

    prediction_correct = []
    task_successes = []

    for episode_idx in range(config.num_episodes):
        # Reset environment
        obs_dict = env.reset()
        prev_action = -1

        episode_pred_correct = []
        done = False

        while not done:
            # Convert to tensors
            obs_t = torch.tensor([obs_dict['observation']], device=device)
            prev_a_t = torch.tensor([prev_action], device=device)
            target_t = torch.tensor([obs_dict['target']], device=device)
            timestep_t = torch.tensor([obs_dict['timestep']], device=device)

            # In Condition B: action IS the predicted next observation
            action, log_prob, entropy = agent.get_action(obs_t, prev_a_t, target_t, timestep_t)

            # Take a RANDOM action in the environment (we're just predicting)
            env_action = np.random.randint(3)
            obs_dict_next, _, done, info = env.step(env_action)

            # Reward = did we predict the next observation correctly?
            predicted_obs = action.item()
            actual_obs = obs_dict_next['observation']
            prediction_reward = 1.0 if predicted_obs == actual_obs else 0.0

            episode_pred_correct.append(prediction_reward)

            # Simple policy gradient update
            loss = -log_prob * prediction_reward + config.entropy_coef * (-entropy)

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            obs_dict = obs_dict_next
            prev_action = env_action

        prediction_correct.extend(episode_pred_correct)
        task_successes.append(float(info['success']))

        # Evaluation
        if (episode_idx + 1) % config.eval_interval == 0:
            pred_acc = np.mean(prediction_correct[-config.eval_interval * 3:])
            task_rate = np.mean(task_successes[-config.eval_interval:])

            metrics['episodes'].append(episode_idx + 1)
            metrics['prediction_accuracies'].append(pred_acc)
            metrics['task_success_rates'].append(task_rate)

            print(f"Episode {episode_idx + 1:5d} | "
                  f"Pred Acc: {pred_acc:.2%} | "
                  f"Task Success: {task_rate:.2%}")

    # Final evaluation
    final_pred_acc = np.mean(prediction_correct[-1000:])
    final_task_rate = np.mean(task_successes[-500:])

    metrics['final_prediction_accuracy'] = final_pred_acc
    metrics['final_task_success_rate'] = final_task_rate

    print(f"\nFinal Prediction Accuracy: {final_pred_acc:.2%}")
    print(f"Final Task Success (random actions): {final_task_rate:.2%}")

    return metrics, agent


def train_condition_c(config: ExperimentConfig) -> Dict:
    """
    Condition C: Prediction + Delayed Reward (PHASE 4 CORE)

    - Full architecture with prediction head
    - Sparse task reward at T=3
    - Auxiliary prediction loss (optional)

    Expected: THIS IS THE QUESTION
    """
    print("\n" + "=" * 60)
    print("CONDITION C: Prediction + Delayed Reward (CORE TEST)")
    print("=" * 60)

    set_seed(config.seed)
    device = torch.device(config.device)

    # Create environment
    env = CausalChainEnv()

    # Create agent WITH prediction head
    agent_config = CausalAgentConfig(
        hidden_dim=config.hidden_dim,
        use_prediction_head=True,
        use_value_head=True,
    )
    agent = CausalAgent(agent_config).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate)

    # Training metrics
    metrics = {
        'episodes': [],
        'success_rates': [],
        'prediction_losses': [],
        'policy_losses': [],
        'value_losses': [],
        'entropies': [],
    }

    episode_successes = []
    all_pred_losses = []

    for episode_idx in range(config.num_episodes):
        # Collect episode
        episode = collect_episode(env, agent, device)
        episode_successes.append(float(episode['success']))

        # Compute returns
        returns = compute_returns(episode['rewards'], config.gamma)

        # Convert to device
        observations = episode['observations'].to(device)
        prev_actions = episode['prev_actions'].to(device)
        targets = episode['targets'].to(device)
        timesteps = episode['timesteps'].to(device)
        actions = episode['actions'].to(device)
        returns = returns.to(device)
        hiddens = episode['hiddens'].to(device)

        # Get log probs and values
        log_probs, entropies, values = agent.evaluate_action(
            observations, prev_actions, targets, timesteps, actions
        )

        # Compute advantages
        advantages = returns - values.detach()

        # Policy loss
        policy_loss = -(log_probs * advantages).mean()

        # Value loss
        value_loss = F.mse_loss(values, returns)

        # Prediction loss (auxiliary)
        # Predict next observation from hidden state and action
        if len(observations) > 1:
            pred_logits = agent.predict_next_obs(hiddens[:-1], actions[:-1])
            next_obs = observations[1:]
            pred_loss = F.cross_entropy(pred_logits, next_obs.long())
        else:
            pred_loss = torch.tensor(0.0, device=device)

        all_pred_losses.append(pred_loss.item())

        # Entropy bonus
        entropy_loss = -entropies.mean()

        # Total loss
        loss = (policy_loss +
                config.value_coef * value_loss +
                config.prediction_coef * pred_loss +
                config.entropy_coef * entropy_loss)

        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluation
        if (episode_idx + 1) % config.eval_interval == 0:
            success_rate = np.mean(episode_successes[-config.eval_interval:])
            pred_loss_avg = np.mean(all_pred_losses[-config.eval_interval:])

            metrics['episodes'].append(episode_idx + 1)
            metrics['success_rates'].append(success_rate)
            metrics['prediction_losses'].append(pred_loss_avg)
            metrics['policy_losses'].append(policy_loss.item())
            metrics['value_losses'].append(value_loss.item())
            metrics['entropies'].append(entropies.mean().item())

            print(f"Episode {episode_idx + 1:5d} | "
                  f"Success: {success_rate:.2%} | "
                  f"Pred Loss: {pred_loss_avg:.4f} | "
                  f"Entropy: {entropies.mean().item():.4f}")

    # Final evaluation
    final_success_rate = evaluate_agent(env, agent, device, config.eval_episodes)
    metrics['final_success_rate'] = final_success_rate

    # Also evaluate prediction accuracy
    pred_accuracy = evaluate_prediction(env, agent, device, 500)
    metrics['final_prediction_accuracy'] = pred_accuracy

    print(f"\nFinal Success Rate: {final_success_rate:.2%}")
    print(f"Final Prediction Accuracy: {pred_accuracy:.2%}")
    print(f"Random Baseline: 25.00%")
    print(f"Relative Improvement: {(final_success_rate - 0.25) / 0.25 * 100:.1f}%")

    return metrics, agent


def evaluate_agent(env, agent, device, num_episodes: int = 200) -> float:
    """Evaluate agent success rate with deterministic policy."""
    successes = 0

    for _ in range(num_episodes):
        obs_dict = env.reset()
        prev_action = -1
        done = False

        while not done:
            obs_t = torch.tensor([obs_dict['observation']], device=device)
            prev_a_t = torch.tensor([prev_action], device=device)
            target_t = torch.tensor([obs_dict['target']], device=device)
            timestep_t = torch.tensor([obs_dict['timestep']], device=device)

            with torch.no_grad():
                action, _, _ = agent.get_action(
                    obs_t, prev_a_t, target_t, timestep_t,
                    deterministic=True
                )

            obs_dict, _, done, info = env.step(action.item())
            prev_action = action.item()

        if info['success']:
            successes += 1

    return successes / num_episodes


def evaluate_prediction(env, agent, device, num_samples: int = 500) -> float:
    """Evaluate world model prediction accuracy."""
    correct = 0
    total = 0

    for _ in range(num_samples):
        obs_dict = env.reset()
        prev_action = -1

        for _ in range(env.config.horizon):
            obs_t = torch.tensor([obs_dict['observation']], device=device)
            prev_a_t = torch.tensor([prev_action], device=device)
            target_t = torch.tensor([obs_dict['target']], device=device)
            timestep_t = torch.tensor([obs_dict['timestep']], device=device)

            with torch.no_grad():
                outputs = agent.forward(obs_t, prev_a_t, target_t, timestep_t)
                hidden = outputs['hidden']

                # Random action
                action = np.random.randint(3)
                action_t = torch.tensor([action], device=device)

                # Predict next observation
                pred_logits = agent.predict_next_obs(hidden, action_t)
                pred_obs = pred_logits.argmax(dim=-1).item()

            # Take action and get actual next observation
            obs_dict_next, _, done, _ = env.step(action)
            actual_obs = obs_dict_next['observation']

            if pred_obs == actual_obs:
                correct += 1
            total += 1

            if done:
                break

            obs_dict = obs_dict_next
            prev_action = action

    return correct / total


def run_all_conditions(base_config: ExperimentConfig):
    """Run all three conditions and compare."""
    results = {}

    # Condition A: Reactive RL
    config_a = ExperimentConfig(**asdict(base_config))
    config_a.condition = "A"
    metrics_a, agent_a = train_condition_a(config_a)
    results['A'] = metrics_a

    # Condition B: Prediction-as-Action
    config_b = ExperimentConfig(**asdict(base_config))
    config_b.condition = "B"
    metrics_b, agent_b = train_condition_b(config_b)
    results['B'] = metrics_b

    # Condition C: Prediction + Delayed Reward
    config_c = ExperimentConfig(**asdict(base_config))
    config_c.condition = "C"
    metrics_c, agent_c = train_condition_c(config_c)
    results['C'] = metrics_c

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 4 EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"\nRandom Baseline: 25.00%")
    print(f"\nCondition A (Reactive RL):        {results['A']['final_success_rate']:.2%}")
    print(f"Condition B (Prediction-as-Act):  Task={results['B']['final_task_success_rate']:.2%}, "
          f"Pred={results['B']['final_prediction_accuracy']:.2%}")
    print(f"Condition C (Prediction+Reward):  {results['C']['final_success_rate']:.2%}")

    # Interpretation
    print("\n" + "-" * 60)
    print("INTERPRETATION:")

    if results['C']['final_success_rate'] > 0.5:
        print("SUCCESS: RL can acquire AND use world knowledge!")
        print("This supports the hypothesis that prediction-driven RL")
        print("can learn world models that enable multi-step planning.")
    elif results['C']['final_success_rate'] > 0.35:
        print("PARTIAL SUCCESS: Some planning ability acquired.")
        print("World model is learned but not fully utilized.")
    else:
        print("FAILURE: Prediction alone insufficient for planning.")
        print("World model may be learned but is not used for decisions.")
        print("This suggests explicit planning mechanisms are needed.")

    return results


def save_results(results: Dict, config: ExperimentConfig):
    """Save results to file."""
    os.makedirs(config.save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{config.save_dir}/phase4_results_{timestamp}.json"

    # Convert numpy values to Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    results_json = {
        k: {k2: convert(v2) for k2, v2 in v.items()}
        for k, v in results.items()
    }

    with open(filename, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to: {filename}")


def plot_results(results: Dict, config: ExperimentConfig):
    """Plot training curves for all conditions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Condition A
    ax = axes[0]
    if 'success_rates' in results['A']:
        ax.plot(results['A']['episodes'], results['A']['success_rates'], 'b-', label='Success Rate')
    ax.axhline(y=0.25, color='r', linestyle='--', label='Random')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.set_title('Condition A: Reactive RL')
    ax.legend()
    ax.set_ylim([0, 1])

    # Condition B
    ax = axes[1]
    if 'prediction_accuracies' in results['B']:
        ax.plot(results['B']['episodes'], results['B']['prediction_accuracies'], 'g-', label='Pred Accuracy')
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Chance (2 obs)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Accuracy')
    ax.set_title('Condition B: Prediction-as-Action')
    ax.legend()
    ax.set_ylim([0, 1])

    # Condition C
    ax = axes[2]
    if 'success_rates' in results['C']:
        ax.plot(results['C']['episodes'], results['C']['success_rates'], 'purple', label='Success Rate')
    ax.axhline(y=0.25, color='r', linestyle='--', label='Random')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.set_title('Condition C: Prediction + Delayed Reward')
    ax.legend()
    ax.set_ylim([0, 1])

    plt.tight_layout()

    os.makedirs(config.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{config.save_dir}/phase4_curves_{timestamp}.png", dpi=150)
    plt.close()

    print(f"Plot saved to: {config.save_dir}/phase4_curves_{timestamp}.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 4 Experiments")
    parser.add_argument("--condition", type=str, default="all",
                        choices=["A", "B", "C", "all"],
                        help="Which condition to run")
    parser.add_argument("--episodes", type=int, default=20000,
                        help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cuda/cpu/auto)")
    args = parser.parse_args()

    # Setup config
    config = ExperimentConfig(
        num_episodes=args.episodes,
        seed=args.seed,
        device="cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu",
    )

    print("Phase 4: World Learning Through Interaction")
    print("=" * 60)
    print(f"Episodes: {config.num_episodes}")
    print(f"Device: {config.device}")
    print(f"Seed: {config.seed}")

    if args.condition == "all":
        results = run_all_conditions(config)
        save_results(results, config)
        plot_results(results, config)
    elif args.condition == "A":
        metrics, agent = train_condition_a(config)
    elif args.condition == "B":
        metrics, agent = train_condition_b(config)
    elif args.condition == "C":
        metrics, agent = train_condition_c(config)
