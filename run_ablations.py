"""
Ablation experiments for Phase 2b: Prediction-as-Action.

This script runs three ablations to strengthen the paper:
1. Delayed reward: Show RL still converges with delayed prediction feedback
2. Representation probe: Compare hidden representations between RL and MLE
3. Auxiliary WM failure: Show that auxiliary WM + RL fails on PredictionTask

Usage:
    python run_ablations.py --ablation delayed_reward
    python run_ablations.py --ablation representation_probe
    python run_ablations.py --ablation auxiliary_wm_failure
    python run_ablations.py --all
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Import our modules
import sys
sys.path.insert(0, ".")

from src.environment.tasks import PredictionTask
from src.models.baby_model import PredictionModel
from src.training.prediction_reinforce import PredictionREINFORCE, PredictionMLE, PredictionExperience
from src.training.prediction_trainer import PredictionConfig, PredictionTrainer


# =============================================================================
# Ablation 1: Delayed Reward
# =============================================================================

@dataclass
class DelayedRewardConfig:
    """Config for delayed reward ablation."""
    num_positions: int = 8
    dynamics_type: str = "circular_shift"
    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    ff_dim: int = 256
    lr: float = 1e-3
    total_steps: int = 15000
    batch_size: int = 32
    eval_interval: int = 500
    entropy_coef: float = 0.01
    baseline_decay: float = 0.99
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    # Delay settings
    sequence_length: int = 3  # Number of predictions before reward


class DelayedPredictionTrainer:
    """
    Trainer for delayed reward ablation.

    Instead of immediate reward after each prediction, the agent makes
    a sequence of predictions and receives cumulative reward at the end.

    This tests credit assignment: can RL still learn when reward is delayed?
    """

    def __init__(self, config: DelayedRewardConfig):
        self.config = config
        self.device = config.device

        random.seed(config.seed)
        torch.manual_seed(config.seed)

        self.task = PredictionTask(
            num_positions=config.num_positions,
            dynamics_type=config.dynamics_type,
        )

        self.model = PredictionModel(
            state_dim=config.num_positions,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.baseline = 0.0
        self.step = 0
        self.best_accuracy = 0.0

        print(f"DelayedPredictionTrainer initialized:")
        print(f"  Sequence length (delay): {config.sequence_length}")
        print(f"  Dynamics: {config.dynamics_type}")
        print(f"  Device: {self.device}")

    def collect_sequence(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], float]:
        """
        Collect a sequence of predictions with delayed reward.

        Returns:
            states: List of state tensors
            predictions: List of prediction tensors
            targets: List of target tensors
            total_reward: Sum of prediction accuracies (given at end)
        """
        states = []
        predictions = []
        targets = []
        log_probs = []
        rewards = []

        # Start position
        position, _ = self.task.generate_instance()

        for t in range(self.config.sequence_length):
            # Encode state
            state = torch.tensor(
                self.task.encode_state(position),
                dtype=torch.float32,
                device=self.device
            )
            states.append(state)

            # Get target
            next_pos = self.task.get_dynamics_next(position)
            target = torch.tensor(
                self.task.encode_state(next_pos),
                dtype=torch.float32,
                device=self.device
            )
            targets.append(target)

            # Sample prediction
            with torch.no_grad():
                pred, log_prob = self.model.sample_prediction(state)
            predictions.append(pred)
            log_probs.append(log_prob)

            # Compute step reward (but don't give it yet)
            mse = ((pred - target) ** 2).mean().item()
            rewards.append(-mse)

            # Advance to next state
            position = next_pos

        # Total reward is sum of step rewards (delayed signal)
        total_reward = sum(rewards)

        return states, predictions, targets, log_probs, total_reward

    def update(self, batch_size: int) -> Dict[str, float]:
        """Run one update with delayed rewards."""
        all_log_probs = []
        all_rewards = []
        all_mses = []
        all_accuracies = []

        for _ in range(batch_size):
            states, preds, targets, log_probs, total_reward = self.collect_sequence()

            # For REINFORCE with delayed reward, we assign the total reward
            # to all actions in the sequence (simplest credit assignment)
            for lp in log_probs:
                all_log_probs.append(lp)
                all_rewards.append(total_reward / self.config.sequence_length)

            # Compute metrics
            for pred, target in zip(preds, targets):
                mse = ((pred - target) ** 2).mean().item()
                all_mses.append(mse)
                acc = 1.0 if pred.argmax().item() == target.argmax().item() else 0.0
                all_accuracies.append(acc)

        # Stack for gradient computation
        log_probs_tensor = torch.stack(all_log_probs)
        rewards_tensor = torch.tensor(all_rewards, device=self.device)

        # Update baseline
        mean_reward = rewards_tensor.mean().item()
        self.baseline = 0.99 * self.baseline + 0.01 * mean_reward

        # Policy gradient
        advantages = rewards_tensor - self.baseline

        # Need to recompute log probs with gradient
        # Collect fresh forward passes
        policy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        for _ in range(batch_size):
            states, preds, targets, _, total_reward = self.collect_sequence()
            seq_loss = torch.tensor(0.0, device=self.device)

            for state, pred, target in zip(states, preds, targets):
                mean, std, _ = self.model(state.unsqueeze(0))
                dist = torch.distributions.Normal(mean.squeeze(0), std.squeeze(0))
                log_prob = dist.log_prob(pred).sum()
                advantage = (total_reward / self.config.sequence_length) - self.baseline
                seq_loss = seq_loss - log_prob * advantage

            policy_loss = policy_loss + seq_loss

        policy_loss = policy_loss / (batch_size * self.config.sequence_length)

        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            "loss": policy_loss.item(),
            "reward": mean_reward,
            "mse": np.mean(all_mses),
            "accuracy": np.mean(all_accuracies),
        }

    def evaluate(self, num_episodes: int = 100) -> Dict[str, float]:
        """Evaluate with deterministic predictions."""
        self.model.eval()
        total_mse = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for _ in range(num_episodes):
                position, _ = self.task.generate_instance()

                for t in range(self.config.sequence_length):
                    state = torch.tensor(
                        self.task.encode_state(position),
                        dtype=torch.float32,
                        device=self.device
                    )

                    next_pos = self.task.get_dynamics_next(position)
                    target = torch.tensor(
                        self.task.encode_state(next_pos),
                        dtype=torch.float32,
                        device=self.device
                    )

                    pred = self.model.get_deterministic_prediction(state)

                    mse = ((pred - target) ** 2).mean().item()
                    total_mse += mse

                    if pred.argmax().item() == target.argmax().item():
                        correct += 1
                    total += 1

                    position = next_pos

        self.model.train()
        return {
            "eval/mse": total_mse / total,
            "eval/accuracy": correct / total,
        }

    def train(self):
        """Main training loop."""
        print(f"\nTraining with delayed reward (sequence_length={self.config.sequence_length})...")
        print("-" * 60)

        while self.step < self.config.total_steps:
            metrics = self.update(self.config.batch_size)
            self.step += self.config.batch_size * self.config.sequence_length

            if self.step % self.config.eval_interval == 0:
                eval_metrics = self.evaluate()
                if eval_metrics["eval/accuracy"] > self.best_accuracy:
                    self.best_accuracy = eval_metrics["eval/accuracy"]

                print(f"Step {self.step:6d} | "
                      f"Loss: {metrics['loss']:.4f} | "
                      f"Reward: {metrics['reward']:.4f} | "
                      f"MSE: {metrics['mse']:.4f} | "
                      f"Acc: {metrics['accuracy']:.2%} | "
                      f"Eval Acc: {eval_metrics['eval/accuracy']:.2%}")

        print("\n" + "=" * 60)
        print("FINAL RESULTS (Delayed Reward)")
        print("=" * 60)
        final_eval = self.evaluate(200)
        print(f"Sequence Length (Delay): {self.config.sequence_length}")
        print(f"Final MSE: {final_eval['eval/mse']:.4f}")
        print(f"Final Accuracy: {final_eval['eval/accuracy']:.2%}")
        print(f"Best Accuracy: {self.best_accuracy:.2%}")

        return final_eval


def run_delayed_reward_ablation():
    """Run delayed reward ablation with different delay lengths."""
    print("\n" + "=" * 70)
    print("ABLATION 1: DELAYED REWARD")
    print("=" * 70)
    print("Testing whether RL converges with delayed prediction feedback.")
    print("Hypothesis: RL still works but requires more steps.\n")

    results = {}

    for delay in [1, 2, 3, 5]:
        print(f"\n{'='*60}")
        print(f"Testing delay = {delay} steps")
        print(f"{'='*60}")

        config = DelayedRewardConfig(
            sequence_length=delay,
            total_steps=20000 if delay > 2 else 15000,
        )
        trainer = DelayedPredictionTrainer(config)
        final_metrics = trainer.train()

        results[delay] = {
            "final_accuracy": final_metrics["eval/accuracy"],
            "final_mse": final_metrics["eval/mse"],
            "best_accuracy": trainer.best_accuracy,
        }

    print("\n" + "=" * 70)
    print("DELAYED REWARD ABLATION SUMMARY")
    print("=" * 70)
    print(f"{'Delay':<10} {'Final Acc':<15} {'Best Acc':<15} {'Final MSE':<15}")
    print("-" * 55)
    for delay, r in results.items():
        print(f"{delay:<10} {r['final_accuracy']:.2%}          {r['best_accuracy']:.2%}          {r['final_mse']:.4f}")

    return results


# =============================================================================
# Ablation 2: Representation Probe (Linear Probe / CCA)
# =============================================================================

def extract_representations(model: PredictionModel, task: PredictionTask,
                           device: str, num_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """Extract hidden representations for all states."""
    model.eval()

    representations = []
    labels = []

    with torch.no_grad():
        for pos in range(task.num_positions):
            for _ in range(num_samples // task.num_positions):
                state = torch.tensor(
                    task.encode_state(pos),
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0)

                # Get hidden representation (before prediction head)
                h = model.encoder(state)
                representations.append(h.cpu().numpy().flatten())
                labels.append(pos)

    return np.array(representations), np.array(labels)


def linear_probe_accuracy(X: np.ndarray, y: np.ndarray) -> float:
    """Train a linear classifier and return accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5)
    return scores.mean()


def cca_similarity(X1: np.ndarray, X2: np.ndarray, n_components: int = 8) -> float:
    """Compute CCA similarity between two representation matrices."""
    from sklearn.cross_decomposition import CCA

    # Ensure same number of samples
    n = min(len(X1), len(X2))
    X1, X2 = X1[:n], X2[:n]

    # Fit CCA
    n_comp = min(n_components, X1.shape[1], X2.shape[1])
    cca = CCA(n_components=n_comp)
    X1_c, X2_c = cca.fit_transform(X1, X2)

    # Compute correlation for each component
    correlations = []
    for i in range(n_comp):
        corr = np.corrcoef(X1_c[:, i], X2_c[:, i])[0, 1]
        correlations.append(abs(corr))

    return np.mean(correlations)


def run_representation_probe_ablation():
    """Compare representations learned by RL vs MLE."""
    print("\n" + "=" * 70)
    print("ABLATION 2: REPRESENTATION PROBE")
    print("=" * 70)
    print("Comparing hidden representations learned by RL vs MLE.")
    print("Hypothesis: Both learn similar representations (same task structure).\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Train RL model
    print("Training RL model...")
    config_rl = PredictionConfig(
        training_mode="rl",
        total_steps=10000,
        dynamics_type="circular_shift",
        device=device,
    )
    trainer_rl = PredictionTrainer(config_rl)
    trainer_rl.train()

    # Train MLE model
    print("\nTraining MLE model...")
    config_mle = PredictionConfig(
        training_mode="mle",
        total_steps=5000,
        dynamics_type="circular_shift",
        device=device,
    )
    trainer_mle = PredictionTrainer(config_mle)
    trainer_mle.train()

    # Extract representations
    print("\nExtracting representations...")
    task = PredictionTask(dynamics_type="circular_shift")

    repr_rl, labels_rl = extract_representations(trainer_rl.model, task, device)
    repr_mle, labels_mle = extract_representations(trainer_mle.model, task, device)

    # Linear probe
    print("\nLinear probe analysis...")
    probe_acc_rl = linear_probe_accuracy(repr_rl, labels_rl)
    probe_acc_mle = linear_probe_accuracy(repr_mle, labels_mle)

    print(f"Linear probe accuracy (RL):  {probe_acc_rl:.2%}")
    print(f"Linear probe accuracy (MLE): {probe_acc_mle:.2%}")

    # CCA similarity
    print("\nCCA similarity analysis...")
    cca_sim = cca_similarity(repr_rl, repr_mle)
    print(f"CCA similarity (RL vs MLE): {cca_sim:.4f}")

    # Random baseline
    random_repr = np.random.randn(*repr_rl.shape)
    cca_random = cca_similarity(repr_rl, random_repr)
    print(f"CCA similarity (RL vs Random): {cca_random:.4f}")

    results = {
        "linear_probe_rl": probe_acc_rl,
        "linear_probe_mle": probe_acc_mle,
        "cca_rl_vs_mle": cca_sim,
        "cca_rl_vs_random": cca_random,
    }

    print("\n" + "=" * 70)
    print("REPRESENTATION PROBE SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<30} {'Value':<15}")
    print("-" * 45)
    for k, v in results.items():
        print(f"{k:<30} {v:.4f}")

    return results


# =============================================================================
# Ablation 3: Auxiliary WM Failure on PredictionTask
# =============================================================================

class AuxiliaryWMPredictionModel(nn.Module):
    """
    Model with auxiliary world-model head for PredictionTask.

    This is the WRONG architecture: prediction is auxiliary, not the action.
    Used to show that auxiliary WM fails even on prediction-focused task.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Policy head: discrete actions (wrong for prediction task!)
        self.policy_head = nn.Linear(hidden_dim, state_dim)

        # Auxiliary WM head: predicts next state
        self.wm_head = nn.Sequential(
            nn.Linear(hidden_dim + state_dim, hidden_dim),  # Takes state + action
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(state)
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value

    def predict_next_state(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h = self.encoder(state)
        # One-hot encode action if needed
        if action.dim() == 1:
            action_onehot = F.one_hot(action.long(), num_classes=state.shape[-1]).float()
        else:
            action_onehot = action
        wm_input = torch.cat([h, action_onehot], dim=-1)
        return self.wm_head(wm_input)


class AuxiliaryWMTrainer:
    """
    Trainer that uses auxiliary WM loss on PredictionTask.

    This demonstrates FAILURE: the policy picks discrete actions,
    the WM predicts states, but prediction accuracy is never the action.
    """

    def __init__(self, config):
        self.config = config
        self.device = config.device

        random.seed(config.seed)
        torch.manual_seed(config.seed)

        self.task = PredictionTask(
            num_positions=config.num_positions,
            dynamics_type=config.dynamics_type,
        )

        self.model = AuxiliaryWMPredictionModel(
            state_dim=config.num_positions,
            hidden_dim=config.hidden_dim,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.baseline = 0.0
        self.step = 0
        self.best_accuracy = 0.0
        self.wm_weight = config.wm_weight if hasattr(config, 'wm_weight') else 1.0

        print(f"AuxiliaryWMTrainer initialized:")
        print(f"  WM weight: {self.wm_weight}")
        print(f"  Device: {self.device}")

    def collect_batch(self, batch_size: int):
        """Collect experiences."""
        states = []
        actions = []
        targets = []
        rewards = []

        for _ in range(batch_size):
            position, _ = self.task.generate_instance()
            state = torch.tensor(
                self.task.encode_state(position),
                dtype=torch.float32,
                device=self.device
            )

            next_pos = self.task.get_dynamics_next(position)
            target = torch.tensor(
                self.task.encode_state(next_pos),
                dtype=torch.float32,
                device=self.device
            )

            # Sample discrete action from policy
            with torch.no_grad():
                logits, _ = self.model(state.unsqueeze(0))
                dist = torch.distributions.Categorical(logits=logits.squeeze(0))
                action = dist.sample()

            # Reward based on whether action matches target argmax
            # (This is the WRONG setup - action should BE the prediction)
            reward = 1.0 if action.item() == target.argmax().item() else 0.0

            states.append(state)
            actions.append(action)
            targets.append(target)
            rewards.append(reward)

        return states, actions, targets, rewards

    def update(self, batch_size: int) -> Dict[str, float]:
        """Run one update."""
        states, actions, targets, rewards = self.collect_batch(batch_size)

        states_t = torch.stack(states)
        actions_t = torch.stack(actions)
        targets_t = torch.stack(targets)
        rewards_t = torch.tensor(rewards, device=self.device)

        # Policy loss (REINFORCE)
        logits, values = self.model(states_t)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions_t)

        mean_reward = rewards_t.mean().item()
        self.baseline = 0.99 * self.baseline + 0.01 * mean_reward
        advantages = rewards_t - self.baseline

        policy_loss = -(log_probs * advantages.detach()).mean()

        # Auxiliary WM loss (MLE on state prediction)
        action_onehot = F.one_hot(actions_t, num_classes=self.task.num_positions).float()
        pred_states = self.model.predict_next_state(states_t, action_onehot)
        wm_loss = F.mse_loss(pred_states, targets_t)

        # Total loss
        total_loss = policy_loss + self.wm_weight * wm_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Compute prediction accuracy (how well does WM predict?)
        with torch.no_grad():
            wm_accuracy = (pred_states.argmax(dim=-1) == targets_t.argmax(dim=-1)).float().mean().item()

        return {
            "loss/total": total_loss.item(),
            "loss/policy": policy_loss.item(),
            "loss/wm": wm_loss.item(),
            "reward": mean_reward,
            "policy_accuracy": mean_reward,  # Same as reward for this task
            "wm_accuracy": wm_accuracy,
        }

    def evaluate(self, num_episodes: int = 100) -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()
        correct_policy = 0
        correct_wm = 0

        with torch.no_grad():
            for _ in range(num_episodes):
                position, _ = self.task.generate_instance()
                state = torch.tensor(
                    self.task.encode_state(position),
                    dtype=torch.float32,
                    device=self.device
                ).unsqueeze(0)

                next_pos = self.task.get_dynamics_next(position)
                target_idx = next_pos

                # Policy prediction
                logits, _ = self.model(state)
                policy_pred = logits.argmax(dim=-1).item()
                if policy_pred == target_idx:
                    correct_policy += 1

                # WM prediction (given the policy's action)
                action = torch.tensor([policy_pred], device=self.device)
                action_onehot = F.one_hot(action, num_classes=self.task.num_positions).float()
                wm_pred = self.model.predict_next_state(state, action_onehot)
                if wm_pred.argmax(dim=-1).item() == target_idx:
                    correct_wm += 1

        self.model.train()
        return {
            "eval/policy_accuracy": correct_policy / num_episodes,
            "eval/wm_accuracy": correct_wm / num_episodes,
        }

    def train(self):
        """Main training loop."""
        print(f"\nTraining with Auxiliary WM (WRONG architecture for prediction task)...")
        print("-" * 60)

        while self.step < self.config.total_steps:
            metrics = self.update(self.config.batch_size)
            self.step += self.config.batch_size

            if self.step % self.config.eval_interval == 0:
                eval_metrics = self.evaluate()
                if eval_metrics["eval/policy_accuracy"] > self.best_accuracy:
                    self.best_accuracy = eval_metrics["eval/policy_accuracy"]

                print(f"Step {self.step:6d} | "
                      f"Policy Acc: {metrics['policy_accuracy']:.2%} | "
                      f"WM Acc: {metrics['wm_accuracy']:.2%} | "
                      f"WM Loss: {metrics['loss/wm']:.4f} | "
                      f"Eval Policy: {eval_metrics['eval/policy_accuracy']:.2%}")

        print("\n" + "=" * 60)
        print("FINAL RESULTS (Auxiliary WM - WRONG Architecture)")
        print("=" * 60)
        final_eval = self.evaluate(200)
        print(f"Final Policy Accuracy: {final_eval['eval/policy_accuracy']:.2%}")
        print(f"Final WM Accuracy: {final_eval['eval/wm_accuracy']:.2%}")
        print(f"Best Policy Accuracy: {self.best_accuracy:.2%}")

        return final_eval


@dataclass
class AuxWMConfig:
    num_positions: int = 8
    dynamics_type: str = "circular_shift"
    hidden_dim: int = 64
    lr: float = 1e-3
    total_steps: int = 10000
    batch_size: int = 32
    eval_interval: int = 500
    wm_weight: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


def run_auxiliary_wm_failure_ablation():
    """Show that auxiliary WM + RL fails on PredictionTask."""
    print("\n" + "=" * 70)
    print("ABLATION 3: AUXILIARY WM FAILURE")
    print("=" * 70)
    print("Demonstrating that auxiliary WM architecture fails on PredictionTask.")
    print("The WM learns to predict (low loss) but policy fails (low accuracy).\n")

    # Run auxiliary WM experiment
    config = AuxWMConfig(total_steps=10000)
    trainer = AuxiliaryWMTrainer(config)
    aux_results = trainer.train()

    # Compare with prediction-as-action (correct architecture)
    print("\n" + "-" * 60)
    print("Comparison: Prediction-as-Action (CORRECT architecture)")
    print("-" * 60)

    pred_config = PredictionConfig(
        training_mode="rl",
        total_steps=10000,
        dynamics_type="circular_shift",
        device=config.device,
    )
    pred_trainer = PredictionTrainer(pred_config)
    pred_results = pred_trainer.train()

    print("\n" + "=" * 70)
    print("AUXILIARY WM FAILURE ABLATION SUMMARY")
    print("=" * 70)
    print(f"{'Architecture':<30} {'Policy Acc':<15} {'WM/Pred Acc':<15}")
    print("-" * 60)
    print(f"{'Auxiliary WM (WRONG)':<30} {aux_results['eval/policy_accuracy']:.2%}          {aux_results['eval/wm_accuracy']:.2%}")
    print(f"{'Pred-as-Action (CORRECT)':<30} {pred_results['eval/argmax_accuracy']:.2%}          {'N/A (unified)':<15}")

    return {
        "auxiliary_wm": aux_results,
        "pred_as_action": pred_results,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run Phase 2b ablations")
    parser.add_argument("--ablation", type=str,
                       choices=["delayed_reward", "representation_probe", "auxiliary_wm_failure"],
                       help="Which ablation to run")
    parser.add_argument("--all", action="store_true", help="Run all ablations")
    args = parser.parse_args()

    if args.all:
        print("\n" + "=" * 70)
        print("RUNNING ALL ABLATIONS")
        print("=" * 70)

        results = {}

        # Ablation 1
        results["delayed_reward"] = run_delayed_reward_ablation()

        # Ablation 2
        try:
            results["representation_probe"] = run_representation_probe_ablation()
        except ImportError as e:
            print(f"Skipping representation probe (missing sklearn): {e}")
            results["representation_probe"] = None

        # Ablation 3
        results["auxiliary_wm_failure"] = run_auxiliary_wm_failure_ablation()

        print("\n" + "=" * 70)
        print("ALL ABLATIONS COMPLETE")
        print("=" * 70)

        return results

    elif args.ablation == "delayed_reward":
        return run_delayed_reward_ablation()
    elif args.ablation == "representation_probe":
        return run_representation_probe_ablation()
    elif args.ablation == "auxiliary_wm_failure":
        return run_auxiliary_wm_failure_ablation()
    else:
        print("Please specify --ablation or --all")
        parser.print_help()


if __name__ == "__main__":
    main()
