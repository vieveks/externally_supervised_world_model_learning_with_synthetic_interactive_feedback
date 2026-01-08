"""Trainer for Prediction-as-Action task (Phase 2b).

This trainer handles:
- PredictionTask environment
- PredictionModel with continuous outputs
- Both RL (REINFORCE) and MLE training modes for comparison
"""

import torch
import random
from typing import Dict, List, Optional
from dataclasses import dataclass

from ..environment.tasks import PredictionTask
from ..models.baby_model import PredictionModel
from ..training.prediction_reinforce import (
    PredictionREINFORCE,
    PredictionMLE,
    PredictionExperience,
)


@dataclass
class PredictionConfig:
    """Configuration for PredictionTask experiments."""

    # Task settings
    num_positions: int = 8
    dynamics_type: str = "circular_shift"  # circular_shift, random_fixed, identity
    reward_type: str = "negative_mse"  # negative_mse, threshold_binary
    threshold: float = 0.1

    # Model settings
    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    ff_dim: int = 256

    # Training settings
    training_mode: str = "rl"  # "rl" or "mle"
    lr: float = 1e-3
    total_steps: int = 10000
    batch_size: int = 32
    eval_interval: int = 500
    log_interval: int = 100

    # RL-specific
    entropy_coef: float = 0.01
    baseline_decay: float = 0.99

    # Device
    device: str = "cpu"

    # Seed
    seed: int = 42


class PredictionTrainer:
    """
    Trainer for Prediction-as-Action experiments.

    Supports two modes:
    - RL mode: Train with REINFORCE, reward = prediction accuracy
    - MLE mode: Train with MSE loss (pretraining baseline)

    The key experiment: Does RL mode match MLE mode?
    If yes, RL gradients can replace pretraining gradients.
    """

    def __init__(self, config: PredictionConfig):
        self.config = config
        self.device = config.device

        # Set seeds
        random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Create task
        self.task = PredictionTask(
            num_positions=config.num_positions,
            dynamics_type=config.dynamics_type,
            reward_type=config.reward_type,
            threshold=config.threshold,
        )

        # Create model
        self.model = PredictionModel(
            state_dim=config.num_positions,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
        ).to(self.device)

        # Create algorithm based on mode
        if config.training_mode == "rl":
            self.algorithm = PredictionREINFORCE(
                model=self.model,
                lr=config.lr,
                baseline_decay=config.baseline_decay,
                entropy_coef=config.entropy_coef,
                device=self.device,
            )
        else:  # mle
            self.algorithm = PredictionMLE(
                model=self.model,
                lr=config.lr,
                device=self.device,
            )

        # Tracking
        self.step = 0
        self.best_accuracy = 0.0
        self.metrics_history: List[Dict] = []

        print(f"PredictionTrainer initialized:")
        print(f"  Task: {self.task.name} ({config.dynamics_type})")
        print(f"  Mode: {config.training_mode.upper()}")
        print(f"  State dim: {config.num_positions}")
        print(f"  Parameters: {self.model.count_parameters():,}")
        print(f"  Device: {self.device}")

    def collect_batch(self) -> List[PredictionExperience]:
        """Collect a batch of experiences."""
        experiences = []

        for _ in range(self.config.batch_size):
            # Generate new episode
            position, _ = self.task.generate_instance()

            # Encode state
            state = torch.tensor(
                self.task.encode_state(position), dtype=torch.float32, device=self.device
            )

            # Get target (for computing metrics)
            target = torch.tensor(
                self.task.get_target_state(), dtype=torch.float32, device=self.device
            )

            # Sample prediction from model
            with torch.no_grad():
                prediction, log_prob = self.model.sample_prediction(state)

            # Compute reward from environment
            pred_list = prediction.cpu().tolist()
            actual_next_pos = self.task.get_dynamics_next(position)
            reward = self.task.compute_reward(pred_list, actual_next_pos)

            # Compute MSE for logging
            mse = ((prediction - target) ** 2).mean().item()

            experiences.append(
                PredictionExperience(
                    state=state,
                    prediction=prediction,
                    target=target,
                    reward=reward,
                    log_prob=log_prob.item(),
                    mse=mse,
                )
            )

        return experiences

    def evaluate(self, num_episodes: int = 100) -> Dict[str, float]:
        """Evaluate model with deterministic predictions."""
        self.model.eval()

        total_mse = 0.0
        correct_argmax = 0

        with torch.no_grad():
            for _ in range(num_episodes):
                # Generate episode
                position, _ = self.task.generate_instance()
                state = torch.tensor(
                    self.task.encode_state(position),
                    dtype=torch.float32,
                    device=self.device,
                )
                target = torch.tensor(
                    self.task.get_target_state(),
                    dtype=torch.float32,
                    device=self.device,
                )

                # Get deterministic prediction (mean)
                prediction = self.model.get_deterministic_prediction(state)

                # Compute metrics
                mse = ((prediction - target) ** 2).mean().item()
                total_mse += mse

                # Argmax accuracy
                if prediction.argmax().item() == target.argmax().item():
                    correct_argmax += 1

        self.model.train()

        return {
            "eval/mse": total_mse / num_episodes,
            "eval/argmax_accuracy": correct_argmax / num_episodes,
        }

    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.config.total_steps} steps...")
        print(f"Mode: {self.config.training_mode.upper()}")
        print("-" * 60)

        while self.step < self.config.total_steps:
            # Collect batch
            experiences = self.collect_batch()

            # Update model
            metrics = self.algorithm.update(experiences)

            self.step += self.config.batch_size

            # Log progress
            if self.step % self.config.log_interval == 0:
                self._log_metrics(metrics)

            # Evaluate
            if self.step % self.config.eval_interval == 0:
                eval_metrics = self.evaluate()
                metrics.update(eval_metrics)

                # Track best
                accuracy = eval_metrics["eval/argmax_accuracy"]
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy

                self._log_eval(eval_metrics)

            self.metrics_history.append({"step": self.step, **metrics})

        # Final evaluation
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        final_eval = self.evaluate(num_episodes=200)
        print(f"Final MSE: {final_eval['eval/mse']:.4f}")
        print(f"Final Argmax Accuracy: {final_eval['eval/argmax_accuracy']:.2%}")
        print(f"Best Argmax Accuracy: {self.best_accuracy:.2%}")

        return final_eval

    def _log_metrics(self, metrics: Dict):
        """Log training metrics."""
        if self.config.training_mode == "rl":
            print(
                f"Step {self.step:6d} | "
                f"Loss: {metrics.get('loss/total', 0):.4f} | "
                f"Reward: {metrics.get('reward/mean', 0):.4f} | "
                f"MSE: {metrics.get('prediction/mse', 0):.4f} | "
                f"Acc: {metrics.get('prediction/argmax_accuracy', 0):.2%}"
            )
        else:
            print(
                f"Step {self.step:6d} | "
                f"Loss: {metrics.get('loss/mse', 0):.4f} | "
                f"Acc: {metrics.get('prediction/argmax_accuracy', 0):.2%}"
            )

    def _log_eval(self, metrics: Dict):
        """Log evaluation metrics."""
        print(
            f"  [EVAL] MSE: {metrics['eval/mse']:.4f} | "
            f"Accuracy: {metrics['eval/argmax_accuracy']:.2%} | "
            f"Best: {self.best_accuracy:.2%}"
        )


def run_prediction_experiment(config_path: str = None, **kwargs):
    """
    Run a prediction-as-action experiment.

    Args:
        config_path: Path to YAML config (optional)
        **kwargs: Override config values
    """
    # Load config from file if provided
    if config_path:
        import yaml

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        # Override with kwargs
        config_dict.update(kwargs)
        config = PredictionConfig(**config_dict)
    else:
        config = PredictionConfig(**kwargs)

    # Run training
    trainer = PredictionTrainer(config)
    results = trainer.train()

    return trainer, results


if __name__ == "__main__":
    import sys

    # Default: run RL mode
    mode = sys.argv[1] if len(sys.argv) > 1 else "rl"

    config = PredictionConfig(
        training_mode=mode,
        total_steps=10000,
        dynamics_type="circular_shift",
        device="cpu",
    )

    trainer = PredictionTrainer(config)
    trainer.train()
