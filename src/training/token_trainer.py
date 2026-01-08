"""Trainer for Token Prediction Task (Phase 3.1).

This trainer handles:
- TokenPredictionTask environment
- TokenPredictionModel with discrete outputs
- Both RL (REINFORCE) and MLE training modes for comparison
- Sequence prediction with delayed reward
- TD(λ) for credit assignment
"""

import torch
import random
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from ..environment.token_prediction import TokenPredictionTask, SequenceTokenTask
from ..models.token_model import TokenPredictionModel
from ..training.token_reinforce import (
    TokenREINFORCE,
    TokenMLE,
    SequenceREINFORCE,
    TDLambdaREINFORCE,
    TokenExperience,
    SequenceExperience,
)


@dataclass
class TokenTrainerConfig:
    """Configuration for TokenPredictionTask experiments."""

    # Task settings
    vocab_size: int = 16
    grammar_type: str = "deterministic_cyclic"  # cyclic, permutation, bigram
    sequence_length: int = 1  # 1 = single step, >1 = sequence
    reward_delay: int = 0  # Additional delay

    # Model settings
    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    ff_dim: int = 256

    # Training settings
    training_mode: str = "rl"  # "rl", "mle", "td_lambda"
    lr: float = 1e-3
    total_steps: int = 10000
    batch_size: int = 32
    eval_interval: int = 250  # More frequent evaluation
    log_interval: int = 50  # More frequent logging

    # RL-specific
    entropy_coef: float = 0.01
    baseline_decay: float = 0.99

    # TD(λ)-specific
    gamma: float = 0.99
    lambda_: float = 0.9
    value_coef: float = 0.5

    # Device
    device: str = "cpu"

    # Seed
    seed: int = 42

    # Logging
    save_dir: Optional[str] = None
    experiment_name: str = "token_prediction"


class TokenTrainer:
    """
    Trainer for Token Prediction experiments (Phase 3.1).

    Supports:
    - RL mode: Train with REINFORCE, reward = prediction correctness
    - MLE mode: Train with cross-entropy (pretraining baseline)
    - TD(λ) mode: RL with eligibility traces for credit assignment

    The key experiment: Does RL mode match MLE mode?
    If yes, RL gradients can replace pretraining for token prediction.
    """

    def __init__(self, config: TokenTrainerConfig):
        self.config = config
        self.device = config.device

        # Set seeds
        random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Create task
        if config.sequence_length > 1:
            self.task = SequenceTokenTask(
                vocab_size=config.vocab_size,
                sequence_length=config.sequence_length,
                grammar_type=config.grammar_type,
            )
            self.is_sequence = True
        else:
            self.task = TokenPredictionTask(
                vocab_size=config.vocab_size,
                grammar_type=config.grammar_type,
                reward_delay=config.reward_delay,
            )
            self.is_sequence = False

        # Create model
        self.model = TokenPredictionModel(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
        ).to(self.device)

        # Create algorithm based on mode
        if config.training_mode == "rl":
            if self.is_sequence:
                self.algorithm = SequenceREINFORCE(
                    model=self.model,
                    lr=config.lr,
                    baseline_decay=config.baseline_decay,
                    entropy_coef=config.entropy_coef,
                    gamma=config.gamma,
                    device=self.device,
                )
            else:
                self.algorithm = TokenREINFORCE(
                    model=self.model,
                    lr=config.lr,
                    baseline_decay=config.baseline_decay,
                    entropy_coef=config.entropy_coef,
                    device=self.device,
                )
        elif config.training_mode == "td_lambda":
            if not self.is_sequence:
                raise ValueError("TD(λ) requires sequence_length > 1")
            self.algorithm = TDLambdaREINFORCE(
                model=self.model,
                lr=config.lr,
                gamma=config.gamma,
                lambda_=config.lambda_,
                entropy_coef=config.entropy_coef,
                value_coef=config.value_coef,
                device=self.device,
            )
        else:  # mle
            self.algorithm = TokenMLE(
                model=self.model,
                lr=config.lr,
                device=self.device,
            )

        # Tracking
        self.step = 0
        self.best_accuracy = 0.0
        self.metrics_history: List[Dict] = []

        # Setup logging directory
        if config.save_dir:
            self.save_dir = Path(config.save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None

        self._print_config()

    def _print_config(self):
        """Print configuration."""
        print("=" * 60)
        print("TokenTrainer Configuration")
        print("=" * 60)
        print(f"  Task: {self.task.grammar_type}")
        print(f"  Vocab size: {self.config.vocab_size}")
        print(f"  Sequence length: {self.config.sequence_length}")
        print(f"  Reward delay: {self.config.reward_delay}")
        print(f"  Training mode: {self.config.training_mode.upper()}")
        print(f"  Model parameters: {self.model.count_parameters():,}")
        print(f"  Device: {self.device}")
        print("=" * 60)

    def collect_single_step_batch(self) -> List[TokenExperience]:
        """Collect a batch of single-step experiences."""
        experiences = []

        for _ in range(self.config.batch_size):
            # Reset task
            current_token, state = self.task.reset()
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)

            # Sample prediction
            with torch.no_grad():
                prediction, log_prob = self.model.sample_prediction(state_tensor)

            # Step environment
            next_state, reward, done, info = self.task.step(prediction.item())

            experiences.append(
                TokenExperience(
                    state=state_tensor,
                    prediction=prediction.item(),
                    target=info["actual"],
                    reward=reward,
                    log_prob=log_prob.item(),
                    correct=info["correct"],
                )
            )

        return experiences

    def collect_sequence_batch(self) -> List[SequenceExperience]:
        """Collect a batch of sequence experiences."""
        experiences = []

        for _ in range(self.config.batch_size):
            # Reset task
            current_token, state = self.task.reset()

            states = []
            predictions = []
            targets = []
            log_probs = []

            done = False
            while not done:
                state_tensor = torch.tensor(
                    state, dtype=torch.float32, device=self.device
                )
                states.append(state_tensor)

                # Sample prediction
                with torch.no_grad():
                    prediction, log_prob = self.model.sample_prediction(state_tensor)

                # Step environment
                next_state, reward, done, info = self.task.step(prediction.item())

                predictions.append(prediction.item())
                targets.append(info["actual"])
                log_probs.append(log_prob.item())

                state = next_state

            # Final reward
            correct_count = sum(1 for p, t in zip(predictions, targets) if p == t)

            experiences.append(
                SequenceExperience(
                    states=states,
                    predictions=predictions,
                    targets=targets,
                    log_probs=log_probs,
                    reward=reward,  # Delayed reward from task
                    correct_count=correct_count,
                )
            )

        return experiences

    def collect_batch(self):
        """Collect appropriate batch based on task type."""
        if self.is_sequence:
            return self.collect_sequence_batch()
        else:
            return self.collect_single_step_batch()

    def evaluate(self, num_episodes: int = 100) -> Dict[str, float]:
        """Evaluate model with deterministic predictions."""
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for _ in range(num_episodes):
                if self.is_sequence:
                    current_token, state = self.task.reset()
                    done = False
                    while not done:
                        state_tensor = torch.tensor(
                            state, dtype=torch.float32, device=self.device
                        )
                        prediction = self.model.get_deterministic_prediction(state_tensor)
                        next_state, _, done, info = self.task.step(prediction.item())
                        if info["correct"]:
                            correct += 1
                        total += 1
                        state = next_state
                else:
                    current_token, state = self.task.reset()
                    state_tensor = torch.tensor(
                        state, dtype=torch.float32, device=self.device
                    )
                    prediction = self.model.get_deterministic_prediction(state_tensor)
                    _, _, _, info = self.task.step(prediction.item())
                    if info["correct"]:
                        correct += 1
                    total += 1

        self.model.train()

        accuracy = correct / total if total > 0 else 0.0
        return {
            "eval/accuracy": accuracy,
            "eval/correct": correct,
            "eval/total": total,
        }

    def train(self) -> Dict[str, float]:
        """Main training loop."""
        print(f"\nStarting training for {self.config.total_steps} steps...")
        print(f"Mode: {self.config.training_mode.upper()}")
        if self.is_sequence:
            print(f"Sequence length: {self.config.sequence_length}")
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
                accuracy = eval_metrics["eval/accuracy"]
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    if self.save_dir:
                        self._save_checkpoint("best")

                self._log_eval(eval_metrics)

            self.metrics_history.append({"step": self.step, **metrics})

        # Final evaluation
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        final_eval = self.evaluate(num_episodes=500)
        print(f"Final Accuracy: {final_eval['eval/accuracy']:.2%}")
        print(f"Best Accuracy: {self.best_accuracy:.2%}")

        if self.save_dir:
            self._save_results(final_eval)

        return final_eval

    def _log_metrics(self, metrics: Dict):
        """Log training metrics."""
        acc = metrics.get("accuracy", 0)
        if self.config.training_mode == "mle":
            loss = metrics.get("loss/ce", 0)
            print(f"Step {self.step:6d} | Loss: {loss:.4f} | Acc: {acc:.2%}")
        else:
            loss = metrics.get("loss/total", 0)
            reward = metrics.get("reward/mean", metrics.get("reward", 0))
            print(
                f"Step {self.step:6d} | Loss: {loss:.4f} | "
                f"Reward: {reward:.4f} | Acc: {acc:.2%}"
            )

    def _log_eval(self, metrics: Dict):
        """Log evaluation metrics."""
        print(
            f"  [EVAL] Accuracy: {metrics['eval/accuracy']:.2%} | "
            f"Best: {self.best_accuracy:.2%}"
        )

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        if self.save_dir:
            path = self.save_dir / f"checkpoint_{name}.pt"
            torch.save(
                {
                    "step": self.step,
                    "model_state_dict": self.model.state_dict(),
                    "best_accuracy": self.best_accuracy,
                },
                path,
            )

    def _save_results(self, final_eval: Dict):
        """Save training results."""
        if self.save_dir:
            results = {
                "config": asdict(self.config),
                "final_eval": final_eval,
                "best_accuracy": self.best_accuracy,
                "metrics_history": self.metrics_history,
                "timestamp": datetime.now().isoformat(),
            }
            path = self.save_dir / "results.json"
            with open(path, "w") as f:
                json.dump(results, f, indent=2, default=str)


def run_token_experiment(
    vocab_size: int = 16,
    grammar_type: str = "deterministic_cyclic",
    training_mode: str = "rl",
    sequence_length: int = 1,
    total_steps: int = 10000,
    seed: int = 42,
    device: str = "cpu",
    save_dir: Optional[str] = None,
    **kwargs,
) -> Dict:
    """
    Run a token prediction experiment.

    This is the main entry point for Phase 3.1 experiments.

    Args:
        vocab_size: Number of tokens in vocabulary
        grammar_type: Type of grammar (cyclic, permutation, bigram)
        training_mode: Training algorithm (rl, mle, td_lambda)
        sequence_length: Number of predictions per episode (1 = single step)
        total_steps: Total training steps
        seed: Random seed
        device: Device to use
        save_dir: Directory to save results
        **kwargs: Additional config overrides

    Returns:
        Final evaluation results
    """
    config = TokenTrainerConfig(
        vocab_size=vocab_size,
        grammar_type=grammar_type,
        training_mode=training_mode,
        sequence_length=sequence_length,
        total_steps=total_steps,
        seed=seed,
        device=device,
        save_dir=save_dir,
        **kwargs,
    )

    trainer = TokenTrainer(config)
    results = trainer.train()

    return {"trainer": trainer, "results": results}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Token Prediction Training")
    parser.add_argument("--mode", type=str, default="rl", choices=["rl", "mle", "td_lambda"])
    parser.add_argument("--grammar", type=str, default="deterministic_cyclic")
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--seq-length", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-dir", type=str, default=None)

    args = parser.parse_args()

    run_token_experiment(
        vocab_size=args.vocab_size,
        grammar_type=args.grammar,
        training_mode=args.mode,
        sequence_length=args.seq_length,
        total_steps=args.steps,
        seed=args.seed,
        device=args.device,
        save_dir=args.save_dir,
    )
