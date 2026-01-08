"""Main trainer for WMIL."""

import torch
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm
import json
import time

# Optional tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

from ..models.baby_model import BabyModel
from ..environment.symbolic_env import SymbolicEnv, State, RewardVector
from ..environment.tasks import PatternEchoTask, SequenceNextTask, DelayedMatchTask, NavigationTask
from ..environment.parent import DeterministicParent
from ..utils.config import Config
from .reinforce import REINFORCE, Experience


def create_task(config: Config):
    """Factory function to create task based on config."""
    if config.task == "pattern_echo":
        return PatternEchoTask(num_actions=config.num_actions)
    elif config.task == "sequence_next":
        return SequenceNextTask(
            num_actions=config.num_actions,
            sequence_length=getattr(config, 'sequence_length', 4)
        )
    elif config.task == "delayed_match":
        return DelayedMatchTask(num_actions=config.num_actions)
    elif config.task == "navigation":
        return NavigationTask(
            num_positions=config.num_actions,
            num_actions=config.num_actions,
            episode_length=getattr(config, 'max_episode_length', 3)
        )
    else:
        raise ValueError(f"Unknown task: {config.task}")


class Trainer:
    """Main training loop for WMIL."""

    def __init__(self, config: Config):
        self.config = config
        self.device = config.device

        # Create task and environment
        self.task = create_task(config)
        self.parent = DeterministicParent(self.task)
        self.env = SymbolicEnv(
            task=self.task,
            parent=self.parent,
            max_steps=config.max_episode_length,
            device=self.device,
        )

        # Create model
        self.model = BabyModel(
            state_dim=config.state_dim,
            num_actions=config.num_actions,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
        ).to(self.device)

        # Create optimizer/algorithm
        self.algorithm = REINFORCE(
            model=self.model,
            lr=config.lr,
            baseline_decay=config.baseline_decay,
            world_model_coef=config.world_model_coef,
            entropy_coef=config.entropy_coef,
            mve_horizon=getattr(config, 'mve_horizon', 0),
            gamma=getattr(config, 'gamma', 0.99),
            device=self.device,
        )

        # Log MVE status
        if getattr(config, 'mve_horizon', 0) > 0:
            print(f"MVE enabled: H={config.mve_horizon}, gamma={getattr(config, 'gamma', 0.99)}")

        # Logging
        self.log_dir = Path(config.log_dir) / f"run_{int(time.time())}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if HAS_TENSORBOARD:
            self.writer = SummaryWriter(self.log_dir)
        else:
            self.writer = None
            print("TensorBoard not available, using console logging only")

        # Save config
        with open(self.log_dir / "config.json", "w") as f:
            json.dump(vars(config), f, indent=2)

        # Metrics tracking
        self.global_step = 0
        self.episode_count = 0
        self.best_success_rate = 0.0

        # Print model info
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Device: {self.device}")
        print(f"Log dir: {self.log_dir}")

    def collect_batch(self) -> List[Experience]:
        """
        Collect a batch of experiences.

        For multi-step episodes (max_episode_length > 1):
        - Collects full episodes
        - Assigns episode reward to all transitions (for delayed reward tasks)

        Returns:
            List of Experience tuples
        """
        experiences = []
        episodes_collected = 0

        while episodes_collected < self.config.batch_size:
            # Reset environment
            state = self.env.reset()
            episode_experiences = []
            done = False

            # Collect full episode
            while not done:
                state_tensor = state.to_tensor(self.device)

                # Get action from policy
                with torch.no_grad():
                    action_dist, _ = self.model(state_tensor)
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action).item()

                # Take step
                next_state, reward, done, info = self.env.step(action.item())
                next_state_tensor = next_state.to_tensor(self.device)

                # Store experience for this step
                episode_experiences.append({
                    "state": state_tensor.cpu(),
                    "action": action.item(),
                    "reward": reward.goal,
                    "next_state": next_state_tensor.cpu(),
                    "log_prob": log_prob,
                    "success": info.get("success", False),
                })

                state = next_state

            # For delayed reward tasks: propagate final reward to all steps
            # This is crucial for credit assignment
            episode_reward = episode_experiences[-1]["reward"]
            episode_success = episode_experiences[-1]["success"]

            for exp in episode_experiences:
                experiences.append(
                    Experience(
                        state=exp["state"],
                        action=exp["action"],
                        reward=episode_reward,  # Use episode reward for all steps
                        next_state=exp["next_state"],
                        log_prob=exp["log_prob"],
                        success=episode_success,
                    )
                )

            episodes_collected += 1
            self.episode_count += 1

        return experiences

    def train(self) -> Dict[str, float]:
        """
        Main training loop.

        Supports two-phase training if wm_warmup_steps > 0:
        - Phase 1: Train WM only for wm_warmup_steps
        - Phase 2: Freeze WM, train policy only

        Returns:
            Final metrics
        """
        # Check for two-phase training
        wm_warmup_steps = getattr(self.config, 'wm_warmup_steps', 0)
        two_phase = wm_warmup_steps > 0

        if two_phase:
            print(f"\n*** Two-Phase Training ***")
            print(f"Phase 1: WM warmup for {wm_warmup_steps} steps")
            print(f"Phase 2: Policy training for {self.config.total_steps} steps")

        pbar = tqdm(total=self.config.total_steps, desc="Training")

        final_metrics = {}

        # Phase 1: WM warmup (if enabled)
        if two_phase:
            self.algorithm.set_training_mode("wm_only")
            warmup_pbar = tqdm(total=wm_warmup_steps, desc="WM Warmup")
            warmup_step = 0

            while warmup_step < wm_warmup_steps:
                experiences = self.collect_batch()
                warmup_step += len(experiences)
                metrics = self.algorithm.update(experiences)

                warmup_pbar.update(len(experiences))
                warmup_pbar.set_postfix({
                    "wm_loss": f"{metrics.get('loss/world_model', 0):.4f}",
                })

            warmup_pbar.close()
            print(f"\nWM warmup complete. Final WM loss: {metrics.get('loss/world_model', 0):.4f}")

            # Switch to policy-only mode
            self.algorithm.set_training_mode("policy_only")
            print("Switching to policy-only training (WM frozen)\n")

        # Main training loop (Phase 2 if two-phase, or normal training)
        while self.global_step < self.config.total_steps:
            # Collect batch
            experiences = self.collect_batch()
            self.global_step += len(experiences)

            # Update model
            metrics = self.algorithm.update(experiences)
            final_metrics = metrics  # Always update final metrics

            # Track best success rate
            success_rate = metrics.get("success_rate", 0)
            if success_rate > self.best_success_rate:
                self.best_success_rate = success_rate

            # Log metrics
            if self.global_step % self.config.log_interval == 0:
                self._log_metrics(metrics)

            # Update progress bar
            pbar.update(len(experiences))
            pbar.set_postfix(
                {
                    "success": f"{metrics.get('success_rate', 0):.2%}",
                    "entropy": f"{metrics.get('entropy', 0):.2f}",
                    "wm_loss": f"{metrics.get('loss/world_model', 0):.4f}",
                }
            )

            # Save checkpoint
            if self.global_step % self.config.save_interval == 0:
                self._save_checkpoint()

        pbar.close()
        if self.writer:
            self.writer.close()

        # Final evaluation
        print("\n" + "=" * 50)
        print("Training Complete!")
        print("=" * 50)
        self._print_final_stats(final_metrics)

        return final_metrics

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to TensorBoard."""
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, self.global_step)

    def _save_checkpoint(self):
        """Save model checkpoint."""
        checkpoint_path = self.log_dir / f"checkpoint_{self.global_step}.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.algorithm.optimizer.state_dict(),
                "global_step": self.global_step,
                "baseline": self.algorithm.baseline,
            },
            checkpoint_path,
        )

    def _print_final_stats(self, metrics: Dict[str, float]):
        """Print final training statistics."""
        print(f"Total steps: {self.global_step:,}")
        print(f"Total episodes: {self.episode_count:,}")
        print(f"Best success rate: {self.best_success_rate:.2%}")
        print(f"Final success rate: {metrics.get('success_rate', 0):.2%}")
        print(f"Final entropy: {metrics.get('entropy', 0):.4f}")
        print(f"Final world-model loss: {metrics.get('loss/world_model', 0):.4f}")

        # Check success criteria
        print("\n" + "-" * 50)
        print("MVP Success Criteria:")
        print("-" * 50)

        random_baseline = 1.0 / self.config.num_actions
        success_rate = metrics.get("success_rate", 0)

        # Criterion 1: Better than random
        c1 = success_rate > random_baseline
        print(f"1. Success > {random_baseline:.1%} (random): {'PASS' if c1 else 'FAIL'} ({success_rate:.2%})")

        # Criterion 2: > 50% by end (relaxed for MVP)
        c2 = success_rate > 0.5
        print(f"2. Success > 50%: {'PASS' if c2 else 'FAIL'} ({success_rate:.2%})")

        # Criterion 3: Entropy decreased (check manually in TensorBoard)
        print(f"3. Entropy: {metrics.get('entropy', 0):.4f}")

        # Criterion 4: World-model MSE decreased (check TensorBoard)
        print(f"4. WM Loss: {metrics.get('loss/world_model', 0):.4f}")

        if c1 and c2:
            print("\n*** MVP PASSED - Ready for Phase 2! ***")
        else:
            print("\n*** MVP FAILED - Debug before proceeding! ***")
