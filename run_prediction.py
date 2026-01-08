"""Run Prediction-as-Action experiments."""

import sys
import argparse

sys.path.insert(0, ".")
from src.training.prediction_trainer import PredictionConfig, PredictionTrainer


def main():
    parser = argparse.ArgumentParser(description="Run Prediction-as-Action experiment")
    parser.add_argument(
        "--mode", type=str, default="rl", choices=["rl", "mle"], help="Training mode"
    )
    parser.add_argument("--steps", type=int, default=10000, help="Total training steps")
    parser.add_argument(
        "--dynamics",
        type=str,
        default="circular_shift",
        choices=["circular_shift", "random_fixed", "identity"],
        help="Dynamics type",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (cuda or cpu)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    config = PredictionConfig(
        training_mode=args.mode,
        total_steps=args.steps,
        dynamics_type=args.dynamics,
        device=args.device,
        seed=args.seed,
    )

    trainer = PredictionTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
