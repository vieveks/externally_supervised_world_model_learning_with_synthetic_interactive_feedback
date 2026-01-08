#!/usr/bin/env python3
"""
WMIL Training Script

Run the MVP experiment:
    python train.py

With custom config:
    python train.py --config configs/custom.yaml

Override specific params:
    python train.py --total_steps 20000 --lr 0.001
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import Config, load_config
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="WMIL Training")

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )

    # Override options
    parser.add_argument("--task", type=str, help="Task name")
    parser.add_argument("--num_actions", type=int, help="Number of actions")
    parser.add_argument("--hidden_dim", type=int, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, help="Number of transformer layers")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--total_steps", type=int, help="Total training steps")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--log_interval", type=int, help="Log interval")
    parser.add_argument("--save_interval", type=int, help="Save interval")
    parser.add_argument("--device", type=str, help="Device (cpu/cuda)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(str(config_path))
        print(f"Loaded config from {config_path}")
    else:
        config = Config()
        print("Using default config")

    # Override with command line args
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            setattr(config, key, value)
            print(f"Override: {key} = {value}")

    # Print config summary
    print("\n" + "=" * 50)
    print("WMIL MVP Configuration")
    print("=" * 50)
    print(f"Task: {config.task}")
    print(f"Actions: {config.num_actions}")
    print(f"Model: {config.num_layers} layers, hidden={config.hidden_dim}")
    print(f"Algorithm: {config.algorithm}")
    print(f"Total steps: {config.total_steps}")
    print(f"Batch size: {config.batch_size}")
    print(f"Device: {config.device}")
    print("=" * 50 + "\n")

    # Create trainer and run
    trainer = Trainer(config)
    metrics = trainer.train()

    return metrics


if __name__ == "__main__":
    main()
