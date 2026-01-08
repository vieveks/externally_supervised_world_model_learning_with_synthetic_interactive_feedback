"""Configuration handling for WMIL."""

from dataclasses import dataclass, field
from typing import Optional
import yaml
import torch


@dataclass
class Config:
    """Configuration for WMIL training."""

    # Task
    task: str = "pattern_echo"
    num_actions: int = 8
    sequence_length: int = 4  # For SequenceNext task

    # Model
    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    ff_dim: int = 256
    dropout: float = 0.0

    # State
    state_dim: int = 8

    # Training
    algorithm: str = "reinforce"
    lr: float = 1e-3
    baseline_decay: float = 0.99
    entropy_coef: float = 0.01
    world_model_coef: float = 1.0

    # Environment
    parent: str = "deterministic"
    max_episode_length: int = 1

    # Rollout
    batch_size: int = 64
    total_steps: int = 10000
    log_interval: int = 100
    save_interval: int = 1000

    # Two-phase training (frozen WM experiment)
    wm_warmup_steps: int = 0  # If > 0, train WM first, then freeze

    # Model-Based Value Expansion (MVE)
    mve_horizon: int = 0  # If > 0, use WM for value expansion (H-step lookahead)
    gamma: float = 0.99  # Discount factor for MVE

    # Paths
    experiment_dir: str = "experiments"
    log_dir: str = "runs"

    # Device
    device: str = "auto"

    def __post_init__(self):
        """Set device after initialization."""
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


def load_config(path: str) -> Config:
    """Load configuration from YAML file."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    # Ensure numeric types are correct (YAML sometimes parses 1e-3 as string)
    float_fields = ['lr', 'baseline_decay', 'entropy_coef', 'world_model_coef', 'dropout']
    for field in float_fields:
        if field in data and isinstance(data[field], str):
            data[field] = float(data[field])

    return Config(**data)
