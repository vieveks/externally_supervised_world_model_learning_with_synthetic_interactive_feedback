"""
Run Token Prediction Experiments (Phase 3.1)

This script runs the core Phase 3.1 experiments:
1. RL vs MLE on deterministic grammar
2. Different grammar types
3. Delayed reward / credit assignment
4. TD(λ) vs vanilla REINFORCE

Usage:
    python run_token_prediction.py                    # Run default RL experiment
    python run_token_prediction.py --mode mle         # Run MLE baseline
    python run_token_prediction.py --compare          # Run RL vs MLE comparison
    python run_token_prediction.py --delay-ablation   # Run delay ablation
    python run_token_prediction.py --all              # Run all experiments
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import torch

from src.training.token_trainer import TokenTrainerConfig, TokenTrainer


def run_single_experiment(config: TokenTrainerConfig) -> dict:
    """Run a single experiment and return results."""
    trainer = TokenTrainer(config)
    results = trainer.train()
    return {
        "config": {
            "training_mode": config.training_mode,
            "grammar_type": config.grammar_type,
            "vocab_size": config.vocab_size,
            "sequence_length": config.sequence_length,
            "reward_delay": config.reward_delay,
        },
        "results": results,
        "best_accuracy": trainer.best_accuracy,
    }


def run_rl_vs_mle_comparison(
    vocab_size: int = 16,
    grammar_type: str = "deterministic_cyclic",
    total_steps: int = 10000,
    device: str = "cpu",
):
    """
    Core experiment: Compare RL and MLE on token prediction.

    If RL matches MLE, we've shown RL gradients can replace pretraining.
    """
    print("\n" + "=" * 70)
    print("PHASE 3.1 CORE EXPERIMENT: RL vs MLE on Token Prediction")
    print("=" * 70)

    results = {}

    # Run MLE
    print("\n[1/2] Running MLE baseline...")
    mle_config = TokenTrainerConfig(
        vocab_size=vocab_size,
        grammar_type=grammar_type,
        training_mode="mle",
        total_steps=total_steps,
        device=device,
        seed=42,
    )
    mle_result = run_single_experiment(mle_config)
    results["mle"] = mle_result
    print(f"MLE Final Accuracy: {mle_result['best_accuracy']:.2%}")

    # Run RL
    print("\n[2/2] Running RL (REINFORCE)...")
    rl_config = TokenTrainerConfig(
        vocab_size=vocab_size,
        grammar_type=grammar_type,
        training_mode="rl",
        total_steps=total_steps,
        device=device,
        seed=42,
    )
    rl_result = run_single_experiment(rl_config)
    results["rl"] = rl_result
    print(f"RL Final Accuracy: {rl_result['best_accuracy']:.2%}")

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"Grammar Type: {grammar_type}")
    print(f"Vocab Size: {vocab_size}")
    print(f"MLE Accuracy: {mle_result['best_accuracy']:.2%}")
    print(f"RL Accuracy:  {rl_result['best_accuracy']:.2%}")
    gap = abs(mle_result['best_accuracy'] - rl_result['best_accuracy'])
    print(f"Gap: {gap:.2%}")

    if rl_result['best_accuracy'] >= 0.95 and mle_result['best_accuracy'] >= 0.95:
        print("\n✓ SUCCESS: Both RL and MLE achieve >95% accuracy!")
        print("  This validates the prediction-as-action hypothesis for tokens.")
    elif rl_result['best_accuracy'] >= 0.95:
        print("\n✓ RL succeeds (>95%), MLE underperforms.")
    elif mle_result['best_accuracy'] >= 0.95:
        print("\n✗ MLE succeeds but RL fails. Investigate hyperparameters.")
    else:
        print("\n✗ Both methods fail. Task may be too hard or bug in implementation.")

    return results


def run_delay_ablation(
    vocab_size: int = 16,
    grammar_type: str = "deterministic_cyclic",
    total_steps: int = 15000,
    device: str = "cpu",
):
    """
    Credit assignment ablation: How does RL perform with delayed rewards?

    Tests sequence lengths 1, 3, 5, 7 with reward at end.
    """
    print("\n" + "=" * 70)
    print("DELAY ABLATION: Credit Assignment Challenge")
    print("=" * 70)

    sequence_lengths = [1, 3, 5, 7]
    results = {}

    for i, seq_len in enumerate(sequence_lengths, 1):
        print(f"\n{'='*60}")
        print(f"[Experiment {i}/4: Sequence Length = {seq_len}]")
        print(f"{'='*60}")

        if seq_len == 1:
            # Single step, no delay
            config = TokenTrainerConfig(
                vocab_size=vocab_size,
                grammar_type=grammar_type,
                training_mode="rl",
                sequence_length=1,
                total_steps=total_steps,
                device=device,
                seed=42,
            )
        else:
            # Multi-step with delayed reward
            config = TokenTrainerConfig(
                vocab_size=vocab_size,
                grammar_type=grammar_type,
                training_mode="rl",
                sequence_length=seq_len,
                total_steps=total_steps,
                device=device,
                seed=42,
            )

        result = run_single_experiment(config)
        results[f"seq_{seq_len}"] = result
        print(f"Sequence {seq_len}: Accuracy = {result['best_accuracy']:.2%}")

    # Summary
    print("\n" + "=" * 70)
    print("DELAY ABLATION SUMMARY")
    print("=" * 70)
    print(f"{'Sequence Length':<20} {'Accuracy':<15}")
    print("-" * 35)
    for seq_len in sequence_lengths:
        acc = results[f"seq_{seq_len}"]["best_accuracy"]
        print(f"{seq_len:<20} {acc:.2%}")

    return results


def run_td_lambda_comparison(
    vocab_size: int = 16,
    grammar_type: str = "deterministic_cyclic",
    sequence_length: int = 5,
    total_steps: int = 20000,
    device: str = "cpu",
):
    """
    Compare TD(λ) vs vanilla REINFORCE on delayed reward.

    TD(λ) should improve credit assignment.
    """
    print("\n" + "=" * 70)
    print("TD(λ) vs REINFORCE: Credit Assignment Comparison")
    print("=" * 70)
    print(f"Sequence Length: {sequence_length}")

    results = {}

    # Vanilla REINFORCE
    print("\n[1/2] Running vanilla REINFORCE...")
    reinforce_config = TokenTrainerConfig(
        vocab_size=vocab_size,
        grammar_type=grammar_type,
        training_mode="rl",
        sequence_length=sequence_length,
        total_steps=total_steps,
        device=device,
        seed=42,
    )
    reinforce_result = run_single_experiment(reinforce_config)
    results["reinforce"] = reinforce_result
    print(f"REINFORCE Accuracy: {reinforce_result['best_accuracy']:.2%}")

    # TD(λ)
    print("\n[2/2] Running TD(λ)...")
    td_config = TokenTrainerConfig(
        vocab_size=vocab_size,
        grammar_type=grammar_type,
        training_mode="td_lambda",
        sequence_length=sequence_length,
        lambda_=0.9,
        total_steps=total_steps,
        device=device,
        seed=42,
    )
    td_result = run_single_experiment(td_config)
    results["td_lambda"] = td_result
    print(f"TD(λ) Accuracy: {td_result['best_accuracy']:.2%}")

    # Summary
    print("\n" + "=" * 70)
    print("TD(λ) COMPARISON SUMMARY")
    print("=" * 70)
    print(f"Sequence Length: {sequence_length}")
    print(f"REINFORCE: {reinforce_result['best_accuracy']:.2%}")
    print(f"TD(λ=0.9): {td_result['best_accuracy']:.2%}")

    improvement = td_result['best_accuracy'] - reinforce_result['best_accuracy']
    if improvement > 0.05:
        print(f"\n✓ TD(λ) improves by {improvement:.2%}")
    elif improvement > 0:
        print(f"\n~ TD(λ) slightly improves by {improvement:.2%}")
    else:
        print(f"\n✗ TD(λ) does not improve over REINFORCE")

    return results


def run_grammar_comparison(
    vocab_size: int = 16,
    total_steps: int = 10000,
    device: str = "cpu",
):
    """
    Compare RL on different grammar types.

    Tests: deterministic_cyclic, deterministic_permutation, bigram
    """
    print("\n" + "=" * 70)
    print("GRAMMAR COMPARISON: Different Transition Dynamics")
    print("=" * 70)

    grammar_types = ["deterministic_cyclic", "deterministic_permutation", "bigram"]
    results = {}

    for grammar in grammar_types:
        print(f"\n[Grammar: {grammar}]")
        config = TokenTrainerConfig(
            vocab_size=vocab_size,
            grammar_type=grammar,
            training_mode="rl",
            total_steps=total_steps,
            device=device,
            seed=42,
        )
        result = run_single_experiment(config)
        results[grammar] = result
        print(f"{grammar}: Accuracy = {result['best_accuracy']:.2%}")

    # Summary
    print("\n" + "=" * 70)
    print("GRAMMAR COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Grammar Type':<30} {'Accuracy':<15}")
    print("-" * 45)
    for grammar in grammar_types:
        acc = results[grammar]["best_accuracy"]
        print(f"{grammar:<30} {acc:.2%}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 3.1 Token Prediction Experiments")
    parser.add_argument("--mode", type=str, default="rl", choices=["rl", "mle"])
    parser.add_argument("--compare", action="store_true", help="Run RL vs MLE comparison")
    parser.add_argument("--delay-ablation", action="store_true", help="Run delay ablation")
    parser.add_argument("--td-lambda", action="store_true", help="Run TD(λ) comparison")
    parser.add_argument("--grammar-compare", action="store_true", help="Run grammar comparison")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="results/phase3_1")

    args = parser.parse_args()

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    if args.all or args.compare:
        all_results["rl_vs_mle"] = run_rl_vs_mle_comparison(
            vocab_size=args.vocab_size,
            total_steps=args.steps,
            device=args.device,
        )

    if args.all or args.delay_ablation:
        all_results["delay_ablation"] = run_delay_ablation(
            vocab_size=args.vocab_size,
            total_steps=args.steps,
            device=args.device,
        )

    if args.all or args.td_lambda:
        all_results["td_lambda"] = run_td_lambda_comparison(
            vocab_size=args.vocab_size,
            total_steps=args.steps * 2,  # More steps for this
            device=args.device,
        )

    if args.all or args.grammar_compare:
        all_results["grammar_comparison"] = run_grammar_comparison(
            vocab_size=args.vocab_size,
            total_steps=args.steps,
            device=args.device,
        )

    # If no special flags, run single experiment
    if not any([args.compare, args.delay_ablation, args.td_lambda, args.grammar_compare, args.all]):
        print(f"\nRunning single {args.mode.upper()} experiment...")
        config = TokenTrainerConfig(
            vocab_size=args.vocab_size,
            training_mode=args.mode,
            total_steps=args.steps,
            device=args.device,
        )
        result = run_single_experiment(config)
        all_results["single"] = result

    # Save all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = save_dir / f"results_{timestamp}.json"

    # Convert tensors to python types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        else:
            return obj

    with open(results_file, "w") as f:
        json.dump(convert_for_json(all_results), f, indent=2, default=str)

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
