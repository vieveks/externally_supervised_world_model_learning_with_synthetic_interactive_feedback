"""
Week 2: Compositional Generalization Experiments

Tests whether RL can generalize to novel combinations of seen primitives,
despite the distributional collapse observed in Week 1.

Critical Question: Does RL learn compositional structure even when
distributions collapse?

Key Metrics:
- Accuracy on held-out compositions (target: > chance = 25%)
- RL vs MLE generalization gap
- Which compositions transfer successfully

Usage:
    python run_compositional_experiments.py --all
    python run_compositional_experiments.py --rl-vs-mle
    python run_compositional_experiments.py --composition-breakdown
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import numpy as np

import torch

from src.environment.compositional_grammar import CompositionalGrammar
from src.training.token_reinforce import TokenExperience


def train_on_compositional_grammar(
    grammar: CompositionalGrammar,
    algorithm,
    total_steps: int,
    batch_size: int,
    eval_interval: int,
    device: str,
    test_mode: bool = False,
) -> float:
    """
    Train a model on compositional grammar.

    Args:
        test_mode: If True, evaluate on held-out compositions

    Returns:
        best_accuracy: Best accuracy achieved
    """
    best_accuracy = 0.0
    step = 0

    while step < total_steps:
        # Collect batch of experiences
        experiences = []
        for _ in range(batch_size):
            # Reset environment (always train on training compositions)
            prefix, state = grammar.reset(test_mode=False)

            # Get model prediction
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            logits, value = algorithm.model.forward(state_tensor)
            probs = torch.softmax(logits, dim=-1)
            predicted_token = torch.multinomial(probs, 1).item()

            # Take step in environment (train mode)
            next_state, reward, done, info = grammar.step(predicted_token, test_mode=False)

            # Store experience
            exp = TokenExperience(
                state=state_tensor,
                prediction=predicted_token,
                target=info["suffix"],
                reward=reward,
                log_prob=torch.log(probs[predicted_token] + 1e-10),
                correct=info["correct"],
            )
            experiences.append(exp)

        # Update model
        metrics = algorithm.update(experiences)
        step += batch_size

        # Evaluate periodically
        if step % eval_interval < batch_size or step >= total_steps:  # Trigger when we cross eval_interval or at end
            eval_accuracy = evaluate_on_compositional_grammar(
                grammar=grammar,
                model=algorithm.model,
                num_episodes=100,
                device=device,
                test_mode=test_mode,
            )
            best_accuracy = max(best_accuracy, eval_accuracy)
            mode_str = "Test (Held-out)" if test_mode else "Train"
            print(f"  Step {step}/{total_steps}: {mode_str} Accuracy = {eval_accuracy:.2%} (best: {best_accuracy:.2%})")

    # Final evaluation if we didn't just do one
    if step % eval_interval >= batch_size:
        eval_accuracy = evaluate_on_compositional_grammar(
            grammar=grammar,
            model=algorithm.model,
            num_episodes=100,
            device=device,
            test_mode=test_mode,
        )
        best_accuracy = max(best_accuracy, eval_accuracy)

    return best_accuracy


def evaluate_on_compositional_grammar(
    grammar: CompositionalGrammar,
    model,
    num_episodes: int,
    device: str,
    test_mode: bool = False,
) -> float:
    """Evaluate model on compositional grammar."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for _ in range(num_episodes):
            prefix, state = grammar.reset(test_mode=test_mode)
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            logits, _ = model.forward(state_tensor)
            # Use argmax for deterministic evaluation
            predicted_token = logits.argmax(dim=-1).item()

            next_state, reward, done, info = grammar.step(predicted_token, test_mode=test_mode)
            correct += int(info["correct"])
            total += 1

    model.train()
    return correct / total if total > 0 else 0.0


def run_rl_vs_mle_compositional(
    vocab_size: int = 16,
    num_prefixes: int = 4,
    num_suffixes: int = 4,
    held_out_fraction: float = 0.25,
    total_steps: int = 20000,
    device: str = "cpu",
) -> Dict:
    """
    Core Week 2 experiment: RL vs MLE on compositional generalization.

    This tests whether RL can generalize to novel combinations despite
    the distributional collapse observed in Week 1.
    """
    print("\n" + "=" * 70)
    print("WEEK 2 CORE EXPERIMENT: RL vs MLE on Compositional Generalization")
    print("=" * 70)
    print(f"\nHeld-out fraction: {held_out_fraction:.0%}")
    print("This tests whether RL learns compositional structure that generalizes.\n")

    # Create compositional grammar
    grammar = CompositionalGrammar(
        vocab_size=vocab_size,
        num_prefixes=num_prefixes,
        num_suffixes=num_suffixes,
        held_out_fraction=held_out_fraction,
    )
    stats = grammar.analyze_splits()

    print("Grammar Structure:")
    print(f"  Total compositions: {stats['total_compositions']}")
    print(f"  Training compositions: {stats['train_compositions']}")
    print(f"  Test compositions: {stats['test_compositions']}")
    print(f"  All primitives in train: {stats['all_primitives_in_train']}")

    results = {"grammar_stats": stats}

    from src.models.token_model import TokenPredictionModel
    from src.training.token_reinforce import TokenREINFORCE, TokenMLE

    # Run MLE
    print("\n[1/2] Training MLE on compositional grammar...")
    mle_model = TokenPredictionModel(
        vocab_size=vocab_size,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        ff_dim=256,
    ).to(device)

    mle_algo = TokenMLE(
        model=mle_model,
        lr=1e-3,
        device=device,
    )

    # Train on training compositions
    print("  Training on seen compositions...")
    print(f"  [DEBUG] Training for {total_steps} steps, eval every {total_steps // 4} steps")
    mle_train_acc = train_on_compositional_grammar(
        grammar=grammar,
        algorithm=mle_algo,
        total_steps=total_steps,
        batch_size=32,
        eval_interval=total_steps // 4,  # Less frequent for clarity
        device=device,
        test_mode=False,
    )
    print(f"  [DEBUG] MLE training complete, final train acc: {mle_train_acc:.2%}")

    # Evaluate on held-out compositions
    print("  Evaluating on held-out compositions...")
    mle_test_acc = evaluate_on_compositional_grammar(
        grammar=grammar,
        model=mle_model,
        num_episodes=200,
        device=device,
        test_mode=True,
    )

    results["mle"] = {
        "train_accuracy": mle_train_acc,
        "test_accuracy": mle_test_acc,
    }
    print(f"MLE Train Accuracy: {mle_train_acc:.2%}")
    print(f"MLE Test Accuracy (Generalization): {mle_test_acc:.2%}")

    # Run RL
    print("\n[2/2] Training RL (REINFORCE) on compositional grammar...")
    rl_model = TokenPredictionModel(
        vocab_size=vocab_size,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        ff_dim=256,
    ).to(device)

    rl_algo = TokenREINFORCE(
        model=rl_model,
        lr=1e-3,
        baseline_decay=0.99,
        entropy_coef=0.01,
        device=device,
    )

    # Train on training compositions
    print("  Training on seen compositions...")
    rl_train_acc = train_on_compositional_grammar(
        grammar=grammar,
        algorithm=rl_algo,
        total_steps=total_steps,
        batch_size=32,
        eval_interval=total_steps // 4,
        device=device,
        test_mode=False,
    )

    # Evaluate on held-out compositions
    print("  Evaluating on held-out compositions...")
    rl_test_acc = evaluate_on_compositional_grammar(
        grammar=grammar,
        model=rl_model,
        num_episodes=200,
        device=device,
        test_mode=True,
    )

    results["rl"] = {
        "train_accuracy": rl_train_acc,
        "test_accuracy": rl_test_acc,
    }
    print(f"RL Train Accuracy: {rl_train_acc:.2%}")
    print(f"RL Test Accuracy (Generalization): {rl_test_acc:.2%}")

    # Summary
    print("\n" + "=" * 70)
    print("COMPOSITIONAL GENERALIZATION RESULTS")
    print("=" * 70)
    print(f"Chance baseline: {1.0 / num_suffixes:.2%} (random guessing)")
    print(f"\nTraining (Seen Compositions):")
    print(f"  MLE: {mle_train_acc:.2%}")
    print(f"  RL:  {rl_train_acc:.2%}")
    print(f"\nGeneralization (Held-Out Compositions):")
    print(f"  MLE: {mle_test_acc:.2%}")
    print(f"  RL:  {rl_test_acc:.2%}")
    print(f"\nGeneralization Gap:")
    print(f"  MLE: {(mle_train_acc - mle_test_acc):.2%}")
    print(f"  RL:  {(rl_train_acc - rl_test_acc):.2%}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    chance = 1.0 / num_suffixes

    # RL generalization
    if rl_test_acc > chance * 2:
        print("[SUCCESS] RL generalizes to novel compositions (>2x chance)")
        print("  RL learned compositional structure, not just memorization")
    elif rl_test_acc > chance * 1.5:
        print("[PARTIAL] RL shows some generalization (1.5-2x chance)")
        print("  RL learned partial compositional structure")
    else:
        print("[FAILURE] RL at chance level (<1.5x chance)")
        print("  RL did not learn compositional structure")

    # RL vs MLE comparison
    test_gap = abs(mle_test_acc - rl_test_acc)
    if test_gap <= 0.10:
        print("[SUCCESS] RL matches MLE on generalization (gap ≤10%)")
        print("  RL generalizes as well as MLE despite distributional collapse")
    elif test_gap <= 0.20:
        print("[PARTIAL] RL weaker than MLE on generalization (10-20% gap)")
        print("  RL learns structure but less robustly than MLE")
    else:
        print("[FAILURE] RL significantly trails MLE (>20% gap)")
        print("  Distributional collapse may hurt compositional learning")

    return results


def main():
    parser = argparse.ArgumentParser(description="Week 2: Compositional Generalization Experiments")
    parser.add_argument("--rl-vs-mle", action="store_true",
                       help="Run RL vs MLE on compositional generalization")
    parser.add_argument("--all", action="store_true",
                       help="Run all experiments")
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--prefixes", type=int, default=4)
    parser.add_argument("--suffixes", type=int, default=4)
    parser.add_argument("--held-out", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    all_results = {}

    if args.all or args.rl_vs_mle:
        print("\n" + "=" * 80)
        print("EXPERIMENT 1: RL vs MLE on Compositional Generalization")
        print("=" * 80)
        all_results["rl_vs_mle_compositional"] = run_rl_vs_mle_compositional(
            vocab_size=args.vocab_size,
            num_prefixes=args.prefixes,
            num_suffixes=args.suffixes,
            held_out_fraction=args.held_out,
            total_steps=args.steps,
            device=args.device,
        )

    # If no flags, show help
    if not any([args.rl_vs_mle, args.all]):
        parser.print_help()
        print("\n" + "=" * 70)
        print("WEEK 2: COMPOSITIONAL GENERALIZATION EXPERIMENTS")
        print("=" * 70)
        print("\nCritical Question: Does RL generalize compositionally despite")
        print("                   the distributional collapse from Week 1?")
        print("\nRun with --all to execute all experiments.")

    # Save results
    if all_results:
        save_dir = Path("results/phase3_language/week2_compositional")
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = save_dir / f"compositional_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
