"""Week 2 v2: Factorized Compositional Generalization Experiments.

Tests true compositional structure with independent slots.
"""

import argparse
import torch
import json
from datetime import datetime
from pathlib import Path

from src.environment.factorized_grammar import FactorizedGrammar
from src.models.token_model import TokenPredictionModel
from src.training.token_reinforce import TokenREINFORCE, TokenMLE, TokenExperience


def train_on_factorized_grammar(
    grammar: FactorizedGrammar,
    algorithm,
    total_steps: int,
    batch_size: int,
    eval_interval: int,
    device: str,
    test_mode: bool = False,
) -> float:
    """
    Train a model on factorized grammar.

    Args:
        test_mode: If True, evaluate on held-out sequences

    Returns:
        best_accuracy: Best accuracy achieved
    """
    best_accuracy = 0.0
    step = 0

    while step < total_steps:
        # Collect batch of experiences
        experiences = []
        for _ in range(batch_size):
            # Reset environment (always train on training sequences)
            a_token, state = grammar.reset(test_mode=False)

            # Get model prediction for B given A
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            logits, value = algorithm.model.forward(state_tensor)
            predicted_token = logits.argmax(dim=-1).item()

            # Take step in environment
            next_state, reward, done, info = grammar.step(predicted_token)

            # Store experience
            exp = TokenExperience(
                state=state_tensor,
                prediction=predicted_token,
                target=info["actual"],
                reward=reward,
                log_prob=0.0,  # Not used for MLE
                correct=info["correct"],
            )
            experiences.append(exp)

        # Update model
        metrics = algorithm.update(experiences)
        step += batch_size

        # Evaluate periodically
        if step % eval_interval < batch_size or step >= total_steps:
            eval_accuracy = evaluate_on_factorized_grammar(
                grammar=grammar,
                model=algorithm.model,
                num_episodes=100,
                device=device,
                test_mode=test_mode,
            )
            best_accuracy = max(best_accuracy, eval_accuracy)
            mode_str = "Test (Held-out)" if test_mode else "Train"
            print(f"  Step {step}/{total_steps}: {mode_str} Accuracy = {eval_accuracy:.2%} (best: {best_accuracy:.2%})")

    # Final evaluation
    if step % eval_interval >= batch_size:
        eval_accuracy = evaluate_on_factorized_grammar(
            grammar=grammar,
            model=algorithm.model,
            num_episodes=100,
            device=device,
            test_mode=test_mode,
        )
        best_accuracy = max(best_accuracy, eval_accuracy)

    return best_accuracy


def evaluate_on_factorized_grammar(
    grammar: FactorizedGrammar,
    model,
    num_episodes: int,
    device: str,
    test_mode: bool = False,
) -> float:
    """Evaluate model on factorized grammar."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for _ in range(num_episodes):
            a_token, state = grammar.reset(test_mode=test_mode)
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            logits, _ = model.forward(state_tensor)
            predicted_token = logits.argmax(dim=-1).item()

            next_state, reward, done, info = grammar.step(predicted_token)
            correct += int(info["correct"])
            total += 1

    model.train()
    return correct / total if total > 0 else 0.0


def run_rl_vs_mle_factorized(
    num_a_values: int = 2,
    num_b_values: int = 2,
    held_out_fraction: float = 0.25,
    total_steps: int = 10000,
    device: str = "cpu",
) -> dict:
    """
    Core Week 2 v2 experiment: RL vs MLE on factorized compositional generalization.

    Tests whether RL can recombine independent factors.
    """
    print("=" * 80)
    print("WEEK 2 v2: RL vs MLE on Factorized Compositional Generalization")
    print("=" * 80)

    # Create grammar
    grammar = FactorizedGrammar(
        vocab_size=16,
        num_a_values=num_a_values,
        num_b_values=num_b_values,
        held_out_fraction=held_out_fraction,
    )

    # Show grammar structure
    stats = grammar.analyze_splits()
    print(f"\nHeld-out fraction: {held_out_fraction:.0%}")
    print("This tests whether RL learns factorized structure that generalizes.\n")
    print("Grammar Structure:")
    print(f"  Total sequences: {stats['total_sequences']}")
    print(f"  Training sequences: {stats['train_sequences']}")
    print(f"  Test sequences: {stats['test_sequences']}")
    print(f"  All primitives in train: {stats['all_primitives_in_train']}")

    results = {"grammar_stats": stats}

    # Run MLE
    print("\n[1/2] Training MLE on factorized grammar...")
    mle_model = TokenPredictionModel(
        vocab_size=16,
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

    # Train on training sequences
    print("  Training on seen sequences...")
    mle_train_acc = train_on_factorized_grammar(
        grammar=grammar,
        algorithm=mle_algo,
        total_steps=total_steps,
        batch_size=32,
        eval_interval=total_steps // 4,
        device=device,
        test_mode=False,
    )

    # Evaluate on held-out sequences
    print("  Evaluating on held-out sequences...")
    mle_test_acc = evaluate_on_factorized_grammar(
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
    print("\n[2/2] Training RL (REINFORCE) on factorized grammar...")
    rl_model = TokenPredictionModel(
        vocab_size=16,
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

    # Train on training sequences
    print("  Training on seen sequences...")
    rl_train_acc = train_on_factorized_grammar(
        grammar=grammar,
        algorithm=rl_algo,
        total_steps=total_steps,
        batch_size=32,
        eval_interval=total_steps // 4,
        device=device,
        test_mode=False,
    )

    # Evaluate on held-out sequences
    print("  Evaluating on held-out sequences...")
    rl_test_acc = evaluate_on_factorized_grammar(
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

    # Compare
    print("\n" + "=" * 80)
    print("FACTORIZED COMPOSITIONAL GENERALIZATION RESULTS")
    print("=" * 80)
    print(f"Chance baseline: {100 / num_b_values:.2f}% (random B given A)\n")

    print("Training (Seen Sequences):")
    print(f"  MLE: {mle_train_acc:.2%}")
    print(f"  RL:  {rl_train_acc:.2%}\n")

    print("Generalization (Held-Out Sequences):")
    print(f"  MLE: {mle_test_acc:.2%}")
    print(f"  RL:  {rl_test_acc:.2%}\n")

    print("Generalization Gap:")
    print(f"  MLE: {mle_train_acc - mle_test_acc:.2%}")
    print(f"  RL:  {rl_train_acc - rl_test_acc:.2%}\n")

    # Interpretation
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    chance = 1.0 / num_b_values
    rl_above_chance = rl_test_acc > (1.5 * chance)
    mle_above_chance = mle_test_acc > (1.5 * chance)
    rl_matches_mle = abs(rl_test_acc - mle_test_acc) <= 0.10

    if rl_above_chance:
        print(f"[SUCCESS] RL above chance on test (>{1.5*chance:.0%})")
        print("  RL learned compositional structure that generalizes")
    else:
        print(f"[FAILURE] RL at/near chance on test (<{1.5*chance:.0%})")
        print("  RL did not learn compositional structure")

    if rl_matches_mle:
        print("[SUCCESS] RL matches MLE on generalization (gap ≤10%)")
        print("  RL generalizes as well as MLE despite distributional collapse")
    else:
        if rl_test_acc > mle_test_acc:
            print("[SURPRISE] RL generalizes BETTER than MLE")
        else:
            print("[MIXED] RL generalizes worse than MLE")

    results["interpretation"] = {
        "rl_above_chance": rl_above_chance,
        "mle_above_chance": mle_above_chance,
        "rl_matches_mle": rl_matches_mle,
    }

    # Save results
    output_dir = Path("results/phase3_language/week2_factorized")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"factorized_results_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to {output_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Week 2 v2: Factorized Compositional Generalization")
    parser.add_argument("--num-a", type=int, default=2, help="Number of values for slot A")
    parser.add_argument("--num-b", type=int, default=2, help="Number of values for slot B")
    parser.add_argument("--held-out", type=float, default=0.25, help="Fraction to hold out")
    parser.add_argument("--steps", type=int, default=10000, help="Training steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--all", action="store_true", help="Run all experiments")

    args = parser.parse_args()

    if args.all:
        print("\n" + "=" * 80)
        print("EXPERIMENT 1: RL vs MLE on Factorized Composition (2x2)")
        print("=" * 80)
        run_rl_vs_mle_factorized(
            num_a_values=2,
            num_b_values=2,
            held_out_fraction=0.25,
            total_steps=args.steps,
            device=args.device,
        )
    else:
        run_rl_vs_mle_factorized(
            num_a_values=args.num_a,
            num_b_values=args.num_b,
            held_out_fraction=args.held_out,
            total_steps=args.steps,
            device=args.device,
        )


if __name__ == "__main__":
    main()
