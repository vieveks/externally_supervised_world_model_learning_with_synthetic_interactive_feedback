"""
Week 1: Ambiguous Grammar Experiments

Tests whether RL can maintain distributions over ambiguous tokens
or collapses to argmax.

Critical Question: Does RL handle genuine ambiguity (multiple valid
continuations with equal probability)?

Metrics:
- Accuracy (should reach ~75% oracle ceiling)
- Policy entropy at ambiguous points (should be >0.5)
- KL divergence from oracle (should be <0.5)
- RL vs MLE comparison

Usage:
    python run_ambiguous_experiments.py --all
    python run_ambiguous_experiments.py --rl-vs-mle
    python run_ambiguous_experiments.py --entropy-analysis
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import numpy as np

import torch

from src.environment.ambiguous_grammar import AmbiguousGrammar
from src.training.token_reinforce import TokenExperience


def train_on_ambiguous_grammar(
    grammar: AmbiguousGrammar,
    algorithm,
    total_steps: int,
    batch_size: int,
    eval_interval: int,
    device: str,
) -> float:
    """
    Train a model on ambiguous grammar.

    Returns best accuracy achieved.
    """
    best_accuracy = 0.0
    step = 0

    while step < total_steps:
        # Collect batch of experiences
        experiences = []
        for _ in range(batch_size):
            # Reset environment
            current_token, state = grammar.reset()

            # Get model prediction
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            logits, value = algorithm.model.forward(state_tensor)
            probs = torch.softmax(logits, dim=-1)
            predicted_token = torch.multinomial(probs, 1).item()

            # Take step in environment
            next_state, reward, done, info = grammar.step(predicted_token)

            # Store experience
            exp = TokenExperience(
                state=state_tensor,
                prediction=predicted_token,
                target=info["actual"],
                reward=reward,
                log_prob=torch.log(probs[predicted_token] + 1e-10),
                correct=info["correct"],
            )
            experiences.append(exp)

        # Update model
        metrics = algorithm.update(experiences)
        step += batch_size

        # Evaluate periodically
        if step % eval_interval == 0:
            eval_accuracy = evaluate_on_ambiguous_grammar(
                grammar=grammar,
                model=algorithm.model,
                num_episodes=100,
                device=device,
            )
            best_accuracy = max(best_accuracy, eval_accuracy)
            print(f"  Step {step}/{total_steps}: Accuracy = {eval_accuracy:.2%} (best: {best_accuracy:.2%})")

    return best_accuracy


def evaluate_on_ambiguous_grammar(
    grammar: AmbiguousGrammar,
    model,
    num_episodes: int,
    device: str,
) -> float:
    """Evaluate model on ambiguous grammar."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for _ in range(num_episodes):
            current_token, state = grammar.reset()
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            logits, _ = model.forward(state_tensor)
            probs = torch.softmax(logits, dim=-1)
            predicted_token = torch.multinomial(probs, 1).item()

            next_state, reward, done, info = grammar.step(predicted_token)
            correct += int(info["correct"])
            total += 1

    model.train()
    return correct / total if total > 0 else 0.0


def run_rl_vs_mle_ambiguous(
    vocab_size: int = 16,
    ambiguity_level: str = "high",
    total_steps: int = 20000,
    device: str = "cpu",
) -> Dict:
    """
    Core Week 1 experiment: RL vs MLE on ambiguous grammar.

    This tests whether RL maintains distribution or collapses to argmax.
    """
    print("\n" + "=" * 70)
    print("WEEK 1 CORE EXPERIMENT: RL vs MLE on Ambiguous Grammar")
    print("=" * 70)
    print(f"\nAmbiguity Level: {ambiguity_level}")
    print("This tests whether RL can maintain distributions over ambiguous tokens.\n")

    # Create ambiguous grammar to get oracle statistics
    grammar = AmbiguousGrammar(
        vocab_size=vocab_size,
        ambiguity_level=ambiguity_level,
    )
    stats = grammar.analyze_grammar()

    print("Grammar Properties:")
    print(f"  Oracle accuracy: {stats['oracle_accuracy']:.2%}")
    print(f"  Ambiguous tokens: {stats['num_ambiguous_tokens']}/{vocab_size}")
    print(f"  Average entropy: {stats['average_entropy']:.3f}")
    print(f"  Normalized entropy: {stats['normalized_entropy']:.3f}")

    results = {"grammar_stats": stats}

    # For now, we need to manually train on ambiguous grammar
    # TODO: Integrate AmbiguousGrammar into TokenTrainer infrastructure

    print("\n[NOTICE] Full integration pending - running standalone experiments")
    print("Will train RL and MLE directly on AmbiguousGrammar task")

    # Placeholder - will implement direct training loop
    from src.models.token_model import TokenPredictionModel
    from src.training.token_reinforce import TokenREINFORCE, TokenMLE

    # Create model for MLE
    print("\n[1/2] Training MLE on ambiguous grammar...")
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

    # Train MLE
    mle_accuracy = train_on_ambiguous_grammar(
        grammar=grammar,
        algorithm=mle_algo,
        total_steps=total_steps,
        batch_size=32,
        eval_interval=250,
        device=device,
    )

    results["mle"] = {"accuracy": mle_accuracy}
    print(f"MLE Accuracy: {mle_accuracy:.2%}")

    # Create model for RL
    print("\n[2/2] Training RL (REINFORCE) on ambiguous grammar...")
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

    # Train RL
    rl_accuracy = train_on_ambiguous_grammar(
        grammar=grammar,
        algorithm=rl_algo,
        total_steps=total_steps,
        batch_size=32,
        eval_interval=250,
        device=device,
    )

    results["rl"] = {"accuracy": rl_accuracy}
    print(f"RL Accuracy: {rl_accuracy:.2%}")

    # Summary
    print("\n" + "=" * 70)
    print("AMBIGUOUS GRAMMAR RESULTS")
    print("=" * 70)
    print(f"Oracle Ceiling: {stats['oracle_accuracy']:.2%}")
    print(f"MLE Accuracy:   {mle_accuracy:.2%}")
    print(f"RL Accuracy:    {rl_accuracy:.2%}")
    gap = abs(mle_accuracy - rl_accuracy)
    print(f"RL vs MLE Gap:  {gap:.2%}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    oracle_acc = stats['oracle_accuracy']
    if rl_accuracy >= oracle_acc * 0.9:
        print("[SUCCESS] RL achieves ~oracle accuracy (≥90% of ceiling)")
        print("  RL learns the distribution, doesn't just memorize argmax")
    elif rl_accuracy >= oracle_acc * 0.7:
        print("[PARTIAL] RL achieves reasonable accuracy (70-90% of ceiling)")
        print("  RL learns some distribution structure")
    else:
        print("[FAILURE] RL significantly underperforms oracle (<70%)")
        print("  RL may be collapsing to argmax or failing to learn")

    if gap <= 0.10:
        print("[SUCCESS] RL matches MLE (gap ≤10%)")
    else:
        print(f"[CAUTION] RL trails MLE by {gap:.2%}")

    return results


def analyze_policy_entropy(
    vocab_size: int = 16,
    ambiguity_level: str = "high",
    total_steps: int = 20000,
    device: str = "cpu",
) -> Dict:
    """
    Analyze policy entropy at ambiguous decision points.

    Key metric: Does RL maintain high entropy or collapse to low entropy?
    """
    print("\n" + "=" * 70)
    print("POLICY ENTROPY ANALYSIS")
    print("=" * 70)

    # Create grammar
    grammar = AmbiguousGrammar(
        vocab_size=vocab_size,
        ambiguity_level=ambiguity_level,
    )

    # Train RL model
    print("\n[1/1] Training RL model...")
    from src.models.token_model import TokenPredictionModel
    from src.training.token_reinforce import TokenREINFORCE

    model = TokenPredictionModel(
        vocab_size=vocab_size,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        ff_dim=256,
    ).to(device)

    algorithm = TokenREINFORCE(
        model=model,
        lr=1e-3,
        baseline_decay=0.99,
        entropy_coef=0.01,
        device=device,
    )

    # Train
    _ = train_on_ambiguous_grammar(
        grammar=grammar,
        algorithm=algorithm,
        total_steps=total_steps,
        batch_size=32,
        eval_interval=250,
        device=device,
    )
    model.eval()

    # Analyze entropy at each token
    print("\n" + "=" * 70)
    print("ENTROPY ANALYSIS")
    print("=" * 70)

    entropies = {}
    kl_divergences = {}

    with torch.no_grad():
        for token in range(vocab_size):
            # Get oracle distribution
            oracle_dist = grammar.get_oracle_distribution(token).to(device)

            # Get learned policy distribution
            state = grammar.encode_token(token)
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            logits, _ = model.forward(state_tensor)
            policy_dist = torch.softmax(logits, dim=-1)

            # Compute entropy
            entropy = -torch.sum(policy_dist * torch.log(policy_dist + 1e-10))
            entropies[token] = entropy.item()

            # Compute KL divergence
            oracle_probs = oracle_dist + 1e-10
            policy_probs = policy_dist + 1e-10
            kl = torch.sum(policy_probs * torch.log(policy_probs / oracle_probs))
            kl_divergences[token] = kl.item()

            # Print for ambiguous tokens
            if grammar.is_ambiguous(token):
                oracle_entropy = grammar.compute_entropy(token)
                print(f"\nToken {token} (AMBIGUOUS):")
                print(f"  Oracle dist: {oracle_dist.cpu().numpy()}")
                print(f"  Policy dist: {policy_dist.cpu().numpy()}")
                print(f"  Oracle entropy: {oracle_entropy:.3f}")
                print(f"  Policy entropy: {entropy:.3f}")
                print(f"  KL divergence: {kl:.3f}")

    # Summary statistics
    ambiguous_tokens = [t for t in range(vocab_size) if grammar.is_ambiguous(t)]
    deterministic_tokens = [t for t in range(vocab_size) if not grammar.is_ambiguous(t)]

    avg_entropy_ambiguous = np.mean([entropies[t] for t in ambiguous_tokens])
    avg_entropy_deterministic = np.mean([entropies[t] for t in deterministic_tokens])
    avg_kl_ambiguous = np.mean([kl_divergences[t] for t in ambiguous_tokens])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Average entropy (ambiguous tokens): {avg_entropy_ambiguous:.3f}")
    print(f"Average entropy (deterministic tokens): {avg_entropy_deterministic:.3f}")
    print(f"Average KL divergence (ambiguous): {avg_kl_ambiguous:.3f}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if avg_entropy_ambiguous > 0.5:
        print("[SUCCESS] High entropy at ambiguous points (>0.5)")
        print("  Policy maintains distribution, doesn't collapse")
    elif avg_entropy_ambiguous > 0.3:
        print("[PARTIAL] Moderate entropy at ambiguous points (0.3-0.5)")
        print("  Policy shows some distribution, but may be collapsing")
    else:
        print("[FAILURE] Low entropy at ambiguous points (<0.3)")
        print("  Policy has collapsed to argmax")

    if avg_kl_ambiguous < 0.5:
        print("[SUCCESS] Low KL divergence (<0.5)")
        print("  Policy closely matches oracle distribution")
    elif avg_kl_ambiguous < 1.0:
        print("[PARTIAL] Moderate KL divergence (0.5-1.0)")
        print("  Policy approximates oracle but with some deviation")
    else:
        print("[FAILURE] High KL divergence (≥1.0)")
        print("  Policy significantly differs from oracle (collapsed?)")

    return {
        "entropies": entropies,
        "kl_divergences": kl_divergences,
        "avg_entropy_ambiguous": avg_entropy_ambiguous,
        "avg_entropy_deterministic": avg_entropy_deterministic,
        "avg_kl_ambiguous": avg_kl_ambiguous,
    }


def main():
    parser = argparse.ArgumentParser(description="Week 1: Ambiguous Grammar Experiments")
    parser.add_argument("--rl-vs-mle", action="store_true",
                       help="Run RL vs MLE comparison")
    parser.add_argument("--entropy-analysis", action="store_true",
                       help="Analyze policy entropy and KL divergence")
    parser.add_argument("--all", action="store_true",
                       help="Run all experiments")
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--ambiguity", type=str, default="high",
                       choices=["low", "medium", "high"])
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    all_results = {}

    if args.all or args.rl_vs_mle:
        print("\n" + "=" * 80)
        print("EXPERIMENT 1: RL vs MLE on Ambiguous Grammar")
        print("=" * 80)
        all_results["rl_vs_mle"] = run_rl_vs_mle_ambiguous(
            vocab_size=args.vocab_size,
            ambiguity_level=args.ambiguity,
            total_steps=args.steps,
            device=args.device,
        )

    if args.all or args.entropy_analysis:
        print("\n" + "=" * 80)
        print("EXPERIMENT 2: Policy Entropy Analysis")
        print("=" * 80)
        all_results["entropy_analysis"] = analyze_policy_entropy(
            vocab_size=args.vocab_size,
            ambiguity_level=args.ambiguity,
            total_steps=args.steps,
            device=args.device,
        )

    # If no flags, show help
    if not any([args.rl_vs_mle, args.entropy_analysis, args.all]):
        parser.print_help()
        print("\n" + "=" * 70)
        print("WEEK 1: AMBIGUOUS GRAMMAR EXPERIMENTS")
        print("=" * 70)
        print("\nCritical Question: Does RL maintain distribution or collapse to argmax?")
        print("\nRun with --all to execute all experiments.")

    # Save results
    if all_results:
        save_dir = Path("results/phase3_language/week1_ambiguous")
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = save_dir / f"ambiguous_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
