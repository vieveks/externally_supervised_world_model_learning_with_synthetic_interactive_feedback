"""
Stage 3.1b: Stochastic Token Prediction Experiments

This script tests whether RL can handle probabilistic token transitions.

Key questions:
1. Does RL accuracy degrade with stochasticity compared to deterministic?
2. How does RL compare to MLE on stochastic grammars?
3. Does credit assignment become harder with stochasticity?
4. Do RL and MLE learn similar representations (CCA analysis)?

Usage:
    python run_stochastic_experiments.py --all                    # Run all experiments
    python run_stochastic_experiments.py --stochastic-compare     # RL vs MLE on stochastic
    python run_stochastic_experiments.py --stochastic-delay       # Delay ablation on stochastic
    python run_stochastic_experiments.py --det-vs-stoch           # Compare deterministic vs stochastic
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import torch
import numpy as np

from src.training.token_trainer import TokenTrainerConfig, TokenTrainer


def run_stochastic_rl_vs_mle(
    vocab_size: int = 16,
    total_steps: int = 15000,
    device: str = "cpu",
    save_results: bool = True,
) -> Dict:
    """
    Core Stage 3.1b experiment: RL vs MLE on stochastic bigram grammar.

    This is the CRITICAL test - determinism was too easy.
    """
    print("\n" + "=" * 70)
    print("STAGE 3.1b CORE EXPERIMENT: RL vs MLE on STOCHASTIC Grammar")
    print("=" * 70)
    print("\nThis is the real test - deterministic grammars were too easy.")
    print("If RL handles stochasticity well, the approach is robust.\n")

    results = {}

    # Run MLE on stochastic
    print("[1/2] Running MLE on stochastic bigram grammar...")
    mle_config = TokenTrainerConfig(
        vocab_size=vocab_size,
        grammar_type="bigram",
        training_mode="mle",
        total_steps=total_steps,
        device=device,
        seed=42,
    )
    mle_trainer = TokenTrainer(mle_config)
    mle_result = mle_trainer.train()
    results["mle_stochastic"] = {
        "config": {
            "grammar_type": "bigram",
            "training_mode": "mle",
            "vocab_size": vocab_size,
        },
        "best_accuracy": mle_trainer.best_accuracy,
        "final_eval": mle_result,
    }
    print(f"MLE (Stochastic) Final Accuracy: {mle_trainer.best_accuracy:.2%}")

    # Run RL on stochastic
    print("\n[2/2] Running RL (REINFORCE) on stochastic bigram grammar...")
    rl_config = TokenTrainerConfig(
        vocab_size=vocab_size,
        grammar_type="bigram",
        training_mode="rl",
        total_steps=total_steps,
        device=device,
        seed=42,
    )
    rl_trainer = TokenTrainer(rl_config)
    rl_result = rl_trainer.train()
    results["rl_stochastic"] = {
        "config": {
            "grammar_type": "bigram",
            "training_mode": "rl",
            "vocab_size": vocab_size,
        },
        "best_accuracy": rl_trainer.best_accuracy,
        "final_eval": rl_result,
    }
    print(f"RL (Stochastic) Final Accuracy: {rl_trainer.best_accuracy:.2%}")

    # Summary
    print("\n" + "=" * 70)
    print("STOCHASTIC COMPARISON SUMMARY")
    print("=" * 70)
    print(f"Grammar Type: bigram (stochastic)")
    print(f"Vocab Size: {vocab_size}")
    print(f"MLE Accuracy: {mle_trainer.best_accuracy:.2%}")
    print(f"RL Accuracy:  {rl_trainer.best_accuracy:.2%}")
    gap = abs(mle_trainer.best_accuracy - rl_trainer.best_accuracy)
    print(f"Gap: {gap:.2%}")

    # Interpret results
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if rl_trainer.best_accuracy >= 0.80:
        print("✓ SUCCESS: RL handles stochasticity well (≥80% accuracy)")
        print("  The prediction-as-action approach extends beyond deterministic grammars.")
        if gap <= 0.10:
            print("✓ RL matches MLE within 10% - strong result!")
        else:
            print(f"~ RL trails MLE by {gap:.2%} - acceptable for stochastic case")
    elif rl_trainer.best_accuracy >= 0.60:
        print("~ PARTIAL SUCCESS: RL learns stochastic patterns (60-80%)")
        print("  Approach works but degrades with stochasticity")
    else:
        print("✗ FAILURE: RL struggles with stochasticity (<60%)")
        print("  Approach may be limited to deterministic grammars")
        print("  Consider: More training steps? Better credit assignment (TD-λ)?")

    if save_results:
        save_dir = Path("results/phase3_1b")
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = save_dir / f"stochastic_rl_vs_mle_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {results_file}")

    return results


def run_deterministic_vs_stochastic_comparison(
    vocab_size: int = 16,
    total_steps: int = 15000,
    device: str = "cpu",
) -> Dict:
    """
    Compare RL performance on deterministic vs stochastic grammars.

    Tests whether stochasticity fundamentally changes learning dynamics.
    """
    print("\n" + "=" * 70)
    print("DETERMINISTIC vs STOCHASTIC COMPARISON")
    print("=" * 70)
    print("\nHow much does stochasticity hurt RL performance?\n")

    results = {}

    # RL on deterministic cyclic
    print("[1/3] Running RL on deterministic cyclic grammar...")
    det_config = TokenTrainerConfig(
        vocab_size=vocab_size,
        grammar_type="deterministic_cyclic",
        training_mode="rl",
        total_steps=total_steps,
        device=device,
        seed=42,
    )
    det_trainer = TokenTrainer(det_config)
    det_result = det_trainer.train()
    results["rl_deterministic_cyclic"] = {
        "grammar_type": "deterministic_cyclic",
        "best_accuracy": det_trainer.best_accuracy,
    }

    # RL on deterministic permutation
    print("\n[2/3] Running RL on deterministic permutation grammar...")
    perm_config = TokenTrainerConfig(
        vocab_size=vocab_size,
        grammar_type="deterministic_permutation",
        training_mode="rl",
        total_steps=total_steps,
        device=device,
        seed=42,
    )
    perm_trainer = TokenTrainer(perm_config)
    perm_result = perm_trainer.train()
    results["rl_deterministic_permutation"] = {
        "grammar_type": "deterministic_permutation",
        "best_accuracy": perm_trainer.best_accuracy,
    }

    # RL on stochastic bigram
    print("\n[3/3] Running RL on stochastic bigram grammar...")
    stoch_config = TokenTrainerConfig(
        vocab_size=vocab_size,
        grammar_type="bigram",
        training_mode="rl",
        total_steps=total_steps,
        device=device,
        seed=42,
    )
    stoch_trainer = TokenTrainer(stoch_config)
    stoch_result = stoch_trainer.train()
    results["rl_stochastic_bigram"] = {
        "grammar_type": "bigram",
        "best_accuracy": stoch_trainer.best_accuracy,
    }

    # Summary
    print("\n" + "=" * 70)
    print("GRAMMAR COMPARISON SUMMARY (RL only)")
    print("=" * 70)
    print(f"{'Grammar Type':<30} {'RL Accuracy':<15}")
    print("-" * 45)
    print(f"{'Deterministic Cyclic':<30} {det_trainer.best_accuracy:.2%}")
    print(f"{'Deterministic Permutation':<30} {perm_trainer.best_accuracy:.2%}")
    print(f"{'Stochastic Bigram':<30} {stoch_trainer.best_accuracy:.2%}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    avg_det = (det_trainer.best_accuracy + perm_trainer.best_accuracy) / 2
    stoch = stoch_trainer.best_accuracy
    degradation = avg_det - stoch

    print(f"Average Deterministic: {avg_det:.2%}")
    print(f"Stochastic: {stoch:.2%}")
    print(f"Degradation: {degradation:.2%}")

    if degradation <= 0.10:
        print("\n✓ Stochasticity has MINIMAL impact (≤10% degradation)")
        print("  RL handles probabilistic transitions robustly")
    elif degradation <= 0.20:
        print("\n~ Stochasticity has MODERATE impact (10-20% degradation)")
        print("  RL works but performance degrades with stochasticity")
    else:
        print("\n✗ Stochasticity has MAJOR impact (>20% degradation)")
        print("  RL may be limited to deterministic or near-deterministic tasks")

    return results


def run_stochastic_delay_ablation(
    vocab_size: int = 16,
    total_steps: int = 20000,
    device: str = "cpu",
) -> Dict:
    """
    Test credit assignment on stochastic grammar with delayed reward.

    Questions:
    - Does stochasticity make credit assignment harder?
    - Do we still get 100% accuracy at 7-step delay?
    """
    print("\n" + "=" * 70)
    print("STOCHASTIC DELAY ABLATION: Credit Assignment Challenge")
    print("=" * 70)
    print("\nDoes stochasticity make credit assignment harder?\n")

    sequence_lengths = [1, 3, 5, 7]
    results = {}

    for i, seq_len in enumerate(sequence_lengths, 1):
        print(f"\n{'='*60}")
        print(f"[Experiment {i}/4: Sequence Length = {seq_len}]")
        print(f"{'='*60}")

        config = TokenTrainerConfig(
            vocab_size=vocab_size,
            grammar_type="bigram",  # Stochastic
            training_mode="rl",
            sequence_length=seq_len,
            total_steps=total_steps,
            device=device,
            seed=42,
        )

        trainer = TokenTrainer(config)
        result = trainer.train()

        results[f"seq_{seq_len}_stochastic"] = {
            "sequence_length": seq_len,
            "grammar_type": "bigram",
            "best_accuracy": trainer.best_accuracy,
        }

        print(f"Sequence {seq_len} (Stochastic): Accuracy = {trainer.best_accuracy:.2%}")

    # Summary
    print("\n" + "=" * 70)
    print("STOCHASTIC DELAY ABLATION SUMMARY")
    print("=" * 70)
    print(f"{'Sequence Length':<20} {'Accuracy (Stochastic)':<25}")
    print("-" * 45)
    for seq_len in sequence_lengths:
        acc = results[f"seq_{seq_len}_stochastic"]["best_accuracy"]
        print(f"{seq_len:<20} {acc:.2%}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS: Stochastic Credit Assignment")
    print("=" * 70)

    # Check if accuracy degrades with sequence length
    accuracies = [results[f"seq_{seq_len}_stochastic"]["best_accuracy"] for seq_len in sequence_lengths]
    degradation = accuracies[0] - accuracies[-1]  # seq_1 - seq_7

    print(f"Accuracy at seq_1: {accuracies[0]:.2%}")
    print(f"Accuracy at seq_7: {accuracies[-1]:.2%}")
    print(f"Degradation: {degradation:.2%}")

    if accuracies[-1] >= 0.70 and degradation <= 0.20:
        print("\n✓ Credit assignment ROBUST with stochasticity")
        print("  Maintains >70% accuracy even at 7-step delay")
    elif accuracies[-1] >= 0.50:
        print("\n~ Credit assignment DEGRADED but functional")
        print("  50-70% accuracy at 7-step delay - TD(λ) might help")
    else:
        print("\n✗ Credit assignment FAILS with stochastic + delay")
        print("  <50% accuracy - need better credit assignment algorithms")

    return results


def run_cca_analysis(
    vocab_size: int = 16,
    total_steps: int = 15000,
    device: str = "cpu",
) -> Dict:
    """
    Canonical Correlation Analysis: Do RL and MLE learn similar representations?

    This tests whether RL discovers the same underlying structure as MLE.
    """
    print("\n" + "=" * 70)
    print("CCA ANALYSIS: Representation Similarity")
    print("=" * 70)
    print("\nDo RL and MLE learn similar internal representations?\n")

    # Train both models
    print("[1/2] Training MLE model on stochastic grammar...")
    mle_config = TokenTrainerConfig(
        vocab_size=vocab_size,
        grammar_type="bigram",
        training_mode="mle",
        total_steps=total_steps,
        device=device,
        seed=42,
    )
    mle_trainer = TokenTrainer(mle_config)
    mle_trainer.train()
    mle_model = mle_trainer.model

    print("\n[2/2] Training RL model on stochastic grammar...")
    rl_config = TokenTrainerConfig(
        vocab_size=vocab_size,
        grammar_type="bigram",
        training_mode="rl",
        total_steps=total_steps,
        device=device,
        seed=42,
    )
    rl_trainer = TokenTrainer(rl_config)
    rl_trainer.train()
    rl_model = rl_trainer.model

    # Extract representations
    print("\nExtracting hidden representations...")
    mle_model.eval()
    rl_model.eval()

    # Collect representations for all tokens
    mle_hiddens = []
    rl_hiddens = []

    with torch.no_grad():
        for token in range(vocab_size):
            # One-hot encode
            state = [0.0] * vocab_size
            state[token] = 1.0
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)

            # Get hidden states (before final layer)
            # Use the model's embedding and transformer, following its forward() logic
            mle_emb = mle_model.embedding(state_tensor.unsqueeze(0))  # Add batch dim
            mle_hidden = mle_model.transformer(mle_emb.unsqueeze(1)).squeeze(1)  # Add seq dim, then remove
            mle_hiddens.append(mle_hidden.squeeze(0).cpu().numpy())  # Remove batch dim

            rl_emb = rl_model.embedding(state_tensor.unsqueeze(0))
            rl_hidden = rl_model.transformer(rl_emb.unsqueeze(1)).squeeze(1)
            rl_hiddens.append(rl_hidden.squeeze(0).cpu().numpy())

    mle_hiddens = np.array(mle_hiddens).squeeze()  # [vocab_size, hidden_dim]
    rl_hiddens = np.array(rl_hiddens).squeeze()

    # Compute CCA
    print("\nComputing CCA similarity...")
    from sklearn.cross_decomposition import CCA

    n_components = min(vocab_size, mle_hiddens.shape[-1]) // 2
    cca = CCA(n_components=n_components)

    try:
        mle_c, rl_c = cca.fit_transform(mle_hiddens, rl_hiddens)

        # Compute correlation for each component
        correlations = []
        for i in range(n_components):
            corr = np.corrcoef(mle_c[:, i], rl_c[:, i])[0, 1]
            correlations.append(corr)

        mean_cca = np.mean(correlations)

        print(f"\nCCA Similarity: {mean_cca:.3f}")
        print(f"Top 3 canonical correlations: {correlations[:3]}")

        # Interpret
        print("\n" + "=" * 70)
        print("INTERPRETATION")
        print("=" * 70)

        if mean_cca >= 0.80:
            print("✓ STRONG SIMILARITY (≥0.80)")
            print("  RL and MLE learn very similar representations")
            print("  Different optimization but same underlying structure")
        elif mean_cca >= 0.60:
            print("~ MODERATE SIMILARITY (0.60-0.80)")
            print("  RL and MLE capture similar structure but with differences")
        else:
            print("✗ LOW SIMILARITY (<0.60)")
            print("  RL and MLE learn different representations")
            print("  May indicate RL finds different solution or underfits")

        results = {
            "mean_cca": mean_cca,
            "correlations": correlations,
            "mle_accuracy": mle_trainer.best_accuracy,
            "rl_accuracy": rl_trainer.best_accuracy,
        }

    except Exception as e:
        print(f"\n✗ CCA computation failed: {e}")
        print("  This might happen with small vocab or underfitting")
        results = {"error": str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(description="Stage 3.1b: Stochastic Token Prediction")
    parser.add_argument("--stochastic-compare", action="store_true",
                       help="Run RL vs MLE on stochastic grammar")
    parser.add_argument("--det-vs-stoch", action="store_true",
                       help="Compare deterministic vs stochastic")
    parser.add_argument("--stochastic-delay", action="store_true",
                       help="Delay ablation on stochastic grammar")
    parser.add_argument("--cca", action="store_true",
                       help="CCA analysis of representations")
    parser.add_argument("--all", action="store_true",
                       help="Run all experiments")
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    all_results = {}

    if args.all or args.stochastic_compare:
        print("\n" + "=" * 80)
        print("EXPERIMENT 1: RL vs MLE on Stochastic Grammar")
        print("=" * 80)
        all_results["stochastic_compare"] = run_stochastic_rl_vs_mle(
            vocab_size=args.vocab_size,
            total_steps=args.steps,
            device=args.device,
        )

    if args.all or args.det_vs_stoch:
        print("\n" + "=" * 80)
        print("EXPERIMENT 2: Deterministic vs Stochastic Comparison")
        print("=" * 80)
        all_results["det_vs_stoch"] = run_deterministic_vs_stochastic_comparison(
            vocab_size=args.vocab_size,
            total_steps=args.steps,
            device=args.device,
        )

    if args.all or args.stochastic_delay:
        print("\n" + "=" * 80)
        print("EXPERIMENT 3: Stochastic Delay Ablation")
        print("=" * 80)
        all_results["stochastic_delay"] = run_stochastic_delay_ablation(
            vocab_size=args.vocab_size,
            total_steps=args.steps + 5000,  # More steps for harder task
            device=args.device,
        )

    if args.all or args.cca:
        print("\n" + "=" * 80)
        print("EXPERIMENT 4: CCA Analysis")
        print("=" * 80)
        all_results["cca"] = run_cca_analysis(
            vocab_size=args.vocab_size,
            total_steps=args.steps,
            device=args.device,
        )

    # If no flags, show help
    if not any([args.stochastic_compare, args.det_vs_stoch,
                args.stochastic_delay, args.cca, args.all]):
        parser.print_help()
        print("\n" + "=" * 70)
        print("STAGE 3.1b: Stochastic Token Prediction")
        print("=" * 70)
        print("\nThis stage tests the CRITICAL question:")
        print("Does RL handle stochastic token transitions?")
        print("\nDeterministic grammars (Stage 3.1a) were too easy - RL got 100%.")
        print("Stochastic grammars are the real test of the approach.")
        print("\nRun with --all to execute all experiments.")

    # Save all results
    if all_results:
        save_dir = Path("results/phase3_1b")
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = save_dir / f"stage3_1b_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n\nAll results saved to {results_file}")


if __name__ == "__main__":
    main()
