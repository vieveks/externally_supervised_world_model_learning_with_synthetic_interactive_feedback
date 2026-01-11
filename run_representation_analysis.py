"""Week 3: Representation Analysis - Do RL and MLE learn similar representations?

Tests hypothesis: RL and MLE learn similar structural representations
despite different output distributions (entropy collapse).
"""

import argparse
import torch
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

from src.environment.ambiguous_grammar import AmbiguousGrammar
from src.models.token_model import TokenPredictionModel
from src.training.token_reinforce import TokenREINFORCE, TokenMLE, TokenExperience
from src.analysis.representation_similarity import (
    representational_similarity_analysis,
    compare_representations,
)


def train_models_for_analysis(
    grammar: AmbiguousGrammar,
    total_steps: int = 20000,
    device: str = 'cpu',
) -> tuple:
    """
    Train both RL and MLE models on ambiguous grammar.

    Returns trained models for representation analysis.
    """
    print("=" * 80)
    print("TRAINING MODELS FOR REPRESENTATION ANALYSIS")
    print("=" * 80)

    # Train MLE
    print("\n[1/2] Training MLE...")
    mle_model = TokenPredictionModel(
        vocab_size=grammar.vocab_size,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        ff_dim=256,
        dropout=0.0,
    ).to(device)

    mle_algo = TokenMLE(model=mle_model, lr=1e-3, device=device)

    step = 0
    log_interval = max(500, total_steps // 10)  # Log ~10 times during training
    while step < total_steps:
        experiences = []
        for _ in range(32):
            token, state = grammar.reset()
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            logits, _ = mle_model.forward(state_tensor)
            predicted_token = logits.argmax(dim=-1).item()
            next_state, reward, done, info = grammar.step(predicted_token)

            exp = TokenExperience(
                state=state_tensor,
                prediction=predicted_token,
                target=info["actual"],
                reward=reward,
                log_prob=0.0,
                correct=info["correct"],
            )
            experiences.append(exp)

        mle_algo.update(experiences)
        step += 32

        if step % log_interval < 32:  # Trigger when crossing log_interval
            acc = sum(1 for e in experiences if e.correct) / len(experiences)
            print(f"  Step {step}/{total_steps}: Batch Acc = {acc:.1%}")

    print("  MLE training complete")

    # Train RL
    print("\n[2/2] Training RL (REINFORCE)...")
    rl_model = TokenPredictionModel(
        vocab_size=grammar.vocab_size,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        ff_dim=256,
        dropout=0.0,
    ).to(device)

    rl_algo = TokenREINFORCE(
        model=rl_model,
        lr=1e-3,
        baseline_decay=0.99,
        entropy_coef=0.01,
        device=device,
    )

    step = 0
    while step < total_steps:
        experiences = []
        for _ in range(32):
            token, state = grammar.reset()
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            prediction, log_prob = rl_model.sample_prediction(state_tensor)

            next_state, reward, done, info = grammar.step(prediction.item())

            exp = TokenExperience(
                state=state_tensor,
                prediction=prediction.item(),
                target=info["actual"],
                reward=reward,
                log_prob=log_prob.item(),
                correct=info["correct"],
            )
            experiences.append(exp)

        rl_algo.update(experiences)
        step += 32

        if step % log_interval < 32:  # Trigger when crossing log_interval
            acc = sum(1 for e in experiences if e.correct) / len(experiences)
            print(f"  Step {step}/{total_steps}: Batch Acc = {acc:.1%}")

    print("  RL training complete\n")

    return mle_model, rl_model


def collect_paired_representations(
    mle_model,
    rl_model,
    grammar: AmbiguousGrammar,
    num_samples: int = 500,
    device: str = 'cpu',
) -> tuple:
    """
    Collect hidden representations from BOTH models on the SAME inputs.

    CRITICAL: We must use the same input states for both models to compute
    valid representation similarity. Comparing representations on different
    inputs would give meaningless results.

    Returns:
        mle_hiddens: MLE hidden representations (num_samples, hidden_dim)
        rl_hiddens: RL hidden representations (num_samples, hidden_dim)
        tokens: Token IDs (num_samples,)
        is_ambiguous: Whether each token is ambiguous (num_samples,)
    """
    mle_model.eval()
    rl_model.eval()

    mle_hiddens_list = []
    rl_hiddens_list = []
    tokens_list = []
    is_ambiguous_list = []

    with torch.no_grad():
        for _ in range(num_samples):
            # Sample input ONCE
            token, state = grammar.reset()
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)

            # Get hidden representations from BOTH models on SAME input
            mle_hidden = mle_model.get_hidden(state_tensor)
            rl_hidden = rl_model.get_hidden(state_tensor)

            mle_hiddens_list.append(mle_hidden.cpu())
            rl_hiddens_list.append(rl_hidden.cpu())
            tokens_list.append(token)
            is_ambiguous_list.append(grammar.is_ambiguous(token))

    mle_hiddens = torch.stack(mle_hiddens_list)
    rl_hiddens = torch.stack(rl_hiddens_list)
    tokens = np.array(tokens_list)
    is_ambiguous = np.array(is_ambiguous_list)

    return mle_hiddens, rl_hiddens, tokens, is_ambiguous


def analyze_representations(
    mle_model,
    rl_model,
    grammar: AmbiguousGrammar,
    num_samples: int = 500,
    device: str = 'cpu',
) -> dict:
    """
    Perform comprehensive representation analysis.

    Compares RL and MLE representations on:
    - All tokens
    - Ambiguous tokens only
    - Deterministic tokens only
    """
    print("=" * 80)
    print("REPRESENTATION SIMILARITY ANALYSIS")
    print("=" * 80)

    # Collect representations from both models ON THE SAME INPUTS
    print("\nCollecting paired representations (same inputs for both models)...")
    mle_hiddens, rl_hiddens, tokens, is_ambiguous = collect_paired_representations(
        mle_model, rl_model, grammar, num_samples, device
    )

    results = {}

    # Analysis 1: All tokens
    print("\n[1/3] Analyzing all tokens...")
    all_results = compare_representations(mle_hiddens, rl_hiddens)
    results['all_tokens'] = all_results

    print(f"  Linear CKA: {all_results['linear_cka']:.3f}")
    print(f"  RBF CKA: {all_results['rbf_cka']:.3f}")
    print(f"  CCA Mean: {all_results['cca_mean']:.3f}")
    print(f"  PWCCA: {all_results['pwcca']:.3f}")

    # Analysis 2: Ambiguous tokens only
    print("\n[2/3] Analyzing ambiguous tokens only...")
    ambig_mask = is_ambiguous
    if ambig_mask.sum() > 0:
        ambig_results = compare_representations(
            mle_hiddens[ambig_mask],
            rl_hiddens[ambig_mask]
        )
        results['ambiguous_tokens'] = ambig_results

        print(f"  Linear CKA: {ambig_results['linear_cka']:.3f}")
        print(f"  RBF CKA: {ambig_results['rbf_cka']:.3f}")
        print(f"  CCA Mean: {ambig_results['cca_mean']:.3f}")
        print(f"  PWCCA: {ambig_results['pwcca']:.3f}")

    # Analysis 3: Deterministic tokens only
    print("\n[3/3] Analyzing deterministic tokens only...")
    determ_mask = ~is_ambiguous
    if determ_mask.sum() > 0:
        determ_results = compare_representations(
            mle_hiddens[determ_mask],
            rl_hiddens[determ_mask]
        )
        results['deterministic_tokens'] = determ_results

        print(f"  Linear CKA: {determ_results['linear_cka']:.3f}")
        print(f"  RBF CKA: {determ_results['rbf_cka']:.3f}")
        print(f"  CCA Mean: {determ_results['cca_mean']:.3f}")
        print(f"  PWCCA: {determ_results['pwcca']:.3f}")

    return results, mle_hiddens, rl_hiddens, tokens, is_ambiguous


def visualize_results(results: dict, output_dir: Path):
    """Create visualization of similarity results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    contexts = ['all_tokens', 'ambiguous_tokens', 'deterministic_tokens']
    context_labels = ['All Tokens', 'Ambiguous Only', 'Deterministic Only']
    metrics = ['linear_cka', 'rbf_cka', 'cca_mean', 'pwcca']
    metric_labels = ['Linear CKA', 'RBF CKA', 'CCA Mean', 'PWCCA']

    for idx, (context, label) in enumerate(zip(contexts, context_labels)):
        if context in results:
            values = [results[context][m] for m in metrics]

            ax = axes[idx]
            ax.bar(metric_labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax.set_ylim(0, 1.0)
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.set_ylabel('Similarity Score', fontsize=10)
            ax.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='High similarity (>0.7)')
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3)

            # Add value labels on bars
            for i, v in enumerate(values):
                ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'representation_similarity.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {output_dir / 'representation_similarity.png'}")


def run_week3_analysis(
    vocab_size: int = 16,
    num_ambiguous: int = 8,
    ambiguity_level: str = "high",
    training_steps: int = 20000,
    num_samples: int = 500,
    device: str = 'cpu',
) -> dict:
    """
    Main Week 3 experiment: Representation similarity analysis.

    Tests whether RL and MLE learn similar representations despite
    different output distributions.
    """
    print("\n" + "=" * 80)
    print("WEEK 3: REPRESENTATION SIMILARITY ANALYSIS")
    print("=" * 80)
    print("\nHypothesis: RL and MLE learn similar structural representations")
    print("            despite RL's distributional collapse\n")

    # Create ambiguous grammar
    grammar = AmbiguousGrammar(
        vocab_size=vocab_size,
        num_ambiguous_tokens=num_ambiguous,
        ambiguity_level=ambiguity_level,
        branching_factor=2,
    )

    stats = grammar.analyze_grammar()
    num_deterministic = stats['vocab_size'] - stats['num_ambiguous_tokens']
    print(f"Grammar: {stats['num_ambiguous_tokens']} ambiguous, "
          f"{num_deterministic} deterministic")
    print(f"Oracle accuracy: {stats['oracle_accuracy']:.1%}\n")

    # Train models
    mle_model, rl_model = train_models_for_analysis(
        grammar=grammar,
        total_steps=training_steps,
        device=device,
    )

    # Analyze representations
    results, mle_hiddens, rl_hiddens, tokens, is_ambiguous = analyze_representations(
        mle_model=mle_model,
        rl_model=rl_model,
        grammar=grammar,
        num_samples=num_samples,
        device=device,
    )

    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    linear_cka = results['all_tokens']['linear_cka']
    rbf_cka = results['all_tokens']['rbf_cka']
    avg_cka = (linear_cka + rbf_cka) / 2

    if avg_cka > 0.6:
        print("[SUCCESS] High representational similarity")
        print(f"  Average CKA: {avg_cka:.3f} (Linear: {linear_cka:.3f}, RBF: {rbf_cka:.3f})")
        print("  RL and MLE learn similar structural representations")
        print("  Distributional collapse is in output layer, not representations")
    elif avg_cka > 0.4:
        print("[PARTIAL] Moderate representational similarity")
        print(f"  Average CKA: {avg_cka:.3f} (Linear: {linear_cka:.3f}, RBF: {rbf_cka:.3f})")
        print("  RL and MLE learn somewhat similar representations")
        print("  Some structural differences exist")
    else:
        print("[SURPRISING] Low representational similarity")
        print(f"  Average CKA: {avg_cka:.3f} (Linear: {linear_cka:.3f}, RBF: {rbf_cka:.3f})")
        print("  RL and MLE learn fundamentally different representations")
        print("  Collapse affects internal representations, not just outputs")

    # Context-specific analysis
    if 'ambiguous_tokens' in results:
        ambig_cka = results['ambiguous_tokens']['linear_cka']
        print(f"\nAmbiguous tokens: CKA = {ambig_cka:.3f}")
        if ambig_cka > 0.7:
            print("  High similarity even for ambiguous contexts")

    if 'deterministic_tokens' in results:
        determ_cka = results['deterministic_tokens']['linear_cka']
        print(f"Deterministic tokens: CKA = {determ_cka:.3f}")
        if determ_cka > 0.7:
            print("  High similarity for deterministic contexts")

    # Save results
    output_dir = Path("results/phase3_language/week3_representations")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prepare results for JSON (convert numpy arrays to lists)
    results_serializable = {}
    for context, metrics in results.items():
        results_serializable[context] = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                results_serializable[context][key] = value.tolist()
            else:
                results_serializable[context][key] = value

    # Save JSON results
    output_file = output_dir / f"representation_results_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\n\nResults saved to {output_file}")

    # Visualize
    visualize_results(results, output_dir)

    return results


def main():
    parser = argparse.ArgumentParser(description="Week 3: Representation Analysis")
    parser.add_argument("--vocab-size", type=int, default=16, help="Vocabulary size")
    parser.add_argument("--num-ambiguous", type=int, default=8, help="Number of ambiguous tokens")
    parser.add_argument("--ambiguity-level", type=str, default="high", help="Ambiguity level: low, medium, high")
    parser.add_argument("--steps", type=int, default=20000, help="Training steps")
    parser.add_argument("--samples", type=int, default=500, help="Samples for analysis")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    run_week3_analysis(
        vocab_size=args.vocab_size,
        num_ambiguous=args.num_ambiguous,
        ambiguity_level=args.ambiguity_level,
        training_steps=args.steps,
        num_samples=args.samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
