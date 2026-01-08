"""
Test script to understand the bigram grammar difficulty.

Questions:
1. What are the actual bigram probabilities?
2. What is the theoretical maximum accuracy (if predicting argmax)?
3. Is the grammar learnable?
"""

import torch
import numpy as np
from src.environment.token_prediction import TokenPredictionTask


def analyze_bigram_grammar(vocab_size=16, seed=42):
    """Analyze the bigram grammar structure."""
    print("=" * 70)
    print("BIGRAM GRAMMAR ANALYSIS")
    print("=" * 70)

    # Create task
    import random
    random.seed(seed)
    task = TokenPredictionTask(
        vocab_size=vocab_size,
        grammar_type="bigram",
    )

    # Get transition matrix
    T = task.get_transition_matrix()
    print(f"\nTransition Matrix Shape: {T.shape}")
    print(f"Transition Matrix:\n{T}")

    # Compute statistics
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)

    # Entropy per token
    entropies = []
    max_probs = []
    for i in range(vocab_size):
        probs = T[i].numpy()
        # Compute entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropies.append(entropy)
        # Max prob
        max_prob = np.max(probs)
        max_probs.append(max_prob)

    mean_entropy = np.mean(entropies)
    max_entropy = np.log(vocab_size)
    mean_max_prob = np.mean(max_probs)

    print(f"Mean Entropy: {mean_entropy:.3f} (max: {max_entropy:.3f})")
    print(f"Normalized Entropy: {mean_entropy / max_entropy:.3f}")
    print(f"Mean Max Probability: {mean_max_prob:.3f}")

    print("\nPer-token Max Probabilities:")
    for i in range(vocab_size):
        print(f"  Token {i:2d} -> max_prob = {max_probs[i]:.3f}")

    # Theoretical oracle accuracy
    oracle_accuracy = mean_max_prob
    print(f"\n{'='*70}")
    print(f"THEORETICAL ORACLE ACCURACY: {oracle_accuracy:.2%}")
    print(f"{'='*70}")
    print("\nThis is the maximum accuracy achievable by always predicting argmax.")
    print("If this is low (~20-30%), the grammar is highly stochastic.")
    print("If this is high (~80-90%), the grammar is nearly deterministic.")

    # Check if approximately uniform
    if mean_entropy / max_entropy > 0.9:
        print("\n⚠️  WARNING: Grammar is nearly uniform random!")
        print("   This is too hard for any prediction method.")
        print("   Consider: Lower entropy distributions")

    # Learnability
    print(f"\n{'='*70}")
    print("LEARNABILITY ASSESSMENT")
    print(f"{'='*70}")

    if oracle_accuracy >= 0.80:
        print("✓ HIGHLY LEARNABLE: Oracle accuracy ≥80%")
        print("  Grammar is mostly deterministic, should be easy to learn")
    elif oracle_accuracy >= 0.50:
        print("~ MODERATELY LEARNABLE: Oracle accuracy 50-80%")
        print("  Grammar has structure but significant stochasticity")
    elif oracle_accuracy >= 0.30:
        print("⚠️  DIFFICULT: Oracle accuracy 30-50%")
        print("  High stochasticity, hard to learn without many samples")
    else:
        print("✗ VERY DIFFICULT: Oracle accuracy <30%")
        print("  Near-uniform distribution, may be unlearnable")

    return {
        "mean_entropy": mean_entropy,
        "max_entropy": max_entropy,
        "oracle_accuracy": oracle_accuracy,
        "transition_matrix": T,
    }


def test_deterministic_baselines(vocab_size=16):
    """Test what baseline strategies achieve."""
    print("\n" + "=" * 70)
    print("BASELINE STRATEGIES")
    print("=" * 70)

    import random
    random.seed(42)
    task = TokenPredictionTask(
        vocab_size=vocab_size,
        grammar_type="bigram",
    )

    # Strategy 1: Random prediction
    random_correct = 0
    random_total = 1000
    for _ in range(random_total):
        current, _ = task.reset()
        predicted = random.randint(0, vocab_size - 1)
        actual = task.get_next_token(current)
        if predicted == actual:
            random_correct += 1

    random_acc = random_correct / random_total
    print(f"Random Prediction: {random_acc:.2%}")

    # Strategy 2: Always predict mode (most common next token overall)
    # Count next token frequencies
    from collections import Counter
    next_token_counts = Counter()
    for _ in range(1000):
        current, _ = task.reset()
        actual = task.get_next_token(current)
        next_token_counts[actual] += 1

    mode_token = next_token_counts.most_common(1)[0][0]
    print(f"\nMost common next token: {mode_token} ({next_token_counts[mode_token]/10:.1f}%)")

    mode_correct = 0
    mode_total = 1000
    for _ in range(mode_total):
        current, _ = task.reset()
        predicted = mode_token
        actual = task.get_next_token(current)
        if predicted == actual:
            mode_correct += 1

    mode_acc = mode_correct / mode_total
    print(f"Always Predict Mode: {mode_acc:.2%}")

    # Strategy 3: Oracle (always predict argmax for current token)
    oracle_correct = 0
    oracle_total = 1000
    for _ in range(oracle_total):
        current, _ = task.reset()
        predicted = task.get_deterministic_next(current)
        actual = task.get_next_token(current)
        if predicted == actual:
            oracle_correct += 1

    oracle_acc = oracle_correct / oracle_total
    print(f"Oracle (argmax per token): {oracle_acc:.2%}")

    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    print(f"Random: {random_acc:.2%}")
    print(f"Mode: {mode_acc:.2%}")
    print(f"Oracle: {oracle_acc:.2%}")
    print(f"\nRL got {14.0:.2%}, MLE got {13.0:.2%}")
    print(f"Oracle ceiling is {oracle_acc:.2%}")

    if random_acc > 0.08:
        print("\n⚠️  Random baseline is suspiciously high!")
        print("   This suggests grammar may be nearly uniform")


if __name__ == "__main__":
    results = analyze_bigram_grammar(vocab_size=16, seed=42)
    test_deterministic_baselines(vocab_size=16)

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("\nIf oracle accuracy is low (<40%), we should:")
    print("1. Generate less uniform bigram distributions")
    print("2. Use controlled stochasticity (e.g., 80% deterministic, 20% noise)")
    print("3. Test on a range of entropy levels")
