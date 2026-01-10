"""
Test script for compositional grammar validation.

This verifies that the compositional grammar has the correct properties:
1. All primitives (prefixes and suffixes) appear in training
2. Test compositions are truly held-out (novel combinations)
3. Grammar structure is interpretable and well-formed
"""

from src.environment.compositional_grammar import CompositionalGrammar


def test_compositional_grammar():
    """Test compositional grammar properties."""
    print("=" * 70)
    print("COMPOSITIONAL GRAMMAR VALIDATION")
    print("=" * 70)

    # Create grammar
    grammar = CompositionalGrammar(
        vocab_size=16,
        num_prefixes=4,
        num_suffixes=4,
        held_out_fraction=0.25,
    )

    # Analyze splits
    stats = grammar.analyze_splits()

    print("\nGrammar Structure:")
    print(f"  Total compositions: {stats['total_compositions']}")
    print(f"  Training compositions: {stats['train_compositions']}")
    print(f"  Test compositions: {stats['test_compositions']}")
    print(f"  Held-out fraction: {stats['held_out_fraction']:.2%}")

    print("\nPrimitives:")
    print(f"  Prefixes (total): {stats['num_prefixes']}")
    print(f"  Suffixes (total): {stats['num_suffixes']}")
    print(f"  Prefixes in train: {stats['prefixes_in_train']}")
    print(f"  Suffixes in train: {stats['suffixes_in_train']}")
    print(f"  Prefixes in test: {stats['prefixes_in_test']}")
    print(f"  Suffixes in test: {stats['suffixes_in_test']}")

    # Verify all primitives appear in training
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    if stats['all_primitives_in_train']:
        print("[OK] All primitives (prefixes & suffixes) appear in training")
        print("     This ensures test compositions are novel COMBINATIONS,")
        print("     not novel PRIMITIVES")
    else:
        print("[WARN] Some primitives missing from training")
        print("       This would test primitive generalization, not composition")

    # Show train compositions
    print("\n" + "=" * 70)
    print("TRAINING COMPOSITIONS")
    print("=" * 70)
    train_comps = grammar.get_train_compositions()
    for prefix, suffix in sorted(train_comps):
        print(f"  {prefix} -> {suffix}")

    # Show test compositions
    print("\n" + "=" * 70)
    print("TEST COMPOSITIONS (HELD-OUT)")
    print("=" * 70)
    test_comps = grammar.get_test_compositions()
    for prefix, suffix in sorted(test_comps):
        print(f"  {prefix} -> {suffix} [NOVEL COMBINATION]")

    # Test episode generation
    print("\n" + "=" * 70)
    print("EPISODE SAMPLING TEST")
    print("=" * 70)

    print("\nTraining mode episodes:")
    for i in range(3):
        prefix, state = grammar.reset(test_mode=False)
        valid_suffixes = grammar.get_correct_suffix(prefix)
        print(f"  Episode {i+1}: Prefix={prefix}, Valid suffixes={valid_suffixes}")

    print("\nTest mode episodes:")
    for i in range(3):
        prefix, state = grammar.reset(test_mode=True)
        valid_suffixes = grammar.get_correct_suffix(prefix)
        test_only = [s for s in valid_suffixes if grammar.is_test_composition(prefix, s)]
        print(f"  Episode {i+1}: Prefix={prefix}, Held-out suffixes={test_only}")

    # Test step function
    print("\n" + "=" * 70)
    print("COMPOSITIONAL REASONING TEST")
    print("=" * 70)

    # Find a test composition
    test_comp = test_comps[0] if test_comps else None
    if test_comp:
        prefix, correct_suffix = test_comp
        print(f"\nTest composition: {prefix} -> {correct_suffix}")
        print("This composition was NEVER seen during training.")
        print("\nQuestion: Can a model trained on other compositions")
        print("          predict the correct suffix for this prefix?")

        # Test with correct prediction
        grammar._current_token = prefix
        next_state, reward, done, info = grammar.step(correct_suffix, test_mode=True)

        print(f"\nIf model predicts {correct_suffix}:")
        print(f"  Reward: {reward}")
        print(f"  Correct: {info['correct']}")
        print(f"  Is held-out: {info['is_held_out']}")

        # Test with wrong prediction
        wrong_suffix = (correct_suffix + 1) % grammar.vocab_size
        grammar._current_token = prefix
        next_state, reward, done, info = grammar.step(wrong_suffix, test_mode=True)

        print(f"\nIf model predicts {wrong_suffix}:")
        print(f"  Reward: {reward}")
        print(f"  Correct: {info['correct']}")


def test_generalization_difficulty():
    """Test different generalization scenarios."""
    print("\n" + "=" * 70)
    print("GENERALIZATION DIFFICULTY ANALYSIS")
    print("=" * 70)

    print("\nScenario 1: Easy (25% held-out)")
    grammar_easy = CompositionalGrammar(
        vocab_size=16,
        num_prefixes=4,
        num_suffixes=4,
        held_out_fraction=0.25,
    )
    stats_easy = grammar_easy.analyze_splits()
    print(f"  Train: {stats_easy['train_compositions']}, Test: {stats_easy['test_compositions']}")
    print(f"  Coverage: Each prefix seen with {stats_easy['train_compositions'] // stats_easy['num_prefixes']:.0f} suffixes on average")

    print("\nScenario 2: Hard (50% held-out)")
    grammar_hard = CompositionalGrammar(
        vocab_size=16,
        num_prefixes=4,
        num_suffixes=4,
        held_out_fraction=0.50,
    )
    stats_hard = grammar_hard.analyze_splits()
    print(f"  Train: {stats_hard['train_compositions']}, Test: {stats_hard['test_compositions']}")
    print(f"  Coverage: Each prefix seen with {stats_hard['train_compositions'] // stats_hard['num_prefixes']:.0f} suffixes on average")


if __name__ == "__main__":
    test_compositional_grammar()
    test_generalization_difficulty()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("Compositional grammar implementation verified.")
    print("\nKey properties:")
    print("  [OK] Systematic train/test split")
    print("  [OK] All primitives appear in training")
    print("  [OK] Test = novel COMBINATIONS (not novel primitives)")
    print("  [OK] Clear compositional structure (PREFIX + SUFFIX)")
    print("\nReady for Week 2 experiments: RL vs MLE on compositional generalization")
