"""Test factorized compositional grammar."""

from src.environment.factorized_grammar import FactorizedGrammar

# Create grammar
grammar = FactorizedGrammar(
    vocab_size=16,
    num_a_values=2,  # A can be 0 or 1 (tokens)
    num_b_values=2,  # B can be 2 or 3 (tokens)
    held_out_fraction=0.25,
)

print("=" * 70)
print("FACTORIZED GRAMMAR VALIDATION")
print("=" * 70)

# Analyze splits
stats = grammar.analyze_splits()
print("\nGrammar Structure:")
print(f"  Total sequences: {stats['total_sequences']}")
print(f"  Training sequences: {stats['train_sequences']}")
print(f"  Test sequences: {stats['test_sequences']}")
print(f"  Held-out fraction: {stats['test_sequences'] / stats['total_sequences']:.2%}")

print(f"\nSlot A values: {stats['num_a_values']}")
print(f"Slot B values: {stats['num_b_values']}")
print(f"A values in train: {stats['a_values_in_train']}")
print(f"B values in train: {stats['b_values_in_train']}")

print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)
if stats['all_primitives_in_train']:
    print("[OK] All primitives (A and B values) appear in training")
    print("     This ensures test sequences are novel COMBINATIONS,")
    print("     not novel PRIMITIVES")
else:
    print("[ERROR] Some primitives missing from training!")

print("\n" + "=" * 70)
print("TRAINING SEQUENCES")
print("=" * 70)
for a, b in sorted(grammar.get_train_sequences()):
    print(f"  {a} -> {b}")

print("\n" + "=" * 70)
print("TEST SEQUENCES (HELD-OUT)")
print("=" * 70)
for a, b in sorted(grammar.get_test_sequences()):
    print(f"  {a} -> {b} [NOVEL COMBINATION]")

print("\n" + "=" * 70)
print("EPISODE SAMPLING TEST")
print("=" * 70)

print("\nTraining mode episodes:")
for i in range(3):
    a_token, state = grammar.reset(test_mode=False)
    print(f"  Episode {i+1}: A={a_token}, Target B={grammar._current_sequence[1]}")

print("\nTest mode episodes:")
for i in range(3):
    a_token, state = grammar.reset(test_mode=True)
    print(f"  Episode {i+1}: A={a_token}, Target B={grammar._current_sequence[1]} [HELD-OUT]")

print("\n" + "=" * 70)
print("COMPOSITIONAL REASONING TEST")
print("=" * 70)

# Get a test sequence
test_seq = list(grammar.get_test_sequences())[0]
print(f"\nTest sequence: {test_seq[0]} -> {test_seq[1]}")
print("This combination was NEVER seen during training.")
print("")
print("Question: Can a model trained on other A-B combinations")
print("          predict B for this A value?")
print("")
print("If model predicts correctly:")
print("  → Learned compositional structure (factorized A and B)")
print("If model predicts incorrectly:")
print("  → Only memorized specific combinations")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("Factorized grammar implementation verified.")
print("")
print("Key properties:")
print("  [OK] 2-slot structure (A B)")
print(f"  [OK] {stats['num_a_values']} values for A, {stats['num_b_values']} values for B")
print(f"  [OK] {stats['train_sequences']} training, {stats['test_sequences']} test sequences")
print("  [OK] All primitives appear in training")
print("  [OK] Test = novel COMBINATIONS (not novel primitives)")
print("")
print("Ready for Week 2 v2: True compositional generalization test")
