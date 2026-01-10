"""Debug train/test split."""

from src.environment.compositional_grammar import CompositionalGrammar

# Create grammar
grammar = CompositionalGrammar(
    vocab_size=16,
    num_prefixes=4,
    num_suffixes=4,
    held_out_fraction=0.25,
)

print("Training compositions:")
train = sorted(grammar.get_train_compositions())
for p, s in train:
    print(f"  {p} -> {s}")

print("\nTest compositions:")
test = sorted(grammar.get_test_compositions())
for p, s in test:
    print(f"  {p} -> {s}")

print("\nTest 10 training episodes:")
for i in range(10):
    prefix, state = grammar.reset(test_mode=False)
    print(f"  Episode {i+1}: Prefix={prefix}, Target={grammar._target_suffix}, Is test? {grammar.is_test_composition(prefix, grammar._target_suffix)}")
