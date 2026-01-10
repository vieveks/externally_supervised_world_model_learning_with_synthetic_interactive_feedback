"""Debug compositional grammar states and rewards."""

from src.environment.compositional_grammar import CompositionalGrammar
import torch

# Create grammar
grammar = CompositionalGrammar(
    vocab_size=16,
    num_prefixes=4,
    num_suffixes=4,
    held_out_fraction=0.25,
)

# Test a few episodes
for i in range(5):
    prefix, state = grammar.reset(test_mode=False)
    print(f'Episode {i+1}:')
    print(f'  Prefix token: {prefix}')
    print(f'  State (one-hot): {state[:8]}... (showing first 8)')
    print(f'  State sum: {sum(state)}')

    # Try predicting each possible suffix
    valid_suffixes = grammar.get_correct_suffix(prefix)
    print(f'  Valid suffixes for prefix {prefix}: {valid_suffixes}')

    # Test with a correct suffix
    if valid_suffixes:
        correct_suffix = valid_suffixes[0]
        next_state, reward, done, info = grammar.step(correct_suffix, test_mode=False)
        print(f'  Predicted suffix {correct_suffix}: reward={reward}, info={info}')
    print()
