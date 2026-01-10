"""Analyze what the model predicts after training."""

import torch
from collections import defaultdict
from src.environment.compositional_grammar import CompositionalGrammar
from src.models.token_model import TokenPredictionModel
from src.training.token_reinforce import TokenMLE, TokenExperience

# Create grammar
grammar = CompositionalGrammar(
    vocab_size=16,
    num_prefixes=4,
    num_suffixes=4,
    held_out_fraction=0.25,
)

print("Training compositions:")
for p, s in sorted(grammar.get_train_compositions()):
    print(f"  {p} -> {s}")

print("\nTest compositions:")
for p, s in sorted(grammar.get_test_compositions()):
    print(f"  {p} -> {s}")

# Create model
model = TokenPredictionModel(
    vocab_size=16,
    hidden_dim=64,
    num_layers=2,
    num_heads=4,
).to('cuda')

# Create algorithm
mle = TokenMLE(model=model, lr=1e-3, device='cuda')

# Train for 5000 steps
print("\nTraining for 5000 steps...")
for step in range(5000):
    # Collect batch
    experiences = []
    for _ in range(32):
        prefix, state = grammar.reset(test_mode=False)
        state_tensor = torch.tensor(state, dtype=torch.float32, device='cuda')
        logits, _ = model.forward(state_tensor)
        predicted_token = logits.argmax(dim=-1).item()
        next_state, reward, done, info = grammar.step(predicted_token, test_mode=False)

        exp = TokenExperience(
            state=state_tensor,
            prediction=predicted_token,
            target=info["actual"],
            reward=reward,
            log_prob=0.0,
            correct=info["correct"],
        )
        experiences.append(exp)

    # Update
    metrics = mle.update(experiences)

    if (step + 1) % 1000 == 0:
        print(f"  Step {step+1}: Loss={metrics['loss/ce']:.4f}")

# Analyze what the model learned
print("\nModel predictions for each prefix:")
model.eval()
with torch.no_grad():
    for prefix in range(4):
        state = grammar.encode_token(prefix)
        state_tensor = torch.tensor(state, dtype=torch.float32, device='cuda')
        logits, _ = model.forward(state_tensor)
        probs = torch.softmax(logits, dim=-1)

        # Get top 5 predictions
        top_probs, top_tokens = torch.topk(probs, k=8)

        print(f"\n  Prefix {prefix}:")
        for prob, token in zip(top_probs, top_tokens):
            marker = ""
            # Check if this is a valid training suffix
            if grammar.is_train_composition(prefix, token.item()):
                marker = " [TRAIN]"
            elif grammar.is_test_composition(prefix, token.item()):
                marker = " [TEST]"

            print(f"    Token {token.item()}: {prob.item():.3f}{marker}")

# Test accuracy on each composition
print("\nAccuracy by composition:")
composition_results = defaultdict(lambda: {"correct": 0, "total": 0})

with torch.no_grad():
    for _ in range(1000):
        prefix, state = grammar.reset(test_mode=False)
        state_tensor = torch.tensor(state, dtype=torch.float32, device='cuda')
        logits, _ = model.forward(state_tensor)
        predicted_token = logits.argmax(dim=-1).item()
        next_state, reward, done, info = grammar.step(predicted_token, test_mode=False)

        composition = (prefix, info["actual"])
        composition_results[composition]["total"] += 1
        composition_results[composition]["correct"] += int(info["correct"])

print("\nTraining compositions:")
for comp in sorted(composition_results.keys()):
    results = composition_results[comp]
    accuracy = results["correct"] / results["total"] if results["total"] > 0 else 0
    print(f"  {comp[0]} -> {comp[1]}: {accuracy:.1%} ({results['correct']}/{results['total']})")
