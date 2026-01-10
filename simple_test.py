"""Simplest possible test."""

import torch
from src.environment.compositional_grammar import CompositionalGrammar
from src.models.token_model import TokenPredictionModel
from src.training.token_reinforce import TokenMLE, TokenExperience

# Create grammar
grammar = CompositionalGrammar(vocab_size=16, num_prefixes=4, num_suffixes=4, held_out_fraction=0.25)

print("Train compositions:", sorted(grammar.get_train_compositions()))
print("Test compositions:", sorted(grammar.get_test_compositions()))

# Create small model
model = TokenPredictionModel(vocab_size=16, hidden_dim=32, num_layers=1, num_heads=2).to('cuda')
mle = TokenMLE(model=model, lr=1e-2, device='cuda')  # Higher LR

# Train for just 100 steps
print("\nTraining for 100 steps...")
for step in range(100):
    experiences = []
    for _ in range(32):
        prefix, state = grammar.reset(test_mode=False)
        state_tensor = torch.tensor(state, dtype=torch.float32, device='cuda')
        logits, _ = model.forward(state_tensor)
        predicted_token = logits.argmax(dim=-1).item()
        next_state, reward, done, info = grammar.step(predicted_token, test_mode=False)
        exp = TokenExperience(state=state_tensor, prediction=predicted_token, target=info["actual"], reward=reward, log_prob=0.0, correct=info["correct"])
        experiences.append(exp)

    metrics = mle.update(experiences)
    if (step + 1) % 20 == 0:
        acc = sum(1 for e in experiences if e.correct) / len(experiences)
        print(f"  Step {step+1}: Loss={metrics['loss/ce']:.3f}, Batch Acc={acc:.1%}")

# Check predictions
print("\nFinal predictions:")
model.eval()
with torch.no_grad():
    for prefix in range(4):
        state_tensor = torch.tensor(grammar.encode_token(prefix), dtype=torch.float32, device='cuda')
        logits, _ = model.forward(state_tensor)
        pred = logits.argmax(dim=-1).item()
        probs = torch.softmax(logits, dim=-1)
        print(f"  Prefix {prefix}: predicts {pred} (prob={probs[pred]:.2f})")
