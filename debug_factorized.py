"""Debug factorized training."""

import torch
from src.environment.factorized_grammar import FactorizedGrammar
from src.models.token_model import TokenPredictionModel
from src.training.token_reinforce import TokenMLE, TokenExperience

# Create grammar
grammar = FactorizedGrammar(vocab_size=16, num_a_values=2, num_b_values=2, held_out_fraction=0.25)

print("Train sequences:", sorted(grammar.get_train_sequences()))
print("Test sequences:", sorted(grammar.get_test_sequences()))

# Create model
model = TokenPredictionModel(vocab_size=16, hidden_dim=32, num_layers=1, num_heads=2).to('cuda')
mle = TokenMLE(model=model, lr=1e-2, device='cuda')

# Train for 200 steps
print("\nTraining for 200 steps...")
for step in range(200):
    experiences = []
    for _ in range(32):
        a_token, state = grammar.reset(test_mode=False)
        state_tensor = torch.tensor(state, dtype=torch.float32, device='cuda')
        logits, _ = model.forward(state_tensor)
        predicted_token = logits.argmax(dim=-1).item()
        next_state, reward, done, info = grammar.step(predicted_token)
        exp = TokenExperience(state=state_tensor, prediction=predicted_token, target=info["actual"], reward=reward, log_prob=0.0, correct=info["correct"])
        experiences.append(exp)

    metrics = mle.update(experiences)
    if (step + 1) % 40 == 0:
        acc = sum(1 for e in experiences if e.correct) / len(experiences)
        print(f"  Step {step+1}: Loss={metrics['loss/ce']:.3f}, Batch Acc={acc:.1%}")

# Check what the model learned
print("\nFinal predictions:")
model.eval()
with torch.no_grad():
    for a in [0, 1]:
        state_tensor = torch.tensor(grammar.encode_token(a), dtype=torch.float32, device='cuda')
        logits, _ = model.forward(state_tensor)
        pred = logits.argmax(dim=-1).item()
        probs = torch.softmax(logits, dim=-1)

        # Check which sequences this A appears in
        train_b_for_a = [b for (a_val, b) in grammar.get_train_sequences() if a_val == a]
        test_b_for_a = [b for (a_val, b) in grammar.get_test_sequences() if a_val == a]

        print(f"  A={a}: predicts B={pred} (prob={probs[pred]:.2f})")
        print(f"    Training: A={a} -> {train_b_for_a}")
        if test_b_for_a:
            print(f"    Test: A={a} -> {test_b_for_a}")
