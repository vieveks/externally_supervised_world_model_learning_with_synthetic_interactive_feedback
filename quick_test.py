"""Quick test of training."""

import torch
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

# Create model
model = TokenPredictionModel(
    vocab_size=16,
    hidden_dim=64,
    num_layers=2,
    num_heads=4,
).to('cuda')

# Create algorithm
mle = TokenMLE(model=model, lr=1e-3, device='cuda')

# Train for 1000 steps
print("Training MLE for 1000 steps...")
for step in range(0, 1000, 100):
    # Collect batch
    experiences = []
    for _ in range(100):
        prefix, state = grammar.reset(test_mode=False)
        state_tensor = torch.tensor(state, dtype=torch.float32, device='cuda')
        logits, _ = model.forward(state_tensor)
        predicted_token = logits.argmax(dim=-1).item()  # Use argmax
        next_state, reward, done, info = grammar.step(predicted_token, test_mode=False)

        exp = TokenExperience(
            state=state_tensor,
            prediction=predicted_token,
            target=info["actual"],
            reward=reward,
            log_prob=0.0,  # Not needed for MLE
            correct=info["correct"],
        )
        experiences.append(exp)

    # Update
    metrics = mle.update(experiences)

    # Evaluate
    model.eval()
    correct = 0
    with torch.no_grad():
        for _ in range(100):
            prefix, state = grammar.reset(test_mode=False)
            state_tensor = torch.tensor(state, dtype=torch.float32, device='cuda')
            logits, _ = model.forward(state_tensor)
            predicted_token = logits.argmax(dim=-1).item()
            next_state, reward, done, info = grammar.step(predicted_token, test_mode=False)
            correct += int(info["correct"])
    model.train()

    accuracy = correct / 100
    print(f"  Step {step+100}: Loss={metrics['loss/ce']:.4f}, Train Acc={accuracy:.2%}")

# Final test on held-out
print("\nEvaluating on held-out compositions...")
model.eval()
correct = 0
with torch.no_grad():
    for _ in range(100):
        prefix, state = grammar.reset(test_mode=True)
        state_tensor = torch.tensor(state, dtype=torch.float32, device='cuda')
        logits, _ = model.forward(state_tensor)
        predicted_token = logits.argmax(dim=-1).item()
        next_state, reward, done, info = grammar.step(predicted_token, test_mode=True)
        correct += int(info["correct"])

test_accuracy = correct / 100
print(f"Test Accuracy: {test_accuracy:.2%}")
