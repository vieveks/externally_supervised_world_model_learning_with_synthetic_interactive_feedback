"""Debug training to see what's happening."""

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

print("Grammar analysis:")
print(f"  Training compositions: {len(grammar.get_train_compositions())}")
print(f"  Test compositions: {len(grammar.get_test_compositions())}")
print()

# Create model
model = TokenPredictionModel(
    vocab_size=16,
    hidden_dim=64,
    num_layers=2,
    num_heads=4,
    dropout=0.0,
)

print(f"Model parameters: {model.count_parameters():,}")
print()

# Create MLE trainer
mle = TokenMLE(model=model, lr=1e-3, device='cpu')

# Collect a batch of experiences
print("Collecting 100 training experiences...")
experiences = []
for i in range(100):
    prefix, state = grammar.reset(test_mode=False)
    state_tensor = torch.tensor(state, dtype=torch.float32)

    # Get model prediction
    logits, value = model.forward(state_tensor)
    probs = torch.softmax(logits, dim=-1)
    predicted_token = torch.multinomial(probs, 1).item()

    # Take step
    next_state, reward, done, info = grammar.step(predicted_token, test_mode=False)

    exp = TokenExperience(
        state=state_tensor,
        prediction=predicted_token,
        target=info["actual"],
        reward=reward,
        log_prob=torch.log(probs[predicted_token] + 1e-10).item(),
        correct=info["correct"],
    )
    experiences.append(exp)

    if i < 5:
        print(f"  Episode {i+1}: Prefix={prefix} -> Target={info['actual']}, Predicted={predicted_token}, Correct={info['correct']}")

# Check distribution of targets
targets = [e.target for e in experiences]
print(f"\nTarget distribution (should cover all training suffixes):")
for t in range(16):
    count = targets.count(t)
    if count > 0:
        print(f"  Token {t}: {count} times ({count/len(targets)*100:.1f}%)")

# Check model's initial predictions
print("\nModel's output distribution:")
with torch.no_grad():
    # Test on each prefix
    for prefix in range(4):
        state = grammar.encode_token(prefix)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        logits, _ = model.forward(state_tensor)
        probs = torch.softmax(logits, dim=-1)

        print(f"  Prefix {prefix}:")
        top_probs, top_tokens = torch.topk(probs, k=5)
        for prob, token in zip(top_probs, top_tokens):
            print(f"    Token {token.item()}: {prob.item():.3f}")

# Train for a few steps
print("\nTraining MLE for 10 steps...")
for step in range(10):
    # Collect batch
    experiences = []
    for _ in range(100):
        prefix, state = grammar.reset(test_mode=False)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        logits, value = model.forward(state_tensor)
        probs = torch.softmax(logits, dim=-1)
        predicted_token = torch.multinomial(probs, 1).item()
        next_state, reward, done, info = grammar.step(predicted_token, test_mode=False)

        exp = TokenExperience(
            state=state_tensor,
            prediction=predicted_token,
            target=info["actual"],
            reward=reward,
            log_prob=torch.log(probs[predicted_token] + 1e-10).item(),
            correct=info["correct"],
        )
        experiences.append(exp)

    # Update
    metrics = mle.update(experiences)
    accuracy = sum(1 for e in experiences if e.correct) / len(experiences)
    print(f"  Step {step+1}: Loss={metrics['loss/ce']:.4f}, Accuracy={accuracy:.2%}")

# Final test
print("\nFinal model predictions:")
with torch.no_grad():
    for prefix in range(4):
        state = grammar.encode_token(prefix)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        logits, _ = model.forward(state_tensor)
        probs = torch.softmax(logits, dim=-1)

        print(f"  Prefix {prefix}:")
        top_probs, top_tokens = torch.topk(probs, k=5)
        for prob, token in zip(top_probs, top_tokens):
            print(f"    Token {token.item()}: {prob.item():.3f}")
