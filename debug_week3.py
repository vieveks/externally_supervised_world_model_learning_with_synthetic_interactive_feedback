"""Debug Week 3 representation analysis - FAST version."""

import torch
import numpy as np
from src.environment.ambiguous_grammar import AmbiguousGrammar
from src.models.token_model import TokenPredictionModel
from src.training.token_reinforce import TokenMLE, TokenREINFORCE, TokenExperience

# Create grammar
grammar = AmbiguousGrammar(
    vocab_size=16,
    num_ambiguous_tokens=8,
    ambiguity_level="high",
    branching_factor=2,
)

print("Training simple models (500 steps each - fast debug)...")

# Train MLE - only 500 steps for quick test
mle_model = TokenPredictionModel(vocab_size=16, hidden_dim=64, num_layers=2, num_heads=4).to('cuda')
mle = TokenMLE(model=mle_model, lr=1e-3, device='cuda')

for step in range(500):
    experiences = []
    for _ in range(32):
        token, state = grammar.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32, device='cuda')
        logits, _ = mle_model.forward(state_tensor)
        predicted_token = logits.argmax(dim=-1).item()
        next_state, reward, done, info = grammar.step(predicted_token)
        exp = TokenExperience(state=state_tensor, prediction=predicted_token, target=info["actual"], reward=reward, log_prob=0.0, correct=info["correct"])
        experiences.append(exp)
    mle.update(experiences)
    if (step + 1) % 100 == 0:
        acc = sum(1 for e in experiences if e.correct) / len(experiences)
        print(f"  MLE step {step+1}: acc={acc:.1%}")

print("MLE trained")

# Train RL - only 500 steps for quick test
rl_model = TokenPredictionModel(vocab_size=16, hidden_dim=64, num_layers=2, num_heads=4).to('cuda')
rl = TokenREINFORCE(model=rl_model, lr=1e-3, baseline_decay=0.99, entropy_coef=0.01, device='cuda')

for step in range(500):
    experiences = []
    for _ in range(32):
        token, state = grammar.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32, device='cuda')
        prediction, log_prob = rl_model.sample_prediction(state_tensor)
        next_state, reward, done, info = grammar.step(prediction.item())
        exp = TokenExperience(state=state_tensor, prediction=prediction.item(), target=info["actual"], reward=reward, log_prob=log_prob.item(), correct=info["correct"])
        experiences.append(exp)
    rl.update(experiences)
    if (step + 1) % 100 == 0:
        acc = sum(1 for e in experiences if e.correct) / len(experiences)
        print(f"  RL step {step+1}: acc={acc:.1%}")

print("RL trained")

# Collect representations
print("\nCollecting representations...")
mle_model.eval()
rl_model.eval()

mle_hiddens = []
rl_hiddens = []

with torch.no_grad():
    for _ in range(100):
        token, state = grammar.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32, device='cuda')

        mle_h = mle_model.get_hidden(state_tensor).cpu().numpy()
        rl_h = rl_model.get_hidden(state_tensor).cpu().numpy()

        mle_hiddens.append(mle_h)
        rl_hiddens.append(rl_h)

mle_hiddens = np.array(mle_hiddens)
rl_hiddens = np.array(rl_hiddens)

print(f"MLE hidden shape: {mle_hiddens.shape}")
print(f"RL hidden shape: {rl_hiddens.shape}")

# Check statistics
print(f"\nMLE hidden stats:")
print(f"  Mean: {mle_hiddens.mean():.3f}")
print(f"  Std: {mle_hiddens.std():.3f}")
print(f"  Min: {mle_hiddens.min():.3f}")
print(f"  Max: {mle_hiddens.max():.3f}")

print(f"\nRL hidden stats:")
print(f"  Mean: {rl_hiddens.mean():.3f}")
print(f"  Std: {rl_hiddens.std():.3f}")
print(f"  Min: {rl_hiddens.min():.3f}")
print(f"  Max: {rl_hiddens.max():.3f}")

# Simple correlation
print("\nSimple correlation test:")
mle_flat = mle_hiddens.flatten()
rl_flat = rl_hiddens.flatten()
corr = np.corrcoef(mle_flat, rl_flat)[0, 1]
print(f"  Pearson correlation: {corr:.3f}")

# Check if representations are diverse
print("\nDiversity check:")
mle_std_per_dim = mle_hiddens.std(axis=0)
rl_std_per_dim = rl_hiddens.std(axis=0)
print(f"  MLE: {(mle_std_per_dim > 0.01).sum()}/{len(mle_std_per_dim)} dimensions active")
print(f"  RL: {(rl_std_per_dim > 0.01).sum()}/{len(rl_std_per_dim)} dimensions active")

# Quick CKA test
print("\nQuick Linear CKA test:")
X = torch.tensor(mle_hiddens, dtype=torch.float32)
Y = torch.tensor(rl_hiddens, dtype=torch.float32)

X_centered = X - X.mean(dim=0, keepdim=True)
Y_centered = Y - Y.mean(dim=0, keepdim=True)

K_X = X_centered @ X_centered.T
K_Y = Y_centered @ Y_centered.T

numerator = torch.sum(K_X * K_Y)
denominator = torch.sqrt(torch.sum(K_X * K_X) * torch.sum(K_Y * K_Y))

cka = (numerator / denominator).item() if denominator > 0 else 0.0
print(f"  Linear CKA: {cka:.3f}")

print("\nDone!")
