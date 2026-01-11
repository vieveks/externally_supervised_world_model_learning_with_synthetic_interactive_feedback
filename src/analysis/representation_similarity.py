"""Representation similarity analysis: CCA and CKA.

For Week 3: Compare RL and MLE learned representations.
"""

import torch
import numpy as np
from typing import Tuple, Dict
from sklearn.cross_decomposition import CCA as sklearn_CCA


def center_kernel(K: torch.Tensor) -> torch.Tensor:
    """Center a kernel matrix."""
    n = K.shape[0]
    H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
    return H @ K @ H


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Compute Linear Centered Kernel Alignment (CKA).

    CKA measures similarity between two sets of representations.
    Range: [0, 1], where 1 = perfectly aligned.

    Args:
        X: Representations from model 1 (n_samples, dim1)
        Y: Representations from model 2 (n_samples, dim2)

    Returns:
        CKA score
    """
    # Compute centered Gram matrices
    X_centered = X - X.mean(dim=0, keepdim=True)
    Y_centered = Y - Y.mean(dim=0, keepdim=True)

    # Gram matrices
    K_X = X_centered @ X_centered.T
    K_Y = Y_centered @ Y_centered.T

    # CKA formula
    numerator = torch.sum(K_X * K_Y)
    denominator = torch.sqrt(torch.sum(K_X * K_X) * torch.sum(K_Y * K_Y))

    if denominator == 0:
        return 0.0

    return (numerator / denominator).item()


def rbf_cka(X: torch.Tensor, Y: torch.Tensor, sigma: float = None) -> float:
    """
    Compute RBF Centered Kernel Alignment (CKA).

    Uses RBF kernel instead of linear kernel for non-linear comparison.

    Args:
        X: Representations from model 1 (n_samples, dim1)
        Y: Representations from model 2 (n_samples, dim2)
        sigma: RBF bandwidth (default: median heuristic)

    Returns:
        CKA score
    """
    def rbf_kernel(X: torch.Tensor, sigma: float) -> torch.Tensor:
        """Compute RBF kernel matrix."""
        # Pairwise squared distances
        X_norm = (X ** 2).sum(dim=1, keepdim=True)
        dists = X_norm + X_norm.T - 2 * X @ X.T

        # RBF kernel
        return torch.exp(-dists / (2 * sigma ** 2))

    # Compute median heuristic for sigma if not provided
    if sigma is None:
        with torch.no_grad():
            X_norm = (X ** 2).sum(dim=1, keepdim=True)
            dists = X_norm + X_norm.T - 2 * X @ X.T
            sigma = torch.median(torch.sqrt(dists[dists > 0])).item()

    # Compute RBF kernels
    K_X = rbf_kernel(X, sigma)
    K_Y = rbf_kernel(Y, sigma)

    # Center kernels
    K_X_centered = center_kernel(K_X)
    K_Y_centered = center_kernel(K_Y)

    # CKA formula
    numerator = torch.sum(K_X_centered * K_Y_centered)
    denominator = torch.sqrt(
        torch.sum(K_X_centered * K_X_centered) *
        torch.sum(K_Y_centered * K_Y_centered)
    )

    if denominator == 0:
        return 0.0

    return (numerator / denominator).item()


def cca_similarity(X: torch.Tensor, Y: torch.Tensor, n_components: int = None) -> Tuple[float, np.ndarray]:
    """
    Compute Canonical Correlation Analysis (CCA) similarity.

    CCA finds linear combinations of features that are maximally correlated.

    Args:
        X: Representations from model 1 (n_samples, dim1)
        Y: Representations from model 2 (n_samples, dim2)
        n_components: Number of canonical components (default: min(dims))

    Returns:
        mean_correlation: Mean of canonical correlations
        correlations: All canonical correlations
    """
    # Convert to numpy
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()

    # Determine number of components
    if n_components is None:
        n_components = min(X_np.shape[1], Y_np.shape[1], X_np.shape[0] - 1)

    # Fit CCA
    cca = sklearn_CCA(n_components=n_components)

    try:
        X_c, Y_c = cca.fit_transform(X_np, Y_np)

        # Compute correlations for each canonical component
        correlations = np.array([
            np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
            for i in range(n_components)
        ])

        mean_correlation = np.mean(correlations)

        return mean_correlation, correlations

    except Exception as e:
        print(f"CCA failed: {e}")
        return 0.0, np.array([])


def pwcca_similarity(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Compute Projection Weighted Canonical Correlation Analysis (PWCCA).

    PWCCA weights canonical correlations by how much variance they explain.
    More robust than simple mean CCA.

    Args:
        X: Representations from model 1 (n_samples, dim1)
        Y: Representations from model 2 (n_samples, dim2)

    Returns:
        PWCCA score
    """
    # Convert to numpy
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()

    n_components = min(X_np.shape[1], Y_np.shape[1], X_np.shape[0] - 1)

    # Fit CCA
    cca = sklearn_CCA(n_components=n_components)

    try:
        X_c, Y_c = cca.fit_transform(X_np, Y_np)

        # Compute correlations
        correlations = np.array([
            np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
            for i in range(n_components)
        ])

        # Compute variance explained in X by each component
        variances = np.var(X_c, axis=0)
        weights = variances / np.sum(variances)

        # Weighted average
        pwcca = np.sum(weights * correlations)

        return float(pwcca)

    except Exception as e:
        print(f"PWCCA failed: {e}")
        return 0.0


def compare_representations(
    X: torch.Tensor,
    Y: torch.Tensor,
    methods: list = None
) -> Dict[str, float]:
    """
    Compare two sets of representations using multiple similarity metrics.

    Args:
        X: Representations from model 1 (n_samples, dim1)
        Y: Representations from model 2 (n_samples, dim2)
        methods: List of methods to use (default: all)

    Returns:
        Dictionary of similarity scores
    """
    if methods is None:
        methods = ['linear_cka', 'rbf_cka', 'cca', 'pwcca']

    results = {}

    if 'linear_cka' in methods:
        try:
            results['linear_cka'] = linear_cka(X, Y)
        except Exception as e:
            print(f"Linear CKA failed: {e}")
            results['linear_cka'] = 0.0

    if 'rbf_cka' in methods:
        try:
            results['rbf_cka'] = rbf_cka(X, Y)
        except Exception as e:
            print(f"RBF CKA failed: {e}")
            results['rbf_cka'] = 0.0

    if 'cca' in methods:
        try:
            mean_corr, correlations = cca_similarity(X, Y)
            # Handle NaN
            if np.isnan(mean_corr):
                mean_corr = 0.0
            results['cca_mean'] = mean_corr
            results['cca_correlations'] = correlations
        except Exception as e:
            print(f"CCA failed: {e}")
            results['cca_mean'] = 0.0
            results['cca_correlations'] = np.array([])

    if 'pwcca' in methods:
        try:
            pwcca_val = pwcca_similarity(X, Y)
            # Handle NaN
            if np.isnan(pwcca_val):
                pwcca_val = 0.0
            results['pwcca'] = pwcca_val
        except Exception as e:
            print(f"PWCCA failed: {e}")
            results['pwcca'] = 0.0

    return results


def representational_similarity_analysis(
    model1,
    model2,
    states: torch.Tensor,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Complete representational similarity analysis between two models.

    Extracts hidden representations and compares them using multiple metrics.

    Args:
        model1: First model (e.g., RL)
        model2: Second model (e.g., MLE)
        states: Input states to probe (n_samples, state_dim)
        device: Device to run on

    Returns:
        Dictionary of similarity metrics
    """
    model1.eval()
    model2.eval()

    with torch.no_grad():
        # Extract hidden representations
        states = states.to(device)

        # Get hidden states from both models
        hidden1 = model1.get_hidden(states)  # (n_samples, hidden_dim)
        hidden2 = model2.get_hidden(states)  # (n_samples, hidden_dim)

        # Compare representations
        results = compare_representations(hidden1, hidden2)

    return results


def analyze_by_context(
    model1,
    model2,
    grammar,
    context_type: str,
    num_samples: int = 100,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Analyze representational similarity in specific contexts.

    Args:
        model1: First model
        model2: Second model
        grammar: Grammar environment
        context_type: 'ambiguous', 'deterministic', or 'all'
        num_samples: Number of samples to collect
        device: Device to run on

    Returns:
        Similarity metrics for this context
    """
    # Collect states from the specified context
    states = []

    for _ in range(num_samples):
        token, state = grammar.reset()
        states.append(state)

    states_tensor = torch.tensor(states, dtype=torch.float32, device=device)

    # Analyze similarity
    results = representational_similarity_analysis(
        model1, model2, states_tensor, device
    )

    return results
