import torch
import torch.nn.functional as F
import numpy as np
from library.config import DEVICE, FLOAT_PRECISION


def predict_proba_gpu(X_numpy, W_numpy, b_numpy):
    """
    Performs GPU-accelerated inference using PyTorch with high precision (float64).

    This kernel implements the linear scoring and softmax normalization steps
    on the configured hardware device. It ensures that the dot product accumulation
    retains numerical stability by strictly adhering to float64 precision.

    Args:
        X_numpy (np.ndarray): Input features of shape (n_samples, n_features).
        W_numpy (np.ndarray): Weights of shape (n_classes, n_features).
        b_numpy (np.ndarray): Biases of shape (n_classes,).

    Returns:
        np.ndarray: Class probabilities of shape (n_samples, n_classes).
    """
    # Use no_grad to disable gradient calculation for inference,
    # which reduces memory consumption and speeds up computations.
    with torch.no_grad():
        # Convert NumPy arrays to PyTorch tensors.
        # We explicitly enforce the precision (float64) and device (CPU/GPU)
        # defined in the configuration to ensure numerical stability.
        X_t = torch.tensor(X_numpy, dtype=FLOAT_PRECISION, device=DEVICE)
        W_t = torch.tensor(W_numpy, dtype=FLOAT_PRECISION, device=DEVICE)
        b_t = torch.tensor(b_numpy, dtype=FLOAT_PRECISION, device=DEVICE)

        # Compute Logits: Z = X @ W.T + b
        # F.linear efficiently computes the matrix multiplication and bias addition.
        # X_t: (N, D), W_t: (K, D), b_t: (K,) -> logits: (N, K)
        logits = F.linear(X_t, W_t, b_t)

        # Apply Softmax to get probabilities.
        # dim=1 ensures the softmax is normalized across the class dimension for each sample.
        probs_t = F.softmax(logits, dim=1)

        # Move the result back to CPU and convert to a NumPy array.
        probs = probs_t.cpu().numpy()

    return probs
