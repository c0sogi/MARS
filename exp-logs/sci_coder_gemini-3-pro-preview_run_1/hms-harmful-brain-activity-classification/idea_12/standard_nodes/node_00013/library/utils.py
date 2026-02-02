import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed: The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def kl_divergence_loss(
    y_pred: np.ndarray, y_true: np.ndarray, epsilon: float = 1e-15
) -> float:
    """
    Computes the Kullback-Leibler (KL) Divergence between predicted and target probabilities.
    This is the official competition metric.

    Formula: D(P || Q) = sum(P * log(P / Q))

    Args:
        y_pred: Predicted probabilities of shape (N, C).
        y_true: Ground truth probabilities of shape (N, C).
        epsilon: Small constant to avoid log(0).

    Returns:
        The mean KL divergence score across all samples.
    """
    # Clip predictions to avoid log(0) errors
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Calculate KL Divergence: P * log(P) - P * log(Q)

    # Term 1: P * log(P)
    # Handle the case where y_true is 0. 0 * log(0) should be 0.
    # We use a mask to calculate log only where y_true > 0.
    term_p = np.zeros_like(y_true)
    mask = y_true > 0
    term_p[mask] = y_true[mask] * np.log(y_true[mask])

    # Term 2: P * log(Q)
    term_q = y_true * np.log(y_pred)

    # Sum over classes (axis=1), then mean over samples (axis=0)
    kl_div = np.sum(term_p - term_q, axis=1)

    return float(np.mean(kl_div))


def generate_coordinate_map(
    height: int, width: int, device: str = "cpu"
) -> torch.Tensor:
    """
    Generates a temporal coordinate map (linear gradient) for the spectrogram.

    This creates a 2D tensor where values vary linearly from -1.0 to 1.0 along the
    time axis (Height). This map is used as an additional channel in the spectrogram
    input to provide the model with explicit temporal position information.

    Args:
        height: The height of the spectrogram (Time dimension).
        width: The width of the spectrogram (Frequency dimension).
        device: The PyTorch device to create the tensor on.

    Returns:
        A tensor of shape (1, height, width) containing the coordinate gradient.
    """
    # Generate a linear gradient from -1 to 1 along the height (time) axis
    # Shape: (Height,)
    gradient = torch.linspace(-1.0, 1.0, steps=height, device=device)

    # Reshape to (Height, 1) and expand to (Height, Width)
    # This creates a map where every row has the same value, but values change across rows
    coordinate_map = gradient.unsqueeze(1).expand(height, width)

    # Add the channel dimension: (1, Height, Width)
    return coordinate_map.unsqueeze(0)
