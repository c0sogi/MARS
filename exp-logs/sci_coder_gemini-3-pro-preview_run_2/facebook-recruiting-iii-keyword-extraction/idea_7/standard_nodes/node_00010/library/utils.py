import os
import random
import numpy as np
import torch
from sklearn.metrics import f1_score
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Saves the model state, optimizer state, epoch, and loss to a file.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "loss": loss,
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, device=Config.DEVICE):
    """
    Loads the model state and optimizer state from a file.
    Returns the checkpoint dictionary if successful, else None.
    """
    if not os.path.exists(path):
        return None

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def optimize_threshold(y_true, y_pred_probs, num_steps=100):
    """
    Calculates the optimal threshold for multi-label classification by maximizing the Mean F1-Score (Samples).
    Uses percentiles of the predicted probabilities to define the search range dynamically.

    Args:
        y_true: Ground truth labels (sparse matrix or numpy array).
        y_pred_probs: Predicted probabilities (numpy array).
        num_steps: Number of steps to search within the range.

    Returns:
        best_threshold: The threshold value that yields the highest F1 score.
        best_score: The highest F1 score achieved.
    """
    # Estimate percentiles from a subset of data to define search range efficiently
    # We take the first 10,000 samples to estimate the distribution of probabilities
    n_subset = min(y_pred_probs.shape[0], 10000)
    subset_probs = y_pred_probs[:n_subset].ravel()

    # Calculate percentiles: 95th to 99.9th
    # This focuses the search on the tail where the "positive" signals likely reside
    # for sparse multi-label problems.
    p_lower = np.percentile(subset_probs, 95)
    p_upper = np.percentile(subset_probs, 99.9)

    # Handle edge cases where probabilities are very clustered or degenerate
    if p_lower >= p_upper:
        p_lower = 0.1
        p_upper = 0.9

    # Generate candidate thresholds
    thresholds = np.linspace(p_lower, p_upper, num_steps)

    best_score = -1.0
    best_threshold = 0.5

    # Search for optimal threshold
    for thresh in thresholds:
        # Apply threshold
        y_pred_bin = (y_pred_probs >= thresh).astype(np.int8)

        # Calculate F1 Score (Samples average)
        # zero_division=0 ensures no warnings/errors for empty predictions
        score = f1_score(y_true, y_pred_bin, average="samples", zero_division=0)

        if score > best_score:
            best_score = score
            best_threshold = thresh

    # Print full precision as requested
    print(f"Best Validation F1 Score: {best_score}")
    print(f"Optimal Threshold: {best_threshold}")

    return best_threshold, best_score
