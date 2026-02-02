import numpy as np
import torch
from library.config import seed_everything


def compute_pearson_correlation(predictions, targets):
    """
    Computes the Pearson correlation coefficient between predictions and targets.

    This function handles inputs as lists, numpy arrays, or torch tensors.
    It flattens the inputs and computes the correlation using numpy.

    Args:
        predictions: Predicted scores (Tensor, Array, or List).
        targets: Ground truth scores (Tensor, Array, or List).

    Returns:
        float: The Pearson correlation coefficient. Returns 0.0 if calculation fails
               (e.g., due to constant values/zero variance).
    """
    # Convert torch tensors to numpy arrays on CPU
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Convert lists to numpy arrays
    predictions = np.array(predictions)
    targets = np.array(targets)

    # Flatten arrays to ensure 1D vectors
    predictions = predictions.flatten()
    targets = targets.flatten()

    # Basic validation
    if len(predictions) != len(targets):
        # In case of mismatch, we cannot compute correlation meaningfully
        # Depending on strictness, we might raise error or return 0.
        # Raising error is safer for debugging.
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
        )

    if len(predictions) < 2:
        return 0.0

    # Compute Pearson Correlation
    # np.corrcoef returns the correlation matrix [[1, r], [r, 1]]
    try:
        correlation_matrix = np.corrcoef(predictions, targets)
        pearson_score = correlation_matrix[0, 1]

        # Handle cases where variance is 0 (resulting in NaN)
        if np.isnan(pearson_score):
            return 0.0

        return float(pearson_score)
    except Exception:
        return 0.0
