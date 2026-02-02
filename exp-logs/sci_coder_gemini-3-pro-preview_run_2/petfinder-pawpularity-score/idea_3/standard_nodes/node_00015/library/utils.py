import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from library.config import seed_everything


def get_score(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE) between the predicted
    probabilities (scaled 0-1) and the ground truth.

    This function handles automatic rescaling. It assumes predictions are in
    the [0, 1] range. It detects if the ground truth is scaled [0, 1] or
    original [1, 100] and computes the RMSE on the original [1, 100] scale.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values.
        y_pred (np.ndarray or torch.Tensor): Predicted values, expected to be in [0, 1].

    Returns:
        float: RMSE score on the original [1, 100] scale.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Heuristic to check if y_true is scaled (max <= 1) or original (max > 1)
    # The dataset minimum is > 1.0, so original values will trigger max > 1.
    if y_true.max() > 1.0:
        # y_true is original scale [1, 100]. Scale y_pred to match.
        y_pred_scaled = y_pred * 100.0
        y_true_scaled = y_true
    else:
        # y_true is scaled [0, 1]. Scale both to original [1, 100] for reporting.
        y_pred_scaled = y_pred * 100.0
        y_true_scaled = y_true * 100.0

    # Calculate MSE
    mse = mean_squared_error(y_true_scaled, y_pred_scaled)

    # Return RMSE
    return np.sqrt(mse)
