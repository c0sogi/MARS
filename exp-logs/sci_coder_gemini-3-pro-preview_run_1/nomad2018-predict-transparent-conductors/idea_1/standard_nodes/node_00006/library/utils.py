import random
import os
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import mean_squared_log_error


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_rmsle(y_true, y_pred_log):
    """
    Computes the Column-wise Root Mean Squared Logarithmic Error (RMSLE).

    This function first applies an exponential transformation (exp(x) - 1)
    to the log-scale predictions to convert them back to the original scale,
    and then calculates the RMSLE against the ground truth.

    Args:
        y_true: Ground truth values in original scale (numpy array or DataFrame).
        y_pred_log: Predicted values in log(1+x) scale (numpy array or tensor).

    Returns:
        float: The mean RMSLE averaged over the target columns.
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    if isinstance(y_pred_log, torch.Tensor):
        y_pred_log = y_pred_log.detach().cpu().numpy()
    if isinstance(y_pred_log, list):
        y_pred_log = np.array(y_pred_log)

    # Inverse transform predictions: exp(x) - 1
    # This reverses the log(1+y) transformation applied during training
    y_pred = np.expm1(y_pred_log)

    # Clip predictions to be non-negative to avoid errors in log calculation
    # (RMSLE is undefined for negative values)
    y_pred = np.maximum(y_pred, 0)

    # Calculate RMSLE for each column
    rmsles = []
    num_cols = y_true.shape[1]

    for i in range(num_cols):
        # mean_squared_log_error calculates mean((log(1+y) - log(1+y_pred))^2)
        # We use the original scale values here
        try:
            msle = mean_squared_log_error(y_true[:, i], y_pred[:, i])
            rmsle = np.sqrt(msle)
        except ValueError:
            # Fallback for potential edge cases, though clipping should prevent this
            rmsle = 0.0
        rmsles.append(rmsle)

    # Return the column-wise mean
    return np.mean(rmsles)


def save_submission(ids, predictions_log, filename):
    """
    Formats and saves the submission file in the required CSV format.

    Args:
        ids: List or array of sample IDs.
        predictions_log: Predicted values in log(1+x) scale.
        filename: Output file path.
    """
    # Ensure predictions are numpy array
    if isinstance(predictions_log, torch.Tensor):
        predictions_log = predictions_log.detach().cpu().numpy()

    # Inverse transform: exp(x) - 1 to get original energy scale
    predictions = np.expm1(predictions_log)

    # Clip to non-negative values as energies cannot be negative in this context
    predictions = np.maximum(predictions, 0)

    # Create DataFrame with required columns
    df = pd.DataFrame(
        predictions, columns=["formation_energy_ev_natom", "bandgap_energy_ev"]
    )
    df.insert(0, "id", ids)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save to CSV without index
    df.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")
