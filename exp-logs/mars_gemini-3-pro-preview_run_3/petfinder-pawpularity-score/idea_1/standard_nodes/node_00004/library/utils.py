import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import mean_squared_error


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_rmse(y_true, y_pred):
    """
    Computes the Root Mean Squared Error (RMSE) between true and predicted values.

    Args:
        y_true (array-like or torch.Tensor): Ground truth target values.
        y_pred (array-like or torch.Tensor): Predicted target values.

    Returns:
        float: The RMSE value.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Calculate RMSE
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)


def save_submission(ids, predictions, output_path):
    """
    Saves the predictions to a CSV file in the required competition format.

    Args:
        ids (list or np.array): List of Pet Profile IDs.
        predictions (list or np.array): List of predicted Pawpularity scores.
        output_path (str): Full path where the CSV should be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame matching the sample submission format
    df = pd.DataFrame({"Id": ids, "Pawpularity": predictions})

    # Save to CSV without the index
    df.to_csv(output_path, index=False)
