import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import log_loss


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_log_loss(y_true, y_pred):
    """
    Calculates the multi-class logarithmic loss.

    Args:
        y_true (array-like): True labels (indices or one-hot).
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: The log loss value.
    """
    # Using sklearn's implementation which handles various input formats robustly
    # labels parameter ensures all classes are accounted for even if not present in the batch
    return log_loss(y_true, y_pred, labels=[0, 1, 2])


def format_submission(ids, predictions, columns, output_path):
    """
    Formats the predictions into a CSV file for submission, applying numerical
    stability clipping as defined in the task metric.

    The clipping rule is: max(min(p, 1-10^-15), 10^-15).

    Args:
        ids (list or array-like): The sequence of IDs corresponding to the predictions.
        predictions (numpy.ndarray): The predicted probabilities (shape: [n_samples, n_classes]).
        columns (list): The column names for the classes (e.g., ['EAP', 'HPL', 'MWS']).
        output_path (str): The file path to save the submission CSV.
    """
    # Ensure predictions are a numpy array
    preds = np.array(predictions)

    # Apply numerical stability clipping
    # predicted probabilities are replaced with max(min(p,1-10^{-15}),10^{-15})
    epsilon = 1e-15
    preds_clipped = np.clip(preds, epsilon, 1 - epsilon)

    # Create DataFrame
    submission_df = pd.DataFrame(preds_clipped, columns=columns)

    # Insert ID column at the beginning
    submission_df.insert(0, "id", ids)

    # Save to CSV
    # index=False avoids writing row numbers
    submission_df.to_csv(output_path, index=False)
