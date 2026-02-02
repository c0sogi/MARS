import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import log_loss
from library.config import SEED


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # PyTorch seeding
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_log_loss(y_true, y_pred, labels=None):
    """
    Calculates the multi-class log loss with specific clipping and normalization
    as defined in the competition metric.

    The metric requires:
    1. Rescaling probabilities so each row sums to 1.
    2. Clipping probabilities to [1e-15, 1-1e-15].

    Args:
        y_true (array-like): Ground truth labels (n_samples,). Can be class names or indices.
        y_pred (array-like): Predicted probabilities (n_samples, n_classes).
        labels (list, optional): List of class labels to index the columns of y_pred.
                                 Required if y_pred doesn't cover all classes in y_true
                                 or to ensure correct column mapping.

    Returns:
        float: The calculated log loss.
    """
    # Ensure inputs are numpy arrays
    y_pred = np.array(y_pred, dtype=np.float64)

    # 1. Rescale: Divide each row by the row sum
    # "The submitted probabilities ... are rescaled prior to being scored"
    row_sums = y_pred.sum(axis=1, keepdims=True)
    # Handle potential zero sums (though unlikely) to avoid NaNs
    row_sums[row_sums == 0] = 1.0
    y_pred = y_pred / row_sums

    # 2. Clip: max(min(p, 1-10^-15), 10^-15)
    # "predicted probabilities are replaced with max(min(p,1-10^-15),10^-15)"
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # 3. Calculate Log Loss
    # sklearn's log_loss handles the mathematical summation and averaging
    score = log_loss(y_true, y_pred, labels=labels)

    return score


def save_submission(ids, probabilities, columns, output_path):
    """
    Saves the submission file in the correct format.

    Args:
        ids (array-like): Image IDs.
        probabilities (array-like): Predicted probabilities matrix.
        columns (list): List of species names (column headers).
        output_path (str): Path to save the CSV.
    """
    # Create DataFrame
    df = pd.DataFrame(probabilities, columns=columns)

    # Insert ID column at the beginning
    df.insert(0, "id", ids)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
