import os
import random
import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
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


def clipped_log_loss(y_true, y_pred):
    """
    Computes the multi-class log loss with rescaling and clipping as per task specifications.

    Metric: Multi-class log loss.
    Rescaling: Each row of probabilities is divided by the row sum.
    Clipping: Probabilities are clipped to [1e-15, 1 - 1e-15].

    Args:
        y_true: Ground truth labels. Can be:
                - 1D array-like of integer class indices (shape: (n_samples,))
                - 2D array-like of one-hot encoded labels (shape: (n_samples, n_classes))
        y_pred: Predicted probabilities (shape: (n_samples, n_classes)).

    Returns:
        float: The computed log loss.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Rescale probabilities: each row divided by row sum
    row_sums = y_pred.sum(axis=1, keepdims=True)
    # Safety check: avoid division by zero if a model outputs all zeros
    row_sums[row_sums == 0] = 1.0
    y_pred = y_pred / row_sums

    # Clip probabilities to avoid log(0) and extremes
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)

    n_samples = y_pred.shape[0]

    if y_true.ndim == 1:
        # Case: y_true are class indices
        # We need to gather the probability assigned to the true class for each sample
        # y_true must be integers in range [0, n_classes-1]

        # Advanced indexing to select p(y_true) for each sample
        # rows: 0 to N-1, cols: y_true values
        relevant_probs = y_pred[np.arange(n_samples), y_true]

        # Log loss is negative mean of log probabilities of true classes
        loss = -np.mean(np.log(relevant_probs))

    else:
        # Case: y_true is one-hot encoded or soft labels
        # Standard formula: -1/N * sum(y_true * log(y_pred))
        loss = -np.sum(y_true * np.log(y_pred)) / n_samples

    return loss


def save_submission(ids, classes, preds, filename):
    """
    Saves the predictions to a CSV file in the required submission format.

    Format:
    id,Class1,Class2,...
    2,0.1,0.5,...

    Args:
        ids: Array-like of image IDs.
        classes: List of class names (strings) corresponding to columns of preds.
        preds: Array-like of predicted probabilities (shape: (n_samples, n_classes)).
        filename: Path where the CSV file will be saved.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Create DataFrame
    df = pd.DataFrame(preds, columns=classes)

    # Insert 'id' column at the start
    df.insert(0, "id", ids)

    # Save to CSV without index
    df.to_csv(filename, index=False)
