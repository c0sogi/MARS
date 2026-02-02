import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=Config.SEED):
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metric(y_true, y_pred):
    """
    Calculates the Mean Column-wise ROC AUC score.

    Args:
        y_true: Ground truth labels. Can be a torch.Tensor or numpy.ndarray.
                Shape can be (N, num_classes) [one-hot] or (N,) [indices].
        y_pred: Predicted probabilities. Can be a torch.Tensor or numpy.ndarray.
                Shape (N, num_classes).

    Returns:
        float: The macro-averaged ROC AUC score.
    """
    # Convert tensors to numpy arrays if necessary
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    # Handle potential NaNs or Infs in predictions by replacing them
    if np.isnan(y_pred).any():
        y_pred = np.nan_to_num(y_pred)

    try:
        # Check target format to decide on parameters
        # If y_true is 1D (indices) or 2D with 1 column, use multi_class='ovr'
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            score = roc_auc_score(y_true, y_pred, average="macro", multi_class="ovr")
        else:
            # If y_true is one-hot encoded
            score = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        # This can happen if a fold/batch has only one class present in y_true
        # Return 0.5 as a neutral score or 0.0 to indicate failure
        score = 0.5

    # Handle cases where roc_auc_score returns NaN (e.g., missing classes in macro average)
    if np.isnan(score):
        score = 0.5

    return score


def get_class_weights(df):
    """
    Calculates inverse class frequency weights for Weighted Cross-Entropy Loss.

    Args:
        df (pd.DataFrame): The training metadata DataFrame containing target columns.

    Returns:
        torch.Tensor: A tensor of shape (num_classes,) containing the weights,
                      moved to the configured device.
    """
    labels = Config.CLASS_LABELS
    class_counts = []
    total_samples = len(df)

    for label in labels:
        # If the dataframe has one-hot encoded columns
        if label in df.columns:
            count = df[label].sum()
        # Fallback to stratify_label if available
        elif "stratify_label" in df.columns:
            count = (df["stratify_label"] == label).sum()
        else:
            # Should not happen given the metadata generation
            count = 0

        # Prevent division by zero
        if count == 0:
            count = 1
        class_counts.append(count)

    class_counts = np.array(class_counts)
    n_classes = len(labels)

    # Calculate "balanced" weights: n_samples / (n_classes * np.bincount(y))
    weights = total_samples / (n_classes * class_counts)

    # Convert to tensor and move to device
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(Config.DEVICE)

    return weight_tensor
