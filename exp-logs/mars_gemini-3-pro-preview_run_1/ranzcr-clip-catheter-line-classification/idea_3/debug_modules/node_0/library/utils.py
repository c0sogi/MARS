import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score


def seed_everything(seed=42):
    """
    Sets the seed for all random number generators to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_score(y_true, y_pred):
    """
    Calculates the average ROC AUC score across all labels.

    Args:
        y_true: Ground truth labels (numpy array or pandas DataFrame).
        y_pred: Predicted probabilities (numpy array or pandas DataFrame).

    Returns:
        float: The average ROC AUC score.
    """
    # Convert to numpy if input is pandas DataFrame
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values

    # Calculate average AUC
    # 'macro' average calculates metrics for each label, and finds their unweighted mean.
    try:
        score = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        # Fallback for edge cases (e.g., only one class present in a batch/subset)
        # Calculate per column and ignore errors
        scores = []
        for i in range(y_true.shape[1]):
            try:
                # Check if there is more than one class in the column
                if len(np.unique(y_true[:, i])) > 1:
                    s = roc_auc_score(y_true[:, i], y_pred[:, i])
                    scores.append(s)
            except ValueError:
                pass

        if len(scores) > 0:
            score = np.mean(scores)
        else:
            score = 0.5  # Default fallback if no columns can be scored

    return float(score)


def get_pos_weights(target_cols, df=None, file_path=None, load_cached_data=True):
    """
    Computes or loads positive class weights for BCEWithLogitsLoss.

    Logic:
    1. If load_cached_data is True and file_path exists, load and return.
    2. Else, compute from df.
       Weight = number_of_negatives / number_of_positives
    3. Save to file_path if provided.

    Args:
        target_cols (list): List of target column names.
        df (pd.DataFrame): DataFrame containing the training data. Required if not loading from cache.
        file_path (str): Path to save/load the weights .npy file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        torch.Tensor: Tensor of weights with shape (num_classes,).
    """

    # 1. Try to load cached data
    if load_cached_data and file_path is not None and os.path.exists(file_path):
        try:
            weights = np.load(file_path)
            return torch.tensor(weights, dtype=torch.float32)
        except Exception:
            # If load fails, proceed to compute
            pass

    # 2. Compute from scratch
    if df is None:
        raise ValueError(
            "DataFrame 'df' is required to compute pos_weights if cache is not found."
        )

    weights = []
    for col in target_cols:
        # Handle cases where column might not exist or be empty
        if col not in df.columns:
            weights.append(1.0)
            continue

        pos = df[col].sum()
        neg = len(df) - pos

        if pos == 0:
            weight = 1.0  # Avoid division by zero, though ideally shouldn't happen in training set
        else:
            weight = neg / pos
        weights.append(weight)

    weights_np = np.array(weights)

    # 3. Save to cache
    if file_path is not None:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, weights_np)

    return torch.tensor(weights_np, dtype=torch.float32)
