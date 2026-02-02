import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_class_weights(df, target_col="label"):
    """
    Calculates the positive class weight for binary classification to handle class imbalance.
    This implements the Inverse Class Frequency strategy by computing the ratio of
    negative samples to positive samples.

    Args:
        df (pd.DataFrame): The dataframe containing the training labels.
        target_col (str): The name of the column containing the labels (0 or 1).

    Returns:
        torch.Tensor: A scalar tensor 'pos_weight' calculated as (num_neg / num_pos).
                      This is intended for use with torch.nn.BCEWithLogitsLoss(pos_weight=...).
    """
    labels = df[target_col].values
    num_pos = np.sum(labels == 1)
    num_neg = np.sum(labels == 0)

    # Avoid division by zero if no positive samples exist
    if num_pos == 0:
        return torch.tensor(1.0, dtype=torch.float32)

    # Calculate pos_weight: scale positive class loss up relative to negative class
    # Equivalent to w_pos = N_neg / N_pos when w_neg = 1.0
    pos_weight = num_neg / num_pos

    return torch.tensor(pos_weight, dtype=torch.float32)


def compute_roc_auc(y_true, y_pred):
    """
    Computes the Area Under the Receiver Operating Characteristic Curve (ROC AUC).
    Handles input conversion from PyTorch tensors to NumPy arrays.

    Args:
        y_true: Array-like or Tensor of ground truth labels (0 or 1).
        y_pred: Array-like or Tensor of predicted probabilities for class 1.

    Returns:
        float: The ROC AUC score. Returns 0.5 if only one class is present in y_true.
    """
    # Detach tensors if necessary and convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Handle edge case where only one class is present in the batch/set
    # ROC AUC is undefined in this case; return 0.5 (random guessing)
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_pred)
