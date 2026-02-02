import os
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score

# Defined based on the dataset analysis which identified 6 unique classes: 1, 2, 3, 4, 6, 7.
# Class 5 is missing from the training data.
TARGET_CLASSES = [1, 2, 3, 4, 6, 7]


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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def inverse_transform_target(preds, classes=None):
    """
    Maps 0-indexed predictions back to the original Cover_Type class labels.

    Args:
        preds: Numpy array or Torch tensor of 0-indexed class predictions.
        classes: List of original class labels. Defaults to TARGET_CLASSES.

    Returns:
        Numpy array of mapped class labels.
    """
    if classes is None:
        classes = TARGET_CLASSES

    # Convert torch tensor to numpy if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()

    preds = np.array(preds)

    # Ensure preds are integers
    preds = preds.astype(int)

    # Map indices to classes
    mapper = np.array(classes)
    return mapper[preds]


def compute_metrics(y_true, y_pred):
    """
    Computes the multi-class classification accuracy.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        float: The accuracy score.
    """
    # Convert torch tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Flatten if necessary to ensure 1D arrays
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    return accuracy_score(y_true, y_pred)
