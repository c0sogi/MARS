import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    Configures cudnn for deterministic execution.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the Mixup loss by mixing the scalar losses of the two targets.
    This implements the 'Mixed Losses' strategy where the loss is a linear
    combination of the losses computed for each target separately.

    Args:
        criterion: The loss function (e.g., nn.BCEWithLogitsLoss).
        pred: Model predictions (logits).
        y_a: First set of targets.
        y_b: Second set of targets (shuffled).
        lam: Lambda value for mixing (scalar).

    Returns:
        The mixed loss value.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def calculate_auc(y_true, y_score):
    """
    Calculates the Area Under the ROC Curve.

    Args:
        y_true: Ground truth labels (numpy array or list).
        y_score: Predicted probabilities (numpy array or list).

    Returns:
        float: The AUC score. Returns 0.5 if only one class is present in y_true.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # Check for binary classification validity (must have at least 2 classes in y_true)
    # Although validation sets should be stratified, this prevents crashes on small debug batches
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_score)
