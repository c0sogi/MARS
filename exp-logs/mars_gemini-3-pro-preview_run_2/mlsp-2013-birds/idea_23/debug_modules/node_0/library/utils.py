import os
import random
import sys
import logging
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN to guarantee reproducible results
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def get_score(y_true, y_pred):
    """
    Calculates the Macro-Averaged ROC AUC score.
    Handles cases where a class might be missing in the target vector for small validation sets.

    Args:
        y_true (np.ndarray): Ground truth labels (N_samples, N_classes).
        y_pred (np.ndarray): Predicted probabilities (N_samples, N_classes).

    Returns:
        float: The mean ROC AUC score across all valid classes.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    scores = []
    num_classes = y_true.shape[1]

    for i in range(num_classes):
        # Only calculate AUC if the class has both positive and negative samples
        # in the provided batch/fold. Scikit-learn's roc_auc_score throws an error
        # if y_true contains only one class.
        if len(np.unique(y_true[:, i])) > 1:
            try:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                scores.append(score)
            except ValueError:
                # Fallback if something unexpected happens
                pass
        else:
            # If a class is constant in the ground truth (e.g., all 0s), AUC is undefined.
            # We exclude it from the macro average for this specific validation step.
            pass

    # If no classes were valid (unlikely in a proper split, but possible in very small debug batches),
    # return a neutral score.
    if len(scores) == 0:
        return 0.5

    return np.mean(scores)


def get_logger(filename):
    """
    Initializes a logger that writes to both a file and the console.

    Args:
        filename (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter

    logger = getLogger(__name__)
    logger.setLevel(INFO)

    # Avoid adding handlers multiple times if get_logger is called repeatedly
    if not logger.handlers:
        # Console Handler
        handler1 = StreamHandler(sys.stdout)
        handler1.setFormatter(Formatter("%(message)s"))

        # File Handler
        handler2 = FileHandler(filename=filename)
        handler2.setFormatter(Formatter("%(message)s"))

        logger.addHandler(handler1)
        logger.addHandler(handler2)

    return logger
