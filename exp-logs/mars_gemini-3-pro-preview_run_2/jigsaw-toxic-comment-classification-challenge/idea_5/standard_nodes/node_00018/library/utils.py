import os
import time
import random
import datetime
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    Used for tracking loss and metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_time(elapsed):
    """
    Formats a time duration in seconds to a string (hh:mm:ss).
    """
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def get_score(y_true, y_pred):
    """
    Calculates the Mean Column-wise ROC AUC.

    Args:
        y_true (np.array): Ground truth labels of shape (N, num_labels).
        y_pred (np.array): Predicted probabilities of shape (N, num_labels).

    Returns:
        float: The mean ROC AUC score across all columns.
    """
    scores = []
    num_cols = y_true.shape[1]

    for i in range(num_cols):
        # Calculate AUC for each column individually
        try:
            # Check if column has more than one class to avoid ValueError
            if len(np.unique(y_true[:, i])) > 1:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                scores.append(score)
            else:
                # If only one class is present in the batch/set, we cannot compute AUC.
                # In a full validation set, this shouldn't happen for these labels.
                pass
        except ValueError:
            pass

    if not scores:
        return 0.0

    return np.mean(scores)


def find_optimal_weights(preds_list, y_true):
    """
    Finds the optimal weights to blend multiple predictions using Nelder-Mead optimization.

    Args:
        preds_list (list of np.array): List of prediction arrays, each of shape (N, num_labels).
        y_true (np.array): Ground truth labels of shape (N, num_labels).

    Returns:
        np.array: Optimal weights summing to 1.
    """
    n_models = len(preds_list)
    if n_models == 1:
        return np.array([1.0])

    def loss_func(weights):
        # Ensure weights are positive and sum to 1
        weights = np.abs(weights)
        w_sum = np.sum(weights)
        if w_sum == 0:
            return 1.0  # Return high loss if weights are invalid

        norm_weights = weights / w_sum

        # Compute weighted average of predictions
        final_pred = np.zeros_like(preds_list[0], dtype=float)
        for i, pred in enumerate(preds_list):
            final_pred += norm_weights[i] * pred

        # We want to maximize AUC, so we minimize negative AUC
        return -get_score(y_true, final_pred)

    # Initial weights: equal distribution
    initial_weights = np.ones(n_models) / n_models

    # Optimize
    result = minimize(
        loss_func,
        initial_weights,
        method="Nelder-Mead",
        tol=1e-5,
        options={"maxiter": 100},
    )

    # Normalize final weights
    final_weights = np.abs(result.x)
    final_weights /= np.sum(final_weights)

    return final_weights
