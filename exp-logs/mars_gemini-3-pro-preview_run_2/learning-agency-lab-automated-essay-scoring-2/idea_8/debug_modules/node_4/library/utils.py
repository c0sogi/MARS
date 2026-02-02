import os
import random
import numpy as np
import torch
import logging
import scipy.optimize as optimize
from functools import partial
from sklearn.metrics import cohen_kappa_score


def seed_everything(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_logger(filename):
    """
    Creates and configures a logger that writes to both a file and the console.

    Args:
        filename (str): The base path/name for the log file (without extension).
    """
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if logger is retrieved multiple times
    if not logger.handlers:
        formatter = logging.Formatter("%(message)s")

        # Console Handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # File Handler
        # Ensure the directory exists
        log_dir = os.path.dirname(filename)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(filename=f"{filename}.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_score(y_true, y_pred):
    """
    Calculates the Quadratic Weighted Kappa score.
    """
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


class OptimizedRounder:
    """
    Optimizes thresholds for rounding continuous predictions to integers
    to maximize the Quadratic Weighted Kappa score.
    """

    def __init__(self):
        # Initial thresholds roughly equidistant for 1-6 scale
        self.coef_ = [1.5, 2.5, 3.5, 4.5, 5.5]

    def _kappa_loss(self, coef, X, y):
        """
        Loss function: Negative Quadratic Weighted Kappa.
        """
        # Enforce ordering of thresholds for logical consistency
        coef_sorted = np.sort(coef)
        # Digitize maps input to bins. Adding 1 converts 0-5 index to 1-6 score.
        X_p = np.digitize(X, coef_sorted) + 1
        ll = cohen_kappa_score(y, X_p, weights="quadratic")
        return -ll

    def fit(self, X, y):
        """
        Optimizes the thresholds using the Nelder-Mead algorithm.

        Args:
            X (np.array): Continuous predictions.
            y (np.array): Ground truth labels.
        """
        loss_partial = partial(self._kappa_loss, X=X, y=y)

        # Initial guess
        initial_coef = self.coef_

        # Minimize the negative kappa
        result = optimize.minimize(loss_partial, initial_coef, method="nelder-mead")

        self.coef_ = np.sort(result.x)

    def predict(self, X, coef=None):
        """
        Applies the learned thresholds to convert continuous predictions to integer scores.

        Args:
            X (np.array): Continuous predictions.
            coef (list, optional): Custom coefficients. Defaults to learned coefficients.
        """
        if coef is None:
            coef = self.coef_

        # Ensure coefficients are sorted
        coef = np.sort(coef)

        # digitize returns indices:
        # < coef[0] -> 0 -> score 1
        # >= coef[0] & < coef[1] -> 1 -> score 2
        # ...
        # >= coef[4] -> 5 -> score 6
        return np.digitize(X, coef) + 1

    def coefficients(self):
        """
        Returns the optimized coefficients.
        """
        return self.coef_
