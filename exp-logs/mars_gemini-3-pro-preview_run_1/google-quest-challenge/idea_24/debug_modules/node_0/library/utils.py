import os
import sys
import random
import numpy as np
import torch
import logging
from scipy.stats import spearmanr
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_spearmanr_metric(preds, targets):
    """
    Computes the mean column-wise Spearman's rank correlation coefficient.

    Args:
        preds (np.ndarray or torch.Tensor): Predicted probabilities of shape (N, 30).
        targets (np.ndarray or torch.Tensor): Ground truth labels of shape (N, 30).

    Returns:
        float: The mean Spearman's correlation coefficient.
    """
    # Ensure inputs are numpy arrays
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Check shapes
    if preds.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: preds {preds.shape} vs targets {targets.shape}"
        )

    # If arrays are 1D, reshape to (N, 1)
    if len(preds.shape) == 1:
        preds = preds.reshape(-1, 1)
        targets = targets.reshape(-1, 1)

    num_cols = preds.shape[1]
    correlations = []

    for col_idx in range(num_cols):
        # Get the column vectors
        pred_col = preds[:, col_idx]
        target_col = targets[:, col_idx]

        # Compute Spearman correlation
        # spearmanr returns a Result object or (correlation, pvalue) tuple depending on version
        # We handle the case where the column might be constant (std=0), which might return nan.
        try:
            corr, _ = spearmanr(pred_col, target_col)
            if np.isnan(corr):
                corr = 0.0
        except Exception:
            corr = 0.0

        correlations.append(corr)

    return float(np.mean(correlations))


def setup_logger(name="logger", log_file=None, level=logging.INFO):
    """
    Sets up a logger that outputs to console and optionally to a file.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file.
        level (int): Logging level.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger
