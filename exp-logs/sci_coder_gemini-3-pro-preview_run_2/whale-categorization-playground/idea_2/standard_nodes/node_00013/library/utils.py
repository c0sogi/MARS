import os
import sys
import logging
import numpy as np
from library.config import WORKING_DIR


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
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


def calculate_map5(predictions, targets):
    """
    Calculates the Mean Average Precision at 5 (MAP@5).

    The metric is defined as:
    MAP@5 = (1/N) * sum_{i=1}^N (1 / rank_i)
    where rank_i is the 1-based rank of the true label in the prediction list.
    If the true label is not in the top 5 predictions, the score for that sample is 0.

    Args:
        predictions (list or np.ndarray): A list of lists or 2D array where each row contains
                                          the top 5 predicted class labels.
        targets (list or np.ndarray): A list or 1D array containing the ground truth class labels.

    Returns:
        float: The MAP@5 score.
    """
    # Ensure inputs are lists for consistent iteration
    if hasattr(predictions, "tolist"):
        predictions = predictions.tolist()
    if hasattr(targets, "tolist"):
        targets = targets.tolist()

    n = len(targets)
    if n == 0:
        return 0.0

    score_sum = 0.0

    for pred_row, target in zip(predictions, targets):
        # Ensure we only consider the top 5 predictions
        top_5_preds = pred_row[:5]

        if target in top_5_preds:
            # Get 0-based index
            rank_idx = top_5_preds.index(target)
            # Score is 1 / (rank + 1)
            score_sum += 1.0 / (rank_idx + 1)

    return score_sum / n


def setup_logger(name="whale_logger", log_file=None, level=logging.INFO):
    """
    Sets up a logger that outputs to console and optionally to a file.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file. If None, defaults to 'train.log' in WORKING_DIR.
        level (int): Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to prevent duplicate logging if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Stream Handler (Console)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # File Handler
    if log_file is None:
        log_file = os.path.join(WORKING_DIR, "train.log")

    # Ensure directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
