import os
import random
import numpy as np
import torch
import logging
import nltk
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to: {seed}")


def compute_levenshtein(predictions, targets):
    """
    Computes the Levenshtein error rate.

    Metric = (Sum of Levenshtein distances) / (Total number of ground truth gestures)

    Args:
        predictions (list of list of int): Predicted gesture sequences.
        targets (list of list of int): Ground truth gesture sequences.

    Returns:
        float: The calculated error rate.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"Predictions ({len(predictions)}) and targets ({len(targets)}) must have the same length."
        )

    total_distance = 0
    total_gestures = 0

    for pred_seq, target_seq in zip(predictions, targets):
        # Ensure sequences are lists of integers
        p_seq = list(pred_seq)
        t_seq = list(target_seq)

        # Calculate Levenshtein distance for this sequence pair
        # nltk.edit_distance works on lists of arbitrary hashable items (ints in this case)
        dist = nltk.edit_distance(p_seq, t_seq)

        total_distance += dist
        total_gestures += len(t_seq)

    if total_gestures == 0:
        return 0.0

    score = total_distance / total_gestures
    return score


def setup_logger(name="experiment", log_file=None):
    """
    Configures a logger to print to console and save to a file.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file. If None, defaults to config working dir.

    Returns:
        logging.Logger: Configured logger instance.
    """
    if log_file is None:
        log_file = os.path.join(Config.WORKING_DIR, f"{name}.log")

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers = []

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File Handler
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and accuracy during training.
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
