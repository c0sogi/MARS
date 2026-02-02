import os
import sys
import random
import numpy as np
import torch
import logging
from library.configuration import Config


def seed_everything(seed=Config.seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
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


def map5(preds, targets):
    """
    Computes Mean Average Precision @ 5 (MAP@5).

    For a single ground truth label per image, this is equivalent to
    Mean Reciprocal Rank (MRR) at 5.

    Args:
        preds (torch.Tensor, np.ndarray, or list): Shape (N, 5) containing top-5 predicted labels/indices.
        targets (torch.Tensor, np.ndarray, or list): Shape (N,) containing true labels/indices.

    Returns:
        float: The MAP@5 score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    score = 0.0
    n = len(targets)

    if n == 0:
        return 0.0

    for i in range(n):
        p = preds[i]
        t = targets[i]

        # Handle case where rows might be numpy arrays
        if isinstance(p, np.ndarray):
            p = p.tolist()

        # Check if target is in predictions
        if t in p:
            # Get rank (0-indexed)
            rank = p.index(t)
            # Only consider top 5
            if rank < 5:
                score += 1.0 / (rank + 1)

    return score / n


def setup_logger(out_file=None):
    """
    Sets up the logger to write to console and optionally to a file.

    Args:
        out_file (str, optional): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplication if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File Handler
    if out_file:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        file_handler = logging.FileHandler(out_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
