import os
import sys
import logging
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import WORKING_DIR, seed_everything


def get_logger(name="training", log_file=None):
    """
    Creates a logger that writes to a file and stdout.
    """
    if log_file is None:
        log_file = os.path.join(WORKING_DIR, "inference.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        # File Handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console Handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


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


def calculate_roc_auc(y_true, y_scores):
    """
    Calculates the Area Under the ROC Curve.
    Handles cases where only one class is present in the batch by returning 0.5.
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Check if we have both classes
    if len(np.unique(y_true)) < 2:
        return 0.5

    try:
        return roc_auc_score(y_true, y_scores)
    except ValueError:
        return 0.5


def save_checkpoint(state, filename="checkpoint.pth", is_best=False):
    """
    Saves the model state to the working directory.
    """
    filepath = os.path.join(WORKING_DIR, filename)
    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(WORKING_DIR, "best_model.pth")
        torch.save(state, best_filepath)


def load_checkpoint(model, filename="best_model.pth", optimizer=None, device="cpu"):
    """
    Loads a model checkpoint.
    """
    filepath = os.path.join(WORKING_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    # Load model weights
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint
