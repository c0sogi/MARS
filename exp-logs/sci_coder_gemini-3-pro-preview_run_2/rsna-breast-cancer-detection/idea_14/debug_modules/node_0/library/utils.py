import os
import random
import shutil
import sys
import logging
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def probabilistic_f1(y_true, y_pred, epsilon=1e-7):
    """
    Computes the Probabilistic F1 score (pF1).

    Args:
        y_true (np.array): Binary ground truth labels (0 or 1).
        y_pred (np.array): Predicted probabilities (between 0 and 1).
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: The pF1 score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # pTP: Probabilistic True Positives (Sum of probs where ground truth is 1)
    pTP = np.sum(y_true * y_pred)

    # pFP: Probabilistic False Positives (Sum of probs where ground truth is 0)
    pFP = np.sum((1 - y_true) * y_pred)

    # TP + FN: Total actual positives
    total_positives = np.sum(y_true)

    # pPrecision = pTP / (pTP + pFP)
    # Note: pTP + pFP is equivalent to sum(y_pred)
    predicted_sum = pTP + pFP
    pPrecision = pTP / (predicted_sum + epsilon)

    # pRecall = pTP / (TP + FN)
    pRecall = pTP / (total_positives + epsilon)

    # pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)
    denominator = pPrecision + pRecall
    if denominator == 0:
        return 0.0

    pf1 = 2 * (pPrecision * pRecall) / denominator
    return pf1


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
    """

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def setup_logger(log_file):
    """
    Sets up a logger to write to console and a file.
    """
    logger = logging.getLogger("BreastCancerDetection")
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File Handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): If True, copies the checkpoint to 'best_model.pth'.
        filename (str): Name of the checkpoint file.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    filepath = os.path.join(Config.WORK_DIR, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(Config.WORK_DIR, "best_model.pth")
        shutil.copyfile(filepath, best_path)


def get_device():
    """
    Returns the appropriate torch device based on Config.
    """
    return torch.device(Config.DEVICE)
