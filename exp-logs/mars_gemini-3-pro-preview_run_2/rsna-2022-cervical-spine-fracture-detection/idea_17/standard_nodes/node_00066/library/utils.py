import os
import sys
import random
import logging
import numpy as np
import torch
import torch.nn.functional as F
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_file: str = None):
    """
    Sets up a logger that writes to a file and stdout.

    Args:
        log_file (str, optional): Path to the log file. Defaults to Config.OUTPUT_DIR/train.log.

    Returns:
        logging.Logger: Configured logger instance.
    """
    if log_file is None:
        log_file = os.path.join(Config.OUTPUT_DIR, "train.log")

    # Ensure the directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("CervicalSpineFracture")
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream Handler (stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


class AverageMeter:
    """
    Computes and stores the average and current value.
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


def get_weighted_log_loss(
    logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor = None
):
    """
    Calculates the weighted multi-label logarithmic loss.

    The loss is calculated as the binary cross entropy weighted by class importance,
    averaged across all samples and classes.

    Args:
        logits (torch.Tensor): Raw model outputs (before sigmoid). Shape (B, 8).
        targets (torch.Tensor): Ground truth labels. Shape (B, 8).
        weights (torch.Tensor, optional): Class weights. Shape (8,).
                                          Defaults to Config.LOSS_WEIGHTS.

    Returns:
        torch.Tensor: Scalar tensor representing the mean weighted loss.
    """
    if weights is None:
        weights = Config.LOSS_WEIGHTS

    # Ensure weights are on the same device as the logits
    if weights.device != logits.device:
        weights = weights.to(logits.device)

    # Calculate weighted BCE.
    # The 'weight' argument in binary_cross_entropy_with_logits broadcasts to the input shape.
    # Since weights is (8,) and logits is (B, 8), the weights are applied column-wise (per class).
    loss = F.binary_cross_entropy_with_logits(
        logits, targets.float(), weight=weights, reduction="mean"
    )

    return loss
