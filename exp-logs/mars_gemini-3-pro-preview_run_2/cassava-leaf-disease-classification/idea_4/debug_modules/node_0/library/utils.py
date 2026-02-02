import os
import sys
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(log_file=None):
    """
    Configures and returns a logger that writes to both console and a file.

    Args:
        log_file (str, optional): Path to the log file.

    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger()

    # Clear existing handlers to prevent duplication if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File Handler
    if log_file is not None:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class SoftTargetCrossEntropy(nn.Module):
    """
    Cross Entropy Loss that handles soft targets (probabilities) usually generated
    by MixUp or CutMix regularization.

    Standard nn.CrossEntropyLoss requires class indices (long), whereas this
    implementation accepts a probability distribution (float) as the target.
    """

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: predictions (logits) of shape [N, C]
            target: soft targets (probabilities) of shape [N, C]

        Returns:
            torch.Tensor: The mean loss scalar.
        """
        # Calculate log probabilities from logits
        log_probs = F.log_softmax(x, dim=-1)

        # Cross entropy formula: -sum(target * log(predicted_prob))
        # We sum over the class dimension (dim=-1) and then mean over the batch
        loss = torch.sum(-target * log_probs, dim=-1)
        return loss.mean()
