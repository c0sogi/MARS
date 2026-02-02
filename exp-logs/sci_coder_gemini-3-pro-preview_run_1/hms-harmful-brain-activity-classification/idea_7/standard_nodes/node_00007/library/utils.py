import os
import sys
import random
import logging
import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed: int):
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def kl_divergence_score(
    y_pred: torch.Tensor, y_true: torch.Tensor, from_logits: bool = True
) -> torch.Tensor:
    """
    Calculates the Kullback-Leibler Divergence score between predictions and targets.

    Args:
        y_pred (torch.Tensor): Predicted logits or probabilities. Shape (Batch, Num_Classes).
        y_true (torch.Tensor): Ground truth probabilities. Shape (Batch, Num_Classes).
        from_logits (bool): If True, applies log_softmax to y_pred.
                            If False, applies log to y_pred (assumes y_pred are probabilities).

    Returns:
        torch.Tensor: The scalar KL Divergence score (averaged over the batch).
    """
    if from_logits:
        # y_pred are logits -> convert to log-probabilities
        input_log_probs = F.log_softmax(y_pred, dim=1)
    else:
        # y_pred are probabilities -> convert to log-probabilities
        # Clamp to avoid log(0)
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, min=epsilon, max=1.0)
        input_log_probs = torch.log(y_pred)

    # PyTorch kl_div expects input=log_probs, target=probs
    # reduction='batchmean' aligns with the mathematical definition of KL divergence averaged over the batch
    loss = F.kl_div(input_log_probs, y_true, reduction="batchmean")

    return loss


def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Configures and returns a logger instance.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to a log file. If provided, logs will be written to this file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times if get_logger is called repeatedly
    if not logger.handlers:
        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Stream Handler (Console)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # File Handler (Optional)
        if log_file:
            # Ensure directory exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
