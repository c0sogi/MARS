import os
import sys
import random
import numpy as np
import torch
import logging
from scipy.stats import pearsonr
from library.config import Config


def seed_everything(seed: int = Config.seed):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str = "Main"):
    """
    Configures and returns a logger instance for tracking progress.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create handler if it doesn't exist to avoid duplicate logs
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def compute_metrics(eval_pred):
    """
    Computes the Pearson correlation coefficient for the evaluation predictions.

    Args:
        eval_pred: A tuple (predictions, labels) from the Trainer.

    Returns:
        dict: A dictionary containing the 'pearson' correlation score.
    """
    predictions, labels = eval_pred

    # Predictions from the regression head usually come in shape (N, 1) or (N,)
    # We flatten them to ensure they are 1D arrays for pearsonr
    predictions = predictions.flatten()
    labels = labels.flatten()

    pearson_corr, _ = pearsonr(predictions, labels)

    return {"pearson": pearson_corr}
