import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_accuracy(predictions: list, references: list) -> float:
    """
    Computes the exact match accuracy between predictions and references.
    The predicted and actual string must match exactly in order to count as correct.

    Args:
        predictions (list): List of predicted normalized text strings.
        references (list): List of ground truth normalized text strings.

    Returns:
        float: The accuracy as a ratio (0.0 to 1.0).
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Shape mismatch: Predictions have {len(predictions)} elements, "
            f"but references have {len(references)} elements."
        )

    if len(predictions) == 0:
        return 0.0

    # Calculate correct matches
    correct_count = 0
    for pred, ref in zip(predictions, references):
        if pred == ref:
            correct_count += 1

    return correct_count / len(predictions)
