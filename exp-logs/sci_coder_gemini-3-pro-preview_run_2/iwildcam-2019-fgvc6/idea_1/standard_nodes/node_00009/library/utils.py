import os
import random
import numpy as np
import torch
from sklearn.metrics import f1_score
from library.config import Config


def set_seed(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_macro_f1(y_true, y_pred) -> float:
    """
    Calculates the Macro F1 score using scikit-learn.

    Args:
        y_true: Array-like of ground truth labels.
        y_pred: Array-like of predicted labels.

    Returns:
        float: The macro F1 score.
    """
    return f1_score(y_true, y_pred, average="macro")
