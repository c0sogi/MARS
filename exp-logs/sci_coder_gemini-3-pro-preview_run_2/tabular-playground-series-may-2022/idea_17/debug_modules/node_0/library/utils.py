import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed: int = Config.RANDOM_STATE):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.RANDOM_STATE.
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


def get_device() -> torch.device:
    """
    Selects the compute device. Prioritizes GPU (CUDA) if available.

    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def compute_auc(y_true, y_scores):
    """
    Computes the Area Under the ROC Curve.

    Args:
        y_true: True binary labels.
        y_scores: Target scores (probability estimates).

    Returns:
        float: The ROC AUC score.
    """
    return roc_auc_score(y_true, y_scores)


def print_metric(name: str, value: float):
    """
    Prints a metric name and its value with full precision.

    Args:
        name (str): The name of the metric.
        value (float): The metric value.
    """
    print(f"{name}: {value}")
