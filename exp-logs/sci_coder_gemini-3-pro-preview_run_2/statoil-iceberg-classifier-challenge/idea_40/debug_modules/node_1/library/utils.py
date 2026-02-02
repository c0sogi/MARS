import os
import random
import numpy as np
import torch
from library.config import SEED


def set_seed(seed=SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to the value in config.py.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def log_metric(phase, metric_name, value, epoch=None):
    """
    Logs a metric with full precision.

    Args:
        phase (str): The phase of execution (e.g., 'Train', 'Val', 'Test').
        metric_name (str): The name of the metric (e.g., 'Loss', 'Accuracy').
        value (float): The value of the metric.
        epoch (int, optional): The current epoch number.
    """
    prefix = f"[{phase}]"
    if epoch is not None:
        prefix += f" Epoch {epoch} -"

    # Print with full precision (repr) or high decimal places
    print(f"{prefix} {metric_name}: {value}")
