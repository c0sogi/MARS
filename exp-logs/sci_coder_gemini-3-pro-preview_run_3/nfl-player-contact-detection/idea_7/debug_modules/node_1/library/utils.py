import os
import time
import random
import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from library.config import Config


class Timer:
    """
    Context manager to measure and print the execution time of a code block.
    """

    def __init__(self, name="Task"):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        print(f"[{self.name}] Start")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        print(f"[{self.name}] Done. Execution time: {elapsed_time} seconds")


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


def compute_mcc(y_true, y_pred):
    """
    Calculates the Matthews Correlation Coefficient (MCC).

    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted binary labels.

    Returns:
        float: The MCC score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    score = matthews_corrcoef(y_true, y_pred)
    return score
