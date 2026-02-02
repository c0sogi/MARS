import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed: int = Config.SEED):
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
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state: dict, is_best: bool, filepath: str = Config.MODEL_PATH):
    """
    Saves the model checkpoint to the specified filepath.

    Args:
        state (dict): The state dictionary containing model weights and optimizer state.
        is_best (bool): Boolean flag indicating if this is the best model so far.
        filepath (str): The destination path for the checkpoint. Defaults to Config.MODEL_PATH.
    """
    if is_best:
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(state, filepath)


def calculate_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve.

    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: The AUC score. Returns 0.5 if only one class is present.
    """
    # Check if there are at least two classes to calculate AUC
    if len(np.unique(y_true)) < 2:
        return 0.5

    # Detach tensors if necessary and convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return roc_auc_score(y_true, y_pred)


def print_metric(name: str, value: float):
    """
    Prints a metric name and its value with full precision (no formatting).

    Args:
        name (str): The name of the metric (e.g., "Validation AUC").
        value (float): The numerical value of the metric.
    """
    print(f"{name}: {value}")
