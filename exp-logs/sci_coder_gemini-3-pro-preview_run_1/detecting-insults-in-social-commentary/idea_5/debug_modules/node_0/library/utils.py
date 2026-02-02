import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed: int = Config.seed):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.seed.
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


def calculate_metric(y_true, y_pred):
    """
    Computes the Area Under the Receiver Operating Curve (AUC).

    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: The ROC AUC score.
    """
    # Ensure inputs are numpy arrays or tensors on CPU
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return roc_auc_score(y_true, y_pred)
