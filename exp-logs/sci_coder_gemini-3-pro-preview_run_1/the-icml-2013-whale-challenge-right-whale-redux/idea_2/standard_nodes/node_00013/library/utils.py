import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_roc_auc(y_true, y_scores) -> float:
    """
    Calculates the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

    Args:
        y_true: Ground truth binary labels (0 or 1). Can be list, numpy array, or torch tensor.
        y_scores: Predicted probabilities for the positive class. Can be list, numpy array, or torch tensor.

    Returns:
        float: The ROC AUC score.
    """
    # Convert torch tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    try:
        score = roc_auc_score(y_true, y_scores)
        return score
    except ValueError:
        # Handle edge cases where only one class is present in y_true
        return 0.5
