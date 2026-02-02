import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets seeds for reproducibility and configures cuDNN for hardware optimization.

    Args:
        seed (int): The random seed to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # As explicitly requested in the task description for hardware optimization
    # on fixed size inputs (32x32).
    torch.backends.cudnn.benchmark = True
    # benchmark=True generally conflicts with deterministic=True.
    # We prioritize the explicit request for benchmark=True.
    torch.backends.cudnn.deterministic = False


def calculate_roc_auc(y_true, y_scores):
    """
    Calculates the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

    Args:
        y_true (array-like): Ground truth binary labels (0 or 1).
        y_scores (array-like): Predicted probabilities for the positive class (1).

    Returns:
        float: The ROC AUC score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.detach().cpu().numpy()

    return roc_auc_score(y_true, y_scores)
