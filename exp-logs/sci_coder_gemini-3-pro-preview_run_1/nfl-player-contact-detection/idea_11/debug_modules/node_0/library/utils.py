import os
import random
import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from library.config import RANDOM_STATE


def seed_everything(seed=RANDOM_STATE):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use. Defaults to RANDOM_STATE from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_mcc(y_true, y_pred):
    """
    Calculates the Matthews Correlation Coefficient between true and predicted labels.

    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.

    Returns:
        float: The MCC score.
    """
    # Ensure inputs are numpy arrays for consistency
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return matthews_corrcoef(y_true, y_pred)
