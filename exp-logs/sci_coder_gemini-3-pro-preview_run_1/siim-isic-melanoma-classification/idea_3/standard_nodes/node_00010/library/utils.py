import os
import random
import shutil
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import WORKING_DIR


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
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


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve (ROC AUC).

    Args:
        y_true (np.array or torch.Tensor): Ground truth binary labels.
        y_pred (np.array or torch.Tensor): Predicted probabilities for the positive class.

    Returns:
        float: The ROC AUC score.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # specific handling for cases where y_true might be float but contains binary values
    y_true = y_true.astype(int)

    try:
        score = roc_auc_score(y_true, y_pred)
    except ValueError:
        # Handle edge cases like only one class present in the batch
        score = 0.5

    return score


def save_checkpoint(state: dict, is_best: bool, filename: str = "checkpoint.pth"):
    """
    Saves the model checkpoint to the working directory.

    Args:
        state (dict): The state dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): The name of the file to save.
    """
    # Ensure the working directory exists (redundant if config handles it, but safe)
    os.makedirs(WORKING_DIR, exist_ok=True)

    filepath = os.path.join(WORKING_DIR, filename)
    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(WORKING_DIR, "best_model.pth")
        shutil.copyfile(filepath, best_filepath)
