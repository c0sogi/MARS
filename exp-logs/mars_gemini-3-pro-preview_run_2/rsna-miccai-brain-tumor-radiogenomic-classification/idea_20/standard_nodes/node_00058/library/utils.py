import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Delegates to the centralized Config configuration.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    Config.setup_reproducibility(seed)


def get_device() -> torch.device:
    """
    Returns the PyTorch device to be used for computation.

    Returns:
        torch.device: The device object (cuda or cpu).
    """
    return torch.device(Config.DEVICE)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training epochs.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve.

    Args:
        y_true (array-like): True binary labels.
        y_pred (array-like): Target scores (probability estimates).

    Returns:
        float: The ROC AUC score.
    """
    # Handle edge case where only one class is present in the batch
    if len(np.unique(y_true)) < 2:
        return 0.5

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return roc_auc_score(y_true, y_pred)


def save_checkpoint(state, is_best, filepath=Config.MODEL_CHECKPOINT_PATH):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model_state_dict, optimizer_state_dict, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filepath (str): Path to save the checkpoint.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save the checkpoint
    if is_best:
        torch.save(state, filepath)
