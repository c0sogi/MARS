import os
import random
import copy
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking losses and metrics during training epochs.
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


def seed_everything(seed=42):
    """
    Sets random seeds for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # Note: torch.backends.cudnn.benchmark is typically handled in the main config
    # to balance reproducibility with hardware optimization.


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

    Args:
        y_true (array-like): Ground truth binary labels (0 or 1).
        y_pred (array-like): Predicted probabilities for the positive class.

    Returns:
        float: The ROC AUC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        # This can happen if y_true contains only one class (e.g. in a small batch)
        return 0.5


def save_checkpoint(model, path):
    """
    Saves the model's state dictionary to a file.
    Uses copy.deepcopy to ensure the saved state is isolated from the live model.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        path (str): The destination file path.
    """
    # Ensure the destination directory exists
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Deep copy the state dict to prevent reference issues if the model continues training
    best_model_wts = copy.deepcopy(model.state_dict())
    torch.save(best_model_wts, path)
