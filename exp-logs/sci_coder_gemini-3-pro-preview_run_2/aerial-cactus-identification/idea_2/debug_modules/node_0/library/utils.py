import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import seed_everything


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility using the library function.

    Args:
        seed (int): The seed value to use.
    """
    seed_everything(seed)


def calculate_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve.

    Args:
        y_true (list or np.array or torch.Tensor): Ground truth binary labels.
        y_pred (list or np.array or torch.Tensor): Predicted probabilities for the positive class.

    Returns:
        float: The ROC AUC score.
    """
    # Detach and move to cpu if they are tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure they are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        # Handle cases where only one class is present in y_true
        return 0.5


def save_checkpoint(state, filepath):
    """
    Saves the model checkpoint to the specified file.

    Args:
        state (dict): State dictionary containing model_state_dict, optimizer_state_dict, etc.
        filepath (str): Path to save the checkpoint file.
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer=None, device=None):
    """
    Loads a model checkpoint.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (torch.device, optional): Device to map the location to.

    Returns:
        dict: The loaded checkpoint dictionary (useful for retrieving epoch, best_score, etc.).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at {filepath}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and accuracy during training.
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
