import os
import random
import copy
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Note: cudnn.benchmark is set to True in Config for performance,
    # so we do not force deterministic algorithms here to avoid conflict.


def save_checkpoint(model, filepath):
    """
    Saves the model's state dictionary to a file.
    Uses copy.deepcopy to ensure the state is immutable at the time of saving.

    Args:
        model (torch.nn.Module): The model to save.
        filepath (str): Path to save the checkpoint.
    """
    # Ensure directory exists
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Deep copy the state dict as per instructions to ensure immutability
    state_dict = copy.deepcopy(model.state_dict())
    torch.save(state_dict, filepath)


def load_checkpoint(model, filepath, device="cpu"):
    """
    Loads a model's state dictionary from a file.

    Args:
        model (torch.nn.Module): The model to load weights into.
        filepath (str): Path to the checkpoint file.
        device (str): Device to map the location to (default: "cpu").

    Returns:
        model: The model with loaded weights.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    # Load with weights_only=True for security where supported (PyTorch 2.x+)
    try:
        state_dict = torch.load(filepath, map_location=device, weights_only=True)
    except TypeError:
        # Fallback for older pytorch versions if weights_only is not supported
        state_dict = torch.load(filepath, map_location=device)

    model.load_state_dict(state_dict)
    return model


def calculate_roc_auc(y_true, y_scores):
    """
    Calculates the Area Under the ROC Curve.

    Args:
        y_true: Ground truth labels (binary). Can be list, numpy array, or torch tensor.
        y_scores: Predicted probabilities. Can be list, numpy array, or torch tensor.

    Returns:
        float: ROC AUC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    try:
        return roc_auc_score(y_true, y_scores)
    except ValueError:
        # Handle edge case where only one class is present in the batch/set
        # This prevents crashing during small batch validation or sanity checks
        return 0.5


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
