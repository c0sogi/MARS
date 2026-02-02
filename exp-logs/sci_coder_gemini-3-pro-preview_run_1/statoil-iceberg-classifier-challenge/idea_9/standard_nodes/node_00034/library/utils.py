import os
import torch
import numpy as np
from sklearn.metrics import log_loss
from library.config import Config, set_seed


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility using the centralized configuration.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    set_seed(seed)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking metrics like loss and accuracy during training epochs.
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


def calculate_log_loss(y_true, y_pred):
    """
    Calculates the Log Loss metric (Binary Cross Entropy).

    Args:
        y_true (array-like): Ground truth labels (0 or 1).
        y_pred (array-like): Predicted probabilities (between 0 and 1).

    Returns:
        float: The log loss value.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Check for NaNs
    if np.isnan(y_pred).any():
        raise ValueError("Predicted probabilities contain NaNs.")

    # Sklearn log_loss handles clipping internally (default eps=1e-15)
    # We enforce labels=[0, 1] to handle cases where a batch might only have one class
    score = log_loss(y_true, y_pred, labels=[0, 1])
    return score


def sigmoid_numpy(x):
    """
    Applies the sigmoid function element-wise to a numpy array.
    Useful for converting model logits to probabilities.

    Args:
        x (np.ndarray): Input array (logits).

    Returns:
        np.ndarray: Probabilities.
    """
    return 1 / (1 + np.exp(-x))


def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pth"):
    """
    Saves the model checkpoint to the specified directory.

    Args:
        state (dict): The state dictionary to save (model weights, optimizer, etc.).
        is_best (bool): If True, saves an additional copy as 'model_best.pth'.
        checkpoint_dir (str): The directory path to save artifacts.
        filename (str): The name of the checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(checkpoint_dir, "model_best.pth")
        torch.save(state, best_filepath)


def count_parameters(model):
    """
    Counts the number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The model.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
