import torch
import numpy as np
from sklearn.metrics import f1_score
from library.config import Config, seed_everything


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility by wrapping the library function.

    Args:
        seed (int): The seed value to use.
    """
    seed_everything(seed)


def get_device():
    """
    Returns the PyTorch device configured in Config.

    Returns:
        torch.device: The device (cpu or cuda).
    """
    return torch.device(Config.DEVICE)


def calculate_macro_f1(y_true, y_pred):
    """
    Calculates the Macro F1 score.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.

    Returns:
        float: The Macro F1 score.
    """
    # Ensure inputs are numpy arrays (handle tensors if passed)
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return f1_score(y_true, y_pred, average="macro")


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
