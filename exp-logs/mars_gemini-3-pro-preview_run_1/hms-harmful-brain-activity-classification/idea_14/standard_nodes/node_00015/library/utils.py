import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def kl_divergence(y_pred, y_true):
    """
    Calculates the Kullback-Leibler (KL) Divergence between predicted and observed target probabilities.

    This implementation handles both PyTorch tensors and NumPy arrays. It ensures numerical
    stability by clipping predictions and uses PyTorch's optimized functional implementation.

    Args:
        y_pred (torch.Tensor or np.ndarray): Predicted probabilities (softmax output). Shape (N, C).
        y_true (torch.Tensor or np.ndarray): Target probabilities (ground truth). Shape (N, C).

    Returns:
        float: The scalar KL divergence score (averaged over the batch).
    """
    # Convert NumPy arrays to PyTorch tensors if necessary
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float32)

    # Ensure targets are on the same device as predictions
    if y_true.device != y_pred.device:
        y_true = y_true.to(y_pred.device)

    # Clip predictions to avoid log(0) which results in NaN/Inf
    # Using a small epsilon value
    epsilon = 1e-15
    y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)

    # Calculate KL Divergence
    # F.kl_div expects input to be log-probabilities (log(P)) and target to be probabilities (Q)
    # reduction='batchmean' ensures the output is the mean KL divergence over the batch
    loss = F.kl_div(torch.log(y_pred), y_true, reduction="batchmean")

    return loss.item()


def save_checkpoint(model, optimizer, scheduler, epoch, score, filename):
    """
    Saves the model state, optimizer state, scheduler state, and current metrics to a file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        scheduler (torch.optim.lr_scheduler._LRScheduler or None): The scheduler state.
        epoch (int): The current epoch number.
        score (float): The validation score at this checkpoint.
        filename (str): The full path where the checkpoint will be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "score": score,
    }

    torch.save(state, filename)
