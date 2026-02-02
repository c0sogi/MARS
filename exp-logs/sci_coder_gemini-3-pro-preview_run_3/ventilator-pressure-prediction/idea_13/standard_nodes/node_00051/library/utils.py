import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Ensures deterministic behavior for CUDNN backends.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Returns the appropriate PyTorch device based on availability.

    Returns:
        torch.device: The device to use (cuda or cpu).
    """
    return torch.device(Config.DEVICE)


def compute_metric(
    preds: torch.Tensor, targets: torch.Tensor, u_out: torch.Tensor
) -> float:
    """
    Computes the Mean Absolute Error (MAE) specifically for the inspiratory phase.
    The metric is calculated only where u_out == 0.

    Args:
        preds (torch.Tensor): Predicted pressures.
        targets (torch.Tensor): Ground truth pressures.
        u_out (torch.Tensor): Control input indicating expiratory phase (1) or inspiratory (0).

    Returns:
        float: The MAE for the inspiratory phase.
    """
    # Ensure inputs are tensors
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)
    if not isinstance(u_out, torch.Tensor):
        u_out = torch.tensor(u_out)

    # Mask: u_out == 0 (Inspiratory phase)
    # We ensure the mask is boolean
    mask = u_out == 0

    # Filter predictions and targets based on the mask
    preds_insp = preds[mask]
    targets_insp = targets[mask]

    # Handle edge case where batch might be empty after masking (unlikely in this dataset)
    if preds_insp.numel() == 0:
        return 0.0

    # Calculate Mean Absolute Error
    mae = torch.abs(preds_insp - targets_insp).mean()

    return mae.item()


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training loops.
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
