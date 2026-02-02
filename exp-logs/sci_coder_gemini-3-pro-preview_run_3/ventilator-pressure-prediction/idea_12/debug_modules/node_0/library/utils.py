import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
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


def get_device():
    """
    Returns the torch device based on configuration.
    """
    return torch.device(Config.DEVICE)


def compute_metric(preds, targets, u_out):
    """
    Computes the Mean Absolute Error (MAE) specifically for the inspiratory phase.
    The competition metric ignores the expiratory phase (where u_out == 1).

    Args:
        preds: Predicted pressure values (Tensor or ndarray).
        targets: Actual pressure values (Tensor or ndarray).
        u_out: Control input indicating expiratory phase (1) or inspiratory (0).

    Returns:
        float: The MAE for the inspiratory phase.
    """
    # Convert to torch tensors if inputs are numpy arrays
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    if isinstance(u_out, np.ndarray):
        u_out = torch.from_numpy(u_out)

    # Ensure inputs are on CPU for metric calculation to avoid unnecessary GPU syncs if just logging
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()
    u_out = u_out.detach().cpu()

    # Create boolean mask: True where u_out is 0 (inspiratory phase)
    mask = u_out == 0

    # Calculate Absolute Error
    abs_error = torch.abs(preds - targets)

    # Filter errors using the mask
    masked_error = abs_error[mask]

    # Return mean
    if masked_error.numel() == 0:
        return 0.0

    return masked_error.mean().item()
