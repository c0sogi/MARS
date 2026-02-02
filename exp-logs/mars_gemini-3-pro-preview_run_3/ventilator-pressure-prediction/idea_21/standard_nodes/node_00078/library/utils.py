import os
import random
import time
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

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


def compute_mae(preds, targets, u_out):
    """
    Computes Mean Absolute Error (MAE) for the inspiratory phase (u_out == 0).

    Args:
        preds (torch.Tensor or np.ndarray): Predicted pressures.
        targets (torch.Tensor or np.ndarray): Ground truth pressures.
        u_out (torch.Tensor or np.ndarray): Control input u_out (0 for inspiration, 1 for expiration).

    Returns:
        float: MAE for the inspiratory phase.
    """
    # Convert inputs to torch tensors if they are numpy arrays
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    if isinstance(u_out, np.ndarray):
        u_out = torch.from_numpy(u_out)

    # Ensure all tensors are on the same device (move to CPU for metric calculation to save GPU mem)
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()
    u_out = u_out.detach().cpu()

    # Create mask for inspiratory phase (u_out == 0)
    # Using boolean indexing
    mask = u_out == 0

    # Filter predictions and targets using the mask
    preds_insp = preds[mask]
    targets_insp = targets[mask]

    # Handle edge case where there are no inspiratory steps (unlikely but safe)
    if len(targets_insp) == 0:
        return 0.0

    # Compute L1 loss (MAE)
    mae = torch.abs(preds_insp - targets_insp).mean().item()

    return mae


class Timer:
    """
    Simple timer context manager to measure execution time of blocks.
    """

    def __init__(self, message="Task"):
        self.message = message

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        duration = self.end - self.start
        print(f"{self.message} finished in {duration:.2f} seconds.")


def log_metrics(epoch, train_loss, val_loss, val_mae, elapsed_time):
    """
    Prints metrics with full precision as requested.
    """
    print(
        f"Epoch {epoch} | Time: {elapsed_time:.2f}s | "
        f"Train Loss: {train_loss} | Val Loss: {val_loss} | Val MAE: {val_mae}"
    )
