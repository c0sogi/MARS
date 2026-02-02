import os
import random
import glob
import numpy as np
import torch
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


def get_device():
    """
    Returns the appropriate torch device (cuda or cpu).

    Returns:
        torch.device: The device object.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_mae(preds, targets, u_out):
    """
    Computes the Mean Absolute Error (MAE) specifically for the inspiratory phase.
    The competition metric ignores the expiratory phase (where u_out == 1).

    Args:
        preds (torch.Tensor): Predicted pressure values.
        targets (torch.Tensor): Ground truth pressure values.
        u_out (torch.Tensor): Control input indicating phase (0=inspiratory, 1=expiratory).

    Returns:
        float: The scalar MAE value for the inspiratory phase.
    """
    # Create a boolean mask for the inspiratory phase (u_out == 0)
    # Ensure u_out is on the same device and comparable
    mask = u_out == 0

    # Apply mask to flatten and select only relevant time steps
    masked_preds = preds[mask]
    masked_targets = targets[mask]

    # Handle edge case where mask might be empty (though unlikely in this dataset)
    if masked_targets.numel() == 0:
        return 0.0

    # Compute L1 loss (MAE)
    mae = torch.abs(masked_preds - masked_targets).mean()

    return mae.item()


def cleanup_cache():
    """
    Removes stale .npy files from the working directory to ensure
    the data pipeline runs from scratch as per the 'Automated Cache Invalidation' requirement.
    """
    cache_dir = Config.CACHE_DIR

    # Ensure directory exists before trying to list files
    if not os.path.exists(cache_dir):
        return

    # Find all .npy files
    pattern = os.path.join(cache_dir, "*.npy")
    files = glob.glob(pattern)

    if files:
        print(
            f"Cleaning up cache: Removing {len(files)} stale .npy files from {cache_dir}..."
        )
        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                print(f"Error removing {f}: {e}")
