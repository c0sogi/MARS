import os
import random
import numpy as np
import torch
from library.config import SCALER_MEAN_PATH, SCALER_SCALE_PATH, MODEL_SAVE_PATH


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
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


# -------------------------------------------------------------------
# Target Scaler Utilities
# -------------------------------------------------------------------
def fit_scaler(targets):
    """
    Computes mean and std of the targets.
    """
    targets = np.array(targets)
    mean = np.mean(targets)
    std = np.std(targets)
    return mean, std


def save_scaler(mean, std, mean_path=SCALER_MEAN_PATH, std_path=SCALER_SCALE_PATH):
    """
    Saves the scaler statistics to .npy files.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(mean_path), exist_ok=True)
    np.save(mean_path, mean)
    np.save(std_path, std)


def load_scaler(mean_path=SCALER_MEAN_PATH, std_path=SCALER_SCALE_PATH):
    """
    Loads the scaler statistics from .npy files.
    Returns (mean, std) as floats.
    """
    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        raise FileNotFoundError(f"Scaler files not found at {mean_path} or {std_path}")

    mean = np.load(mean_path)
    std = np.load(std_path)
    return float(mean), float(std)


def scale_target(targets, mean, std):
    """
    Applies Standard Scaling: (x - mean) / std
    """
    # Add epsilon to std to prevent division by zero, though unlikely here
    return (targets - mean) / (std + 1e-8)


def unscale_target(targets, mean, std):
    """
    Reverses Standard Scaling: x * std + mean
    """
    return targets * std + mean


# -------------------------------------------------------------------
# Model Checkpointing
# -------------------------------------------------------------------
def save_model(model, path=MODEL_SAVE_PATH):
    """
    Saves the model state dictionary to the specified path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path=MODEL_SAVE_PATH, device="cpu"):
    """
    Loads the model state dictionary from the specified path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model
