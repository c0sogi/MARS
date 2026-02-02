import os
import random
import shutil
import numpy as np
import torch


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

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_checkpoint(state, is_best, checkpoint_dir, fold):
    """
    Saves the training checkpoint.

    Args:
        state (dict): The state dictionary to save (model, optimizer, epoch, etc.).
        is_best (bool): True if this is the best model so far.
        checkpoint_dir (str): Directory to save the checkpoint.
        fold (int): The current fold number.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save the current checkpoint
    filename = os.path.join(checkpoint_dir, f"checkpoint_fold_{fold}.pth")
    torch.save(state, filename)

    # If it's the best model, make a copy
    if is_best:
        best_filename = os.path.join(checkpoint_dir, f"model_best_fold_{fold}.pth")
        shutil.copyfile(filename, best_filename)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Loads a checkpoint into the model and optionally the optimizer.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.

    Returns:
        epoch (int): The epoch to resume from (default 0).
        best_score (float): The best score recorded (default None).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load to CPU first to avoid device mismatch issues
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle both full checkpoint dicts and raw state dicts
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    epoch = checkpoint.get("epoch", 0)
    best_score = checkpoint.get("best_score", None)

    return epoch, best_score


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
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
