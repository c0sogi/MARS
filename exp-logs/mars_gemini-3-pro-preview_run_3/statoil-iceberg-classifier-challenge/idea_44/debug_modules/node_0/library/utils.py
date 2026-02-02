import os
import shutil
import random
import numpy as np
import torch
from library import config


def set_seed(seed=config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to set. Defaults to config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        if self.count > 0:
            self.avg = self.sum / self.count


def save_checkpoint(
    state, is_best, fold_idx, checkpoint_dir=config.CHECKPOINT_DIR, filename=None
):
    """
    Saves the model checkpoint to the specified directory.

    Args:
        state (dict): The state dictionary to save (containing model_state_dict, optimizer, epoch, etc.).
        is_best (bool): If True, copies the checkpoint to a 'model_best' file.
        fold_idx (int): The current fold index (used for naming).
        checkpoint_dir (str): The directory to save checkpoints in. Defaults to config.CHECKPOINT_DIR.
        filename (str, optional): Specific filename. If None, defaults to 'checkpoint_fold_{fold_idx}.pth'.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    if filename is None:
        filename = f"checkpoint_fold_{fold_idx}.pth"

    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_filename = f"model_best_fold_{fold_idx}.pth"
        best_filepath = os.path.join(checkpoint_dir, best_filename)
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(model, checkpoint_path, optimizer=None, device=None):
    """
    Loads a model checkpoint.

    Args:
        model (torch.nn.Module): The model to load weights into.
        checkpoint_path (str): Path to the checkpoint file.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        device (torch.device, optional): Device to map the location to.

    Returns:
        dict: The full checkpoint dictionary loaded from the file.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract state_dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Handle DataParallel wrapping if necessary (remove 'module.' prefix if model doesn't have it)
    if list(state_dict.keys())[0].startswith("module.") and not hasattr(
        model, "module"
    ):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint
