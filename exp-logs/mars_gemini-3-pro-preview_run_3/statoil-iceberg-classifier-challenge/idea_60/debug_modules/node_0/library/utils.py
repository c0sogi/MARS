import os
import random
import shutil
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, is_best, fold):
    """
    Saves the model checkpoint to the configured checkpoint directory.

    Args:
        state (dict): The state dictionary containing model parameters, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        fold (int): The current cross-validation fold index.
    """
    # Ensure the checkpoint directory exists
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    filename = f"checkpoint_fold_{fold}.pth"
    filepath = os.path.join(Config.CHECKPOINT_DIR, filename)

    torch.save(state, filepath)

    if is_best:
        best_filename = f"model_best_fold_{fold}.pth"
        best_filepath = os.path.join(Config.CHECKPOINT_DIR, best_filename)
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(model, checkpoint_path, optimizer=None):
    """
    Loads a checkpoint into the model and optionally the optimizer.

    Args:
        model (torch.nn.Module): The model to load weights into.
        checkpoint_path (str): Path to the checkpoint file.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.

    Returns:
        dict: The full checkpoint dictionary (useful for retrieving epoch, best_score, etc.).
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)

    # Load model weights
    model.load_state_dict(checkpoint["state_dict"])

    # Load optimizer state if provided
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint


class AverageMeter:
    """
    Computes and stores the average and current value of a metric.
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
