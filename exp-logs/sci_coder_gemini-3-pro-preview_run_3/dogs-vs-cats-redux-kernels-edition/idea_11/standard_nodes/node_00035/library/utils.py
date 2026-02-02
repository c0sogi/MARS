import os
import random
import shutil
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Seeds all random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def ensure_dir(path: str):
    """
    Ensures that the directory exists. If not, creates it.

    Args:
        path (str): The directory path to check/create.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pth"):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model parameters, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        checkpoint_dir (str): Directory to save the checkpoint.
        filename (str): Filename for the checkpoint.
    """
    ensure_dir(checkpoint_dir)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "model_best.pth")
        shutil.copyfile(filepath, best_path)


def get_device():
    """
    Returns the appropriate torch device.

    Returns:
        torch.device: 'cuda' if available, else 'cpu'.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
