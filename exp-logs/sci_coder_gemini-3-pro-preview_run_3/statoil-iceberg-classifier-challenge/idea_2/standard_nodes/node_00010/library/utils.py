import os
import random
import shutil
import numpy as np
import torch
from library.config import SEED


def seed_everything(seed=SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN
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
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, checkpoint_dir):
    """
    Saves the model checkpoint.

    Args:
        state (dict): The model state dictionary (and optimizer state, etc.).
        is_best (bool): Whether this checkpoint represents the best model so far.
        checkpoint_dir (str): Directory to save the checkpoint files.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, "checkpoint.pth")
    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(checkpoint_dir, "model_best.pth")
        shutil.copyfile(filepath, best_filepath)
