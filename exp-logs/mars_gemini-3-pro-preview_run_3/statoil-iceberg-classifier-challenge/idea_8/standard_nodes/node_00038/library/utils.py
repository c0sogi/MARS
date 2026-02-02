import os
import random
import shutil
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
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


def save_checkpoint(state, is_best, fold, filename="checkpoint.pth"):
    """
    Saves the training checkpoint.

    Args:
        state (dict): The state dictionary containing model parameters, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        fold (int): The current cross-validation fold index.
        filename (str): The name of the checkpoint file.
    """
    # Construct the directory path for the specific fold
    fold_dir = os.path.join(Config.WORKING_DIR, f"fold_{fold}")

    # Ensure the directory exists (redundant if Config.setup() is called, but safe)
    os.makedirs(fold_dir, exist_ok=True)

    filepath = os.path.join(fold_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(fold_dir, "model_best.pth")
        shutil.copyfile(filepath, best_filepath)


class AverageMeter(object):
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
