import os
import random
import shutil
import numpy as np
import torch
from library import config


def set_seed(seed=config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    """
    Saves the model checkpoint to the configured checkpoint directory.

    Args:
        state (dict): The state dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): The name of the file to save.
    """
    # Ensure the directory exists
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    filepath = os.path.join(config.CHECKPOINT_DIR, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(config.CHECKPOINT_DIR, "model_best.pth")
        shutil.copyfile(filepath, best_path)


def load_checkpoint(filename="model_best.pth"):
    """
    Loads a model checkpoint from the configured checkpoint directory.

    Args:
        filename (str): The name of the file to load. Defaults to 'model_best.pth'.

    Returns:
        dict: The loaded state dictionary.
    """
    filepath = os.path.join(config.CHECKPOINT_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at: {filepath}")

    # Load to the configured device
    checkpoint = torch.load(filepath, map_location=config.DEVICE)
    return checkpoint
