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

    # Set environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    """
    Selects the available device (GPU or CPU).

    Returns:
        torch.device: The device to be used for computation.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking metrics like loss and accuracy during training.
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


def save_checkpoint(state, is_best, checkpoint_dir, fold_idx):
    """
    Saves the model checkpoint to the specified directory.

    Args:
        state (dict): The model state dictionary (and optimizer state, etc.).
        is_best (bool): Whether this checkpoint represents the best model so far.
        checkpoint_dir (str): Directory to save the checkpoint.
        fold_idx (int): The current fold index.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = os.path.join(checkpoint_dir, f"checkpoint_fold_{fold_idx}.pth")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(checkpoint_dir, f"model_best_fold_{fold_idx}.pth")
        shutil.copyfile(filename, best_filename)
