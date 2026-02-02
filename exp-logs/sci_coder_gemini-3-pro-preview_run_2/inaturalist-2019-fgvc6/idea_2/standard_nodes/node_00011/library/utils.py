import os
import shutil
import random
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

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Enforce deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    state,
    is_best,
    checkpoint_path=Config.CHECKPOINT_PATH,
    best_model_path=Config.BEST_MODEL_PATH,
):
    """
    Saves the current training state as a checkpoint. If the current state represents
    the best model found so far, it also creates a copy at the best_model_path.

    Args:
        state (dict): Dictionary containing model state_dict, optimizer state_dict, epoch, etc.
        is_best (bool): True if this checkpoint has the best validation metric so far.
        checkpoint_path (str): File path to save the checkpoint.
        best_model_path (str): File path to save the best model copy.
    """
    # Ensure the directory exists
    directory = os.path.dirname(checkpoint_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Save the checkpoint
    torch.save(state, checkpoint_path)

    # If it's the best model, copy it to the best model path
    if is_best:
        shutil.copyfile(checkpoint_path, best_model_path)


class AverageMeter(object):
    """
    Computes and stores the average and current value of a metric.
    Used to track loss and accuracy during training and validation.
    """

    def __init__(self, name="Metric", fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """Resets the meter to initial state."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the meter with a new value.

        Args:
            val (float): The current value to add.
            n (int): The number of samples associated with this value (weight).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """Returns a formatted string representation of the meter."""
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
