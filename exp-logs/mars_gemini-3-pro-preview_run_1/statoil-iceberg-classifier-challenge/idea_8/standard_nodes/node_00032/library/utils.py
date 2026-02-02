import os
import random
import numpy as np
import torch
import sys

# Ensure we can import from the library directory
sys.path.append(os.getcwd())

try:
    from library.config import Config
except ImportError:
    # Fallback/Mock config if running in an environment where library is not set up yet
    class Config:
        SEED = 42


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
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
        """Resets all internal statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the meter with a new value.

        Args:
            val (float): The value to add.
            n (int): The weight/count of the value (default: 1).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def sigmoid(x):
    """
    Applies the sigmoid function to the input array or scalar.

    Args:
        x (np.ndarray or float): Input logits.

    Returns:
        np.ndarray or float: Probabilities in range [0, 1].
    """
    # Clip x to prevent overflow/underflow in exp
    # Although standard sigmoid usually handles it, for float32 stability clipping is safe
    # However, standard implementation:
    return 1 / (1 + np.exp(-x))
