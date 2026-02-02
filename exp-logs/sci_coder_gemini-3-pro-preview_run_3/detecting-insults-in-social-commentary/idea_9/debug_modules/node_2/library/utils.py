import os
import random
import numpy as np
import torch
import codecs
import pandas as pd


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

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


def decode_text(text):
    """
    Decodes unicode-escaped text from the dataset.

    The dataset contains text with escaped characters (e.g., \\xe2\\x80\\x99).
    This function attempts to decode them into proper unicode characters.

    Args:
        text (str or object): The text to decode.

    Returns:
        str: The decoded text, or an empty string if input is NaN.
    """
    if pd.isna(text):
        return ""
    try:
        # Convert to string and decode unicode escapes
        return codecs.decode(str(text), "unicode_escape")
    except Exception:
        # Return original string if decoding fails
        return str(text)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking metrics like loss and accuracy during training.
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
        Updates the moving average.

        Args:
            val (float): The current value to update.
            n (int): The weight/count of the current value (usually batch size).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
