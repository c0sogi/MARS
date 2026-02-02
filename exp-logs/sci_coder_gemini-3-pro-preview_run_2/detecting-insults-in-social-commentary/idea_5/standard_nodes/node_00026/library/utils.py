import os
import random
import numpy as np
import torch
import pandas as pd
import ast
from sklearn.metrics import roc_auc_score


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
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


def get_score(y_true, y_pred):
    """
    Calculates the Area Under the Receiver Operating Curve (AUC).

    Args:
        y_true (array-like): Ground truth labels (0 or 1).
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: The AUC score.
    """
    # Ensure inputs are numpy arrays for stability
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Handle case with single class in batch to avoid errors
    # (though typically not an issue on full validation sets)
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_pred)


def clean_text(text):
    """
    Cleans the text column.
    Handles unicode-escaped text surrounded by double-quotes as described in the dataset.

    Args:
        text (str or object): The raw text from the dataframe.

    Returns:
        str: The cleaned text.
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Attempt to use literal_eval to handle python-style string escaping and quotes.
    # The dataset format is described as unicode-escaped text surrounded by double-quotes.
    # e.g. "User said: \"Hello\"" -> User said: "Hello"
    try:
        if text.startswith('"') and text.endswith('"'):
            cleaned = ast.literal_eval(text)
            return cleaned
    except (ValueError, SyntaxError):
        pass

    # Fallback cleanup if literal_eval fails
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    # Basic unicode unescape if needed
    try:
        text = text.encode("utf-8").decode("unicode_escape")
    except (ValueError, UnicodeError):
        pass

    return text
