import os
import random
import numpy as np
import torch
from sklearn.metrics import f1_score
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_score(y_true, y_pred, threshold=Config.THRESHOLD):
    """
    Calculates the Mean F1-Score (macro-averaged) for multi-label classification.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels (binary).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities or binary labels.
        threshold (float): Threshold to convert probabilities to binary labels.

    Returns:
        float: The macro-averaged F1 score.
    """
    # Convert tensors to numpy if necessary
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    # Apply threshold if predictions are probabilities (float)
    if np.issubdtype(y_pred.dtype, np.floating):
        y_pred = (y_pred > threshold).astype(int)

    return f1_score(y_true, y_pred, average="macro")


class Logger:
    """
    Logger class to write messages to both standard output and a log file.
    """

    def __init__(self, file_path=Config.LOG_PATH):
        self.file_path = file_path

        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

        # Initialize the log file (overwrite existing)
        with open(self.file_path, "w") as f:
            f.write("")

    def print(self, message):
        """
        Prints a message to stdout and appends it to the log file.
        """
        print(message)
        with open(self.file_path, "a") as f:
            f.write(str(message) + "\n")


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
