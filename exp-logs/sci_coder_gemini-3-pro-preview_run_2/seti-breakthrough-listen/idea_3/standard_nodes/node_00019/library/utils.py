import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed: int = Config.seed) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking loss and other metrics during training.
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


def get_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Area Under the ROC Curve (ROC AUC).

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted probabilities.

    Returns:
        float: The ROC AUC score.
    """
    return roc_auc_score(y_true, y_pred)


def save_checkpoint(state: dict, filepath: str) -> None:
    """
    Saves the model checkpoint to the specified filepath.

    Args:
        state: A dictionary containing model state, optimizer state, etc.
        filepath: Path to save the checkpoint file.
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filepath: str, device: torch.device) -> dict:
    """
    Loads a model checkpoint from the specified filepath.

    Args:
        filepath: Path to the checkpoint file.
        device: The device to map the location to (cpu or cuda).

    Returns:
        dict: The loaded checkpoint state dictionary.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at {filepath}")

    return torch.load(filepath, map_location=device)
