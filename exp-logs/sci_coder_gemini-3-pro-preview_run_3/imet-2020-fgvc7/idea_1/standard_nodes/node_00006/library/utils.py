import os
import random
import numpy as np
import torch
from sklearn.metrics import f1_score
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

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
        # Deterministic mode ensures reproducibility but may impact performance slightly
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_micro_f1(probs, targets, threshold: float = Config.DEFAULT_THRESHOLD):
    """
    Calculates the Micro-Averaged F1 score given probabilities and binary targets.

    Args:
        probs (torch.Tensor or np.ndarray): Predicted probabilities (N, C).
        targets (torch.Tensor or np.ndarray): Ground truth binary labels (N, C).
        threshold (float): Threshold to binarize probabilities. Defaults to Config.DEFAULT_THRESHOLD.

    Returns:
        float: The micro-averaged F1 score.
    """
    # Convert PyTorch tensors to NumPy arrays if needed
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()

    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Binarize predictions based on the threshold
    preds = (probs >= threshold).astype(int)

    # Ensure targets are integers
    targets = targets.astype(int)

    # Calculate Micro F1 score
    # average='micro': Calculate metrics globally by counting the total true positives,
    # false negatives and false positives.
    score = f1_score(targets, preds, average="micro")

    return float(score)
