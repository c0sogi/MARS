import os
import random
import numpy as np
import torch
from sklearn.metrics import f1_score


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Ensure deterministic behavior for cuDNN
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


def calculate_micro_f1(preds, targets, threshold=0.5, from_logits=True):
    """
    Calculates the Micro-averaged F1 score.

    Args:
        preds (torch.Tensor or np.ndarray): Model predictions.
        targets (torch.Tensor or np.ndarray): Ground truth labels.
        threshold (float): Threshold for converting probabilities to binary labels.
        from_logits (bool): If True, applies sigmoid to preds before thresholding.

    Returns:
        float: The micro F1 score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().float()
        if from_logits:
            preds = torch.sigmoid(preds)
        preds = preds.numpy()
    elif from_logits:
        # If numpy and from_logits, apply sigmoid manually
        preds = 1 / (1 + np.exp(-preds))

    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Binarize predictions
    binary_preds = (preds > threshold).astype(int)

    # Ensure targets are integers
    targets = targets.astype(int)

    # Calculate Micro F1
    return f1_score(targets, binary_preds, average="micro")
