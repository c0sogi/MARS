import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training loops.
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
    Calculates the Mean Column-wise ROC AUC score.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth binary labels. Shape: (N, num_classes)
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities. Shape: (N, num_classes)

    Returns:
        float: The mean column-wise ROC AUC.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Calculate ROC AUC for each column and average them (macro)
    # This matches the "Mean column-wise ROC AUC" metric
    try:
        return roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        # Fallback in case a specific class is not present in the batch
        # This iterates over columns and averages the valid AUCs
        aucs = []
        for i in range(y_true.shape[1]):
            try:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                aucs.append(score)
            except ValueError:
                pass

        if len(aucs) == 0:
            return 0.5
        return np.mean(aucs)
