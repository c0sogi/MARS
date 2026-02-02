import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
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
    Calculates the average ROC AUC score across all target columns.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels (N, C).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities (N, C).

    Returns:
        float: The average ROC AUC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Calculate AUC for each column
    aucs = []
    num_cols = y_true.shape[1]

    for i in range(num_cols):
        try:
            # Check if the column has more than one unique class to avoid errors
            if len(np.unique(y_true[:, i])) > 1:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                aucs.append(auc)
            else:
                # If only one class is present in y_true, AUC is undefined.
                # In a full validation set, this shouldn't happen for these labels,
                # but we handle it gracefully by skipping or treating as neutral (0.5).
                # Here we skip to strictly follow the "average of individual AUCs" logic for valid columns.
                pass
        except ValueError:
            pass

    if len(aucs) == 0:
        return 0.5

    return np.mean(aucs)
