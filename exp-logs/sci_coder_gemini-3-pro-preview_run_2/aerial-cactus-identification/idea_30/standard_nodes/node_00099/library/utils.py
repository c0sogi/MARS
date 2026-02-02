import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int):
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
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss during training.
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


class RocAucMeter:
    """
    Accumulates predictions and targets to calculate the Area Under the ROC Curve.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = []
        self.y_pred = []

    def update(self, y_true, y_pred):
        """
        Update the meter with new predictions and targets.

        Args:
            y_true: Ground truth labels (numpy array or torch tensor).
            y_pred: Predicted probabilities (numpy array or torch tensor).
        """
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        self.y_true.extend(y_true.flatten())
        self.y_pred.extend(y_pred.flatten())

    def score(self):
        """
        Calculates the ROC AUC score based on accumulated data.

        Returns:
            float: The ROC AUC score. Returns 0.5 if only one class is present.
        """
        if not self.y_true:
            return 0.0

        # roc_auc_score requires at least two classes to be present in y_true
        if len(np.unique(self.y_true)) < 2:
            return 0.5

        return roc_auc_score(self.y_true, self.y_pred)
