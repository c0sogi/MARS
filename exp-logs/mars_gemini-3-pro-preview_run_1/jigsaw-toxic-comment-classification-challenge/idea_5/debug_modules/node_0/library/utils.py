import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking losses and metrics during training.
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


def seed_everything(seed=Config.seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.seed.
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


def get_score(y_true, y_pred):
    """
    Calculates the Mean Column-wise ROC AUC.

    Args:
        y_true (np.ndarray): Ground truth binary labels, shape (N, num_classes).
        y_pred (np.ndarray): Predicted probabilities, shape (N, num_classes).

    Returns:
        float: The mean ROC AUC score across all columns.
    """
    scores = []
    num_classes = y_true.shape[1]

    for i in range(num_classes):
        # Extract columns for the current class
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        # ROC AUC requires both positive and negative classes to be present
        if len(np.unique(y_t)) > 1:
            score = roc_auc_score(y_t, y_p)
            scores.append(score)
        else:
            # If a specific batch or split lacks one class, we cannot compute AUC for it.
            # In a proper validation set, this should not happen due to stratification.
            pass

    if not scores:
        return 0.0

    return np.mean(scores)
