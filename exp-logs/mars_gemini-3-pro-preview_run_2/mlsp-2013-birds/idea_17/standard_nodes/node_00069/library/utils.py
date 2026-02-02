import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training epochs.
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
        Updates the meter with a new value.

        Args:
            val (float): The current value to record.
            n (int): The weight or number of items associated with this value.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_score(y_true, y_pred):
    """
    Computes the Macro-Averaged Area Under the ROC Curve (AUC) for multi-label data.

    This implementation is robust to cases where a specific class might only have
    one label type (all 0s or all 1s) in the provided batch or split, which
    would typically cause sklearn's roc_auc_score to raise a ValueError.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels of shape (N_samples, N_classes).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities of shape (N_samples, N_classes).

    Returns:
        float: The macro-averaged AUC score. Returns 0.0 if no valid classes are found.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    auc_scores = []
    num_classes = y_true.shape[1]

    for i in range(num_classes):
        # We can only compute ROC AUC if there are both positive and negative samples
        # for this specific class in the ground truth.
        if len(np.unique(y_true[:, i])) == 2:
            try:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                auc_scores.append(score)
            except ValueError:
                # In case of any other underlying metric error, skip this class
                continue

    if not auc_scores:
        return 0.0

    return np.mean(auc_scores)
