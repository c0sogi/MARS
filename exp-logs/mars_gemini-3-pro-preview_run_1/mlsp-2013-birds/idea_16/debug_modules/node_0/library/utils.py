import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking loss and metrics during training loops.
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


def seed_everything(seed=Config.SEED):
    """
    Sets the seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_multilabel_auc(y_true, y_pred):
    """
    Computes the macro-averaged ROC AUC for multi-label classification.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels (N, NumClasses).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities (N, NumClasses).

    Returns:
        float: Macro-averaged ROC AUC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Check for shape mismatch
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch in AUC computation: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    try:
        # sklearn's roc_auc_score handles multi-label with average='macro'
        # It requires that each class has at least one positive and one negative sample in y_true
        score = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        # Fallback for cases where some classes might be constant in the current batch/set
        aucs = []
        num_classes = y_true.shape[1]
        for i in range(num_classes):
            # Check if class i has both 0 and 1 in the ground truth
            if len(np.unique(y_true[:, i])) > 1:
                try:
                    class_auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                    aucs.append(class_auc)
                except ValueError:
                    # Skip if calculation fails for this specific class
                    pass

        if len(aucs) == 0:
            # If no classes are valid (e.g. extremely small batch with constant labels), return 0.5
            score = 0.5
        else:
            score = np.mean(aucs)

    return score
