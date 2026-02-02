import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.

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


def compute_auc(y_true, y_pred):
    """
    Computes the Mean Column-wise ROC AUC score.

    Args:
        y_true: Ground truth labels (N, num_classes). Can be numpy array or torch Tensor.
        y_pred: Predicted probabilities (N, num_classes). Can be numpy array or torch Tensor.

    Returns:
        float: The mean ROC AUC score across all columns.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Check shapes
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    num_classes = y_true.shape[1]
    aucs = []

    for i in range(num_classes):
        # Check if the column has only one class (all 0s or all 1s)
        # roc_auc_score requires both classes to be present (binary classification)
        unique_classes = np.unique(y_true[:, i])
        if len(unique_classes) < 2:
            # If only one class is present in the ground truth for this batch/set,
            # AUC is undefined. We return 0.5 (random guess performance) to avoid
            # crashing, assuming this is a rare edge case in batched validation.
            aucs.append(0.5)
        else:
            score = roc_auc_score(y_true[:, i], y_pred[:, i])
            aucs.append(score)

    return np.mean(aucs)


def save_checkpoint(state, filepath):
    """
    Saves the model state to a checkpoint file.

    Args:
        state (dict): The state dictionary to save (model, optimizer, epoch, etc.)
        filepath (str): The path where the checkpoint will be saved.
    """
    # Ensure the directory exists
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(state, filepath)
