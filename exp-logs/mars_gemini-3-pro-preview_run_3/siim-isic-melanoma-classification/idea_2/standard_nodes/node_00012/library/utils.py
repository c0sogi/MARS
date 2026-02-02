import os
import random
import shutil
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import WORKING_DIR, DEVICE


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted probabilities for the positive class.

    Returns:
        float: The ROC AUC score. Returns 0.5 if only one class is present in y_true.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ROC AUC is undefined if there is only one class in the targets
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_pred)


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


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model, optimizer, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): Name of the checkpoint file.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)
    filepath = os.path.join(WORKING_DIR, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(WORKING_DIR, "best_model.pth")
        shutil.copyfile(filepath, best_path)


def load_checkpoint(model, optimizer=None, scheduler=None, filename="best_model.pth"):
    """
    Loads a model checkpoint.

    Args:
        model: The model to load weights into.
        optimizer: (Optional) The optimizer to load state into.
        scheduler: (Optional) The scheduler to load state into.
        filename (str): The filename of the checkpoint to load.

    Returns:
        float: The best score recorded in the checkpoint, or 0.0 if not found.
    """
    filepath = os.path.join(WORKING_DIR, filename)
    if not os.path.exists(filepath):
        # Return 0.0 or raise error depending on preference;
        # here we return 0.0 to indicate no previous best score.
        return 0.0

    checkpoint = torch.load(filepath, map_location=DEVICE)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint.get("best_score", 0.0)
