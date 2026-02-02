import os
import random
import numpy as np
import torch
import shutil
from sklearn.metrics import log_loss, accuracy_score
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(state, is_best, fold, filename="checkpoint.pth"):
    """
    Saves the training checkpoint.

    Args:
        state (dict): Dictionary containing model state, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        fold (int): The current cross-validation fold.
        filename (str): Base filename for the checkpoint.
    """
    # Ensure checkpoint directory exists
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Define file paths
    checkpoint_name = f"checkpoint_fold_{fold}.pth"
    filepath = os.path.join(Config.CHECKPOINT_DIR, checkpoint_name)

    # Save the state
    torch.save(state, filepath)

    # If it's the best model, create a copy
    if is_best:
        best_name = f"model_best_fold_{fold}.pth"
        best_filepath = os.path.join(Config.CHECKPOINT_DIR, best_name)
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(model, optimizer, fold, load_best=True, device=Config.DEVICE):
    """
    Loads a checkpoint into the model and optimizer.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer): The optimizer to load state into (can be None).
        fold (int): The fold number to load.
        load_best (bool): If True, loads the best model. Otherwise loads the latest checkpoint.
        device (str): Device to load the model onto.

    Returns:
        dict: The loaded checkpoint dictionary, or None if not found.
    """
    filename = (
        f"model_best_fold_{fold}.pth" if load_best else f"checkpoint_fold_{fold}.pth"
    )
    filepath = os.path.join(Config.CHECKPOINT_DIR, filename)

    if not os.path.exists(filepath):
        return None

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint


def calculate_log_loss(y_true, y_pred_prob):
    """
    Calculates the Log Loss metric.

    Args:
        y_true (array-like): Ground truth labels (0 or 1).
        y_pred_prob (array-like): Predicted probabilities for class 1.

    Returns:
        float: The log loss value.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)

    # Log loss requires probabilities for both classes if labels are provided,
    # or just the probability of the positive class if y_true is binary.
    # Sklearn handles 1D probability array for binary classification correctly.
    return log_loss(y_true, y_pred_prob, labels=[0, 1])


def calculate_accuracy(y_true, y_pred_prob, threshold=0.5):
    """
    Calculates the classification accuracy.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred_prob (array-like): Predicted probabilities.
        threshold (float): Threshold to convert probabilities to class labels.

    Returns:
        float: The accuracy score.
    """
    y_pred_bin = (np.array(y_pred_prob) > threshold).astype(int)
    return accuracy_score(y_true, y_pred_bin)


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
