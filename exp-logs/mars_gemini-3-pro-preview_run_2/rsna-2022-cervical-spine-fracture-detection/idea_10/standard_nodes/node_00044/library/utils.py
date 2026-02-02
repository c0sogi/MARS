import os
import random
import numpy as np
import torch
import shutil
from sklearn.metrics import log_loss


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def get_weighted_log_loss_score(y_true, y_pred, weights=None, epsilon=1e-15):
    """
    Calculates the weighted multi-label logarithmic loss.

    Args:
        y_true (np.array): Binary ground truth labels of shape (N_samples, N_classes).
        y_pred (np.array): Predicted probabilities of shape (N_samples, N_classes).
        weights (list or np.array, optional): Weights for each class.
                                              Defaults to [1, 1, 1, 1, 1, 1, 1, 7]
                                              (C1-C7 weighted 1, patient_overall weighted 7).
        epsilon (float): Small value to avoid log(0).

    Returns:
        float: The weighted log loss averaged across all individual predictions.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Clip predictions to avoid log(0) or log(1)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Default weights: C1-C7 = 1.0, patient_overall = 7.0
    # Assuming the last column is patient_overall based on standard ordering logic
    if weights is None:
        if y_true.shape[1] == 8:
            weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0])
        else:
            weights = np.ones(y_true.shape[1])
    else:
        weights = np.asarray(weights)

    # Calculate binary log loss for each element:
    # L = -[y * log(p) + (1-y) * log(1-p)]
    loss_matrix = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # Apply weights to each column
    weighted_loss_matrix = loss_matrix * weights

    # The metric is the average loss across all rows (predictions)
    # Note: The prompt says "loss is averaged across all rows".
    # In the submission file, each subtype is a row.
    # So we sum all weighted losses and divide by the total number of elements.
    return np.mean(weighted_loss_matrix)


def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pth"):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model_state_dict, optimizer_state_dict, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        checkpoint_dir (str): Directory to save the checkpoint.
        filename (str): Filename for the checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(checkpoint_dir, "best_model.pth")
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(
    checkpoint_path, model, optimizer=None, scheduler=None, device="cpu"
):
    """
    Loads a model checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler, optional): The scheduler to load state into.
        device (str): Device to map the storage to.

    Returns:
        dict: The full checkpoint dictionary (useful for retrieving epoch/score).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
