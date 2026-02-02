import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from library import config


def set_seed(seed=config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_roc_auc(y_true, y_scores):
    """
    Calculates the Area Under the ROC Curve.

    Args:
        y_true (array-like): Ground truth labels (0 or 1).
        y_scores (array-like): Predicted probabilities for class 1.

    Returns:
        float: ROC AUC score. Returns 0.5 if only one class is present in y_true.
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Check if both classes are present
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_scores)


def save_checkpoint(model, optimizer, epoch, best_score, filepath):
    """
    Saves the model checkpoint containing model state, optimizer state, epoch, and score.

    Args:
        model: PyTorch model.
        optimizer: PyTorch optimizer.
        epoch (int): Current epoch.
        best_score (float): Best validation score achieved so far.
        filepath (str): Path to save the checkpoint file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_score": best_score,
    }
    torch.save(state, filepath)


def load_checkpoint(model, filepath, optimizer=None, device=config.DEVICE):
    """
    Loads a model checkpoint.

    Args:
        model: PyTorch model to load weights into.
        filepath (str): Path to the checkpoint file.
        optimizer: (Optional) Optimizer to load state into.
        device (str): Device to map the location to (e.g., 'cpu', 'cuda').

    Returns:
        tuple: (start_epoch, best_score)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at {filepath}")

    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    start_epoch = checkpoint.get("epoch", 0)
    best_score = checkpoint.get("best_score", 0.0)

    return start_epoch, best_score


def save_submission(clips, probabilities, filepath=config.SUBMISSION_PATH):
    """
    Saves the submission file in the required format.

    Args:
        clips (list): List of clip filenames.
        probabilities (list): List of predicted probabilities.
        filepath (str): Path to save the CSV file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    df = pd.DataFrame({"clip": clips, "probability": probabilities})

    # Ensure correct column order
    df = df[["clip", "probability"]]

    # Save to CSV
    df.to_csv(filepath, index=False)


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
