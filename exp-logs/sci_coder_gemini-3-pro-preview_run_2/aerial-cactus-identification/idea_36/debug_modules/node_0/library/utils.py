import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed=42):
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

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve.

    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted probabilities for the positive class.

    Returns:
        float: The ROC AUC score. Returns 0.5 if only one class is present in y_true.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Check if there are at least two classes to calculate ROC AUC
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_pred)


def save_checkpoint(state, filepath):
    """
    Saves the model checkpoint to the specified file path.

    Args:
        state (dict): The state dictionary to save (e.g., {'state_dict': ..., 'epoch': ...}).
        filepath (str): The destination path for the checkpoint file.
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(state, filepath)


def load_checkpoint(model, filepath, device, optimizer=None):
    """
    Loads a model checkpoint from the specified file path.

    Args:
        model (torch.nn.Module): The model to load weights into.
        filepath (str): The path to the checkpoint file.
        device (str or torch.device): The device to map the checkpoint to.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.

    Returns:
        dict: The loaded checkpoint dictionary (useful for retrieving epoch or best score).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    # Support both direct state_dict saving and wrapped dict saving
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint


def save_submission(ids, predictions, filepath=Config.SUBMISSION_PATH):
    """
    Saves the predictions to a CSV file in the format required for submission.

    Args:
        ids (list or array-like): List of image IDs (filenames).
        predictions (list or array-like): List of predicted probabilities.
        filepath (str): Path to save the submission CSV.
    """
    # Ensure the directory exists
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    df = pd.DataFrame({"id": ids, "has_cactus": predictions})

    df.to_csv(filepath, index=False)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking metrics like loss and accuracy during training loops.
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
