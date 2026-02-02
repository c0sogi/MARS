import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import seed_everything


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and accuracy during training.
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


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve.

    Args:
        y_true: Ground truth labels (numpy array or torch tensor).
        y_pred: Predicted probabilities (numpy array or torch tensor).

    Returns:
        float: AUC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are 1D arrays
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    return roc_auc_score(y_true, y_pred)


def save_checkpoint(model, path):
    """
    Saves the model state dictionary to the specified path.

    Args:
        model: PyTorch model.
        path: Destination file path.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device):
    """
    Loads the model state dictionary from the specified path.

    Args:
        model: PyTorch model instance.
        path: Path to the checkpoint file.
        device: Device to load the tensors onto.

    Returns:
        model: The model with loaded weights.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at {path}")

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def save_submission(ids, predictions, path):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        ids: List or array of image IDs (filenames).
        predictions: List or array of predicted probabilities.
        path: Path to save the submission CSV.
    """
    # Ensure inputs are flat lists/arrays
    ids = np.array(ids).flatten()
    predictions = np.array(predictions).flatten()

    df = pd.DataFrame({"id": ids, "has_cactus": predictions})

    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    df.to_csv(path, index=False)
