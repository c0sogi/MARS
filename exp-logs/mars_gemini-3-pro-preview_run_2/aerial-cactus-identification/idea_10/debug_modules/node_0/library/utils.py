import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import set_seed, MODEL_DIR, OUTPUT_DIR


def get_device() -> torch.device:
    """
    Returns the appropriate torch device (CUDA if available, else CPU).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def calculate_roc_auc(y_true, y_scores):
    """
    Calculates the Area Under the ROC Curve.

    Args:
        y_true (array-like): True binary labels.
        y_scores (array-like): Target scores (probability estimates of the positive class).

    Returns:
        float: ROC AUC score. Returns 0.5 if only one class is present in y_true.
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # ROC AUC is not defined if there is only one class in the true labels
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_scores)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking metrics (loss, accuracy, etc.) during training.
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


def save_checkpoint(state, filename):
    """
    Saves the model checkpoint to the configured model directory.

    Args:
        state (dict): The state dictionary to save (model weights, optimizer state, etc.).
        filename (str): The name of the file (e.g., 'model_seed_0.pth').
    """
    filepath = os.path.join(MODEL_DIR, filename)
    torch.save(state, filepath)


def save_submission(ids, predictions, filename="submission.csv"):
    """
    Saves the predictions to a CSV file in the format required for submission.

    Args:
        ids (list or array): List of image IDs (filenames).
        predictions (list or array): List of predicted probabilities for 'has_cactus'.
        filename (str): Name of the output CSV file.
    """
    df = pd.DataFrame({"id": ids, "has_cactus": predictions})
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)
