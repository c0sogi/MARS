import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from library.config import WORKING_DIR, seed_everything


def compute_log_loss(y_true, y_pred):
    """
    Computes the Log Loss metric.

    Args:
        y_true: Array-like of ground truth labels (0 or 1).
        y_pred: Array-like of predicted probabilities for class 1 (dog).

    Returns:
        float: The calculated log loss.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate log loss. Sklearn handles clipping internally (eps=1e-15 by default)
    # labels parameter ensures that even if a batch has only one class,
    # it knows it's a binary problem with classes 0 and 1.
    loss = log_loss(y_true, y_pred, labels=[0, 1])
    return loss


def save_checkpoint(state, filename="checkpoint.pth"):
    """
    Saves the model training checkpoint.

    Args:
        state (dict): Dictionary containing model_state_dict, optimizer_state_dict, etc.
        filename (str): Name of the file to save within the WORKING_DIR.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)
    filepath = os.path.join(WORKING_DIR, filename)
    torch.save(state, filepath)
    # We do not print here to keep output clean, unless debugging is needed.


def load_checkpoint(filename, model, optimizer=None, device="cpu"):
    """
    Loads a model checkpoint.

    Args:
        filename (str): Name of the file to load from WORKING_DIR.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str): Device to map the location to ('cpu' or 'cuda').

    Returns:
        dict: The full checkpoint dictionary (useful for retrieving epoch, best_score, etc.)
    """
    filepath = os.path.join(WORKING_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def save_submission(ids, probs, output_path):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        ids (list or np.array): List of image IDs.
        probs (list or np.array): List of predicted probabilities for class 1.
        output_path (str): Full path to save the submission CSV.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame({"id": ids, "label": probs})

    # Sort by ID just in case, though usually order is preserved
    # The sample submission shows IDs 1, 2, 3...
    # Ensure IDs are integers if they are numeric
    try:
        df["id"] = df["id"].astype(int)
    except:
        pass  # Keep as is if not convertible

    df.to_csv(output_path, index=False)
