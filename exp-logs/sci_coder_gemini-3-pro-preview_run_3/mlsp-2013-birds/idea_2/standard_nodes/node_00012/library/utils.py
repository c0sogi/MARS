import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config, set_seed


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility.
    Wraps the set_seed function from library.config.
    """
    set_seed(seed)


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve (ROC AUC) for multi-label classification.
    Uses macro-averaging as per standard multi-label evaluation.

    Args:
        y_true (np.ndarray): Ground truth labels of shape (N, num_classes).
        y_pred (np.ndarray): Predicted probabilities of shape (N, num_classes).

    Returns:
        float: The macro-averaged ROC AUC score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    try:
        # standard macro-averaged ROC AUC
        score = roc_auc_score(y_true, y_pred, average="macro")
        if np.isnan(score):
            raise ValueError("roc_auc_score returned NaN")
    except ValueError:
        # This block handles edge cases, e.g., when a class has no positive samples
        # in the current validation batch. We calculate AUC per class and average
        # only over classes that are present.
        aucs = []
        for i in range(y_true.shape[1]):
            # Check if the class has both 0s and 1s
            if len(np.unique(y_true[:, i])) > 1:
                try:
                    auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                    if not np.isnan(auc):
                        aucs.append(auc)
                except ValueError:
                    pass

        if len(aucs) > 0:
            score = np.mean(aucs)
        else:
            score = 0.5  # Default random guess score if no classes can be evaluated

    return float(score)


def save_checkpoint(model, optimizer, epoch, score, filename):
    """
    Saves the model checkpoint including optimizer state.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): Current epoch.
        score (float): Validation score (AUC).
        filename (str): Path to save the checkpoint.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "score": float(score),
    }
    torch.save(state, filename)


def load_checkpoint(model, optimizer, filename, device):
    """
    Loads the model checkpoint.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer): The optimizer to load state into (can be None).
        filename (str): Path to the checkpoint file.
        device (torch.device): Device to map location.

    Returns:
        tuple: (start_epoch, score)
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint.get("epoch", 0), checkpoint.get("score", 0.0)


def format_and_save_submission(rec_ids, predictions, output_path):
    """
    Formats predictions and saves the submission file in the required format.
    Format: Id,Probability where Id = rec_id * 100 + species_number.

    Args:
        rec_ids (list or np.ndarray): List of recording IDs.
        predictions (np.ndarray): Predicted probabilities matrix (N_samples, 19_species).
        output_path (str): Path to save the CSV.
    """
    # Ensure rec_ids are integers
    rec_ids = np.array(rec_ids).astype(int)

    submission_rows = []
    num_species = predictions.shape[1]

    # Iterate through each recording and each species to flatten the results
    for i, rid in enumerate(rec_ids):
        for species_idx in range(num_species):
            # Construct Id: rec_id * 100 + species_id
            row_id = rid * 100 + species_idx
            prob = predictions[i, species_idx]
            submission_rows.append({"Id": row_id, "Probability": prob})

    df_submission = pd.DataFrame(submission_rows)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_submission.to_csv(output_path, index=False)


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
