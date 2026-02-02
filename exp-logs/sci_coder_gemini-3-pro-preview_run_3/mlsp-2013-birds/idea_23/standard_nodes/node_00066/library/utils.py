import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import set_seed

# Alias for reproducibility to be used by other modules
seed_everything = set_seed


def calculate_multilabel_auc(y_true, y_pred):
    """
    Calculates the macro-averaged ROC AUC for multi-label classification.
    Explicitly handles missing classes in batches by skipping them to ensure robustness.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels (N, C).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities (N, C).

    Returns:
        float: Macro-averaged ROC AUC score. Returns 0.0 if no valid classes are found.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    num_classes = y_true.shape[1]
    aucs = []

    for i in range(num_classes):
        # A class must have both positive (1) and negative (0) samples to calculate ROC AUC
        # This check prevents ValueError from sklearn when a batch lacks a specific class
        if len(np.unique(y_true[:, i])) > 1:
            try:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                aucs.append(auc)
            except ValueError:
                # Fallback for any unexpected edge cases
                pass

    if not aucs:
        return 0.0

    return np.mean(aucs)


def save_checkpoint(state, filepath):
    """
    Saves the model checkpoint state to a file.

    Args:
        state (dict): Dictionary containing model parameters, optimizer state, epoch, etc.
        filepath (str): Destination path for the checkpoint.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device="cpu"):
    """
    Loads a checkpoint into the model and optionally optimizer/scheduler.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to update.
        optimizer (torch.optim.Optimizer, optional): The optimizer to update.
        scheduler (torch.optim.lr_scheduler, optional): The scheduler to update.
        device (str or torch.device): Device to map the checkpoint to.

    Returns:
        dict: The raw checkpoint dictionary (useful for retrieving epoch or best score).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def save_logs(logs, filepath):
    """
    Saves a list of log dictionaries to a CSV file.

    Args:
        logs (list[dict]): List of dictionaries containing log metrics.
        filepath (str): Destination path for the log CSV.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = pd.DataFrame(logs)
    df.to_csv(filepath, index=False)


def save_formatted_submission(rec_ids, predictions, filepath):
    """
    Formats predictions into the competition submission format and saves to CSV.
    Format: Id (rec_id * 100 + species_id), Probability

    Args:
        rec_ids (np.ndarray or list): List of recording IDs corresponding to the predictions.
        predictions (np.ndarray or torch.Tensor): Matrix of probabilities (N_samples, N_classes).
        filepath (str): Destination path for the submission CSV.
    """
    output_ids = []
    output_probs = []

    # Ensure numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()

    num_classes = predictions.shape[1]

    for idx, rec_id in enumerate(rec_ids):
        for species_id in range(num_classes):
            # Construct composite Id as per task description
            composite_id = int(rec_id * 100 + species_id)
            prob = float(predictions[idx, species_id])

            output_ids.append(composite_id)
            output_probs.append(prob)

    df = pd.DataFrame({"Id": output_ids, "Probability": output_probs})

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
