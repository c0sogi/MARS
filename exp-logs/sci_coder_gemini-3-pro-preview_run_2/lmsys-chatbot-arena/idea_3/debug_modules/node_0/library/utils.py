import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from library.config import Config, seed_everything


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, path=None):
    """
    Saves the model checkpoint including optimizer and scheduler states.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        epoch (int): Current epoch.
        val_loss (float): Validation loss.
        path (str, optional): Path to save the checkpoint. Defaults to Config.MODEL_SAVE_PATH.
    """
    if path is None:
        path = Config.MODEL_SAVE_PATH

    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "val_loss": val_loss,
    }

    torch.save(state, path)


def load_checkpoint(
    model, optimizer=None, scheduler=None, path=None, device=Config.DEVICE
):
    """
    Loads a model checkpoint.

    Args:
        model: The PyTorch model to load weights into.
        optimizer (optional): Optimizer to load state into.
        scheduler (optional): Scheduler to load state into.
        path (str, optional): Path to the checkpoint. Defaults to Config.MODEL_SAVE_PATH.
        device (torch.device, optional): Device to map the location to.

    Returns:
        tuple: (start_epoch, best_val_loss)
    """
    if path is None:
        path = Config.MODEL_SAVE_PATH

    if not os.path.exists(path):
        return 0, float("inf")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if (
        scheduler is not None
        and "scheduler_state_dict" in checkpoint
        and checkpoint["scheduler_state_dict"] is not None
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint.get("epoch", 0), checkpoint.get("val_loss", float("inf"))


def compute_score(y_true, y_pred):
    """
    Computes the Log Loss metric.

    Args:
        y_true: Ground truth labels (n_samples, 3) or class indices.
        y_pred: Predicted probabilities (n_samples, 3).

    Returns:
        float: The calculated log loss.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate log loss with eps='auto' as per requirements
    return log_loss(y_true, y_pred, eps="auto")


def write_submission(ids, predictions, output_path=None):
    """
    Writes the submission file to CSV in the required format.

    Args:
        ids: List or array of sample IDs.
        predictions: Array of probabilities (n_samples, 3).
        output_path (str, optional): Path to save the file. Defaults to Config.SUBMISSION_PATH.
    """
    if output_path is None:
        output_path = Config.SUBMISSION_PATH

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission_df = pd.DataFrame(
        {
            "id": ids,
            "winner_model_a": predictions[:, 0],
            "winner_model_b": predictions[:, 1],
            "winner_tie": predictions[:, 2],
        }
    )

    submission_df.to_csv(output_path, index=False)
