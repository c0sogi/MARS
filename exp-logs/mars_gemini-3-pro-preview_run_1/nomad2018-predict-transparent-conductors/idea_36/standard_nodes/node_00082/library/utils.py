import os
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(
    model, optimizer, scheduler, epoch, val_loss, filename="best_model.pt"
):
    """
    Saves the model state and training parameters to a checkpoint file.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "val_loss": val_loss,
    }
    path = os.path.join(Config.WORKING_DIR, filename)
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer=None, scheduler=None, filename="best_model.pt"):
    """
    Loads the model state and training parameters from a checkpoint file.

    Returns:
        tuple: (epoch, val_loss)
    """
    path = os.path.join(Config.WORKING_DIR, filename)
    if not os.path.exists(path):
        return 0, float("inf")

    checkpoint = torch.load(path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if (
        scheduler
        and "scheduler_state_dict" in checkpoint
        and checkpoint["scheduler_state_dict"]
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint.get("epoch", 0), checkpoint.get("val_loss", float("inf"))


def compute_column_wise_rmsle(preds, targets):
    """
    Computes the mean of the column-wise Root Mean Squared Logarithmic Error.

    Assumes inputs (preds and targets) are already log-transformed (log(1+x)).
    Therefore, it computes the RMSE of the inputs, which is mathematically equivalent
    to RMSLE of the original values.

    Args:
        preds (torch.Tensor or np.ndarray): Predictions in log space.
        targets (torch.Tensor or np.ndarray): Ground truth in log space.

    Returns:
        float: The mean column-wise RMSLE.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Calculate MSE for each column (formation energy, bandgap)
    mse_col = np.mean((preds - targets) ** 2, axis=0)
    # RMSE for each column
    rmsle_col = np.sqrt(mse_col)
    # Mean of the column-wise metrics
    return np.mean(rmsle_col)


def cache_data(func, filename, load_cached_data=True, *args, **kwargs):
    """
    Generic caching wrapper for deterministic data processing.
    Uses np.savez for storage to avoid pickle issues and ensure efficiency.

    Args:
        func (callable): The function that computes the data.
        filename (str): The filename to save/load the data (e.g., 'data.npz').
        load_cached_data (bool): Whether to try loading from cache.
        *args, **kwargs: Arguments passed to func.

    Returns:
        dict: The data dictionary returned by func or loaded from cache.
    """
    cache_path = os.path.join(Config.WORKING_DIR, filename)

    if load_cached_data and os.path.exists(cache_path):
        try:
            # allow_pickle=True is used here to handle potential object arrays (like string IDs)
            # stored within the npz archive, but the primary format is numpy's native zip.
            data = np.load(cache_path, allow_pickle=True)
            return dict(data)
        except Exception:
            # If loading fails (e.g. corrupt file), proceed to recompute
            pass

    # Compute data
    data = func(*args, **kwargs)

    # Save data
    if isinstance(data, dict):
        np.savez(cache_path, **data)

    return data


def save_submission(ids, preds, filename="submission.csv"):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        ids (list or np.array): List of test IDs.
        preds (np.array): Array of shape (N, 2) containing predicted [formation_energy, bandgap].
                          Values must be in the original scale (not log).
        filename (str): Output filename.
    """
    df = pd.DataFrame(preds, columns=["formation_energy_ev_natom", "bandgap_energy_ev"])
    df.insert(0, "id", ids)
    path = os.path.join(Config.SUBMISSION_DIR, filename)
    df.to_csv(path, index=False)


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(
        self,
        patience=Config.EARLY_STOPPING_PATIENCE,
        delta=0,
        verbose=False,
        path="best_model.pt",
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model, optimizer, scheduler, epoch):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, scheduler, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, scheduler, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, scheduler, epoch):
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_loss, filename=self.path
        )
        self.val_loss_min = val_loss
