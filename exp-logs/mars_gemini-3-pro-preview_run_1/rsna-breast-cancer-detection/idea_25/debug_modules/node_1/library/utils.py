import os
import torch
import numpy as np
from library import config


def seed_everything(seed=config.SEED):
    """
    Sets the random seed for reproducibility using the config utility.
    """
    config.set_seed(seed)


def probabilistic_f1(y_true, y_pred, epsilon=1e-7):
    """
    Computes the Probabilistic F1 score (pF1).

    Formula:
        pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)
        pPrecision = pTP / (pTP + pFP)
        pRecall = pTP / (TP + FN)

    Args:
        y_true (torch.Tensor or np.ndarray): Ground truth labels (binary 0 or 1).
        y_pred (torch.Tensor or np.ndarray): Predicted probabilities (0 to 1).
        epsilon (float): Small value to prevent division by zero.

    Returns:
        float: The computed pF1 score.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Flatten arrays to ensure 1D vectors
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Calculate Probabilistic True Positives (pTP)
    # pTP = Sum(y_pred_i * y_true_i)
    p_tp = np.sum(y_pred * y_true)

    # Calculate Probabilistic False Positives (pFP)
    # pFP = Sum(y_pred_i * (1 - y_true_i))
    p_fp = np.sum(y_pred * (1 - y_true))

    # Calculate Total Ground Truth Positives (TP + FN)
    total_positives = np.sum(y_true)

    # Calculate pPrecision
    # pPrecision = pTP / (pTP + pFP)
    # Note: pTP + pFP = Sum(y_pred)
    sum_pred = np.sum(y_pred)
    p_precision = p_tp / (sum_pred + epsilon)

    # Calculate pRecall
    # pRecall = pTP / (TP + FN)
    p_recall = p_tp / (total_positives + epsilon)

    # Calculate pF1
    p_f1 = 2 * (p_precision * p_recall) / (p_precision + p_recall + epsilon)

    return float(p_f1)


def save_checkpoint(state, filename="checkpoint.pth"):
    """
    Saves the model and optimizer state to the configured checkpoint directory.

    Args:
        state (dict): Dictionary containing 'state_dict', 'optimizer', 'epoch', etc.
        filename (str): Name of the checkpoint file.
    """
    filepath = os.path.join(config.CHECKPOINT_DIR, filename)
    torch.save(state, filepath)


def load_checkpoint(checkpoint_path, model, optimizer=None, device=config.DEVICE):
    """
    Loads a checkpoint into the model and optional optimizer.

    Args:
        checkpoint_path (str): Full path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str): Computation device.

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model weights handling common key variations
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided and available
    if optimizer is not None:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        elif "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
