import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the torch device defined in Config.

    Returns:
        torch.device: The device to perform computations on.
    """
    return Config.DEVICE


def save_checkpoint(model, optimizer, epoch, score, path):
    """
    Saves the model checkpoint including model state, optimizer state, epoch, and validation score.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): Current epoch number.
        score (float): Validation metric score.
        path (str): File path to save the checkpoint.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "score": score,
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, device=Config.DEVICE):
    """
    Loads a model checkpoint.

    Args:
        path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (torch.device): Device to map the location to.

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def calculate_f1_samples(y_true, y_pred):
    """
    Fast vectorized calculation of F1 score with average='samples'.

    Args:
        y_true (np.ndarray): Binary ground truth labels (N, C).
        y_pred (np.ndarray): Binary predicted labels (N, C).

    Returns:
        float: The mean samples F1 score.
    """
    # Calculate True Positives
    tp = (y_true * y_pred).sum(axis=1)

    # Calculate sum of predicted positives and true positives
    pred_sum = y_pred.sum(axis=1)
    true_sum = y_true.sum(axis=1)

    # F1 = 2*TP / (Pred_Sum + True_Sum)
    # Add epsilon to avoid division by zero
    epsilon = 1e-9
    f1_scores = (2 * tp) / (pred_sum + true_sum + epsilon)

    return f1_scores.mean()


def optimize_f1_threshold(y_true, y_probs, step=0.01):
    """
    Finds the optimal probability threshold for multi-label classification
    that maximizes the samples-averaged F1 score.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth binary labels.
        y_probs (np.ndarray or torch.Tensor): Predicted probabilities (sigmoid output).
        step (float): Step size for threshold search.

    Returns:
        tuple: (best_threshold, best_score)
    """
    # Convert to numpy if inputs are tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_probs, torch.Tensor):
        y_probs = y_probs.detach().cpu().numpy()

    best_threshold = 0.5
    best_score = -1.0

    # Iterate over thresholds to find the optimum
    # Range is typically within [0.1, 0.9] for sigmoid outputs
    thresholds = np.arange(0.1, 0.9 + step, step)

    for thr in thresholds:
        # Binarize predictions based on current threshold
        y_pred = (y_probs >= thr).astype(int)

        # Calculate score
        score = calculate_f1_samples(y_true, y_pred)

        if score > best_score:
            best_score = score
            best_threshold = thr

    return best_threshold, best_score
