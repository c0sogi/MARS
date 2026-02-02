import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Macro-Averaged ROC AUC score for multi-label classification.
    Handles cases where a class might not be present in the target vector by skipping it.

    Args:
        y_true (np.array or torch.Tensor): Ground truth labels (N, num_classes).
        y_pred (np.array or torch.Tensor): Predicted probabilities (N, num_classes).

    Returns:
        float: Macro-averaged ROC AUC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    n_classes = y_true.shape[1]
    auc_scores = []

    for i in range(n_classes):
        try:
            # Only calculate ROC AUC if both classes (0 and 1) are present in y_true
            # sklearn roc_auc_score throws ValueError if only one class is present
            if len(np.unique(y_true[:, i])) == 2:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                auc_scores.append(score)
        except ValueError:
            # Skip if calculation fails
            continue

    if not auc_scores:
        return 0.5  # Default fallback if no classes can be evaluated

    return np.mean(auc_scores)


def save_checkpoint(state, filename):
    """
    Saves the model checkpoint to the specified file.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        filename (str): Path to save the checkpoint.
    """
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None, device="cpu"):
    """
    Loads a model checkpoint.

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str): Device to map the location to.

    Returns:
        tuple: (epoch, best_score)
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    checkpoint = torch.load(filename, map_location=device)

    # Load model state
    # Handle different saving conventions
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided
    if optimizer is not None:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        elif "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    best_score = checkpoint.get("best_score", 0.0)

    return epoch, best_score
