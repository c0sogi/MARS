import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from library.config import Config


def seed_everything(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rank_normalize(probs):
    """
    Converts a tensor or numpy array of probabilities into normalized ranks (0 to 1).
    This is useful for ensembling models with different calibration distributions.

    Args:
        probs (torch.Tensor or np.ndarray): Input probabilities of shape (N, C) or (N,).

    Returns:
        torch.Tensor or np.ndarray: Rank-normalized probabilities with the same type and shape as input.
    """
    if isinstance(probs, torch.Tensor):
        device = probs.device
        probs_np = probs.detach().cpu().numpy()
        is_tensor = True
    else:
        probs_np = probs
        is_tensor = False

    # Initialize output container
    ranked = np.zeros_like(probs_np, dtype=np.float32)

    # Handle 1D array
    if probs_np.ndim == 1:
        n = len(probs_np)
        if n > 1:
            # method='average' assigns the average rank to ties
            ranks = rankdata(probs_np, method="average")
            # Normalize to [0, 1]
            ranked = (ranks - 1) / (n - 1)
        else:
            ranked[:] = 0.5
    else:
        # Handle 2D array (N, C)
        n = probs_np.shape[0]
        if n > 1:
            for i in range(probs_np.shape[1]):
                ranks = rankdata(probs_np[:, i], method="average")
                ranked[:, i] = (ranks - 1) / (n - 1)
        else:
            ranked[:] = 0.5

    if is_tensor:
        return torch.tensor(ranked, dtype=torch.float32, device=device)

    return ranked


def calculate_metric(y_true, y_pred):
    """
    Computes the Mean Column-wise ROC AUC metric.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels (one-hot or binary indicators).
                                             Expected shape: (N, Num_Classes).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities.
                                             Expected shape: (N, Num_Classes).

    Returns:
        float: The mean column-wise ROC AUC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Check shapes
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # Calculate Macro Average ROC AUC
    # We handle potential errors if a class is not present in the batch (e.g. during debugging with small batches)
    try:
        score = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        # Fallback: calculate per column manually to handle constant columns
        scores = []
        for i in range(y_true.shape[1]):
            try:
                # Check if column has more than 1 unique class
                if len(np.unique(y_true[:, i])) > 1:
                    s = roc_auc_score(y_true[:, i], y_pred[:, i])
                    scores.append(s)
            except ValueError:
                # Skip columns where AUC is undefined (only one class present)
                pass

        if scores:
            score = np.mean(scores)
        else:
            score = 0.5  # Default if evaluation is impossible

    return score
