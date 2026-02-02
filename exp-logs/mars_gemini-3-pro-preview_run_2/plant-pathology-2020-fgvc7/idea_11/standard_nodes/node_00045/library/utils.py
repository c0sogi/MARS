import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Mean Column-wise ROC AUC.

    Args:
        y_true (np.array): Ground truth labels (one-hot encoded or binary).
        y_pred (np.array): Predicted probabilities.

    Returns:
        float: The mean column-wise ROC AUC score.
    """
    # Handle edge case where a class might not be present in y_true
    try:
        # average='macro' calculates metrics for each label, and finds their unweighted mean.
        # This matches "Mean column-wise ROC AUC".
        score = roc_auc_score(y_true, y_pred, average="macro", multi_class="ovr")
        return score
    except ValueError as e:
        print(f"Warning: Error calculating ROC AUC: {e}")
        return 0.0


def rank_normalize(y_pred):
    """
    Converts raw probabilities to normalized ranks (0 to 1) column-wise.
    Used for Rank-Calibrated Averaging.

    Args:
        y_pred (np.array): Input probabilities of shape (N, C).

    Returns:
        np.array: Rank-normalized probabilities of shape (N, C).
    """
    # Initialize output array
    y_rank = np.zeros_like(y_pred)

    # Iterate over each column (class)
    for i in range(y_pred.shape[1]):
        # rankdata returns ranks starting from 1
        # method='average' handles ties by assigning the average rank
        ranks = rankdata(y_pred[:, i], method="average")

        # Normalize to [0, 1]
        # Subtract 1 to start at 0, divide by (N-1) to end at 1
        if len(ranks) > 1:
            y_rank[:, i] = (ranks - 1) / (len(ranks) - 1)
        else:
            y_rank[:, i] = 0.0  # Fallback for single sample

    return y_rank


def reconstruct_probabilities(rust_probs, scab_probs):
    """
    Reconstructs the 4-class probabilities from the decomposed Rust and Scab binary predictions.

    Mapping logic:
    - Healthy: (1 - Rust) * (1 - Scab)
    - Multiple: Rust * Scab
    - Rust: Rust * (1 - Scab)
    - Scab: (1 - Rust) * Scab

    Args:
        rust_probs (np.array): Probability of Rust (N,).
        scab_probs (np.array): Probability of Scab (N,).

    Returns:
        np.array: Array of shape (N, 4) with columns [healthy, multiple_diseases, rust, scab].
    """
    # Ensure inputs are numpy arrays
    r = np.array(rust_probs)
    s = np.array(scab_probs)

    # Calculate derived classes
    healthy = (1 - r) * (1 - s)
    multiple = r * s
    rust_only = r * (1 - s)
    scab_only = (1 - r) * s

    # Stack into (N, 4) array
    # Order must match sample_submission: healthy, multiple_diseases, rust, scab
    # Based on task description: | healthy | multiple_diseases | rust | scab |

    final_preds = np.column_stack([healthy, multiple, rust_only, scab_only])

    return final_preds
