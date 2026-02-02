import os
import re
import random
import numpy as np
import torch
from sklearn.metrics import f1_score


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_text(text):
    """
    Cleans the input text by removing HTML tags, converting to lowercase,
    removing non-alphanumeric characters, and collapsing whitespace.

    Args:
        text (str): The raw input text (e.g., HTML content).

    Returns:
        str: The cleaned and normalized text.
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML tags using regex
    # <[^>]+> matches any string starting with < and ending with >
    text = re.sub(r"<[^>]+>", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Remove non-alphanumeric characters (keep spaces)
    # This regex replaces any character that is NOT a-z, 0-9, or whitespace with a space
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse multiple spaces into a single space and strip leading/trailing whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def calculate_f1_score(y_true, y_pred):
    """
    Calculates the Mean F1-Score (samples average).

    Args:
        y_true (np.ndarray): Ground truth binary labels (n_samples, n_classes).
        y_pred (np.ndarray): Predicted binary labels (n_samples, n_classes).

    Returns:
        float: The samples-averaged F1 score.
    """
    # zero_division=0 ensures that if a sample has no true or predicted labels, the score is 0
    return f1_score(y_true, y_pred, average="samples", zero_division=0)


def optimize_threshold(y_true, y_pred_probs, steps=20):
    """
    Finds the optimal probability threshold that maximizes the Mean F1-Score.

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_pred_probs (np.ndarray): Predicted probabilities.
        steps (int): Number of steps to search between 0.05 and 0.5.

    Returns:
        tuple: (best_threshold, best_score)
    """
    # Define range of thresholds to check.
    # For multi-label tasks with sparse tags, optimal thresholds are typically low (< 0.5).
    # We search from 0.05 to 0.5.
    thresholds = np.linspace(0.05, 0.5, steps)

    best_threshold = 0.25
    best_score = -1.0

    # Ensure inputs are numpy arrays for efficient computation
    if hasattr(y_true, "cpu"):
        y_true = y_true.detach().cpu().numpy()
    elif not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)

    if hasattr(y_pred_probs, "cpu"):
        y_pred_probs = y_pred_probs.detach().cpu().numpy()
    elif not isinstance(y_pred_probs, np.ndarray):
        y_pred_probs = np.array(y_pred_probs)

    for thresh in thresholds:
        # Binarize predictions based on current threshold
        y_pred_bin = (y_pred_probs > thresh).astype(int)

        # Calculate score
        score = calculate_f1_score(y_true, y_pred_bin)

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score
