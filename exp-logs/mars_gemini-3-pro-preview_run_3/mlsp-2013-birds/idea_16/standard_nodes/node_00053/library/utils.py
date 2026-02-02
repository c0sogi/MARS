import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_multilabel_auc(y_true, y_pred):
    """
    Calculates the macro-averaged ROC AUC for multi-label classification.
    Robust to batches where some classes might be missing (all 0s or all 1s).

    Args:
        y_true (np.ndarray): Ground truth binary labels, shape (N, num_classes).
        y_pred (np.ndarray): Predicted probabilities, shape (N, num_classes).

    Returns:
        float: The average ROC AUC score over valid classes.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n_classes = y_true.shape[1]
    aucs = []

    for i in range(n_classes):
        # Check if the class has both positive and negative samples
        if len(np.unique(y_true[:, i])) > 1:
            try:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                aucs.append(score)
            except ValueError:
                # Fallback for edge cases not caught by unique check
                continue

    if not aucs:
        return 0.5  # Return baseline if no classes are valid

    return np.mean(aucs)


def average_checkpoints(checkpoint_paths):
    """
    Averages the weights of multiple model checkpoints.

    Args:
        checkpoint_paths (list of str): List of file paths to the checkpoints.

    Returns:
        dict: A state dictionary containing the averaged weights.
    """
    if not checkpoint_paths:
        raise ValueError("No checkpoint paths provided for averaging.")

    # Load the first checkpoint to initialize the average dictionary
    # Map to CPU to avoid OOM on GPU during averaging
    first_ckpt = torch.load(checkpoint_paths[0], map_location="cpu")

    # Handle cases where checkpoint is a dict containing 'state_dict' or 'model'
    if isinstance(first_ckpt, dict) and "state_dict" in first_ckpt:
        avg_state_dict = first_ckpt["state_dict"]
    elif isinstance(first_ckpt, dict) and "model" in first_ckpt:
        avg_state_dict = first_ckpt["model"]
    else:
        avg_state_dict = first_ckpt

    # Clone tensors to avoid modifying the loaded object and ensure float precision
    # We use a separate dict to accumulate sums
    sum_state_dict = {key: val.clone().float() for key, val in avg_state_dict.items()}

    num_checkpoints = len(checkpoint_paths)

    # Iterate over the remaining checkpoints
    for path in checkpoint_paths[1:]:
        ckpt = torch.load(path, map_location="cpu")

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        for key in sum_state_dict.keys():
            if key in state_dict:
                sum_state_dict[key] += state_dict[key]
            else:
                raise KeyError(f"Key {key} missing in checkpoint {path}")

    # Divide by number of checkpoints to get the average
    final_avg_state_dict = {}
    for key, val in sum_state_dict.items():
        # Convert back to the original type if necessary, but usually keeping as float/tensor is fine
        # We perform the division in place
        final_avg_state_dict[key] = val / num_checkpoints

    return final_avg_state_dict
