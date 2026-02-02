import os
import random
import numpy as np
import torch
from sklearn.metrics import f1_score


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, filename):
    """
    Saves the model state dictionary and optimizer state to a file.

    Args:
        state (dict): State dictionary containing model and optimizer parameters.
        filename (str): Path to save the checkpoint.
    """
    print(f"Saving checkpoint to {filename}")
    torch.save(state, filename)


def load_checkpoint(checkpoint_path, model, optimizer=None, device="cpu"):
    """
    Loads the model state dictionary (and optimizer state if provided) from a file.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str): Device to map the location to (e.g., 'cpu', 'cuda').

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def optimize_threshold(y_true, y_pred_probs):
    """
    Finds the optimal probability threshold that maximizes the Mean F1-Score (samples average).

    Args:
        y_true (np.array): Binary ground truth labels (N, NumTags).
        y_pred_probs (np.array): Predicted probabilities (N, NumTags).

    Returns:
        tuple: (best_threshold, best_score)
    """
    # Define range of thresholds to test
    thresholds = np.arange(0.1, 0.95, 0.05)
    best_threshold = 0.5
    best_score = -1.0

    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred_probs, torch.Tensor):
        y_pred_probs = y_pred_probs.detach().cpu().numpy()

    # Iterate through thresholds
    for thresh in thresholds:
        # Binarize predictions based on current threshold
        y_pred_bin = (y_pred_probs >= thresh).astype(int)

        # Calculate F1-Score with 'samples' average (standard for multi-label)
        score = f1_score(y_true, y_pred_bin, average="samples", zero_division=0)

        if score > best_score:
            best_score = score
            best_threshold = thresh

    print("Optimization Results:")
    print(f"Best Threshold: {best_threshold}")
    print(f"Best Validation F1-Score: {best_score}")

    return best_threshold, best_score
