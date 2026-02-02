import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss


def seed_everything(seed=42):
    """
    Sets the seed for reproducibility across Python, NumPy, and PyTorch.

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


def get_score(y_true, y_pred):
    """
    Calculates the Log Loss score with standard clipping.

    Args:
        y_true (array-like): Ground truth labels (0 or 1).
        y_pred (array-like): Predicted probabilities for class 1.

    Returns:
        float: The calculated log loss.
    """
    # Clip predictions to avoid log(0) errors
    # Standard practice is [1e-15, 1 - 1e-15]
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    score = log_loss(y_true, y_pred)
    return score


def generate_model_soup(checkpoint_paths, save_path):
    """
    Averages the weights of multiple checkpoints to create a Model Soup.

    Args:
        checkpoint_paths (list): List of file paths to the checkpoints.
        save_path (str): Path to save the averaged model state dictionary.
    """
    if not checkpoint_paths:
        raise ValueError("No checkpoint paths provided for Model Soup.")

    print(f"Generating Model Soup from {len(checkpoint_paths)} checkpoints...")

    # Load the first checkpoint to serve as the base structure
    # Map to CPU to avoid filling GPU memory during aggregation
    base_checkpoint = torch.load(checkpoint_paths[0], map_location="cpu")

    # Handle different saving formats (raw state_dict vs dict with metadata)
    if isinstance(base_checkpoint, dict) and "state_dict" in base_checkpoint:
        base_state = base_checkpoint["state_dict"]
    elif isinstance(base_checkpoint, dict) and "model" in base_checkpoint:
        base_state = base_checkpoint["model"]
    else:
        base_state = base_checkpoint

    # Initialize soup dictionary with float64 for precision during summation
    soup_state = {}
    for k, v in base_state.items():
        if isinstance(v, torch.Tensor):
            # Only average floating point tensors (weights/biases), keep others (like LongTensor buffers) as is
            if torch.is_floating_point(v):
                soup_state[k] = v.clone().to(torch.float64)
            else:
                soup_state[k] = v.clone()

    num_checkpoints = len(checkpoint_paths)

    # Iterate over the remaining checkpoints and accumulate
    for i in range(1, num_checkpoints):
        path = checkpoint_paths[i]
        checkpoint = torch.load(path, map_location="cpu")

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state = checkpoint["model"]
        else:
            state = checkpoint

        for k, v in state.items():
            if (
                k in soup_state
                and isinstance(v, torch.Tensor)
                and torch.is_floating_point(v)
            ):
                soup_state[k] += v.to(torch.float64)

    # Compute mean and revert to float32
    for k in soup_state:
        if isinstance(soup_state[k], torch.Tensor) and torch.is_floating_point(
            soup_state[k]
        ):
            soup_state[k] = (soup_state[k] / num_checkpoints).to(torch.float32)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the souped state dictionary
    torch.save(soup_state, save_path)
    print(f"Model Soup saved to {save_path}")
