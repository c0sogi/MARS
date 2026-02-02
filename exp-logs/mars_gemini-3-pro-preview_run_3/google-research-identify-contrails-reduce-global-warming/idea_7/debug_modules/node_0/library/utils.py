import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.

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


def rle_encode(mask):
    """
    Run-length encoding for a binary mask.
    Pixels are numbered from top to bottom, then left to right.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: Space-delimited list of pairs (start, length).
    """
    # Flatten column-wise (Fortran style) to match "top to bottom, then left to right"
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def dice_coef(y_pred, y_true, smooth=1e-6):
    """
    Computes the Dice coefficient.

    Args:
        y_pred (torch.Tensor): Predicted probabilities or binary mask.
        y_true (torch.Tensor): Ground truth binary mask.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: Dice coefficient.
    """
    # Flatten tensors to compute global Dice for the batch/input
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)

    intersection = (y_pred * y_true).sum()
    return (2.0 * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)


def average_checkpoints(checkpoint_paths, device="cpu"):
    """
    Averages the state dictionaries of multiple checkpoints.

    Args:
        checkpoint_paths (list): List of paths to .pth checkpoint files.
        device (str): Device to load checkpoints onto.

    Returns:
        dict: Averaged state dictionary.
    """
    if not checkpoint_paths:
        raise ValueError("No checkpoints provided for averaging.")

    # Load the first checkpoint
    first_path = checkpoint_paths[0]
    if not os.path.exists(first_path):
        raise FileNotFoundError(f"Checkpoint not found: {first_path}")

    avg_state_dict = torch.load(first_path, map_location=device)

    # Handle wrapped state dicts (e.g., if saved as {'model_state_dict': ...})
    if "model_state_dict" in avg_state_dict:
        avg_state_dict = avg_state_dict["model_state_dict"]
    elif "state_dict" in avg_state_dict:
        avg_state_dict = avg_state_dict["state_dict"]

    # Convert to float for averaging
    for key in avg_state_dict:
        avg_state_dict[key] = avg_state_dict[key].to(torch.float32)

    num_checkpoints = len(checkpoint_paths)

    if num_checkpoints > 1:
        for i in range(1, num_checkpoints):
            path = checkpoint_paths[i]
            if not os.path.exists(path):
                continue

            state_dict = torch.load(path, map_location=device)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            for key in avg_state_dict:
                avg_state_dict[key] += state_dict[key].to(torch.float32)

    # Divide by number of checkpoints
    for key in avg_state_dict:
        avg_state_dict[key] /= num_checkpoints

    return avg_state_dict
