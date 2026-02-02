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
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).

    The pixels are numbered from top to bottom, then left to right:
    1 is pixel (1,1), 2 is pixel (2,1), etc. This corresponds to
    Column-Major (Fortran-style) flattening.

    Args:
        mask (np.ndarray): Binary mask of shape (Height, Width).
                           0 indicates background, 1 indicates object.

    Returns:
        str: Space-delimited list of pairs (start, length).
             Returns '-' if the mask is empty.
    """
    # Flatten column-major as per task description
    pixels = mask.flatten(order="F")

    # If empty
    if not np.any(pixels):
        return "-"

    # Prepend and append 0 to detect start and end of runs
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where value changes
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths: runs[1::2] are ends, runs[::2] are starts
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def dice_score(y_pred, y_true, threshold=0.5, smooth=1e-6):
    """
    Computes the Global Dice coefficient for a batch or dataset.

    The metric is defined as 2 * |X n Y| / (|X| + |Y|).
    This function flattens the entire input to calculate the metric globally
    over the provided tensors, effectively performing a micro-average.

    Args:
        y_pred (torch.Tensor or np.ndarray): Predicted probabilities (after sigmoid).
        y_true (torch.Tensor or np.ndarray): Ground truth binary mask.
        threshold (float): Threshold to convert probabilities to binary mask.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: The Dice coefficient.
    """
    # Convert to tensor if numpy
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)

    # Apply threshold to get binary predictions
    y_pred_bin = (y_pred > threshold).float()
    y_true_bin = y_true.float()

    # Flatten all dimensions to treat as a single set of pixels
    y_pred_flat = y_pred_bin.reshape(-1)
    y_true_flat = y_true_bin.reshape(-1)

    intersection = (y_pred_flat * y_true_flat).sum()
    union = y_pred_flat.sum() + y_true_flat.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return dice.item()


def load_checkpoint_weights(model, checkpoint_path, device=Config.DEVICE):
    """
    Loads weights from a checkpoint file into the model.
    Handles cases where the checkpoint is a full dictionary or just weights.

    Args:
        model (torch.nn.Module): The model instance.
        checkpoint_path (str): Path to the checkpoint file.
        device (str): Device to map the location to.

    Returns:
        model (torch.nn.Module): Model with loaded weights.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Check if it's a dict containing state_dict or just the state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return model


def average_checkpoints(model, checkpoint_paths, device=Config.DEVICE):
    """
    Averages the weights of multiple checkpoints and loads them into the model.
    This implements the model averaging strategy (SWA-like) to improve generalization.

    Args:
        model (torch.nn.Module): The model architecture instance.
        checkpoint_paths (list): List of file paths to checkpoints.
        device (str): Device to map location.

    Returns:
        model (torch.nn.Module): Model with averaged weights.
    """
    if not checkpoint_paths:
        raise ValueError("No checkpoint paths provided for averaging.")

    avg_state_dict = None
    n_checkpoints = 0

    print(f"Averaging weights from {len(checkpoint_paths)} checkpoints...")

    for path in checkpoint_paths:
        if not os.path.exists(path):
            print(f"Warning: Checkpoint {path} not found. Skipping.")
            continue

        checkpoint = torch.load(path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        if avg_state_dict is None:
            # Initialize with the first checkpoint
            avg_state_dict = {
                k: v.clone().to(device).float() for k, v in state_dict.items()
            }
        else:
            # Accumulate
            for k, v in state_dict.items():
                if k in avg_state_dict:
                    avg_state_dict[k] += v.to(device).float()

        n_checkpoints += 1

    if avg_state_dict is None or n_checkpoints == 0:
        raise ValueError("No valid checkpoints found to average.")

    # Divide by number of checkpoints
    for k in avg_state_dict.keys():
        avg_state_dict[k] /= n_checkpoints

    # Load into model
    model.load_state_dict(avg_state_dict)
    print(f"Successfully loaded averaged weights from {n_checkpoints} checkpoints.")
    return model
