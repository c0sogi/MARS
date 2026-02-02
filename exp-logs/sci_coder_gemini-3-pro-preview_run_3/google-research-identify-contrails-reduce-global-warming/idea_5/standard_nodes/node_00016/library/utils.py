import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=None):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int, optional): The seed value to set. If None, uses Config.SEED.
    """
    if seed is None:
        seed = Config.SEED

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE) suitable for submission.
    The mask is flattened in column-major order (top-to-bottom, then left-to-right).

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: Space-delimited list of pairs (start length). Returns '-' if mask is empty.
    """
    # Flatten in column-major order (Fortran-style) as per task description
    # "pixels are numbered from top to bottom, then left to right"
    pixels = mask.flatten(order="F")

    # Prepend and append 0 to detect start and end of runs
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths (end - start)
    runs[1::2] -= runs[::2]

    if len(runs) == 0:
        return "-"

    return " ".join(str(x) for x in runs)


def dice_coeff(pred, target, smooth=1e-6):
    """
    Computes the Dice coefficient between prediction and target tensors.

    Args:
        pred (torch.Tensor): Predicted probabilities or binary mask.
        target (torch.Tensor): Ground truth binary mask.
        smooth (float): Smoothing factor to prevent division by zero.

    Returns:
        torch.Tensor: Scalar Dice score.
    """
    # Flatten tensors
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()

    return (2.0 * intersection + smooth) / (union + smooth)


def average_checkpoints(checkpoint_paths, output_path=None):
    """
    Averages the state dictionaries of multiple checkpoints to create a robust model.
    This implements the 'Convergence-Aware Averaging' strategy.

    Args:
        checkpoint_paths (list): List of file paths to .pth checkpoints.
        output_path (str, optional): Path to save the averaged checkpoint.

    Returns:
        dict: The averaged state dictionary.
    """
    if not checkpoint_paths:
        raise ValueError("No checkpoints provided for averaging.")

    # Load the first checkpoint to initialize the average dict
    # Use map_location='cpu' to avoid GPU memory spikes
    first_ckpt = torch.load(checkpoint_paths[0], map_location="cpu")

    # Handle case where checkpoint saves more than just state_dict (e.g., optimizer state)
    if "model_state_dict" in first_ckpt:
        avg_state_dict = first_ckpt["model_state_dict"]
    else:
        avg_state_dict = first_ckpt

    # Convert parameters to float for precise averaging and clone to avoid side effects
    avg_state_dict = {k: v.clone().float() for k, v in avg_state_dict.items()}

    num_checkpoints = len(checkpoint_paths)

    # Sum parameters from remaining checkpoints
    for i in range(1, num_checkpoints):
        ckpt = torch.load(checkpoint_paths[i], map_location="cpu")
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        for k, v in state.items():
            if k in avg_state_dict:
                avg_state_dict[k] += v.float()

    # Average the parameters
    for k in avg_state_dict:
        avg_state_dict[k] /= num_checkpoints

    # Save if output path is provided
    if output_path:
        torch.save(avg_state_dict, output_path)

    return avg_state_dict


class MetricMonitor:
    """
    A helper class to track and average metrics during training epochs.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {}

    def update(self, metric_name, val):
        if metric_name not in self.metrics:
            self.metrics[metric_name] = {"sum": 0, "count": 0, "avg": 0}

        self.metrics[metric_name]["sum"] += val
        self.metrics[metric_name]["count"] += 1
        self.metrics[metric_name]["avg"] = (
            self.metrics[metric_name]["sum"] / self.metrics[metric_name]["count"]
        )

    def __str__(self):
        # Print full precision as requested
        return " | ".join(
            [
                f"{metric_name}: {stats['avg']}"
                for metric_name, stats in self.metrics.items()
            ]
        )
