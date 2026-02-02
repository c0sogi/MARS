import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    Ensures deterministic behavior for CUDA operations if available.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def dice_coef(y_pred, y_true, smooth=1.0):
    """
    Calculates the Dice Coefficient.

    Args:
        y_pred: Prediction tensor (probabilities or binary).
        y_true: Ground truth tensor (binary).
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        Dice coefficient score.
    """
    # Flatten the tensors to 1D vectors
    y_pred = y_pred.contiguous().view(-1)
    y_true = y_true.contiguous().view(-1)

    intersection = (y_pred * y_true).sum()
    dice = (2.0 * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)

    return dice


def rle_encode(mask):
    """
    Run-Length Encode a binary mask.
    The pixels are numbered from top to bottom, then left to right (Column-Major).

    Args:
        mask: Binary mask (numpy array), 1 - mask, 0 - background.

    Returns:
        Space delimited string of pairs (start, length).
    """
    # Flatten column-wise (Fortran-style) to match "top to bottom, then left to right"
    pixels = mask.flatten(order="F")

    # Prepend and append 0 to detect start and end of runs efficiently
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where value changes (0 to 1 or 1 to 0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # The 'runs' array contains start indices of 1s and start indices of 0s (end of 1s)
    # Calculate lengths: end_index - start_index
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def average_weights(state_dicts):
    """
    Averages model weights from a list of state dictionaries.
    Explicitly filters for floating-point parameters to avoid corrupting integer buffers
    (like Batch Normalization num_batches_tracked).

    Args:
        state_dicts: List of model state dictionaries (e.g., from top-k checkpoints).

    Returns:
        A single averaged state dictionary.
    """
    if not state_dicts:
        return {}

    avg_state = {}
    reference_dict = state_dicts[0]

    for key in reference_dict.keys():
        # Check the type of the parameter in the first state_dict
        param = reference_dict[key]

        if torch.is_floating_point(param):
            # Stack and average floating point parameters (weights, biases, running means)
            stacked_params = torch.stack([d[key] for d in state_dicts])
            avg_state[key] = torch.mean(stacked_params, dim=0)
        else:
            # For integer/long parameters (e.g., BN num_batches_tracked),
            # take the value from the last checkpoint to maintain correct state
            avg_state[key] = state_dicts[-1][key]

    return avg_state
