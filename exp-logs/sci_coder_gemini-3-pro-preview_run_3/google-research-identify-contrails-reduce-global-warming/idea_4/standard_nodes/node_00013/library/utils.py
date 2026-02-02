import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Determines the available computational device.

    Returns:
        torch.device: The device object ('cuda' if available, else 'cpu').
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).

    The metric expects pixels to be numbered from top to bottom, then left to right.
    This corresponds to Fortran-style (column-major) flattening.

    Args:
        mask (np.ndarray): Binary mask of shape (Height, Width).
                           1 indicates mask, 0 indicates background.

    Returns:
        str: Space-delimited list of pairs 'start length'.
             Returns '-' if the mask is empty.
    """
    # Flatten the mask in column-major order (top-to-bottom, then left-to-right)
    pixels = mask.flatten(order="F")

    # We need to find the starts and ends of runs of 1s.
    # Concatenate [0] at the beginning and end to detect transitions at edges.
    # pixels is expected to be binary (0 or 1).
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # 'runs' now contains indices of transitions.
    # Even indices (0, 2, 4...) in 'runs' correspond to 0->1 transitions (Starts).
    # Odd indices (1, 3, 5...) in 'runs' correspond to 1->0 transitions (Ends).

    # If no runs found, return '-'
    if len(runs) == 0:
        return "-"

    # The start indices are already 1-based relative to the original flattened array
    # because of the prepended 0 and the way np.where works on the diff.
    # However, let's trace carefully:
    # Original: [1, 1, 0]
    # Padded:   [0, 1, 1, 0, 0]
    # Diff:     [1, 0, -1, 0] (indices 0, 1, 2, 3 of comparison)
    # Non-zero at index 0 (0->1) and index 2 (1->0).
    # runs = [0+1, 2+1] = [1, 3].
    # Start is 1. End is 3. Length is 3 - 1 = 2. Correct.

    # Extract starts and ends
    starts = runs[0::2]
    ends = runs[1::2]

    # Calculate lengths
    lengths = ends - starts

    # Interleave starts and lengths
    # We want [start1, length1, start2, length2, ...]
    rle_pairs = []
    for s, l in zip(starts, lengths):
        rle_pairs.append(str(s))
        rle_pairs.append(str(l))

    return " ".join(rle_pairs)
