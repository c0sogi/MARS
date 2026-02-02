import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.

    The encoding uses 1-based indexing and column-major order (top-to-bottom, then left-to-right).
    Empty masks are encoded as '-'.

    Args:
        mask (np.ndarray or torch.Tensor): A binary mask of shape (Height, Width).
                                           Values should be 0 (background) or 1 (foreground).

    Returns:
        str: A space-delimited string of start positions and run lengths, or '-' if the mask is empty.
    """
    # Convert tensor to numpy if necessary
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # Ensure the mask is binary integer type for processing
    mask = mask.astype(np.int8)

    # Flatten in column-major order (Fortran-style) as per competition spec
    pixels = mask.flatten(order="F")

    # Check for empty mask
    if np.sum(pixels) == 0:
        return "-"

    # Pad with zeros at the start and end to detect all transitions
    # This handles edge cases where the mask starts or ends with a 1
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes (0->1 or 1->0)
    # np.where returns indices in the padded array.
    # The '+1' adjustment aligns the index to the start of the run (1-based)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # The 'runs' array currently contains [start1, end1, start2, end2, ...]
    # We need to convert end positions to lengths: length = end - start
    runs[1::2] -= runs[::2]

    # Convert to space-delimited string
    return " ".join(str(x) for x in runs)
