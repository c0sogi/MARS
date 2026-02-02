import os
import random
import re
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_image_id(filename):
    """
    Extracts the integer ID from a DICOM filename to enable numerical sorting.

    Example: 'Image-10.dcm' -> 10

    Args:
        filename (str): The filename string.

    Returns:
        int: The extracted image ID. Returns -1 if no digits are found.
    """
    # Look for the pattern "Image-" followed by digits
    match = re.search(r"Image-(\d+)", filename)
    if match:
        return int(match.group(1))

    # Fallback: find any sequence of digits in the filename
    digits = re.findall(r"\d+", filename)
    if digits:
        # Return the last number found, as this is typically the instance number
        return int(digits[-1])

    return -1
