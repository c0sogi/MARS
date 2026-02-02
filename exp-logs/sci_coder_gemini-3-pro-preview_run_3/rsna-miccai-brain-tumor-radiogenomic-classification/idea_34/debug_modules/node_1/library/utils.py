import os
import random
import re
import numpy as np
import torch


def seed_everything(seed=42):
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


def get_slice_number(filename):
    """
    Parses the integer slice index from a DICOM filename.

    This function extracts the first sequence of digits found in the filename,
    which corresponds to the slice number in the BraTS21 dataset format
    (e.g., 'Image-10.dcm' -> 10). This allows for correct numerical sorting
    instead of lexicographical sorting.

    Args:
        filename (str): The file path or filename.

    Returns:
        int: The extracted slice number. Returns -1 if no digits are found.
    """
    # Extract the basename to handle full paths
    base = os.path.basename(filename)

    # Search for the first sequence of digits in the filename
    match = re.search(r"(\d+)", base)

    if match:
        return int(match.group(1))

    return -1
