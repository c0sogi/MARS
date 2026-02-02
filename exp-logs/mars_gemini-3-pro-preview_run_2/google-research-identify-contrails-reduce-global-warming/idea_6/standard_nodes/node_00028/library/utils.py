import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


class AverageMeter:
    """
    Computes and stores the average and current value of a metric.
    Useful for tracking loss and scores during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update the meter with a new value.

        Args:
            val (float): The current value to add.
            n (int): The weight/count of the value (usually batch size).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE) for submission.

    The metric checks that the pairs are sorted, positive, and the decoded pixel
    values are not duplicated. The pixels are numbered from top to bottom,
    then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W) where 1 indicates mask.

    Returns:
        str: Space delimited list of pairs 'start length' or '-' if empty.
    """
    # Flatten column-wise (Fortran-style) as per competition spec
    pixels = mask.flatten(order="F")

    # Pad with 0s at start and end to detect transitions at boundaries
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs[0] is start of first run, runs[1] is end of first run, etc.
    # We want [start, length, start, length...]
    # Current runs: [start1, end1, start2, end2, ...]
    # Length = end - start
    runs[1::2] -= runs[::2]

    encoded = " ".join(str(x) for x in runs)

    if encoded == "":
        return "-"

    return encoded
