import os
import random
import numpy as np
import torch
from collections import defaultdict


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
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.
    The pixels are numbered from top to bottom, then left to right (column-major).

    Args:
        mask (np.ndarray): Binary mask of shape (Height, Width).

    Returns:
        str: Space-delimited RLE string.
    """
    # Flatten column-wise (Fortran-style)
    pixels = mask.flatten(order="F")
    # Pad with 0s at start and end to detect transitions
    pixels = np.concatenate([[0], pixels, [0]])
    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # Calculate lengths (end - start)
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded string into a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): The shape of the output mask (Height, Width).

    Returns:
        np.ndarray: Binary mask of the specified shape.
    """
    if not isinstance(mask_rle, str) or mask_rle.strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    # Parse starts and lengths
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    # Convert 1-indexed to 0-indexed
    starts -= 1
    ends = starts + lengths

    # Create flattened array
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape column-wise
    return img.reshape(shape, order="F")


class MetricMonitor:
    """
    Tracks and averages metrics during training loops.
    """

    def __init__(self, float_precision=4):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.val = defaultdict(float)
        self.avg = defaultdict(float)
        self.sum = defaultdict(float)
        self.count = defaultdict(int)

    def update(self, metric_name, val):
        self.val[metric_name] = val
        self.sum[metric_name] += val
        self.count[metric_name] += 1
        self.avg[metric_name] = self.sum[metric_name] / self.count[metric_name]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{precision}f}".format(
                    metric_name=metric_name, avg=avg, precision=self.float_precision
                )
                for (metric_name, avg) in self.avg.items()
            ]
        )
