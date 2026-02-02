import os
import random
import numpy as np
import torch
from collections import defaultdict


def set_seed(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def rle_encode(mask):
    """
    Encodes a binary mask to Run-Length Encoding (RLE).
    The pixels are one-indexed and numbered from top to bottom, then left to right.

    Args:
        mask (np.array): Binary mask of shape (height, width).

    Returns:
        str: Space delimited list of pairs (start, length).
    """
    # Flatten in column-major order (Fortran-style)
    pixels = mask.flatten(order="F")

    # We need to find where the pixels change from 0 to 1 or 1 to 0.
    # We pad the pixels with 0 at the beginning and end to handle edge cases easily.
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # The 'runs' array contains indices of starts of 1s and starts of 0s alternating.
    # Since we padded with 0 at start, the first change must be 0->1 (start of a run).
    # Even indices in 'runs' are starts, odd indices are ends (exclusive).

    # Calculate lengths: end - start
    runs[1::2] -= runs[::2]

    # Return string
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded string to a binary mask.

    Args:
        mask_rle (str): Space delimited list of pairs (start, length).
        shape (tuple): The shape of the output mask (height, width).

    Returns:
        np.array: Binary mask of given shape.
    """
    if not isinstance(mask_rle, str) or mask_rle.strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    # Convert 1-indexed to 0-indexed
    starts -= 1

    # Calculate end indices
    ends = starts + lengths

    # Create flat array
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    # Fill runs
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape to original shape using column-major order
    return img.reshape(shape, order="F")


class MetricMonitor:
    """
    A helper class to track metrics (like Loss, IoU, Accuracy) during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all metrics."""
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        """
        Updates the metric with a new value.

        Args:
            metric_name (str): Name of the metric.
            val (float): Value to add.
        """
        metric = self.metrics[metric_name]
        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        """
        Returns a string representation of the average metrics.
        Prints full precision as requested.
        """
        return " | ".join(
            [
                "{metric_name}: {avg}".format(
                    metric_name=metric_name, avg=metric["avg"]
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )
