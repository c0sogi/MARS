import os
import random
import numpy as np
import torch
from library import config


def seed_everything(seed=config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).

    Args:
        mask (np.ndarray or torch.Tensor): Binary mask of shape (H, W).
                                           1 - mask, 0 - background.

    Returns:
        str: Space-delimited list of pairs (start, length) or '-' if empty.
             Pixels are numbered from top to bottom, then left to right.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # Ensure mask is binary
    mask = mask.astype(np.int8)

    # Flatten column-wise (Fortran-style) as per competition spec
    pixels = mask.flatten(order="F")

    # Add padding to detect start and end of runs
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where value changes
    # pixels[1:] != pixels[:-1] finds transitions
    # +1 shifts index to match 1-based indexing requirement
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths: runs[1::2] (ends) - runs[::2] (starts)
    runs[1::2] -= runs[::2]

    if len(runs) == 0:
        return "-"

    return " ".join(str(x) for x in runs)


class MetricMonitor:
    """
    A utility class to track and average metrics (loss, accuracy, etc.) over a running phase.
    """

    def __init__(self, float_precision=4):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = {}

    def update(self, metric_name, val, count=1):
        """
        Update a specific metric.

        Args:
            metric_name (str): Name of the metric.
            val (float): Value to accumulate (e.g., batch mean loss).
            count (int): Weight of the value (e.g., batch size).
        """
        previous_data = self.metrics.get(metric_name, {"sum": 0.0, "count": 0})
        previous_data["sum"] += val * count
        previous_data["count"] += count
        self.metrics[metric_name] = previous_data

    def get_avg(self, metric_name):
        """
        Returns the current average for the specified metric.
        """
        data = self.metrics.get(metric_name)
        if not data or data["count"] == 0:
            return 0.0
        return data["sum"] / data["count"]

    def __str__(self):
        """
        Returns a formatted string of current averages.
        """
        return " | ".join(
            [
                "{}: {:.{prec}f}".format(
                    metric_name, self.get_avg(metric_name), prec=self.float_precision
                )
                for metric_name in self.metrics
            ]
        )
