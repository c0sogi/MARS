import os
import random
import warnings
import numpy as np
import torch
from collections import defaultdict
from library.config import Config

# Suppress warnings to keep output clean
warnings.filterwarnings("ignore")


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Enforces deterministic behavior in CuDNN.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Ensure deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MetricMonitor:
    """
    A utility class to track and average metrics (like Loss, F1 score) over epochs.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the internal metrics storage."""
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        """
        Updates the metric with a new value.

        Args:
            metric_name (str): The name of the metric (e.g., 'Loss', 'F1').
            val (float or torch.Tensor): The value to add.
        """
        metric = self.metrics[metric_name]

        # Detach and convert tensor to float if necessary
        if torch.is_tensor(val):
            val = val.detach().cpu().item()

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        """
        Returns a string representation of the averaged metrics.
        Prints full precision without rounding or formatting as required.
        """
        return " | ".join(
            [
                "{}: {}".format(metric_name, metric["avg"])
                for (metric_name, metric) in self.metrics.items()
            ]
        )
