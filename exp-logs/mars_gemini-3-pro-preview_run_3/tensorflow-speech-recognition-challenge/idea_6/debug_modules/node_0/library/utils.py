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
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    """
    Returns the appropriate torch device (cuda or cpu).

    Returns:
        torch.device: The device to perform computations on.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_accuracy(output, target):
    """
    Calculates the multiclass accuracy for a batch of predictions.

    Args:
        output (torch.Tensor): Predicted logits or probabilities of shape (batch_size, num_classes).
        target (torch.Tensor): Ground truth labels of shape (batch_size).

    Returns:
        float: The accuracy for the batch (0.0 to 1.0).
    """
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.max(dim=1)
        correct = pred.eq(target).sum().item()
        return correct / batch_size


class MetricMonitor:
    """
    A utility class to track and update metrics (like loss and accuracy)
    using a running average during training or validation.
    """

    def __init__(self, float_precision=4):
        """
        Args:
            float_precision (int): The number of decimal places for the string representation.
        """
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        """Resets all tracked metrics."""
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val, n=1):
        """
        Updates the running average for a specific metric.

        Args:
            metric_name (str): The name of the metric (e.g., 'Loss', 'Accuracy').
            val (float): The value to record (e.g., batch loss).
            n (int): The weight of the value (e.g., batch size). Default is 1.
        """
        metric = self.metrics[metric_name]
        metric["val"] += val * n
        metric["count"] += n
        metric["avg"] = metric["val"] / metric["count"]

    def get_avg(self, metric_name):
        """
        Returns the current running average of the specified metric.

        Args:
            metric_name (str): The name of the metric.

        Returns:
            float: The average value.
        """
        return self.metrics[metric_name]["avg"]

    def __str__(self):
        """
        Returns a formatted string representation of all tracked metrics.
        """
        return " | ".join(
            [
                "{}: {:.{prec}f}".format(name, metric["avg"], prec=self.float_precision)
                for name, metric in self.metrics.items()
            ]
        )
