import os
import random
import numpy as np
import torch
from collections import defaultdict


def set_seed(seed=42):
    """
    Sets the seed for random, numpy, and torch to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """
    Applies Mixup augmentation to the batch.
    Returns:
        mixed_x: The mixed input tensor.
        y_a: The labels of the first set of samples.
        y_b: The labels of the second set of samples (shuffled).
        lam: The mixing coefficient lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the mixup loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class MetricMonitor:
    """
    A helper class to track and average metrics (loss, accuracy, etc.)
    over the course of an epoch.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets the metric storage.
        """
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        """
        Updates the metric with a new value.
        Args:
            metric_name (str): The name of the metric (e.g., 'Loss').
            val (float): The value to add.
        """
        metric = self.metrics[metric_name]
        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        """
        Returns a string representation of the current averages with full precision.
        """
        return " | ".join(
            [
                "{metric_name}: {avg}".format(
                    metric_name=metric_name, avg=metric["avg"]
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )
