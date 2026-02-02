import torch
import numpy as np
from collections import defaultdict
from library.config import Config


def set_seed(seed=None):
    """
    Sets the random seed for reproducibility.
    Wraps the Config.set_seed method to ensure consistency across the pipeline.
    """
    if seed is None:
        seed = Config.SEED
    Config.set_seed(seed)


def mixup_data(x, y, alpha=1.0, device=None):
    """
    Returns mixed inputs, pairs of targets, and lambda.

    Args:
        x (torch.Tensor): Input batch.
        y (torch.Tensor): Target batch.
        alpha (float): Mixup alpha parameter.
        device (torch.device): Device to perform computations on.

    Returns:
        mixed_x (torch.Tensor): Mixed input batch.
        y_a (torch.Tensor): Original targets.
        y_b (torch.Tensor): Permuted targets.
        lam (float): Mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if device is None:
        device = x.device

    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Computes the loss based on the mixed targets.

    Args:
        criterion (callable): Loss function.
        pred (torch.Tensor): Model predictions.
        y_a (torch.Tensor): Original targets.
        y_b (torch.Tensor): Permuted targets.
        lam (float): Mixing coefficient.

    Returns:
        loss (torch.Tensor): Weighted loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class MetricMonitor:
    """
    A helper class to track and average metrics (e.g., loss, accuracy)
    over a period (usually an epoch).
    """

    def __init__(self, float_precision=4):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        """Resets the internal state."""
        self.metrics = defaultdict(lambda: {"sum": 0, "count": 0})

    def update(self, metric_name, val, n=1):
        """
        Updates the metric with a new value.

        Args:
            metric_name (str): Name of the metric.
            val (float): Value to add.
            n (int): Number of samples associated with this value.
        """
        self.metrics[metric_name]["sum"] += val * n
        self.metrics[metric_name]["count"] += n

    def get_avg(self, metric_name):
        """Returns the average value of a specific metric."""
        metric = self.metrics[metric_name]
        if metric["count"] == 0:
            return 0.0
        return metric["sum"] / metric["count"]

    def __str__(self):
        """Returns a formatted string of current average metrics."""
        return " | ".join(
            [
                "{}: {:.{prec}f}".format(
                    metric_name,
                    metric["sum"] / metric["count"],
                    prec=self.float_precision,
                )
                for metric_name, metric in self.metrics.items()
            ]
        )
