import torch
import numpy as np
from collections import defaultdict
from library.config import Config


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """
    Applies Mixup augmentation to a batch of data.

    Args:
        x (torch.Tensor): Input images.
        y (torch.Tensor): Target labels.
        alpha (float): Parameter for the Beta distribution.
        use_cuda (bool): Whether to use CUDA for index generation.

    Returns:
        mixed_x (torch.Tensor): Mixed images.
        y_a (torch.Tensor): Original labels.
        y_b (torch.Tensor): Shuffled labels.
        lam (float): Mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the Mixup loss.

    Args:
        criterion (callable): Loss function.
        pred (torch.Tensor): Model predictions.
        y_a (torch.Tensor): Original labels.
        y_b (torch.Tensor): Shuffled labels.
        lam (float): Mixing coefficient.

    Returns:
        torch.Tensor: Calculated loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def update_bn(loader, model, device=None):
    """
    Updates the running statistics of the Batch Normalization modules in the model
    by performing a forward pass on the data from the loader.
    This is essential for SWA (Stochastic Weight Averaging) to ensure the
    BN statistics match the averaged weights.

    Args:
        loader (torch.utils.data.DataLoader): Data loader.
        model (torch.nn.Module): The model.
        device (str, optional): Device to run on. Defaults to Config.DEVICE.
    """
    if device is None:
        device = Config.DEVICE

    model.to(device)
    model.train()

    # Use PyTorch's optimized SWA utility to update BN statistics.
    # This resets running stats and computes mean/var over the dataset
    # by iterating through the loader.
    torch.optim.swa_utils.update_bn(loader, model, device=device)


class MetricMonitor:
    """
    A utility class to track and calculate average metrics during training/evaluation.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all metrics."""
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        """
        Updates a specific metric.

        Args:
            metric_name (str): Name of the metric.
            val (float or torch.Tensor): Value to update.
        """
        metric = self.metrics[metric_name]

        if isinstance(val, torch.Tensor):
            val = val.item()

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        """
        Returns a string representation of the metrics.
        Prints full precision without rounding as requested.
        """
        return " | ".join(
            [
                "{}: {}".format(metric_name, metric["avg"])
                for metric_name, metric in self.metrics.items()
            ]
        )
