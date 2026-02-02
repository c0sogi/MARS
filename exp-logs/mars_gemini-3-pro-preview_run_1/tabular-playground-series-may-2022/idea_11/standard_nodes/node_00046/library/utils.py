import os
import random
import numpy as np
import torch
from collections import defaultdict


def seed_everything(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MetricMonitor:
    """
    Tracks and averages metrics (losses, scores) during training/validation.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        """
        Updates the metric with a new value.
        Args:
            metric_name (str): Name of the metric.
            val (float or torch.Tensor): Value to add.
        """
        metric = self.metrics[metric_name]

        if torch.is_tensor(val):
            val = val.item()

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def get(self, metric_name):
        return self.metrics[metric_name]["avg"]

    def __str__(self):
        """
        Returns a string representation of the averaged metrics with full precision.
        """
        return " | ".join(
            [
                "{}: {}".format(metric_name, metric["avg"])
                for (metric_name, metric) in self.metrics.items()
            ]
        )


def save_checkpoint(
    model, path, optimizer=None, scheduler=None, epoch=None, score=None
):
    """
    Saves the model state and optional training artifacts.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    state = {
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if epoch is not None:
        state["epoch"] = epoch
    if score is not None:
        state["score"] = score

    torch.save(state, path)
