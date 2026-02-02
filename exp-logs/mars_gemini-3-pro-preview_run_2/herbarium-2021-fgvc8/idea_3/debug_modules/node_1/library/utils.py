import os
import random
import numpy as np
import torch
from collections import defaultdict
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the seed for reproducibility across random, numpy, and torch.
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


class MetricMonitor:
    """
    A helper class to track metrics (loss, accuracy, F1, etc.) during training/validation.
    """

    def __init__(self, float_precision=4):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        """
        Updates the metric with a new value.
        """
        metric = self.metrics[metric_name]

        # Handle tensors
        if isinstance(val, torch.Tensor):
            val = val.item()

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def get_avg(self, metric_name):
        """Returns the average value of the metric."""
        return self.metrics[metric_name]["avg"]

    def __str__(self):
        """
        Returns a formatted string of the current average metrics.
        """
        return " | ".join(
            [
                "{}: {:.{prec}f}".format(
                    metric_name, metric["avg"], prec=self.float_precision
                )
                for metric_name, metric in self.metrics.items()
            ]
        )


def save_checkpoint(model, optimizer, epoch, score, filename):
    """
    Saves the model checkpoint.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "best_score": score,
    }
    torch.save(state, filename)


def load_checkpoint(model, filename, optimizer=None, device=Config.DEVICE):
    """
    Loads the model checkpoint.
    Returns the epoch and best_score stored in the checkpoint.
    """
    if not os.path.isfile(filename):
        return 0, 0.0

    checkpoint = torch.load(filename, map_location=device)

    # Load model weights
    model.load_state_dict(checkpoint["state_dict"])

    # Load optimizer state if provided
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    epoch = checkpoint.get("epoch", 0)
    best_score = checkpoint.get("best_score", 0.0)

    return epoch, best_score
