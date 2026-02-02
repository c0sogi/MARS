import os
import random
import numpy as np
import torch
from copy import deepcopy
from sklearn.metrics import f1_score


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_f1_score(y_true, y_pred, threshold=0.5, average="macro"):
    """
    Calculates the F1 score for multi-label classification.

    Args:
        y_true: Ground truth labels (numpy array).
        y_pred: Predicted probabilities (numpy array).
        threshold: Threshold for converting probabilities to binary labels.
        average: The type of averaging performed on the data (default: 'macro').

    Returns:
        float: The calculated F1 score.
    """
    # Convert probabilities to binary predictions
    y_pred_bin = (y_pred > threshold).astype(int)
    return f1_score(y_true, y_pred_bin, average=average)


class MetricMonitor:
    """
    A class to track and average metrics during training/validation.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the metric storage."""
        self.metrics = {}

    def update(self, metric_name, val):
        """
        Updates the metric with a new value.

        Args:
            metric_name (str): Name of the metric.
            val (float or torch.Tensor): Value to add.
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = {"count": 0, "sum": 0.0}

        if isinstance(val, torch.Tensor):
            val = val.item()

        self.metrics[metric_name]["count"] += 1
        self.metrics[metric_name]["sum"] += val

    def get_avg(self, metric_name):
        """Returns the average value of the metric."""
        return self.metrics[metric_name]["sum"] / self.metrics[metric_name]["count"]

    def __str__(self):
        """
        Returns a string representation of the metrics with full precision.
        """
        return " | ".join(
            [
                "{}: {}".format(metric_name, self.get_avg(metric_name))
                for metric_name in self.metrics
            ]
        )


class ModelEMA:
    """
    Implements Exponential Moving Average (EMA) of model parameters.
    """

    def __init__(self, model, decay=0.999, device=None):
        """
        Args:
            model: The model to track.
            decay: The decay factor for EMA.
            device: The device to store the shadow model on.
        """
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device:
            self.module.to(device)

    def update(self, model):
        """
        Updates the shadow model parameters using the current model parameters.

        Args:
            model: The current training model.
        """
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.module.state_dict().items():
                if k in msd:
                    # Update: shadow = decay * shadow + (1 - decay) * new
                    v.copy_(self.decay * v + (1.0 - self.decay) * msd[k])
