import os
import random
import copy
import numpy as np
import torch
from collections import defaultdict


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


class MetricMonitor:
    """
    A utility class to track metrics (like Loss, Accuracy, AUC) during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the metrics."""
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        """
        Updates the metric with a new value.

        Args:
            metric_name (str): Name of the metric.
            val (float): Value to update.
        """
        metric = self.metrics[metric_name]
        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        """
        Returns a string representation of the metrics with full precision.
        """
        return " | ".join(
            [
                "{}: {}".format(metric_name, metric["avg"])
                for metric_name, metric in self.metrics.items()
            ]
        )


class ModelEMA:
    """
    Implements Exponential Moving Average (EMA) of model weights.
    """

    def __init__(self, model, decay=0.9999, device=None):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device:
            self.module.to(device)

    def update(self, model):
        """
        Update the EMA model parameters.

        Args:
            model (nn.Module): The current training model.
        """
        with torch.no_grad():
            # Update parameters
            for ema_v, model_v in zip(self.module.parameters(), model.parameters()):
                if self.device:
                    model_v = model_v.to(self.device)
                ema_v.data.mul_(self.decay).add_(model_v.data, alpha=1 - self.decay)

            # Update buffers (e.g. BatchNorm running mean/var)
            for ema_v, model_v in zip(self.module.buffers(), model.buffers()):
                if self.device:
                    model_v = model_v.to(self.device)
                ema_v.data.copy_(model_v.data)


def save_checkpoint(model, optimizer, scheduler, epoch, score, path):
    """
    Saves the model checkpoint.
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "score": score,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    """
    Loads a model checkpoint.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if (
        scheduler
        and "scheduler_state_dict" in checkpoint
        and checkpoint["scheduler_state_dict"]
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint.get("epoch", 0), checkpoint.get("score", 0.0)


class EarlyStopping:
    """
    Early stopping to stop the training when the metric does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, mode="max", min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score = np.inf if mode == "min" else -np.inf

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == "min":
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == "max"
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
