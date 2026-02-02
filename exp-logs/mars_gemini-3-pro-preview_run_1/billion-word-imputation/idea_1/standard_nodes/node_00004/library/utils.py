import os
import sys
import logging
import torch
import numpy as np
from library.config import Config, set_seed

# Configure logging to output to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def get_device():
    """
    Returns the PyTorch device as defined in the configuration.
    """
    return torch.device(Config.DEVICE)


def save_checkpoint(model, optimizer, epoch, loss, filename, scheduler=None):
    """
    Saves the model checkpoint, including model state, optimizer state, epoch, and loss.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): Current epoch.
        loss (float): Current loss.
        filename (str): Path to save the checkpoint.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler state.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None, scheduler=None, device=None):
    """
    Loads a model checkpoint.

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load state into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler to load state into.
        device (torch.device, optional): Device to map the location to.

    Returns:
        dict: The loaded checkpoint dictionary if successful, None otherwise.
    """
    if device is None:
        device = get_device()

    if not os.path.exists(filename):
        return None

    checkpoint = torch.load(filename, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def log_metrics(metrics, prefix=""):
    """
    Logs a dictionary of metrics to stdout with full precision.

    Args:
        metrics (dict): Dictionary of metric names and values.
        prefix (str, optional): Prefix string for the log message.
    """
    parts = []
    if prefix:
        parts.append(f"[{prefix}]")

    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            v = v.item()

        # Print full precision for floats
        if isinstance(v, (float, np.floating)):
            parts.append(f"{k}: {v}")
        else:
            parts.append(f"{k}: {v}")

    logger.info(" ".join(parts))


class MetricTracker:
    """
    A utility class to track and average metrics over a training epoch.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {}
        self.counts = {}

    def update(self, metrics, n=1):
        """
        Update metrics with a new batch of values.

        Args:
            metrics (dict): Dictionary of metric names and values.
            n (int): Number of samples in the batch (weight).
        """
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if k not in self.metrics:
                self.metrics[k] = 0.0
                self.counts[k] = 0

            self.metrics[k] += v * n
            self.counts[k] += n

    def get_averages(self):
        """
        Returns the average of all tracked metrics.
        """
        return {
            k: v / self.counts[k] for k, v in self.metrics.items() if self.counts[k] > 0
        }
