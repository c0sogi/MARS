import os
import random
import shutil
import numpy as np
import torch
from collections import defaultdict
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    Sets seeds for random, os, numpy, and torch.
    Enforces deterministic CuDNN backend.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for full reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MetricMonitor:
    """
    A utility class to track and average metrics (loss, MAE, etc.) during training/validation.
    """

    def __init__(self, float_precision=None):
        """
        Args:
            float_precision (int, optional): Number of decimal places to format.
                                             If None, prints full precision.
        """
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        """Resets the internal metric storage."""
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val, n=1):
        """
        Update the metric with a new value.

        Args:
            metric_name (str): Name of the metric.
            val (float): The value to add (e.g., batch loss).
            n (int): The number of samples associated with this value (e.g., batch size).
        """
        metric = self.metrics[metric_name]
        metric["val"] += val * n
        metric["count"] += n
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        """
        Returns a string representation of the averages of all tracked metrics.
        Prints full precision if float_precision is None.
        """
        return " | ".join(
            [
                (
                    "{}: {:.{prec}f}".format(
                        metric_name, metric["avg"], prec=self.float_precision
                    )
                    if self.float_precision is not None
                    else "{}: {}".format(metric_name, metric["avg"])
                )
                for metric_name, metric in self.metrics.items()
            ]
        )

    @property
    def avg_metrics(self):
        """Returns a dictionary of current average values for all metrics."""
        return {k: v["avg"] for k, v in self.metrics.items()}


def save_checkpoint(
    state, is_best, checkpoint_dir=None, best_model_path=None, filename="checkpoint.pth"
):
    """
    Saves a model checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        checkpoint_dir (str): Directory to save the checkpoint. Defaults to Config.CHECKPOINT_DIR.
        best_model_path (str): Path to save the best model. Defaults to Config.BEST_MODEL_PATH.
        filename (str): Filename for the standard checkpoint.
    """
    if checkpoint_dir is None:
        checkpoint_dir = Config.CHECKPOINT_DIR

    if best_model_path is None:
        best_model_path = Config.BEST_MODEL_PATH

    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        shutil.copyfile(filepath, best_model_path)


def load_checkpoint(
    model, checkpoint_path, optimizer=None, scheduler=None, device=Config.DEVICE
):
    """
    Loads a model checkpoint.

    Args:
        model (torch.nn.Module): The model to load weights into.
        checkpoint_path (str): Path to the checkpoint file.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler to load state into.
        device (str): Device to map the location to.

    Returns:
        checkpoint (dict): The loaded checkpoint dictionary, or None if file not found.
    """
    if not os.path.exists(checkpoint_path):
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle state dict (remove 'module.' prefix if saved from DataParallel)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint
