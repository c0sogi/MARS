import os
import shutil
import torch
from collections import defaultdict
from library.config import set_seed


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility by wrapping the configuration's set_seed.

    Args:
        seed (int): The seed value to set.
    """
    set_seed(seed)


class MetricMonitor:
    """
    A class to track and average metrics (like loss) during training.
    Maintains a running average for each metric updated.
    """

    def __init__(self, float_precision=4):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        """
        Resets all metrics to their initial state.
        """
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        """
        Update the metric with a new value.

        Args:
            metric_name (str): The name of the metric (e.g., 'Loss', 'Accuracy').
            val (float): The value to update (typically the average of a batch).
        """
        metric = self.metrics[metric_name]

        metric["val"] = val
        metric["count"] += 1
        # Compute running average: new_avg = old_avg + (val - old_avg) / count
        metric["avg"] = metric["avg"] + (val - metric["avg"]) / metric["count"]

    def __str__(self):
        """
        Returns a formatted string of the current average metrics.
        """
        return " | ".join(
            [
                "{}: {:.{prec}f}".format(
                    metric_name, metric["avg"], prec=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pth"):
    """
    Saves the model state to a checkpoint file.

    Args:
        state (dict): The state dictionary to save (model, optimizer, epoch, etc.).
        is_best (bool): Whether this checkpoint represents the best model so far.
        checkpoint_dir (str): Directory to save the checkpoint.
        filename (str): Name of the checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        shutil.copyfile(filepath, best_path)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Loads the model state (and optimizer state if provided) from a checkpoint file.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Load to CPU first to avoid mapping issues if GPU configuration differs
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Load model state
    # Checks for common state dict keys
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Assume the checkpoint itself is the state dict
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided and present in checkpoint
    if optimizer is not None:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        elif "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
